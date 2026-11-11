# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# controllers/stocsy_analytes.py
"""
The STOCSY window by analyte, without Tk.

The window audits a quantification. An analyte's reference is, by default,
what the quantification measured for it -- the per-sample areas it reported,
whatever the mode (one multiplet integrated, a shape fitted, the cascade's
joint fit) -- with the protons it divided them by. The analyte is then read
through the signals its listing expects: the ones the annotation matched to
detected peaks (with the protons each holds) and the ones it did not find,
which stay in the list as not detected. A signal is the sum of the detected
peaks that are its lines -- a doublet is two peaks -- so it is compared whole,
never by one line.

Pick a signal and every detected peak around it is read as a possible line of
it, with the share of the signal's protons that line holds
(``processing.signal_comparison.line_shares``): a line of the analyte gives
the reference's quantification, another compound's line does not. The lines
ticked make up the signal, compared whole. Nothing here changes the project:
the analyst's choice is written by ``controllers.signal_lines``.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.
"""
import numpy as np

from controllers.stocsy_controller import StocsyAnalysis
from processing.annotation_scoring import PATTERN_MULT
from processing.multiplet_line_selector import (
    LINE_MATCH_FLOOR, line_match_tolerance, multiplet_line_peaks)
from controllers.analysis_defaults import QUANTIFICATION
from processing.quant_engine import measure_signal
from processing.quant_integration import MIN_TRUSTED_CAPTURE, pattern_evidence
from processing.signal_comparison import compare_to_reference, pattern_share, per_proton
from processing.signal_lines import manual_lines
from processing.spectral_regions import FALLBACK_LINEWIDTH_PPM, measure_linewidth_ppm
from processing.stocsy import SKIP_REASONS
from processing.stocsy_evidence import (
    STOCSY_LIFT_ALPHA, STOCSY_LIFT_MIN_PAIRS, STOCSY_LIFT_R, own_signal_pairs)

# A db signal and a matched one are the same when their listed δ agree to this.
_SAME_DB_PPM = 1e-3


class Target:
    """What is compared: a signal of an analyte (``kind='signal'``, the sum of
    its line peaks) or one detected peak (``kind='peak'``). ``vector`` is None
    when nothing of it can be read across the samples (``why`` says what)."""

    def __init__(self, kind, label, ppm, ranks, vector, sigma, protons, snr=0.0,
                 why='', anchor=None):
        self.kind, self.label, self.ppm = kind, label, ppm
        self.ranks, self.vector, self.sigma = list(ranks), vector, sigma
        self.protons, self.snr, self.why, self.anchor = protons, snr, why, anchor


class AnalyteBrowser:
    """The analytes of a study over a ``StocsyAnalysis``.

    ``metabolites`` are the annotation's dicts (``matched_peaks``,
    ``db_peaks``); ``quant_rows`` the quantification's, for the dots and
    the reference. ``measure(peaks, evidence, local_baseline)`` gives a
    signal's ``(areas, sd, why)``; by default the quantification's own
    measurement, so every number is on the scale of the reference."""

    def __init__(self, analysis: StocsyAnalysis, metabolites=None, quant_rows=None, *,
                 measure=None):
        self.an = analysis
        self._mets = [m for m in metabolites or () if m.get('matched_peaks')]
        self._by_id = {m.get('id'): m for m in self._mets}
        self._quant = {r.get('hmdb_id'): r for r in quant_rows or ()
                       if not r.get('_is_is') and not r.get('_not_quantified')}
        self._dots = {k: int(r.get('_confidence') or 0) for k, r in self._quant.items()}
        self._signals = {}
        self._summary = {}
        self._measured = {}
        self._lw = None
        self.measure = measure or self._engine_measure

    def reset(self):
        """Forget what depends on the correlation settings (the mean r and the
        pairs of each analyte); the signals do not."""
        self._summary.clear()

    def forget(self, met_id=None):
        """Read the analyte's signals again (its lines or verdict were
        changed); every analyte's, without one."""
        if met_id is None:
            self._signals.clear()
            self._summary.clear()
            return
        self._signals.pop(met_id, None)
        self._summary.pop(met_id, None)

    def metabolite(self, met_id) -> dict:
        return self._by_id.get(met_id) or {}

    @property
    def has_annotation(self) -> bool:
        return bool(self._mets)

    @property
    def has_quantification(self) -> bool:
        return bool(self._dots)

    # ── analytes ─────────────────────────────────────────────────────────────

    def analytes(self, *, only_quantified=False, text='') -> list:
        """One row per analyte -- ``id, name, dots, detected, expected, pairs,
        mean_r, score`` -- the quantified ones first (most dots first), then by
        annotation score."""
        text = (text or '').strip().lower()
        out = []
        for m in self._mets:
            if only_quantified and m['id'] not in self._dots:
                continue
            if text and text not in str(m.get('name', '')).lower():
                continue
            out.append({'id': m['id'], 'name': m.get('name') or m['id'],
                        'dots': self._dots.get(m['id'], 0),
                        'score': float(m.get('composite_score') or 0.0),
                        **self._summarise(m['id'])})
        out.sort(key=lambda d: (-d['dots'], -d['score'], d['name']))
        return out

    def _summarise(self, met_id) -> dict:
        """Signals found, and the pairs among the anchor peaks of the signals
        that co-vary: the count the confidence-dot rule reads
        (``processing.stocsy_evidence``)."""
        if met_id not in self._summary:
            rows = self.signals(met_id)
            anchors = []
            for s in rows:
                if s['anchor_usable'] and s['rank'] not in anchors:
                    anchors.append(s['rank'])
            mean_r, pairs = np.nan, 0
            ranks, R, P = self.an.matrix(anchors)
            if len(ranks) >= 2:
                iu = np.triu_indices(len(ranks), 1)
                fin = R[iu][np.isfinite(R[iu])]
                mean_r = float(fin.mean()) if fin.size else np.nan
                pairs = own_signal_pairs([{'peak_rank': r} for r in ranks],
                                         {r: i for i, r in enumerate(ranks)}, R, P)
            self._summary[met_id] = {
                'detected': sum(1 for s in rows if s['detected']),
                'expected': len(rows), 'pairs': pairs, 'mean_r': mean_r}
        return self._summary[met_id]

    @staticmethod
    def lift_rule() -> str:
        return (f'≥ {STOCSY_LIFT_MIN_PAIRS} pairs with r ≥ {STOCSY_LIFT_R:g} '
                f'and p < {STOCSY_LIFT_ALPHA:g}')

    # ── the signals of one analyte ───────────────────────────────────────────

    def signals(self, met_id) -> list:
        """The signals the listing expects, high δ first: ``rank`` (the
        annotation's anchor peak, None when not detected), ``lines`` (the
        detected peaks that are its lines and can be read), ``db_ppm``, ``ppm``
        (where its lines stand), ``protons``, ``type``, ``quality`` (the
        annotation's verdict), ``snr`` (of its clearest line), ``detected``,
        ``usable`` (some line varies across the samples), ``anchor_usable``,
        ``why`` (what keeps an unusable one out), ``evidence`` (its predicted
        lines and their weights, as the quantification reads them:
        ``pattern_evidence``), ``manual`` (its lines were chosen by the
        analyst), ``ppm_diff`` and ``delta_j_hz`` (the annotation's Δδ and ΔJ;
        NaN where it has none) and ``mp_index`` (its place among the annotation's matched peaks; None for
        a listed signal the annotation did not match)."""
        if met_id in self._signals:
            return self._signals[met_id]
        met = self._by_id.get(met_id) or {}
        rows, seen = [], []
        for k, mp in enumerate(met.get('matched_peaks') or ()):
            rank = mp.get('peak_rank')
            rank = int(rank) if rank is not None else None
            region = self._pattern_region(mp)
            found = ([r for r in sorted(self.an.peaks, key=lambda r: -self.an.ppm_of(r))
                      if region[1] <= self.an.ppm_of(r) <= region[0]] if region else
                     [int(p['rank']) for p in multiplet_line_peaks(mp, self.an.peaks)
                      if p.get('rank') is not None])
            stored = len(found)
            found = self._letter_lines(mp, found) if region is None else found
            lines = [r for r in found if self.an.can_seed(r)]
            seen.append(float(mp.get('db_ppm', np.nan)))
            peaks = [self.an.peaks[r] for r in (lines or found) if r in self.an.peaks]
            rows.append({
                'rank': rank, 'lines': lines, 'db_ppm': float(mp.get('db_ppm', np.nan)),
                'ppm': (float(np.mean([p['ppm'] for p in peaks])) if peaks
                        else float(mp.get('peak_ppm', np.nan))),
                'protons': _count(mp.get('protons')), 'type': mp.get('type') or '',
                'quality': mp.get('annotation_quality') or '',
                'snr': max((float(p.get('snr') or 0.0) for p in peaks), default=np.nan),
                'detected': bool(peaks), 'usable': bool(lines),
                'anchor_usable': rank is not None and self.an.can_seed(rank),
                'why': '' if lines else self._why(found[0] if found else rank),
                'evidence': pattern_evidence(mp), 'met_id': met_id,
                'ppm_diff': _num(mp.get('ppm_diff')), 'delta_j_hz': _num(mp.get('delta_j_hz')),
                'manual': manual_lines(mp) is not None, 'mp_index': k, 'region': region,
                'stored_lines': stored if stored > len(found) else None})
        for dp in met.get('db_peaks') or ():
            ppm = float(dp.get('ppm_center', dp.get('ppm_start', np.nan)))
            if any(abs(ppm - s) <= _SAME_DB_PPM for s in seen):
                continue
            rows.append({'rank': None, 'lines': [], 'db_ppm': ppm, 'ppm': np.nan,
                         'protons': _count(dp.get('protons')), 'type': dp.get('type') or '',
                         'quality': '', 'snr': np.nan, 'detected': False, 'usable': False,
                         'anchor_usable': False, 'why': 'not detected', 'evidence': {},
                         'met_id': met_id,
                         'ppm_diff': np.nan, 'delta_j_hz': np.nan,
                         'manual': False, 'mp_index': None, 'region': None, 'stored_lines': None})
        rows.sort(key=lambda s: -s['db_ppm'] if np.isfinite(s['db_ppm']) else 0.0)
        self._signals[met_id] = rows
        return rows

    @staticmethod
    def _pattern_region(mp):
        """(high, low) ppm of a complex multiplet (``m*``): the span its
        measured pattern's lines cover, plus the tightest line tolerance on
        each side. A pattern is read as a region, every peak in it a line, as
        the quantification measures it: line by line, its close lines fall on
        one apex and it shrank to one or two peaks. None for a letter, and
        for one the analyst gave its lines by hand."""
        if (mp.get('type') or '').strip() != PATTERN_MULT or manual_lines(mp) is not None:
            return None
        ppms = [float(l['ppm']) for l in mp.get('predicted_lines') or () if l.get('ppm') is not None]
        if not ppms:
            return None
        return max(ppms) + LINE_MATCH_FLOOR, min(ppms) - LINE_MATCH_FLOOR

    def _letter_lines(self, mp, found):
        """No more lines than the letter has: the ones nearest where the
        signal stands, high δ first. In a crowded window the annotation can
        keep every peak around the anchor as a line (taurine's triplet at
        3.25 ppm on the 100-analyte bench tube kept five). Only shown so here:
        capping them in the annotation itself moved the election of the whole
        tube and raised its false identifications (ADR-102)."""
        n = mp.get('_n_exp_lines')
        if not n or len(found) <= int(n) or manual_lines(mp) is not None:
            return found
        at = float(mp.get('peak_ppm') or mp.get('db_ppm') or 0.0)
        near = sorted(found, key=lambda r: abs(self.an.ppm_of(r) - at))[:int(n)]
        return sorted(near, key=lambda r: -self.an.ppm_of(r))

    def _why(self, rank) -> str:
        if rank is None or rank not in self.an.peaks:
            return 'not detected'
        key = self.an.skipped.get(rank)
        return SKIP_REASONS.get(key, key) if key else 'no per-sample areas'

    def default_reference(self, met_id):
        """Index (in ``signals``) of the analyte's clearest signal: a singlet
        when it has one, which no neighbouring line can disturb, by SNR; the one
        with more protons on a tie. None when none can be read."""
        best = None
        for i, s in enumerate(self.signals(met_id)):
            if s['usable']:
                key = (s['type'] == 's', s['snr'], s['protons'] or 0)
                if best is None or key > best[0]:
                    best = (key, i)
        return best[1] if best else None

    def quantification(self, met_id) -> dict | None:
        """The quantification's row for the analyte (None: not quantified)."""
        return self._quant.get(met_id)

    def quant_target(self, met_id):
        """What the quantification measured for the analyte, as a target: the
        per-sample areas it reported and the protons it divided them by. None
        when the analyte was not quantified or its areas are not kept."""
        row = self._quant.get(met_id)
        if not row:
            return None
        areas = np.asarray(row.get('_areas_per_sample') or (), dtype=float)
        nh = _count(row.get('_nh'))
        if not areas.size or not nh:
            return None
        cols = [int(i) for i in row.get('_included') or ()]
        by_sample = (dict(zip(cols, areas)) if len(cols) == areas.size
                     else dict(enumerate(areas)))
        vec = np.array([by_sample.get(i, np.nan) for i in self.an.included], dtype=float)
        if not np.isfinite(vec).any() or np.nanstd(vec) <= 0:
            return None
        lines = [float(x) for x in row.get('_line_ppms') or ()]
        win = row.get('_multiplet_window')
        width = (abs(float(win[1]) - float(win[0])) if win and len(win) == 2
                 else 0.01 * max(len(lines), 1))
        mode = row.get('_integ_mode') or 'quantified'
        label = (f"quantification: {mode} {row.get('_multiplicity') or ''} "
                 f"δ {float(row.get('ppm') or np.nan):.3f}").replace('  ', ' ')
        return Target('quant', label, float(row.get('ppm') or np.nan), [], vec,
                      self.an.window_sigma(width), nh, float(row.get('snr') or 0.0))

    # ── what is compared ─────────────────────────────────────────────────────

    @property
    def linewidth_ppm(self) -> float:
        """The study's linewidth, measured as the quantification measures it
        (``measure_linewidth_ppm``, the same fallback)."""
        if self._lw is None:
            self._lw = float(measure_linewidth_ppm(
                self.an.ppm, self.an.X, list(self.an.peaks.values())) or FALLBACK_LINEWIDTH_PPM)
        return self._lw

    def _engine_measure(self, peaks, evidence, local_baseline):
        """``(per-sample areas, sd, why)`` of a signal made of ``peaks``,
        measured by the quantification's own function (``measure_signal``:
        its window, its baseline, the tails a window leaves out, a partial
        pattern scaled to the whole or refused). The areas of the detection
        are not on that scale -- integrated foot to foot they held 0.40-0.58
        of what the quantification measured on clean singlets (100-analyte
        bench tube) -- so a ratio between the two would be off before any
        overlap. ``why`` says why there are no areas."""
        key = (tuple(sorted(int(p['rank']) for p in peaks)),
               tuple(evidence.get('predicted_ppms') or ()),
               tuple(evidence.get('line_weights') or ()), bool(local_baseline))
        if key not in self._measured:
            n = self.an.X.shape[1]
            info, why = measure_signal(
                list(peaks), evidence, ppm=self.an.ppm, intensities=self.an.X,
                included=list(range(n)), local_baseline=bool(local_baseline),
                align=QUANTIFICATION['align'], blank_indices=[],
                linewidth_ppm=self.linewidth_ppm, other_peaks=list(self.an.peaks.values()),
                noise_sigma=self.an.point_noise if self.an.ranks else None)
            if why == 'partial':
                found, expected, share, _missing = info
                res = (None, np.nan, f'{found} of {expected} lines, {100 * share:.0f}% of the '
                                     f'pattern: under the {100 * MIN_TRUSTED_CAPTURE:.0f}% the '
                                     'quantification scales to the whole signal')
            elif why or not info or not info.get('areas'):
                res = (None, np.nan, 'no window the quantification could measure')
            else:
                vec = np.array([info['areas'].get(i, np.nan) for i in range(n)], dtype=float)
                lo, hi = info['window']
                sd = self.an.window_sigma(abs(hi - lo)) / float(info.get('captured') or 1.0)
                res = (vec, sd, '')
            self._measured[key] = res
        return self._measured[key]

    def _local_baseline(self, signal):
        row = self._quant.get(signal.get('met_id')) or {}
        return row.get('_local_baseline', QUANTIFICATION['local_baseline'])

    def signal_target(self, met_id, index):
        """The ``index``-th signal of the analyte as a target: its readable
        lines measured together. None when it has no readable line."""
        s = self.signals(met_id)[index]
        return self.lines_target(s, s['lines'])

    def lines_target(self, signal, ranks, label=None):
        """The detected peaks ``ranks`` measured as the signal (its pattern,
        its protons). None when none of them can be read; a stub with ``why``
        when the quantification would not measure them."""
        ranks = [r for r in ranks if self.an.can_seed(r)]
        if not ranks:
            return None
        ppm = float(np.mean([self.an.ppm_of(r) for r in ranks]))
        label = label or (f"{signal['type'] or 'signal'} δ {ppm:.3f}"
                          + (f" ({len(ranks)} lines)" if len(ranks) > 1 else ''))
        vec, sd, why = self.measure([self.an.peaks[r] for r in ranks],
                                    dict(signal.get('evidence') or {}, protons=signal['protons']),
                                    self._local_baseline(signal))
        return Target('signal', label, ppm, ranks, vec, sd, signal['protons'],
                      max(float(self.an.peaks[r].get('snr') or 0.0) for r in ranks),
                      why=why, anchor=signal['rank'])

    def peak_target(self, rank, protons=None, signal=None):
        """One detected peak as a target, measured as one line (a stub with
        ``why`` when it cannot be read across the samples)."""
        p = self.an.peaks[rank]
        ppm, snr = float(p['ppm']), float(p.get('snr') or 0.0)
        if not self.an.can_seed(rank):
            return Target('peak', f'peak #{rank}', ppm, [rank], None, np.nan, protons, snr,
                          why=self._why(rank), anchor=rank)
        vec, sd, why = self.measure([p], {'protons': protons, 'predicted_ppms': [ppm]},
                                    self._local_baseline(signal or {}))
        return Target('peak', f'peak #{rank}', ppm, [rank], vec, sd, protons, snr,
                      why=why, anchor=rank)

    def peaks_around(self, ppm, half_width) -> list:
        """Every detected peak within ``half_width`` ppm of ``ppm``, high δ
        first."""
        return sorted((r for r, p in self.an.peaks.items()
                       if abs(float(p['ppm']) - ppm) <= half_width),
                      key=lambda r: -self.an.ppm_of(r))

    # ── numbers against the reference ────────────────────────────────────────

    def compare(self, ref: Target, targets, *, current=()) -> list:
        """One row per target against the reference: ``label, kind, rank (the
        anchor), ranks, ppm, snr, protons, area_mean, r, p, ratio, sd, n, gap,
        distance, is_ref, current, assigned, usable, why, target``. A target
        that cannot be read keeps its row, with NaN where it has no number."""
        current = set(current)
        read = [t for t in targets if t.vector is not None]
        stats = {}
        if read:
            V = np.vstack([t.vector for t in read])
            n = self.an.M.shape[1]
            res = compare_to_reference(
                ref.vector, V, ref.sigma, [t.sigma for t in read], ref.protons,
                [t.protons for t in read], min_valid=min(5, max(n - 1, 1)))
            r = self.an.basis.correlate(ref.vector[None, :], V)[0]
            p = self.an.basis.stats(r)[0]
            for i, t in enumerate(read):
                stats[id(t)] = {**res[i], 'r': float(r[i]), 'p': float(p[i])}
        nan = {k: np.nan for k in ('ratio', 'sd', 'gap', 'distance', 'r', 'p')}
        out = []
        for t in targets:
            st = stats.get(id(t), {**nan, 'n': 0})
            names = {self._assigned_name(r) for r in t.ranks} - {''}
            area = (float(np.nanmean(t.vector)) if t.vector is not None
                    and np.isfinite(t.vector).any() else np.nan)
            out.append({
                'label': t.label, 'kind': t.kind, 'rank': t.anchor, 'ranks': t.ranks,
                'ppm': t.ppm, 'snr': t.snr, 'protons': t.protons, 'area_mean': area,
                'is_ref': t is ref, 'usable': t.vector is not None, 'why': t.why,
                'current': t is ref or bool(current & set(t.ranks)),
                'assigned': ', '.join(sorted(names)), 'target': t,
                **{k: st[k] for k in ('r', 'p', 'ratio', 'sd', 'n', 'gap', 'distance')}})
            if t is ref:
                out[-1].update({'r': np.nan, 'p': np.nan, 'ratio': 1.0, 'sd': 0.0,
                                'gap': 0.0, 'distance': 0.0})
        return out

    def _assigned_name(self, rank) -> str:
        return str(self.an.peaks[rank].get('assigned_metabolite') or '')

    def signals_vs(self, met_id, ref_index, ref=None) -> list:
        """``(index, row)`` for every signal of the analyte against its
        ``ref_index``-th, in listing order; the ones that cannot be read have
        no row (None). ``ref`` is that signal's target when the caller changed
        it (the protons typed by the analyst)."""
        ref = ref or self.signal_target(met_id, ref_index)
        if ref is None:
            return []
        rows = []
        for i, _s in enumerate(self.signals(met_id)):
            t = ref if i == ref_index else self.signal_target(met_id, i)
            rows.append((i, self.compare(ref, [t])[0] if t is not None else None))
        return rows

    def pattern_share(self, signal, ranks):
        """The fraction of the signal's pattern the peaks ``ranks`` hold, as
        the quantification weighs it (``signal_comparison.pattern_share``);
        None when none of them stands on a predicted line."""
        pred = [float(v) for v in (signal.get('evidence') or {}).get('predicted_ppms') or ()
                if v is not None]
        if not pred or not ranks:
            return None
        return pattern_share(signal['evidence'], [self.an.ppm_of(r) for r in ranks],
                             line_match_tolerance(pred))

    def peak_signal(self, rank, protons=None) -> dict:
        """A detected peak as a signal of unknown multiplicity, for reading
        without an analyte (``protons`` typed by the analyst)."""
        return {'rank': rank, 'lines': [rank] if self.an.can_seed(rank) else [],
                'ppm': self.an.ppm_of(rank), 'db_ppm': self.an.ppm_of(rank),
                'protons': protons, 'type': '', 'evidence': {}, 'manual': False,
                'ppm_diff': np.nan, 'delta_j_hz': np.nan,
                'mp_index': None, 'region': None, 'stored_lines': None}

    def line_rows(self, ref: Target, signal, ticked, half_width) -> list:
        """``signal`` (a row of ``signals`` or ``peak_signal``) read line by
        line against the reference. First the signal the ticked peaks make
        (``kind 'whole'``, its protons), then every detected peak within
        ``half_width`` of the signal, high δ first, each with the protons its
        share of the pattern gives (``pattern_share``; None where no share is
        known, and then no gap). Each row also says ``ticked`` and ``share``;
        the whole row's ``share`` is the part of the pattern the ticked lines
        hold, which the quantification needs at MIN_TRUSTED_CAPTURE or more."""
        s = signal
        centre = s['ppm'] if np.isfinite(s['ppm']) else s['db_ppm']
        ticked = [int(r) for r in ticked]
        peaks = self.peaks_around(centre, half_width)
        targets, shares = [], []
        for r in peaks:
            share = self.pattern_share(s, [r])
            shares.append(share)
            protons = share * s['protons'] if share is not None and s['protons'] else None
            targets.append(self.peak_target(r, protons, s))
        whole = self.lines_target(s, ticked, label=f'ticked lines ({len(ticked)})')
        rows = self.compare(ref, ([whole] if whole is not None else []) + targets,
                            current=ticked)
        out = []
        if whole is not None:
            rows[0].update({'ticked': True, 'share': self.pattern_share(s, ticked), 'kind': 'whole'})
            out.append(rows.pop(0))
        for row, share in zip(rows, shares):
            row.update({'ticked': row['rank'] in ticked, 'share': share})
            out.append(row)
        return out

    def matrix(self, targets):
        """r and p among the targets that can be read: ``(labels, R, P)``."""
        read = [t for t in targets if t.vector is not None]
        if len(read) < 2:
            return [], np.empty((0, 0)), np.empty((0, 0))
        V = np.vstack([t.vector for t in read])
        R = self.an.basis.correlate(V, V)
        P = np.vstack([self.an.basis.stats(row)[0] for row in R])
        return [t.label for t in read], R, P

    # ── the numbers sample by sample ─────────────────────────────────────────

    def sample_order(self) -> list:
        """Sample positions grouped, in the order the groups first appear."""
        first = {}
        for i, g in enumerate(self.an.groups):
            first.setdefault(g, i)
        return sorted(range(len(self.an.groups)),
                      key=lambda i: (first[self.an.groups[i]], i))

    def grid(self, rows, *, per='area') -> dict:
        """The numbers of ``rows`` (``compare`` rows) in each sample, samples
        grouped; ``per='proton'`` divides each by its protons. Returns
        ``names``, ``groups`` and ``values`` (one list per row, NaN where the
        row has no vector or, per proton, no protons)."""
        order = self.sample_order()
        values = []
        for r in rows:
            v = r['target'].vector
            if v is None:
                vals = np.full(len(order), np.nan)
            elif per == 'proton':
                vals = per_proton(v, r['protons']) if r['protons'] else np.full(len(order), np.nan)
            else:
                vals = np.asarray(v, dtype=float)
            values.append([float(vals[i]) for i in order])
        return {'names': [self.an.sample_names[i] for i in order],
                'groups': [self.an.groups[i] for i in order], 'values': values}

    def export_rows(self, rows, ref: Target, *, per='area') -> list:
        """A table of ``rows`` against the reference with the numbers of every
        sample, for a file."""
        g = self.grid(rows, per=per)
        out = []
        for r, vals in zip(rows, g['values']):
            row = {'reference': ref.label, 'candidate': r['label'], 'peak': r['rank'] or '',
                   'in_signal': bool(r.get('ticked')),
                   'line_share': '' if r.get('share') is None else r['share'],
                   'ppm': round(r['ppm'], 5),
                   'snr': round(r['snr'], 1) if np.isfinite(r['snr']) else '',
                   'protons': r['protons'] or '', 'mean_area': r['area_mean'], 'r': r['r'],
                   'p': r['p'], 'ratio_to_reference': r['ratio'], 'sd_log_ratio': r['sd'],
                   'gap_to_reference': r['gap'], 'distance': r['distance'],
                   'annotated_as': r['assigned']}
            row.update({f'{n} ({gr})': v for n, gr, v in zip(g['names'], g['groups'], vals)})
            out.append(row)
        return out


def _count(v):
    """Protons of a listed signal: a positive number or None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
