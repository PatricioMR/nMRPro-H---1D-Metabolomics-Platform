# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/signal_links.py
"""
Whether the rest of an analyte's listing moves with what was measured for it.

One molecule has one amount in each sample, so every signal it lists rises and
falls with that amount, in the proportion its protons call for. Once a study is
quantified this can be asked of every analyte in the table, with the amounts just
measured as the reference:

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

    Wei, S. et al. (2011). Ratio analysis nuclear magnetic resonance
    spectroscopy for selective metabolite identification in complex samples.
    Analytical Chemistry 83(20):7616-7623.

Three things the measurement forced (tubes whose answer is known; vault
STOCSY-singleto-robusto-ao-deslocamento):

* Each sample's value is the peak's apex wherever it stands near its position,
  not the height at one point of the axis. A peak moves between samples by a
  fraction of its width (1.2 mppm against a half-width of 2.2), and at a fixed
  point that moves the height more than the amount does: clean singlets gave
  r 0.12-0.78 with their true amount at a fixed point, 0.94-1.00 at the apex.
* Co-varying says "the same molecule", not "this molecule": a peak of another
  compound that shares a signal with this one can co-vary too. The protons tell
  them apart, so the ratio has to agree as well.
* A window of several peaks gives chance several tries: the critical r is
  corrected for how many were tried.

Correlation is within the sample groups, so that two compounds that merely
differ between the groups do not read as one molecule.

Why after the quantification and not before it (vault ADR-100). Handed to the
annotation, the same test made a Good verdict the election and the study's own
position band read: a few singlets moved the band 0.04 mppm and 17 analytes they
had nothing to do with entered or left the table. And the choice has little to
give there: with every true singlet on its true peak and Good, the table found no
more analytes and held more false ones. So nothing here decides what is found.
It says what the table's own rows are worth:

* A row at two dots, whose identity is what holds it there, with another signal
  of its listing linked is raised to three. Tube of 100, 600 MHz, on six seeds no
  decision had seen: three dots 14.7 +- 4.0 -> 19.8 +- 5.0 real per tube, 0.3 false
  before and after, precision 97.8% -> 98.3%, the analytes and their amounts
  identical. Over twelve seeds 5.1 of the 5.2 raised per tube are cascade rows,
  whose two-dot cap this lifts (ADR-097, ADR-101).
* A singlet linked on the very peak the quantification's position step stands on,
  or the singlet the amount was measured on with another signal linked, is marked
  Good, on that peak. Where the peak that co-varies is not the one position says,
  nothing moves: over twelve seeds position was right more often (moving would have
  corrected 1 singlet and spoiled 4).

What the marks change on the annotation is undone by ``forget_marks`` before the
annotation or the quantification reads it again, so a study quantified twice gives
the same table. The analyst's marks are not touched, and a Bad verdict is not raised.
"""
import numpy as np

from processing.annotation_quality import QUALITY_BAD, QUALITY_GOOD
from processing.annotation_scoring import is_singlet_like, study_drift
from processing.multiplet_line_selector import multiplet_line_peaks
from processing.quant_engine import measure_signal, raise_confidence
from processing.quant_integration import pattern_evidence
from processing.sample_selection import included_sample_indices
from processing.signal_comparison import compare_to_reference
from processing.signal_lines import manual_lines, singlet_on_peak
from processing.spectral_regions import FALLBACK_LINEWIDTH_PPM, measure_linewidth_ppm
from processing.stocsy import (
    STOCSY_MIN_SAMPLES, area_noise, build_basis, critical_r, peak_vectors)

SIGNAL_LINKS = True

# How far from its position a peak's apex is looked for in each sample.
APEX_HALF_PPM = 0.003
# How far a signal's quantification may stand from the row's, as a fraction
# (``signal_comparison.quantity_gap``).
H_RATIO_GAP = 0.5
_ALPHA = 0.05
# A peak that may be the measured signal itself stands this close to its window.
_OWN_WINDOW_PPM = 0.001
# A listing whose position falls this close to the window is the signal measured.
_LISTED_IN_WINDOW_PPM = 0.002

LINKS_KEY = '_signal_links'
_MARK_KEY = '_singlet_mark'
# What standing on another peak changes on a matched singlet (signal_lines.singlet_on_peak).
_MOVED = ('peak_ppm', 'peak_rank', 'ppm_diff', 'predicted_lines', 'ppm_off_drift', 'obs_snr')


def _protons(mp) -> float:
    try:
        v = float(mp.get('protons'))
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) and v > 0 else 0.0


class _Study:
    """The study as this step reads it: each peak's apex and its measured area
    in every sample, the basis of the correlation."""

    def __init__(self, peaks, ppm, intensities, samples_info, *, noise_sigma,
                 local_baseline, align):
        self.ppm = np.asarray(ppm, dtype=float)
        X = np.asarray(intensities, dtype=float)
        self.ok = X.ndim == 2 and X.shape[0] == self.ppm.size and self.ppm.size > 1
        if not self.ok:
            return
        self.included = included_sample_indices(samples_info, X.shape[1])
        info = list(samples_info or ())
        self.X = X[:, self.included]
        self.n = self.X.shape[1]
        self.peaks = {int(p['rank']): p for p in peaks or ()
                      if isinstance(p.get('rank'), (int, np.integer))}
        ranks, M, _skipped = peak_vectors(list(self.peaks.values()), included=self.included,
                                          min_samples=3)
        if (self.n < STOCSY_MIN_SAMPLES or noise_sigma is None
                or not np.isfinite(float(noise_sigma)) or len(ranks) < 2):
            self.ok = False
            return
        self.readable = {r for r, sd in zip(ranks, np.std(M, axis=1)) if sd > 0}
        groups = [str((info[i] if i < len(info) else {}).get(
            'label', (info[i] if i < len(info) else {}).get('group', ''))) for i in self.included]
        self.basis = build_basis(M, groups=groups, normalize='pqn', within_groups=True)
        self.noise = float(noise_sigma)
        self.dx = abs(float(self.ppm[1] - self.ppm[0]))
        self.linewidth = float(measure_linewidth_ppm(self.ppm, self.X, list(self.peaks.values()))
                               or FALLBACK_LINEWIDTH_PPM)
        self.local_baseline, self.align = bool(local_baseline), bool(align)
        self._apex, self._measured = {}, {}

    def apex(self, rank):
        if rank not in self._apex:
            x0 = float(self.peaks[rank]['ppm'])
            m = (self.ppm >= x0 - APEX_HALF_PPM) & (self.ppm <= x0 + APEX_HALF_PPM)
            self._apex[rank] = (self.X[m].max(axis=0) if m.any()
                                else np.full(self.n, np.nan))
        return self._apex[rank]

    def course(self, lines):
        """A multiplet's course across the samples: the median of its lines,
        each on its own scale, so one overlapped line does not lead it."""
        vs = [self.apex(r) for r in lines]
        vs = [v / v.mean() for v in vs if np.isfinite(v).all() and v.mean() > 0]
        return np.median(vs, axis=0) if vs else None

    def measure(self, lines, evidence):
        """``(areas, sd)`` measured as the quantification measures them, or
        ``(None, nan)``; the ratio of two signals is only worth reading on one
        scale."""
        key = (tuple(sorted(lines)), tuple(evidence.get('predicted_ppms') or ()),
               tuple(evidence.get('line_weights') or ()))
        if key not in self._measured:
            info, why = measure_signal(
                [self.peaks[r] for r in lines], evidence, ppm=self.ppm, intensities=self.X,
                included=list(range(self.n)), local_baseline=self.local_baseline,
                align=self.align, blank_indices=[], linewidth_ppm=self.linewidth,
                other_peaks=list(self.peaks.values()), noise_sigma=self.noise)
            if why or not info or not info.get('areas'):
                self._measured[key] = (None, np.nan)
            else:
                lo, hi = info['window']
                npts = max(int(round(abs(hi - lo) / self.dx)) + 1, 1)
                sd = (float(area_noise(self.noise, self.dx, np.array([npts]))[0])
                      / float(info.get('captured') or 1.0))
                self._measured[key] = (np.array([info['areas'].get(i, np.nan)
                                                 for i in range(self.n)], dtype=float), sd)
        return self._measured[key]

    def window_noise(self, window) -> float:
        npts = (max(int(round(abs(float(window[1]) - float(window[0])) / self.dx)) + 1, 1)
                if window else max(int(round(self.linewidth / self.dx)), 1))
        return float(area_noise(self.noise, self.dx, np.array([npts]))[0])


def _measured_vector(row, study):
    """The row's areas over the study's samples, or None."""
    areas = row.get('_areas_per_sample')
    if not isinstance(areas, (list, tuple)) or not areas:
        return None
    if len(areas) == study.n:
        return np.asarray(areas, dtype=float)
    if max(study.included, default=-1) >= len(areas):
        return None
    return np.asarray([areas[i] for i in study.included], dtype=float)


def _signals_of(met) -> list:
    """One matched peak per position the analyte's listing was matched at."""
    seen, out = set(), []
    for mp in met.get('matched_peaks') or ():
        if mp.get('db_ppm') is None:
            continue
        key = round(float(mp['db_ppm']), 3)
        if key not in seen:
            seen.add(key)
            out.append((key, mp))
    return sorted(out, key=lambda kv: kv[0])


def _link_of(mp, db, row_areas, row_sd, row_nh, window, study, tolerance):
    """``{db_ppm, type, rank, r, gap, ppm}`` of the peak of one listed signal that
    co-varies with the row's amounts in the proportion the protons call for, the
    best first, or None. The signal the amount was measured on is not asked: it
    co-varies with itself."""
    def in_window(rank):
        return bool(window) and (float(window[0]) - _OWN_WINDOW_PPM
                                 <= float(study.peaks[rank]['ppm'])
                                 <= float(window[1]) + _OWN_WINDOW_PPM)

    protons = _protons(mp)
    if not protons or (window and float(window[0]) - _LISTED_IN_WINDOW_PPM <= db
                       <= float(window[1]) + _LISTED_IN_WINDOW_PPM):
        return None
    kind = str(mp.get('type') or mp.get('multiplicity') or '').strip().lower()
    if is_singlet_like(kind or 's', strict=True):
        ranks = [r for r in study.readable
                 if abs(float(study.peaks[r]['ppm']) - db) <= tolerance and not in_window(r)]
        tries = [([r], {'protons': protons, 'predicted_ppms': [float(study.peaks[r]['ppm'])]},
                  study.apex(r)) for r in ranks]
    else:
        lines = [int(p['rank']) for p in multiplet_line_peaks(mp, study.peaks)
                 if p.get('rank') in study.readable]
        course = (study.course(lines)
                  if len(lines) >= 2 and not any(in_window(r) for r in lines) else None)
        tries = [(lines, dict(pattern_evidence(mp)), course)] if course is not None else []
    if not tries:
        return None
    R = study.basis.correlate(row_areas[None, :], np.vstack([t[2] for t in tries]))[0]
    bar = critical_r(study.basis.dof, alpha=_ALPHA / len(tries))
    for i in np.argsort(-np.nan_to_num(R, nan=-2.0)):
        if not np.isfinite(R[i]) or R[i] < bar:
            break
        lines, evidence, _course = tries[i]
        areas, sd = study.measure(lines, evidence)
        if areas is None:
            continue
        gap = compare_to_reference(row_areas, areas, row_sd, [sd], row_nh, [protons],
                                   min_valid=min(5, study.n - 1))[0]['gap']
        if np.isfinite(gap) and abs(gap) <= H_RATIO_GAP:
            return {'db_ppm': float(db), 'type': kind, 'rank': int(lines[0]), 'r': float(R[i]),
                    'gap': float(gap), 'ppm': float(study.peaks[lines[0]]['ppm'])}
    return None


def _raisable(row) -> bool:
    """Two dots and its identity is what holds it there. A share resolved out of
    a window rests on another analyte's amount, and a shape that fell short says
    nothing a second signal could answer: those stay. So does a row the lift raised
    (stocsy_evidence): co-variation is one kind of evidence and is worth one dot.
    Twice over it took 25 rows of 54 tubes from one dot to three, all real, but the
    bench could not have shown the cost: its compounds vary independently, and in a
    real study they do not."""
    return (int(row.get('_confidence') or 0) == 2 and not row.get('_shared_with')
            and not row.get('_deconv') and not row.get('_stocsy_lift')
            and (row.get('_confidence_axes') or {}).get('shape') in (None, 3)
            and row.get('_identity_level') in (None, 2))


def _raise(row, links) -> None:
    raise_confidence(row, 3)
    row['_signal_links_from'] = 2
    why = (f"{len(links)} other signal(s) of its listing move with the amount measured "
           f"across the samples, in the ratio of their protons (r of at least "
           f"{min(x['r'] for x in links):.2f}): one molecule has one amount per sample")
    row['_identity_why'] = f"{row['_identity_why']}; {why}" if row.get('_identity_why') else why


def forget_marks(metabolites, manual_marks=None) -> None:
    """Put back what ``_mark`` changed on the matched peaks. A signal the analyst
    has marked since keeps the analyst's mark."""
    manual_marks = manual_marks or {}
    for met in metabolites or ():
        mid = met.get('id', met.get('name'))
        for mp in met.get('matched_peaks') or ():
            mark = mp.pop(_MARK_KEY, None)
            if mark is None or (mid, mp.get('peak_rank')) in manual_marks:
                continue
            for key in mark['added']:
                mp.pop(key, None)
            mp.update(mark['before'])
            reasons = mp.get('_status_reasons')
            if reasons and mark['reason'] in reasons:
                reasons.remove(mark['reason'])


def _mark(mp, peak, reason, common_drift) -> None:
    """Good, on ``peak``; what it was is kept for ``forget_marks``."""
    changes = {'annotation_quality': QUALITY_GOOD}
    if peak.get('rank') != mp.get('peak_rank'):
        moved = singlet_on_peak(mp, peak, common_drift)
        changes.update({k: moved[k] for k in _MOVED if k in moved})
    mark = {'before': {k: mp[k] for k in changes if k in mp},
            'added': [k for k in changes if k not in mp], 'reason': reason}
    if '_status_reasons' not in mp:
        mark['added'].append('_status_reasons')
    mp.update(changes)
    mp[_MARK_KEY] = mark
    mp.setdefault('_status_reasons', []).append(reason)


def _mark_singlets(annotated, worked, rows, study, manual_marks) -> None:
    """Mark on the annotation's own matches what the rows' links say of its
    singlets, where the quantification's position step stands on the same peak."""
    common = study_drift(worked)
    own = {}
    for met in annotated or ():
        for mp in met.get('matched_peaks') or ():
            if mp.get('db_ppm') is not None:
                own.setdefault((str(met.get('id')), round(float(mp['db_ppm']), 3)), (met, mp))
    by_id = {str(m.get('id')): m for m in worked or ()}
    for row in rows:
        links = row.get(LINKS_KEY)
        met = by_id.get(str(row.get('hmdb_id')))
        if not links or met is None:
            continue
        linked = {x['db_ppm']: x for x in links}
        window = row.get('_multiplet_window')
        for db, mp in _signals_of(met):
            if not is_singlet_like(str(mp.get('type') or mp.get('multiplicity') or 's'), strict=True):
                continue
            rank = mp.get('peak_rank')
            if rank not in study.peaks:
                continue
            x = linked.get(float(db))
            at = float(study.peaks[rank]['ppm'])
            measured_here = bool(window) and (float(window[0]) - _OWN_WINDOW_PPM <= at
                                              <= float(window[1]) + _OWN_WINDOW_PPM)
            if x is not None and x['rank'] == rank:
                reason = (f"Singlet moves with the amount measured for this analyte "
                          f"(r {x['r']:.2f}), in the proportion its protons call for")
            elif x is None and measured_here:
                other = links[0]
                reason = (f"Singlet the amount was measured on; the {other['type']} at "
                          f"{other['db_ppm']:.3f} ppm moves with it, in the proportion "
                          f"their protons call for")
            else:
                continue
            found = own.get((str(row.get('hmdb_id')), db))
            if found is None:
                continue
            owner, target = found
            if (target.get('annotation_quality') in (QUALITY_GOOD, QUALITY_BAD)
                    or manual_lines(target) is not None
                    or (owner.get('id'), target.get('peak_rank')) in manual_marks):
                continue
            _mark(target, study.peaks[rank], reason, common)


def apply_links(rows, annotated, worked, peaks, ppm, intensities, samples_info, *, tolerance,
                noise_sigma, local_baseline=False, align=False, manual_marks=None) -> list:
    """Ask each row of the table about the rest of its listing.

    ``annotated`` is the annotation's own list, which the marks are written on;
    ``worked`` the copy the quantification ran on, whose singlets stand where its
    position step left them. Linked rows get ``LINKS_KEY``; those that qualify are
    raised to three dots with a sentence in ``_identity_why``. Rows are changed in
    place and returned."""
    study = _Study(peaks, ppm, intensities, samples_info, noise_sigma=noise_sigma,
                   local_baseline=local_baseline, align=align)
    if not study.ok:
        return rows
    by_id = {str(m.get('id')): m for m in worked or ()}
    for row in rows:
        met = by_id.get(str(row.get('hmdb_id')))
        if row.get('_not_quantified') or row.get('_is_is') or met is None:
            continue
        areas = _measured_vector(row, study)
        if areas is None:
            continue
        window = row.get('_multiplet_window')
        sd = study.window_noise(window)
        nh = float(row.get('_nh') or 0.0) or 1.0
        links = [x for x in (_link_of(mp, db, areas, sd, nh, window, study, float(tolerance))
                             for db, mp in _signals_of(met)) if x is not None]
        if not links:
            continue
        row[LINKS_KEY] = links
        if _raisable(row):
            _raise(row, links)
    _mark_singlets(annotated, worked, rows, study, manual_marks or {})
    return rows
