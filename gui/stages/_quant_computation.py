# gui/stages/_quant_computation.py
"""_QuantCompMixin — what the Quantification stage does around the engine.

``_run`` reads what the analyst chose, hands it to
``processing.quant_engine`` through ``_quant_adapter``, and puts the answer on
screen. Nothing scientific is decided here.

It used to be otherwise: the engine lived inside ``_run`` as 2876 lines that
settled which signal quantified which metabolite in four places that
disagreed with each other. Peaks were handed out by anchor while the lines
underneath were shared, so two analytes could be given the same signal and
neither call site could see it.

What stays: VIP scores, the manual per-sample region update, the tree refresh,
and the exports — display and analysis over results that already exist.
"""
from tkinter import filedialog, messagebox
import numpy as np
import csv
import os

from ._quant_adapter import (
    blank_indices, dilution_and_correction, identity_columns as _identity_columns,
    quantify_project, reported_values, sample_groups, standard_vector)
from app_paths import DECONVOLUTION_ENABLED
from gui.project_vars import collect_vars
from processing.quant_concentration import (
    pqn_group_divergence_of_rows, pqn_group_note)
from processing.sample_selection import included_sample_indices
from processing.spectral_regions import format_regions, parse_regions

# ── Debug logging ─────────────────────────────────────────────────────────────
# Set to True to write a detailed quant_debug.log in the project root.
_QUANT_DEBUG = True
_QUANT_DEBUG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'quant_debug.log')

def _dbg_open():
    """Open the debug log without allowing logging to break quantification."""
    if not _QUANT_DEBUG:
        return None

    # The frozen application may live under Program Files, which is normally
    # read-only. Try the project/bundle path first for development, then the
    # per-user NMRPro-H data directory used by installed builds.
    candidates = [_QUANT_DEBUG_PATH]
    try:
        from app_paths import user_data_dir
        candidates.insert(
            0, os.path.join(user_data_dir(), 'logs', 'quant_debug.log'))
    except Exception:
        pass

    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return open(path, 'w', encoding='utf-8')
        except (OSError, PermissionError):
            continue

    # Debug logging is optional; a read-only installation must still quantify.
    return None

def _dbg(f, *args):
    if f is None:
        return
    print(*args, file=f)
# ─────────────────────────────────────────────────────────────────────────────


def _per_sample_snr(p, all_int, n_s):
    """Per-sample SNR at a detected peak's integration window.

    Integration in the quant pipeline uses ONE fixed window per analyte for
    every sample.  In a sample where the analyte is weak/absent the window
    still integrates noise, so a reported area/concentration on its own cannot
    distinguish "present but small" from "not there".  This returns the SNR of
    the peak window in EACH sample (length ``n_s``) so callers can flag samples
    below the detection limit instead of silently reporting noise as signal.

    Returns
    -------
    (snr_list, noise_lvl)
        ``snr_list`` has one float per sample (NaN when noise is unknown);
        ``noise_lvl`` is the absolute noise level used (or None).
    """
    if p is None or all_int is None or n_s <= 0:
        return [float('nan')] * max(n_s, 0), None
    lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
    hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))
    noise_lvl = p.get('noise', None)
    if (not noise_lvl or noise_lvl <= 0) and p.get('snr', 0) > 0:
        peak_max_all = float(np.max(np.mean(all_int[lo_i:hi_i + 1, :], axis=1)))
        noise_lvl = peak_max_all / p['snr'] if p['snr'] else None
    if not noise_lvl or noise_lvl <= 0:
        return [float('nan')] * n_s, noise_lvl
    snr_list = [float(np.max(all_int[lo_i:hi_i + 1, s])) / noise_lvl
                for s in range(n_s)]
    return snr_list, noise_lvl
# ─────────────────────────────────────────────────────────────────────────────

from ._quant_helpers import (
    _trapz, _pls_vip, _cv, bootstrap_fraction_ci as _bootstrap_fraction_ci)
from processing.multiplet_line_selector import (
    multiplet_line_peaks as _multiplet_line_peaks)








def _competitor_counts(metabolites) -> dict:
    """How many annotated signals claim each detected peak.

    The Quant Peaks window shows this so a peak handed to one analyte does not
    look uncontested when four others wanted it. It counts claims, not winners:
    the election is elsewhere and this is the evidence the analyst reviews it
    against.
    """
    counts: dict = {}
    for met in metabolites or ():
        for mp in met.get('matched_peaks', []) or ():
            pr = mp.get('peak_rank')
            if pr is not None:
                counts[pr] = counts.get(pr, 0) + 1
    return counts


def _write_debug_log(rows, groups, *, norm_mode, proton_corr, blk_sub, align,
                     ref_peak, ref_conc, ref_nh, ext_is, n_samples,
                     excluded_regions, local_baseline, shapes=None,
                     shared=None, diagnostics=None):
    """The run, written down where it can be read after the window closes.

    Two sections earn their place. The summary table is what was measured. The
    NOT QUANTIFIED list is what was not, grouped by reason — grouped because
    the reason decides what to do next and one action serves everyone under it:
    a withdrawn multiplet returns when detection resolves its lines, an
    ambiguous rank when the analyst marks a candidate, one beaten by a rival
    only if the rival turns out to be wrong.
    """
    f = _dbg_open()
    if f is None:
        return
    try:
        _dbg(f, '=' * 70)
        _dbg(f, 'QUANT DEBUG LOG')
        _dbg(f, '=' * 70)
        _dbg(f, f'norm_mode    : {norm_mode}')
        _dbg(f, f'proton_corr  : {proton_corr}')
        _dbg(f, f'blk_sub      : {blk_sub}')
        _dbg(f, f'local_basel. : {local_baseline}')
        _dbg(f, f'window_align : {align}')
        _dbg(f, f'shape_fit m* : {shapes}')
        _dbg(f, f'shared deconv: {shared}')
        # What was excluded is part of what the peak list means: the same
        # spectrum detected with a different window is a different study.
        _dbg(f, f'excluded     : {excluded_regions}')
        _dbg(f, f'n_samples    : {n_samples}')
        _dbg(f, f'groups       : { {g: len(i) for g, i in groups.items()} }')
        if ref_peak is not None:
            _dbg(f, f'ref_peak     : rank={ref_peak.get("rank")} '
                    f'ppm={float(ref_peak.get("ppm", 0)):.4f} '
                    f'conc={ref_conc} nH={ref_nh}')
        else:
            _dbg(f, 'ref_peak     : NOT FOUND — no IS normalisation')
        _dbg(f, f'external_IS  : {ext_is}')

        ok = [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]
        censored = [r for r in rows if r.get('_censored')]
        nq = [r for r in rows
              if r.get('_not_quantified') and not r.get('_censored')]
        _dbg(f, '')
        _dbg(f, '=' * 70)
        _dbg(f, f'SUMMARY TABLE  ({len(ok)} quantified, sorted by mean mM)')
        _dbg(f, '=' * 70)
        gnames = [g for g in groups if g != 'BLK']
        _dbg(f, f'  {"#":>3}  {"Metabolite":<28s}  {"region (ppm)":>16}'
                f'  {"mean mM":>9}  {"nH":>4}  {"SNR":>7}  {"conf":>4}'
                + ''.join(f'  {g[:8]:<10s}' for g in gnames))
        _dbg(f, '  ' + '-' * 128)
        for i, r in enumerate(sorted(
                (r for r in ok if not r.get('_is_is')),
                key=lambda r: (r['conc_mm'] if np.isfinite(r['conc_mm'])
                               else -1), reverse=True), 1):
            cbg = r.get('_concs_by_group', {})
            gstr = ''
            for g in gnames:
                v = cbg.get(g, [])
                m = float(np.mean(v)) if v else float('nan')
                cv = r.get('cv_by_group', {}).get(g, float('nan'))
                gstr += f'  {m:8.3f}(CV{cv:.0f}%)'
            _w_r = r.get('_multiplet_window')
            _reg = (f'{_w_r[0]:7.4f}-{_w_r[1]:<7.4f}' if _w_r
                    else f'{r.get("ppm", 0):>16.4f}')
            _dbg(f, f'  {i:>3}.  {r["metabolite"][:28]:<28s}  {_reg:>16}'
                    f'  {r["conc_mm"]:>9.4f}'
                    f'  {r.get("_nh", float("nan")):>4.2f}'
                    f'  {r.get("snr", float("nan")):>7.1f}'
                    f'  {str(r.get("_confidence", "?")):>4}' + gstr)
        is_row = next((r for r in rows if r.get('_is_is')), None)
        if is_row is not None and ref_conc:
            rec = (is_row['conc_mm'] / ref_conc * 100
                   if np.isfinite(is_row['conc_mm']) else float('nan'))
            _dbg(f, '  ' + '-' * 118)
            _dbg(f, f'   IS.  {is_row["metabolite"][:28]:<28s}'
                    f'  {is_row.get("ppm", 0):>7.4f}'
                    f'  {is_row["conc_mm"]:>9.4f}'
                    f'   recovery={rec:.1f}%')

        if censored:
            _dbg(f, '')
            _dbg(f, '── BELOW THE LIMIT ──────────────────────────────────')
            _dbg(f, '  Bounds, not measurements. An analyte here was looked '
                    'for where the')
            _dbg(f, '  database predicts it and not found; the limit is what '
                    'this experiment')
            _dbg(f, '  could have seen there.')
            _dbg(f, '')
            _dbg(f, f'  {"Metabolite":<30s}  {"region (ppm)":>16}'
                    f'  {"LOD mM":>10}  {"LOQ mM":>10}  verdict')
            _dbg(f, '  ' + '-' * 84)
            for r in sorted(censored, key=lambda r: (r.get('_censored', ''),
                                                     r.get('metabolite', ''))):
                w = r.get('_multiplet_window') or (float('nan'),) * 2
                _dbg(f, f'  {str(r.get("metabolite", "?"))[:30]:<30s}'
                        f'  {w[0]:7.4f}-{w[1]:<7.4f}'
                        f'  {r.get("_lod_mm", float("nan")):>10.4g}'
                        f'  {r.get("_loq_mm", float("nan")):>10.4g}'
                        f'  {r.get("status", "")}')
            _n_ov = sum(1 for r in censored
                        if r.get('_censored') == 'unresolved')
            if _n_ov:
                _dbg(f, '')
                _dbg(f, f'  {_n_ov} marked overlap?: area well above the '
                        'quantification limit with')
                _dbg(f, '  no peak detected in the window. Something is there '
                        'that detection could')
                _dbg(f, '  not resolve. Look at those regions before trusting '
                        'their neighbours.')

        resolved = [r for r in rows if r.get('_deconv')]
        sd = (diagnostics or {}).get('shape_deconvolution')
        if resolved or sd is not None:
            _dbg(f, '')
            _dbg(f, '── WINDOWS RESOLVED BY SHAPE ────────────────────────')
            _dbg(f, "  Windows holding another accepted analyte's signal, "
                    'modelled as the sum of')
            _dbg(f, '  the accepted shapes. Each area is what its own shape '
                    'accounts for; the')
            _dbg(f, '  remainder is what no accepted shape explains.')
            _dbg(f, '')
            _dbg(f, f'  {"Metabolite":<30s}  {"window (ppm)":>16}'
                    f'  {"mean mM":>9}  {"remainder":>9}  resolved with')
            _dbg(f, '  ' + '-' * 90)
            for r in sorted(resolved, key=lambda r: str(r.get('metabolite', ''))):
                d = r['_deconv']
                w = d.get('window') or (float('nan'),) * 2
                rem = list((d.get('remainder') or {}).values())
                med = float(np.median(rem)) if rem else float('nan')
                others = [c['label'] for c in d.get('components', ())
                          if c.get('met_id') != r.get('hmdb_id')]
                _dbg(f, f'  {str(r.get("metabolite", "?"))[:30]:<30s}'
                        f'  {w[0]:7.4f}-{w[1]:<7.4f}'
                        f'  {r.get("conc_mm", float("nan")):>9.4f}'
                        f'  {100 * med:>8.0f}%  {", ".join(others)}')
            if not resolved:
                _dbg(f, "  (none: no winner's window holds another accepted "
                        "analyte's signal)")

        shared = [r for r in rows if r.get('_shared_with')
                  and not r.get('_deconv')]
        sh = (diagnostics or {}).get('shared')
        if sh and sh.get('off'):
            shared = []
        if shared or sh:
            _dbg(f, '')
            _dbg(f, '── SHARED WINDOWS ───────────────────────────────────')
            _dbg(f, '  Peaks two or more analytes sit on. The area is split '
                    'by what each')
            _dbg(f, '  claimant\'s own clean signal says it must be '
                    'contributing here, so the')
            _dbg(f, '  numbers below depend on those signals as well as on '
                    'this window.')
            _dbg(f, '  A split area never reads three dots.')
            _dbg(f, '')
            _dbg(f, f'  {"Metabolite":<30s}  {"region (ppm)":>16}'
                    f'  {"mean mM":>9}  shares the window with')
            _dbg(f, '  ' + '-' * 90)
            for r in sorted(shared, key=lambda r: str(r.get('metabolite', ''))):
                w = r.get('_multiplet_window') or (float('nan'),) * 2
                _dbg(f, f'  {str(r.get("metabolite", "?"))[:30]:<30s}'
                        f'  {w[0]:7.4f}-{w[1]:<7.4f}'
                        f'  {r.get("conc_mm", float("nan")):>9.4f}'
                        f'  {", ".join(r.get("_shared_with") or ())}')

            # What the step did when it did nothing. A stage that silently
            # changes no number is indistinguishable from a stage that is not
            # wired in, and telling those two apart cost a whole run.
            if sh and sh.get('off'):
                _dbg(f, '')
                _dbg(f, '  Not applied: the split is switched off, so every '
                        'winner keeps the whole')
                _dbg(f, '  of its window and the analytes that lost one have '
                        'no number at all.')
            elif sh:
                _dbg(f, '')
                _dbg(f, f'  {sh.get("candidates", 0)} candidate signals -> '
                        f'{sh.get("components", 0)} windows, of which '
                        f'{sh.get("contested", 0)} are claimed by more than '
                        f'one analyte.')
                _dbg(f, f'  {sh.get("resolved", 0)} split, '
                        f'{sh.get("unsolved", 0)} not.')
                _dbg(f, f'  Predicted signals with no detected peak: '
                        f'{sh.get("no_lines", 0)}. Used as bounds: '
                        f'{sh.get("bounds", 0)} (a window at the predicted '
                        f'position, slid over the drift, holding whatever is '
                        f'there). Not used: '
                        f'{sh.get("bounds_no_anchor", 0)} because the analyte '
                        f'has no detected signal anywhere, '
                        f'{sh.get("bounds_foreign", 0)} because an unmatched '
                        f'peak sits in the window, '
                        f'{sh.get("bounds_excluded", 0)} inside an excluded '
                        f'region.')
                _dbg(f, f'  System: {sh.get("rows", 0)} equations, '
                        f'{sh.get("columns", 0)} unknowns. Refused '
                        f'{sh.get("refused_indeterminate", 0)} as '
                        f'indeterminate (no window separates them from '
                        f'another analyte), '
                        f'{sh.get("refused_imprecise", 0)} as too imprecise '
                        f'(own standard error over half the value), '
                        f'{sh.get("refused_empty", 0)} left nothing by the '
                        f'other claimants.')
                _dbg(f, f'  Variance inflation: median '
                        f'{sh.get("vif_median", 0):.1f}, 90th percentile '
                        f'{sh.get("vif_p90", 0):.1f}, '
                        f'{sh.get("vif_infinite", 0)} infinite. Median '
                        f'relative standard error '
                        f'{100 * sh.get("rsd_median", 0):.0f}%.')
                _dbg(f, f'  Widest contest: {sh.get("largest", 0)} analytes '
                        f'over {sh.get("largest_lines", 0)} lines. Overlap is '
                        f'transitive, so a crowded region can chain into one '
                        f'window; a window with N claimants has N-1 unknowns.')
                _skipped = [
                    ('multiplicity never verified', sh.get('not_competitor', 0)),
                    ('no proton count', sh.get('no_protons', 0)),
                ]
                _skipped = [t for t in _skipped if t[1]]
                if _skipped:
                    _dbg(f, '  Candidates left out: '
                            + ', '.join(f'{n} {w}' for w, n in _skipped) + '.')
                for why, n in sorted((sh.get('reasons') or {}).items(),
                                     key=lambda kv: -kv[1]):
                    _dbg(f, f'  [{n}] {why}')
                if sh.get('contests'):
                    _dbg(f, '')
                    _dbg(f, '  THE WIDEST CONTESTS, claimant by claimant. '
                            'What annotation said about each one.')
                    _dbg(f, '  A window with twelve claimants is not a '
                            'quantification problem; which gate let')
                    _dbg(f, '  them through can be read here.')
                    for c in sh['contests']:
                        lo, hi = c['window']
                        _dbg(f, '')
                        _dbg(f, f'  window {lo:.4f}-{hi:.4f} ppm, '
                                f'{len(c["claimants"])} claimants')
                        _dbg(f, f'    {"metabolite":<30s} {"mult":>4} '
                                f'{"dppm":>7} {"dJ Hz":>6} {"lines":>5} '
                                f'{"quality":>8} {"in win":>6} {"sigs":>4}')
                        for r in c['claimants']:
                            dppm = r['dppm']; dj = r['dj']
                            lines = (f'{r["found"]}/{r["expected"]}'
                                     if r['found'] is not None else '-')
                            _dbg(f, f'    {str(r["name"])[:30]:<30s} '
                                    f'{str(r["mult"] or "-"):>4} '
                                    f'{(f"{dppm:+.3f}" if dppm is not None else "-"):>7} '
                                    f'{(f"{dj:.2f}" if dj is not None else "-"):>6} '
                                    f'{lines:>5} {str(r["quality"] or "-"):>8} '
                                    f'{100*r["fraction"]:>5.0f}% '
                                    f'{r["signals"]:>4}')

        _dbg(f, '')
        _dbg(f, '── NOT QUANTIFIED ───────────────────────────────────')
        by_reason: dict = {}
        for r in nq:
            # Grouped on the first reason; every further reason -- the share
            # that could not be resolved, which peak was lost to whom and on
            # what -- is printed under the name. Grouping on the first alone
            # once hid the second entirely; joining them all made every
            # analyte its own group.
            reasons = r.get('_status_reasons') or ['(no reason given)']
            by_reason.setdefault(reasons[0], []).append(
                (r.get('metabolite', r.get('hmdb_id', '?')), reasons[1:]))
        for why, who in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            _dbg(f, f'  [{len(who)}] {why}')
            for nm, more in sorted(who, key=lambda t: str(t[0])):
                _dbg(f, f'        {nm}')
                for line in more:
                    _dbg(f, f'            - {line}')
        _dbg(f, '')
        _dbg(f, 'END')
    finally:
        f.close()


class _QuantCompMixin:
    """Mixin — mixed into QuantificationStage.  Uses self.* freely."""

    def _detection_min_snr(self):
        """SNR threshold used by peak detection — reused to flag, per sample,
        whether an analyte is actually above the detection limit. Falls back to
        the pipeline default (3.0) when the peak-detection stage is unavailable."""
        try:
            return float(self.app.stages['peak_detection'].min_snr_var.get())
        except Exception:
            return 3.0

    # ── Core quantification ───────────────────────────────────────────────────

    def _run(self):
        """Quantify the project: elect, integrate, concentrate, then display.

        Everything scientific happens in ``processing``. What is left here is
        the three jobs a stage actually has — read what the analyst chose, hand
        it to the engine, and put the answer on screen. The engine that used to
        live in this method decided the same question in four places that
        disagreed with each other, and no defect it carried was visible from
        any single one of them.
        """
        self._quant_results = []
        self._all_peak_quant = {}
        # NOT reset: user multiplet choices from the Quant Peaks window must
        # survive a re-quantification. They are cleared when annotation re-runs.
        if not hasattr(self, '_quant_peak_overrides'):
            self._quant_peak_overrides = {}
        if not hasattr(self, '_deconv_overrides'):
            self._deconv_overrides = {}
        self._peak_competitor_count = {}
        self._pqn_factor = 1.0

        ann = self.annotation_stage
        metabolites = list(getattr(ann, '_metabolites', []) or [])
        if not metabolites:
            messagebox.showinfo(
                'Info',
                'No annotation results found.\n\n'
                'Run the Annotation stage first.')
            return

        ppm_arr = self.app.ppm
        # The baseline model found the peaks; the areas are measured on the
        # data unless the analyst asks for the model's spectrum here too.
        _bmq = getattr(self, 'bl_model_quant_var', None)
        _on_model = bool(_bmq.get()) if _bmq is not None else False
        all_int = self.app.all_intensities
        if _on_model:
            try:
                all_int = self.app.corrected_intensities()
            except Exception as _bl_exc:
                print(f'[Quant] baseline model unavailable, integrating the '
                      f'data: {_bl_exc}')
        if ppm_arr is None or all_int is None or np.asarray(all_int).ndim < 2:
            messagebox.showinfo('Info', 'No spectral data loaded.')
            return
        all_int = np.asarray(all_int, dtype=float)
        n_s = all_int.shape[1]
        si = self.app.samples_info or []

        # 'Multiplet' is the only integration mode. A saved project can still
        # carry one of the retired ones, so migrate rather than refuse to open.
        if self.integ_var.get() != 'Multiplet':
            self.integ_var.set('Multiplet')

        norm_mode = self.norm_var.get()
        use_pqn = (norm_mode == 'IS+PQN')
        proton_corr = bool(self.proton_corr_var.get())
        local_baseline = bool(getattr(self, 'local_baseline_var', None)
                              and self.local_baseline_var.get())
        subtract_blanks = bool(self.blk_sub_var.get())
        align = bool(getattr(self, 'icoshift_var', None)
                     and self.icoshift_var.get())
        if align:
            self._status('Pre-align: per-signal window alignment active', 'blue')

        if self.ext_is_var.get() and self._ext_is_indices:
            if not getattr(self, '_ext_is_confirmed', False):
                if not messagebox.askyesno(
                        'External IS — Confirm',
                        'External IS mode is active.\n\n'
                        'Please confirm that the IS concentration and nH are the same\n'
                        'for both the External IS samples and the analytical samples.\n\n'
                        'Proceed with quantification?'):
                    return
        ext_is = (list(self._ext_is_indices)
                  if (self.ext_is_var.get() and self._ext_is_indices) else None)

        ref_peak = None
        try:
            rank = int(self.ref_rank_var.get())
            ref_peak = next((p for p in (self.peak_stage.peaks or [])
                             if p.get('rank') == rank), None)
        except (ValueError, TypeError):
            pass
        if ref_peak is None:
            try:
                ref_peak = self._find_peak(float(self.ref_ppm_var.get()), tol=0.05)
            except ValueError:
                pass
        self.ref_peak = ref_peak
        try:
            ref_conc = float(self.ref_conc_var.get())
        except ValueError:
            ref_conc = 1.0
        try:
            ref_nh = max(1, int(self.ref_nh_var.get()))
        except ValueError:
            ref_nh = 9
        try:
            ref_name = self.std_var.get() or 'Internal standard'
        except Exception:
            ref_name = 'Internal standard'

        quality_marks = getattr(ann, '_quality_marks', {}) or {}
        # What the annotation scored with, from what the project holds -- not
        # from the results window, which a reopened project may not have.
        from processing.annotation_scoring import effective_annotation_params
        try:
            _tol = float(ann.tolerance_var.get())
        except Exception:
            _tol = None
        _ann_params = effective_annotation_params(
            getattr(ann, '_annotation_params', None), _tol)
        try:
            from gui.stages._annotation_helpers import (
                DJ_GOLD_HZ_DEFAULT, ambiguous_peak_ranks)
            # The gate the annotation ranked with. The params dict itself
            # used to be passed here, in dj_gold_hz's position, and the
            # check failed on every run -- quietly, into an empty set.
            _gold = float(_ann_params.get('dj_gold_hz', DJ_GOLD_HZ_DEFAULT))
            ambiguous = ambiguous_peak_ranks(
                metabolites,
                {(str(k[0]), k[1]): v for k, v in quality_marks.items()},
                dj_gold_hz=_gold)
        except Exception as _amb_exc:
            print(f'[Quant] ambiguity check unavailable: {_amb_exc}')
            ambiguous = set()

        # What detection applied, when this session ran it or a save recorded
        # it. Otherwise the field, which is what a loaded project's detection
        # was given -- silently using nothing is worse than using the best
        # evidence available and saying where it came from.
        _excl = getattr(self.peak_stage, '_excluded_regions', None)
        _excl_src = 'as detected'
        if _excl is None:
            _excl_src = 'from the field, not recorded by the run'
            try:
                _excl = parse_regions(
                    self.peak_stage.excluded_regions_var.get())
            except (ValueError, AttributeError):
                _excl = []
        _excluded_used = (f'{format_regions(_excl)} ({_excl_src})' if _excl
                          else 'none')

        self._prog_apply(20, 'Electing signals…')
        _diag: dict = {}
        # The overlap correction is the analyst's choice, and off it must
        # leave every number exactly where the engine put it before the
        # balance existed -- otherwise the two settings cannot be compared.
        _dv = getattr(self, 'deconv_var', None)
        _deconv = (DECONVOLUTION_ENABLED if _dv is None
                   else bool(_dv.get()) and DECONVOLUTION_ENABLED)
        # Complex multiplets by shape fit (ADR-013) are the analyst's choice
        # too; off, they are reported as not quantified and nothing else
        # in the table moves.
        _sv = getattr(self, 'shape_var', None)
        _shapes = bool(_sv.get()) if _sv is not None else False
        _rv = getattr(self, 'shared_var', None)
        _shared = bool(_rv.get()) if _rv is not None else False
        _ssv = getattr(self, 'singlet_snr_var', None)
        try:
            _singlet_snr = float(_ssv.get()) if _ssv is not None else None
        except (tk.TclError, ValueError):
            _singlet_snr = None
        _jv = getattr(self, 'joint_fit_var', None)
        _joint = bool(_jv.get()) if _jv is not None else False
        _cv = getattr(self, 'cascade_var', None)
        _cascade = bool(_cv.get()) if _cv is not None else False
        _mdv = getattr(self, 'max_disagreement_var', None)
        try:
            _max_dis = float(_mdv.get()) / 100.0 if _mdv is not None else None
        except (tk.TclError, ValueError):
            _max_dis = None
        rows = quantify_project(
            metabolites=metabolites,
            peaks=self.peak_stage.peaks or [],
            ppm=ppm_arr,
            intensities=all_int,
            samples_info=si,
            ambiguous_ranks=ambiguous,
            quality_marks=quality_marks,
            anchor_lines=getattr(ann, '_anchor_lines', {}),
            ref_peak=ref_peak, ref_nh=ref_nh, ref_conc=ref_conc,
            ref_name=ref_name,
            external_is_samples=ext_is,
            group_dilution=getattr(self.app, 'group_dilution_factors', {}),
            annotation_params=_ann_params,
            use_pqn=use_pqn,
            local_baseline=local_baseline,
            proton_correction=proton_corr,
            align=align,
            subtract_blanks=subtract_blanks,
            # The spectrum's noise as detection estimated it. It lives on the
            # detection stage, not on the peaks, and without it no analyte can
            # be given a detection limit.
            noise_sigma=getattr(self.peak_stage, 'noise_std', None),
            excluded_regions=_excl or (),
            deconvolve=_deconv,
            quantify_shapes=_shapes,
            resolve_shared=_shared,
            shared_overrides=dict(self._deconv_overrides or {}),
            diagnostics=_diag,
            singlet_min_snr=_singlet_snr,
            cascade=_cascade,
            joint_fit=_joint,
            max_disagreement=_max_dis,
            # the study before the project's baseline model, so every
            # window can say what the model took out of it
            raw_intensities=getattr(self.app, 'all_intensities', None),
        )

        self._prog_apply(70, 'Building results…')
        self._quant_results = rows
        # Every option this run was made with, kept with its results.
        self._last_run_settings = collect_vars(self)
        self._quant_diagnostics = _diag
        self._peak_competitor_count = _competitor_counts(metabolites)
        _pqn_vals = [v for r in rows
                     for v in (r.get('_pqn_factors') or {}).values()]
        if _pqn_vals:
            self._pqn_factor = float(np.median(_pqn_vals))
            _div = pqn_group_divergence_of_rows(
                rows, sample_groups(si, n_s))
            if _div and _div['diverges']:
                messagebox.showwarning('PQN', pqn_group_note(_div))
        # What the manual region editor needs to recompute one row the same
        # way the run computed all of them.
        self._last_norm_mode = norm_mode
        self._last_ref_conc = ref_conc
        self._last_ref_areas = standard_vector(
            ref_peak, ppm_arr, all_int, ref_nh,
            external_is_samples=ext_is, local_baseline=local_baseline)
        _dil, _corr = dilution_and_correction(
            si, getattr(self.app, 'group_dilution_factors', {}), n_s)
        _apply = _dil if (ext_is or norm_mode not in ('IS', 'IS+PQN')) else _corr
        self._last_sample_dil = np.array([_apply.get(i, 1.0)
                                          for i in range(n_s)])
        # What detection actually applied, not what the field currently reads
        # — the analyst can edit the field after detecting, and the peak list
        # being quantified is the one the earlier value produced.
        self._last_groups = sample_groups(si, n_s)
        self._last_blk_indices = blank_indices(si, n_s)
        self._last_non_blk_indices = included_sample_indices(si, n_s)

        rows.sort(key=lambda r: (r['conc_mm']
                                 if np.isfinite(r['conc_mm']) else -1),
                  reverse=True)
        self._sync_all_peak_quant()
        _refused = self._reapply_manual_regions()
        if _refused:
            messagebox.showwarning(
                'Manual regions',
                f'{len(_refused)} sample window(s) set by hand could not be '
                f'applied to this run (the number now comes from a fit):\n'
                + '\n'.join(f'  signal #{k}, sample {i}' for k, i, _w in _refused[:12]))
        _write_debug_log(rows, self._last_groups, norm_mode=norm_mode,
                         proton_corr=proton_corr, blk_sub=subtract_blanks,
                         align=align, ref_peak=ref_peak, ref_conc=ref_conc,
                         ref_nh=ref_nh, ext_is=ext_is, n_samples=n_s,
                         excluded_regions=_excluded_used,
                         local_baseline=local_baseline,
                         shapes=_shapes, shared=_shared,
                         diagnostics=_diag)

        _norm_f = norm_mode

        def _finish_ui():
            self.app.refresh_deconv_button()
            self._prog_apply(93, 'Computing VIP scores…')
            try:
                self._compute_vip_scores()
            except Exception as _vip_err:
                print(f'[Quant] VIP scores skipped: {_vip_err}')
                for _r in self._quant_results:
                    _r.setdefault('vip_score', float('nan'))

            self._prog_apply(99, 'Done.')
            # Counted once, for both places that report it. Two labels reading
            # the same list two different ways is how one of them kept saying
            # 100 after the other was taught to say 65.
            # The internal standard is a row, not an analyte: it stood at
            # 70 quantified here against 69 in the report, which leaves it
            # out.
            _analytes = [_r for _r in self._quant_results
                         if not _r.get('_is_is')]
            _n_nq = sum(1 for _r in _analytes if _r.get('_not_quantified'))
            _n_ok = len(_analytes) - _n_nq
            # Censored analytes are counted apart from both. They are neither
            # a result nor a failure: the experiment looked and reported what
            # it could have seen, which is a different thing from not looking.
            _n_cens = sum(1 for _r in _analytes if _r.get('_censored'))
            _n_nq -= _n_cens
            _note = (f' PQN×{self._pqn_factor:.3f}'
                     if _norm_f == 'IS+PQN' else '')
            self.run_status.config(
                text=(f'{_n_ok} quantified  ·  {_n_cens} < LOQ  ·  '
                      f'{_n_nq} not [{_norm_f}{_note}]'),
                foreground='green')
            for _b in ('btn_csv', 'btn_metaboanalyst', 'btn_pdf'):
                try:
                    getattr(self, _b).config(state='normal')
                except Exception:
                    pass
            # The list holds every analyte that was considered, quantified or
            # not, so its length is not the number of results.
            _plural = '' if _n_ok == 1 else 's'
            _msg = (f'{_n_ok} metabolite{_plural} identified and '
                    f'quantified successfully')
            if _n_cens:
                _msg += f'  ·  {_n_cens} below the limit of quantification'
            if _n_nq:
                _msg += f'  ·  {_n_nq} not quantified'
            self._status_ok(_msg)

            self._update_peak_list_labels()
            self._close_progress_dialog()
            self._show_right_panel()
            if self._tree:
                children = self._tree.get_children()
                if children:
                    try:
                        self._tree.selection_set(children[0])
                        self._tree.focus(children[0])
                        self._on_tree_select()
                    except Exception:
                        pass

        self.app.root.after(0, _finish_ui)

    def _sync_all_peak_quant(self):
        """Mirror the results into ``_all_peak_quant`` for the Quant Peaks window.

        Both key forms are written — the bare rank and ``(rank, ppm)`` —
        because the lookup tries the annotation ppm first and the detected apex
        second, and they are not always the same number.
        """
        for r in self._quant_results:
            mid = str(r.get('hmdb_id') or r.get('metabolite', ''))
            pr = r.get('_peak_rank')
            if not mid or pr is None:
                continue
            entry = {
                'conc_mm': r['conc_mm'],
                'concs_by_group': r.get('_concs_by_group', {}),
                'ppm': float(r.get('ppm', 0)),
            }
            d = self._all_peak_quant.setdefault(mid, {})
            for k in list(d.keys()):
                if k == pr or (isinstance(k, tuple) and k[0] == pr):
                    d[k] = entry
            d[(pr, round(float(r.get('ppm', 0)), 3))] = entry
            d[pr] = entry

    # ── Peak List labels ─────────────────────────────────────────────────────

    def _update_peak_list_labels(self):
        """After quantification, relabel each quantified peak in the Peak List
        with the metabolite that actually won it for integration.

        Annotation Results is intentionally NOT touched — it keeps its own
        score-based ownership (_assignment_sort_key). Only the Peak List follows
        the quantification winner, written as 'assigned_metabolite' (the highest
        priority field in PeakListWindow._populate). The '_quant_assigned' marker
        lets a later run clear only the labels this method set, never an
        annotation- or user-provided assignment.

        Must run on the main thread (called from _finish_ui) because it refreshes
        a tkinter window.
        """
        try:
            peaks = (self.peak_stage.peaks or []) if self.peak_stage else []
            if not peaks:
                return
            pk_by_rank = {p['rank']: p for p in peaks if p.get('rank') is not None}

            # Clear labels written by a previous quantification run only.
            for p in peaks:
                if p.pop('_quant_assigned', False):
                    p.pop('assigned_metabolite', None)

            rank_to_name = {}
            for r in (self._quant_results or []):
                if r.get('_is_is'):
                    continue
                name = r.get('metabolite')
                if not name:
                    continue
                pr = r.get('_peak_rank')
                if pr is not None and pr in pk_by_rank:
                    rank_to_name[pr] = name
                # Multiplet member peaks: every detected peak inside the winning
                # hit's predicted-line window belongs to the same metabolite,
                # mirroring the greedy claim in _run so the Peak List shows the
                # full multiplet under one analyte.
                for h in r.get('_hits', []):
                    _pls = [float(l['ppm']) for l in (h.get('predicted_lines') or [])
                            if 'ppm' in l]
                    if len(_pls) < 2:
                        continue
                    _pk = pk_by_rank.get(pr, {})
                    _w = float(_pk.get('width_ppm', _pk.get('width_at_half', 0.008)) or 0.008)
                    _mar = max(_w * 0.7, 0.010)
                    _lo, _hi = min(_pls) - _mar, max(_pls) + _mar
                    for _orank, _op in pk_by_rank.items():
                        if _lo <= float(_op.get('ppm', -999)) <= _hi:
                            rank_to_name.setdefault(_orank, name)

            for _orank, _nm in rank_to_name.items():
                _p = pk_by_rank.get(_orank)
                if _p is not None:
                    _p['assigned_metabolite'] = _nm
                    _p['_quant_assigned'] = True

            plw = getattr(self.app, '_peak_list_win', None)
            if plw is not None and plw.win.winfo_exists():
                plw.peaks = self.peak_stage.peaks
                plw._populate()
        except Exception as _pll_err:
            print(f'[Quant] peak-list relabel skipped: {_pll_err}')

    # ── VIP scores ───────────────────────────────────────────────────────────

    def _compute_vip_scores(self):
        """Compute PLS-DA VIP scores and store in each result dict as 'vip_score'."""
        if not self._quant_results:
            return

        si      = self.app.samples_info or []
        results = self._quant_results

        n_s = max((len(r.get('_concs_per_sample', [])) for r in results), default=0)
        if n_s < 3:
            for r in results:
                r['vip_score'] = float('nan')
            return

        # Per-sample vectors can differ in length across analytes (e.g. an analyte
        # quantified before a sample was added). Pad/truncate each to exactly n_s
        # so the matrix is rectangular — a ragged list raises "inhomogeneous shape"
        # under numpy and silently skipped all VIP scores.
        rows = [reported_values(r, n_s)[0] for r in results]
        X = np.array(rows, dtype=float).T

        blk_idx = set(getattr(self, '_last_blk_indices', []))
        groups  = getattr(self, '_last_groups', {})

        non_blk_groups = sorted(g for g in groups if g != 'BLK')
        if len(non_blk_groups) < 2:
            for r in results:
                r['vip_score'] = float('nan')
            return

        g_label = {g: i for i, g in enumerate(non_blk_groups)}
        y_full  = np.full(n_s, float('nan'))
        for g, idxs in groups.items():
            if g == 'BLK':
                continue
            lbl = g_label.get(g, 0)
            for i in idxs:
                if i < n_s:
                    y_full[i] = lbl

        mask = np.isfinite(y_full)
        if mask.sum() < 3:
            for r in results:
                r['vip_score'] = float('nan')
            return

        X_sub = X[mask, :]
        y_sub = y_full[mask]

        n_comp = min(2, len(non_blk_groups) - 1, X_sub.shape[0] - 1, X_sub.shape[1])
        n_comp = max(1, n_comp)

        vip = _pls_vip(X_sub, y_sub, n_components=n_comp)

        for r, v in zip(results, vip):
            r['vip_score'] = float(v) if np.isfinite(v) else float('nan')

    # ── Per-sample area update (from Peak View manual adjustment) ────────────

    def _row_of_peak(self, peak_obj):
        """The quantification row an elected window belongs to: by the peak
        itself, else by its rank (a project reopened has other objects)."""
        results = self._quant_results or []
        r = next((x for x in results if x.get('_peak') is peak_obj), None)
        if r is None and peak_obj.get('rank') is not None:
            r = next((x for x in results
                      if x.get('_peak_rank') == peak_obj.get('rank')
                      and x.get('_multiplet_window')), None)
        return r

    @staticmethod
    def _from_a_fit(r) -> bool:
        """Whether a row's number is a fit's (joint fit, cascade, a window
        resolved by shape) and not the integral of a window."""
        return bool(r.get('_by_cascade') or r.get('_joint_fit')
                    or r.get('_fit_curve')
                    or (r.get('_deconv') and r.get('_used_j_resolve')))

    def sample_area_editable(self, peak_obj):
        """``(True, '')`` when a sample's window on this signal can be set by
        the analyst; else ``(False, why)``. No row yet is editable: the
        region waits and is applied after the next run."""
        r = self._row_of_peak(peak_obj)
        if r is None:
            return True, ''
        if self._from_a_fit(r):
            return False, ('Its number comes from a fit (joint fit, cascade or '
                           'a window resolved by shape), not from integrating '
                           'a window: there is no window to move. Change the '
                           'signal\'s boundaries or re-run with the fit off.')
        if not r.get('_multiplet_window'):
            return False, 'The engine kept no integration window for this signal.'
        return True, ''

    def _window_integral(self, row, sample_idx, lo, hi):
        """The integral of one sample over ``[lo, hi]`` on the spectrum the
        run integrated, with the run's baseline choice: the same function
        for the engine's window and for the analyst's, so only the window
        differs between them."""
        from processing.quant_integration import integrate_sample
        use_model = bool((getattr(self, '_last_run_settings', None) or {}).get(
            'bl_model_quant_var', self.bl_model_quant_var.get()))
        mat = (self.app.corrected_intensities() if use_model
               else self.app.all_intensities)
        col = np.asarray(mat, dtype=float)[:, sample_idx]
        return integrate_sample(np.asarray(self.app.ppm, dtype=float), col, lo, hi,
                                local_baseline=bool(row.get('_local_baseline')))

    def update_sample_area(self, peak_obj, sample_idx, lo_ppm, hi_ppm):
        """The analyst sets the window one sample of a signal is integrated
        on. Returns None when done, or why it could not be.

        The engine's number carries every correction its chain applies (the
        capture of the line, the neighbours' skirts, the proton count, the
        standard, PQN, the dilution). Re-doing them here would be a second,
        different engine, and the first version of this was off by the proton
        count (a factor of 9 on betaine). So the engine's number for that
        sample is scaled by how much the integral changed: the same integral,
        on the engine's window and on the analyst's. The same window gives the
        engine's number back exactly. A number that came from a fit has no
        window to move and is refused.
        """
        ok, why = self.sample_area_editable(peak_obj)
        if not ok:
            return why
        r = self._row_of_peak(peak_obj)
        if r is None:
            return None             # applied after the next run
        n_s = len(r.get('_concs_per_sample') or [])
        if not 0 <= int(sample_idx) < n_s:
            return 'That sample is not in the results.'
        s_i = int(sample_idx)
        # the engine's own numbers, kept the first time and never compounded
        if '_engine_concs_per_sample' not in r:
            r['_engine_concs_per_sample'] = list(r['_concs_per_sample'])
            r['_engine_areas_per_sample'] = list(r.get('_areas_per_sample') or [])
        engine_win = tuple(float(v) for v in r['_multiplet_window'])
        new_win = (float(min(lo_ppm, hi_ppm)), float(max(lo_ppm, hi_ppm)))
        manual = r.setdefault('_manual_regions', {})
        same = (abs(new_win[0] - min(engine_win)) < 1e-9
                and abs(new_win[1] - max(engine_win)) < 1e-9)
        ratio = 1.0
        if not same:
            i_old = self._window_integral(r, s_i, min(engine_win), max(engine_win))
            i_new = self._window_integral(r, s_i, *new_win)
            c0 = r['_engine_concs_per_sample'][s_i]
            if (i_old is None or i_new is None or not np.isfinite(i_old)
                    or i_old <= 0 or not np.isfinite(c0) or c0 <= 0):
                return ('The engine measured nothing in this sample on its own '
                        'window, so there is no number to rescale.')
            ratio = float(i_new) / float(i_old)
            manual[s_i] = new_win
        else:
            manual.pop(s_i, None)
        r['_concs_per_sample'][s_i] = (
            r['_engine_concs_per_sample'][s_i] * ratio)
        if r.get('_areas_per_sample') and s_i < len(r['_areas_per_sample']):
            r['_areas_per_sample'][s_i] = (
                r['_engine_areas_per_sample'][s_i] * ratio)
        self._refresh_row_statistics(r)
        return None

    def reset_sample_area(self, peak_obj, sample_idx):
        """Back to the engine's window and number for one sample."""
        r = self._row_of_peak(peak_obj)
        if r is None or '_engine_concs_per_sample' not in r:
            return
        return self.update_sample_area(peak_obj, sample_idx,
                                       *r['_multiplet_window'])

    def _refresh_row_statistics(self, r):
        """What follows from the per-sample numbers: the group statistics, the
        mean and the note that some samples were re-measured by hand. The
        identity, the status and the detection of the row are the engine's and
        are left as they are."""
        s_concs = np.asarray(r['_concs_per_sample'], dtype=float)
        groups = getattr(self, '_last_groups', {})
        blk = getattr(self, '_last_blk_indices', [])
        n_s = s_concs.size
        cv_by_group, concs_by_group, ci_by_group = {}, {}, {}
        for g, idxs in groups.items():
            vals = [float(s_concs[i]) for i in idxs
                    if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
            cv_by_group[g] = _cv(vals)
            concs_by_group[g] = vals
            _pt, _lo, _hi = _bootstrap_fraction_ci(vals)
            ci_by_group[g] = (_lo, _hi)
        if blk:
            blk_vals = [float(s_concs[i]) for i in blk
                        if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
            if blk_vals:
                concs_by_group['BLK'] = blk_vals
                cv_by_group['BLK'] = _cv(blk_vals)
        all_vals = [v for g, vv in concs_by_group.items()
                    if g != 'BLK' for v in vv if np.isfinite(v) and v > 0]
        r['conc_mm'] = float(np.mean(all_vals)) if all_vals else float('nan')
        r['cv_by_group'] = cv_by_group
        r['_concs_by_group'] = concs_by_group
        r['_conc_ci_by_group'] = ci_by_group
        areas = [a for a in (r.get('_areas_per_sample') or [])
                 if a is not None and np.isfinite(a)]
        if areas:
            r['raw_area'] = r['corr_area'] = float(np.mean(areas))
        reasons = [t for t in (r.get('_status_reasons') or [])
                   if not str(t).startswith('Re-measured by hand')]
        manual = r.get('_manual_regions') or {}
        if manual:
            reasons.insert(0, (
                f'Re-measured by hand in {len(manual)} sample(s) on a window '
                'the analyst set: the engine\'s number for each was scaled by '
                'the change in the integral (its corrections are kept); the '
                'other samples are the engine\'s.'))
        r['_status_reasons'] = reasons
        if self._right_widget:
            self._refresh_tree_row(r)

    def _reapply_manual_regions(self):
        """A run measures every sample on the engine's window; the windows the
        analyst set (stored on the signals) are applied on top of it, so they
        survive a re-run. Returns what could not be applied."""
        refused = []
        for pk in (getattr(self.peak_stage, 'peaks', None) or []):
            for s_i, adj in (pk.get('sample_regions') or {}).items():
                why = self.update_sample_area(
                    pk, int(s_i), adj.get('start_ppm'), adj.get('end_ppm'))
                if why:
                    refused.append((pk.get('rank'), int(s_i), why))
        return refused

    # ── Tree row refresh ─────────────────────────────────────────────────────

    def _refresh_tree_row(self, r):
        """Update the single tree row that corresponds to result r."""
        tree = self._tree
        iid = r.get('hmdb_id', r['metabolite'])
        if not tree.exists(iid):
            return
        _nm   = getattr(self, 'norm_var', None)
        _norm = _nm.get() if _nm else 'IS'
        conc  = r.get('conc_mm', float('nan'))
        if not np.isfinite(conc):
            conc_s = 'N/A'
        elif _norm in ('IS', 'IS+PQN'):
            conc_s = f'{conc:.4f}'
        elif _norm == 'Total':
            conc_s = f'{conc:.4f}'
        else:
            conc_s = f'{conc:.3e}'
        stat = r.get('status', 'good')
        sym  = '✓' if stat == 'good' else ('△' if stat == 'warn' else '✗')
        peak_rank = r.get('_peak', {})
        peak_rank = peak_rank.get('rank', '–') if peak_rank else '–'
        vip = r.get('vip_score', float('nan'))
        vip_s = f'{vip:.2f}' if np.isfinite(vip) else '—'
        tree.item(iid, tags=(stat,), values=(
            peak_rank,
            r['metabolite'],
            f"{r.get('ppm', 0):.4f}",
            conc_s,
            f"{r['raw_area']:.4e}",
            f"{r['snr']:.1f}",
            vip_s,
            f'{sym} {stat.capitalize()}',
        ))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, mode):
        if not self._quant_results:
            messagebox.showinfo('Export', 'No results to export.')
            return

        si      = self.app.samples_info or []
        n_s     = (self.app.all_intensities.shape[1]
                   if self.app.all_intensities is not None else 0)
        metabolites = [r['metabolite'] for r in self._quant_results]

        if mode == 'csv':
            path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV', '*.csv')],
                initialfile='quantification_results.csv')
            if not path:
                return
            try:
                # Collect all group names (excluding BLK), preserving insertion order
                all_groups = []
                for r in self._quant_results:
                    for g in r.get('cv_by_group', {}):
                        if g != 'BLK' and g not in all_groups:
                            all_groups.append(g)

                # Normalisation mode label
                norm_mode_lbl = getattr(self, '_last_norm_mode', '') or ''

                # Sample names and labels
                snames  = [si[i].get('name',  f'S{i+1}') if i < len(si) else f'S{i+1}'
                           for i in range(n_s)]
                slabels = [si[i].get('label', 'Unknown') if i < len(si) else 'Unknown'
                           for i in range(n_s)]

                # Quality columns (fixed), then one column per sample
                cv_hdrs  = [f'CV%_{g}' for g in all_groups]
                # Non-parametric bootstrap CI95 of each group mean (Bootstrap-2)
                ci_hdrs  = [h for g in all_groups
                            for h in (f'CI95lo_{g}', f'CI95hi_{g}')]
                quality_cols = (['ChemShift(ppm)', 'Width(ppm)', 'SNR', 'R2', 'Score',
                                 'IntegMode', 'Status', 'Norm',
                                 'Confidence', 'Origin', 'IdentityBasis']
                                + cv_hdrs + ci_hdrs)

                def _fmt(v):
                    try:
                        return f'{float(v):.4f}' if np.isfinite(float(v)) else 'NA'
                    except Exception:
                        return 'NA'

                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)

                    # Header row: Metabolite | HMDB_ID | quality cols | sample names
                    w.writerow(['Metabolite', 'HMDB_ID'] + quality_cols + snames)

                    # Group row: blank for quality cols | sample group labels
                    w.writerow(['Group', ''] + [''] * len(quality_cols) + slabels)

                    # One row per analyte (skip IS internal standard row and
                    # não-quantificados — these carry no measurable signal).
                    for r in self._quant_results:
                        if r.get('_is_is') or r.get('_not_quantified'):
                            continue
                        p = r.get('_peak') or {}
                        _im = r.get('_integ_mode', '')

                        ppm_val   = r.get('ppm', p.get('ppm', float('nan')))
                        # Bug 5 fix: for Multiplet mode use actual integration span
                        _mw = r.get('_multiplet_window')
                        if _im == 'Multiplet' and _mw and _mw[0] is not None:
                            width_val = abs(float(_mw[1]) - float(_mw[0]))
                        else:
                            width_val = p.get('width_ppm', float('nan'))
                        snr_val   = r.get('snr', p.get('snr', float('nan')))
                        r2_val    = p.get('gaussian_r2', float('nan'))
                        score_val = r.get('_ann_score', float('nan'))
                        status    = r.get('status', '')
                        hmdb_id   = r.get('hmdb_id', '')

                        cv_vals = [_fmt(r.get('cv_by_group', {}).get(g, float('nan')))
                                   for g in all_groups]
                        _ci_g   = r.get('_conc_ci_by_group', {})
                        ci_vals = []
                        for g in all_groups:
                            _lo, _hi = _ci_g.get(g, (float('nan'), float('nan')))
                            ci_vals += [_fmt(_lo), _fmt(_hi)]

                        qual_row = [
                            _fmt(ppm_val),
                            _fmt(width_val),
                            _fmt(snr_val),
                            _fmt(r2_val),
                            _fmt(score_val),
                            _im or norm_mode_lbl,
                            status,
                            norm_mode_lbl,
                            *_identity_columns(r),
                        ] + cv_vals + ci_vals

                        # 'n.d.' where the analyte was not detected (or its
                        # amount is not positive): a number there would be a
                        # measurement that does not exist. 'NA' where it was
                        # never measured (a blank, an excluded sample).
                        _vals, _nd = reported_values(r, n_s)
                        sample_vals = ['n.d.' if nd else _fmt(v)
                                       for v, nd in zip(_vals, _nd)]

                        w.writerow([r['metabolite'], hmdb_id] + qual_row + sample_vals)

                self._status_ok(f'CSV → {os.path.basename(path)}')
                messagebox.showinfo('Export', f'Saved:\n{path}')
            except Exception as e:
                messagebox.showerror('Export Error', str(e))

        elif mode == 'metaboanalyst':
            path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV', '*.csv')],
                initialfile='metaboanalyst_input.csv')
            if not path:
                return
            try:
                # Bug 7 fix: exclude BLK samples — MetaboAnalyst uses only
                # experimental samples. Indices of non-BLK samples only.
                # Samples the analyst excluded from the study are not columns
                # either: an all-NA sample is a column MetaboAnalyst would
                # have to guess about.
                _non_blk_idx = [
                    i for i, s in enumerate(si)
                    if str(s.get('type', s.get('label', ''))).upper() != 'BLK'
                    and not s.get('excluded', False)
                    and i < n_s
                ]
                names  = [si[i].get('name',  f'S{i+1}') for i in _non_blk_idx]
                labels = [si[i].get('label', 'Unknown')  for i in _non_blk_idx]
                _seen, _annot, _no_value = {}, [], []
                _table = []
                for r in self._quant_results:
                    # skip the internal standard and lipid index rows, and
                    # what has no measurable signal
                    if (r.get('_is_is') or r.get('_is_lipid_index')
                            or r.get('_not_quantified')):
                        continue
                    _vals, _nd = reported_values(r, n_s)
                    vals = [_vals[i] for i in _non_blk_idx]
                    # MetaboAnalyst needs one name per feature
                    name = str(r['metabolite'])
                    if name in _seen:
                        name = f"{name} [{r.get('hmdb_id') or _seen[name]}]"
                    _seen[name] = _seen.get(name, 0) + 1
                    if not any(np.isfinite(v) for v in vals):
                        _no_value.append(name)    # nothing for it to model
                        continue
                    conf, origin, basis = _identity_columns(r)
                    _table.append([name] + [f'{v:.6e}' if np.isfinite(v) else 'NA'
                                            for v in vals])
                    _annot.append([name, r.get('hmdb_id', ''), conf, origin, basis,
                                   sum(1 for v in vals if np.isfinite(v)),
                                   len(vals)])
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    # MetaboAnalyst format: row1=Sample+names, row2=Label+groups
                    w.writerow(['Sample'] + names)
                    w.writerow(['Label']  + labels)
                    w.writerows(_table)
                # The dots cannot ride inside MetaboAnalyst's format, so they
                # travel beside it: the analyst reads the confidence of every
                # feature before trusting what the statistics make of it.
                _ann_path = os.path.splitext(path)[0] + '_annotations.csv'
                with open(_ann_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['Feature', 'HMDB_ID', 'Confidence', 'Origin',
                                'IdentityBasis', 'N_detected', 'N_samples'])
                    w.writerows(_annot)
                self._status_ok(f'MetaboAnalyst → {os.path.basename(path)}')
                _note = (f'\n{len(_no_value)} metabolite(s) left out: not '
                         f'detected in any exported sample.' if _no_value else '')
                messagebox.showinfo(
                    'Export',
                    f'Saved:\n{path}\n\nConfidence of every feature (1 = a '
                    f'candidate to check, 3 = confirmed):\n{_ann_path}\n\n'
                    f'n.d. samples are NA.{_note}')
            except Exception as e:
                messagebox.showerror('Export Error', str(e))

    def _export_pdf(self):
        if not self._quant_results:
            messagebox.showinfo('Export PDF', 'No quantification results to export.')
            return

        path = filedialog.asksaveasfilename(
            defaultextension='.pdf',
            filetypes=[('PDF', '*.pdf')],
            initialfile='nmr_quantification_report.pdf',
            title='Export PDF Report')
        if not path:
            return

        self._status('Generating PDF report…')
        self.frame.update_idletasks()

        try:
            from gui.widgets.pdf_report import generate_pdf
            ann_stage = getattr(self, 'annotation_stage', None)
            generate_pdf(path, self, self.app, annotation_stage=ann_stage)
            self._status_ok(f'PDF saved: {path}')
            messagebox.showinfo('PDF Export', f'Report saved:\n{path}')
        except Exception as exc:
            self._status_err(f'PDF export failed: {exc}')
            messagebox.showerror('PDF Export Error', str(exc))
