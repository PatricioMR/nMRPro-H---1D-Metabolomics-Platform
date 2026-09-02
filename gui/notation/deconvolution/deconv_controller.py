# gui/notation/deconvolution/deconv_controller.py
"""
Peak-centric Deconvolution Controller.

Key design principle:
  processor.peaks is NEVER modified during deconvolution.
  Results are stored in self.results only.
  processor.peaks is updated ONLY when the user confirms Save in DeconvWindow.
"""
import numpy as np
from .deconv_algorithm import deconvolve_region as deconvolve_algorithm


def _extend_to_baseline(ppm, intensity, lo, hi, noise_std, max_ext_factor=2.0):
    """Extend (lo, hi) outward until spectrum drops to ≤ 3×noise.
    Caps extension at max_ext_factor × region_width per side."""
    region_mask = (ppm >= lo) & (ppm <= hi)
    if not np.any(region_mask):
        return lo, hi
    peak_max = float(intensity[region_mask].max())
    threshold = max(3.0 * noise_std, 0.02 * peak_max)
    width = hi - lo
    cap   = width * max_ext_factor

    # Scan left
    left_pts = ppm[(ppm < lo) & (ppm >= lo - cap)]
    new_lo = lo
    for p in left_pts[::-1]:   # descending from lo outward
        v = float(intensity[np.argmin(np.abs(ppm - p))])
        if v <= threshold:
            new_lo = float(p)
            break

    # Scan right
    right_pts = ppm[(ppm > hi) & (ppm <= hi + cap)]
    new_hi = hi
    for p in right_pts:        # ascending from hi outward
        v = float(intensity[np.argmin(np.abs(ppm - p))])
        if v <= threshold:
            new_hi = float(p)
            break

    return new_lo, new_hi


def _baseline_edge_metrics(x, y, noise_std, edge_fraction=0.12, min_edge_points=6):
    """Compute baseline-edge stability metrics for a selected region."""
    n = len(x)
    if n < (2 * min_edge_points + 2):
        return {
            'valid': False,
            'reason': 'region_too_narrow',
        }

    n_edge = max(min_edge_points, int(round(n * edge_fraction)))
    n_edge = min(n_edge, max(min_edge_points, n // 3))
    left = np.asarray(y[:n_edge], dtype=float)
    right = np.asarray(y[-n_edge:], dtype=float)

    left_level = float(np.median(np.abs(left)))
    right_level = float(np.median(np.abs(right)))
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    tilt = float(abs(np.median(right) - np.median(left)))

    sigma = float(noise_std) if noise_std and noise_std > 0 else float(
        np.median(np.abs(np.asarray(y) - np.median(y))) * 1.4826 + 1e-12)
    edge_limit = 2.5 * sigma
    tilt_limit = 1.5 * sigma

    left_ok = (left_level <= edge_limit) and (left_std <= edge_limit)
    right_ok = (right_level <= edge_limit) and (right_std <= edge_limit)
    tilt_ok = tilt <= tilt_limit
    valid = bool(left_ok and right_ok and tilt_ok)

    reason = 'ok'
    if not valid:
        if not (left_ok and right_ok):
            reason = 'edge_not_at_baseline'
        elif not tilt_ok:
            reason = 'edge_tilt_too_high'

    return {
        'valid': valid,
        'reason': reason,
        'left_level': left_level,
        'right_level': right_level,
        'left_std': left_std,
        'right_std': right_std,
        'tilt': tilt,
        'edge_limit': float(edge_limit),
        'tilt_limit': float(tilt_limit),
        'noise_std': float(sigma),
        'window_points': int(n),
    }

CONTEXT_PPM = 0.08  # display context around each peak


def _get_processor_axes_and_mean(processor):
    ppm = getattr(processor, 'ppm', None)
    mean_intensity = getattr(processor, 'mean_intensity', None)
    if ppm is not None and mean_intensity is not None:
        return ppm, mean_intensity

    app = getattr(processor, 'app', None)
    if app is None:
        return ppm, mean_intensity

    if ppm is None:
        ppm = getattr(app, 'ppm', None)

    if mean_intensity is None:
        raw = getattr(app, 'all_intensities', None)
        if raw is not None:
            if getattr(raw, 'ndim', 1) == 2:
                mean_intensity = np.mean(raw, axis=1)
            else:
                mean_intensity = raw

    if ppm is not None:
        processor.ppm = ppm
    if mean_intensity is not None:
        processor.mean_intensity = mean_intensity

    return ppm, mean_intensity


def _local_align_to_ref(y_samples, ref_idx):
    """
    Locally aligns the columns of y_samples by finding integer shifts 
    that maximize cross-correlation to a reference sample.
    """
    n_pts, n_samples = y_samples.shape
    if n_pts < 10 or n_samples < 2:
        return y_samples.copy()

    ref_sig = y_samples[:, ref_idx]
    aligned = np.zeros_like(y_samples)
    aligned[:, ref_idx] = ref_sig

    # Maximum allowed shift: 20% of the window
    max_shift = max(1, n_pts // 5)

    for k in range(n_samples):
        if k == ref_idx:
            continue
        sig = y_samples[:, k]
        corr = np.correlate(ref_sig, sig, mode='full')
        lag0 = len(sig) - 1

        search_start = max(0, lag0 - max_shift)
        search_end = min(len(corr), lag0 + max_shift + 1)

        best_local_idx = int(np.argmax(corr[search_start:search_end]))
        best_lag = (search_start + best_local_idx) - lag0

        if best_lag == 0:
            aligned[:, k] = sig
        else:
            shifted = np.empty_like(sig)
            if best_lag > 0:
                shifted[:best_lag] = sig[0]
                shifted[best_lag:] = sig[:-best_lag]
            else:
                s = -best_lag
                shifted[-s:] = sig[-1]
                shifted[:-s] = sig[s:]
            aligned[:, k] = shifted

    return aligned


class DeconvController:
    def __init__(self, processor):
        self.processor = processor
        self.peaks = []  # peak dicts queued for deconvolution
        self.results = {}  # peak_idx → result dict (active/last result)
        self.model_results = {}  # peak_idx → {model_name → result dict}
        self.params_used = {}  # peak_idx → params dict
        # excluded_samples: peak_idx → set of sample indices to exclude
        self.excluded_samples = {}

    # ── Peak management ───────────────────────────────────────────────────────

    def set_peaks(self, peaks):
        self.peaks = list(peaks)
        self.results = {}
        self.model_results = {}
        self.excluded_samples = {}

    def get_model_result(self, peak_idx, model_name):
        """Return cached result for (peak_idx, model_name), or None."""
        return self.model_results.get(peak_idx, {}).get(model_name)

    def get_peaks(self):
        return self.peaks

    def add_peak(self, peak, target_ppm=None):
        for i, p in enumerate(self.peaks):
            if abs(p['ppm'] - peak['ppm']) < 0.001:
                return i, self._find_sub(i, target_ppm)
        self.peaks.append(peak)
        return len(self.peaks) - 1, 0

    def add_region(self, lo, hi, target_ppm=None):
        """Add a custom (lo, hi) region as a new controller peak.
        Used by DeconvWindow.select_region when creating a manual region.
        Returns (region_idx, child_idx).
        """
        lo, hi = min(lo, hi), max(lo, hi)
        mid = (lo + hi) / 2
        # Check if this region already exists (within 1 mDa tolerance)
        for i, p in enumerate(self.peaks):
            p_lo = min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
            p_hi = max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
            if abs(p_lo - lo) < 0.001 and abs(p_hi - hi) < 0.001:
                return i, self._find_sub(i, target_ppm)
        # Match to existing processor peaks inside the region
        matched = [p for p in (self.processor.peaks or [])
                   if lo <= p['ppm'] <= hi]
        if matched:
            rep = max(matched, key=lambda p: p.get('height', 0))
            pk = dict(rep)
        else:
            pk = {
                'ppm':       mid,
                'height':    1.0,
                'snr':       1.0,
                'width_ppm': hi - lo,
                'width_at_half': hi - lo,
                'rank':      '?',
            }
        pk['start_ppm']           = hi    # NMR convention: start > end
        pk['end_ppm']             = lo
        pk['overlapping']         = True
        pk['from_overlap_search'] = False
        self.peaks.append(pk)
        return len(self.peaks) - 1, 0

    def _find_sub(self, peak_idx, target_ppm):
        if target_ppm is None:
            return 0
        subs = self.results.get(peak_idx, {}).get('peaks', [])
        if not subs:
            return 0
        return min(range(len(subs)),
                   key=lambda j: abs(subs[j]['ppm'] - target_ppm))

    def set_excluded_samples(self, peak_idx, excluded):
        """Set which sample indices to exclude for peak[peak_idx]."""
        self.excluded_samples[peak_idx] = set(excluded)

    def get_excluded_samples(self, peak_idx):
        return self.excluded_samples.get(peak_idx, set())

    # ── Spectrum extraction ───────────────────────────────────────────────────

    def get_peak_data(self, peak_idx, context_ppm=None):
        """Spectrum slice around peak[peak_idx] using processor.mean_intensity."""
        if peak_idx >= len(self.peaks):
            return None
        p = self.peaks[peak_idx]
        ppm_axis, mean_intensity = _get_processor_axes_and_mean(self.processor)
        if ppm_axis is None or mean_intensity is None:
            return None
        ctx = context_ppm if context_ppm is not None else CONTEXT_PPM
        lo = min(p.get('start_ppm', p['ppm']),
                 p.get('end_ppm', p['ppm'])) - ctx
        hi = max(p.get('start_ppm', p['ppm']),
                 p.get('end_ppm', p['ppm'])) + ctx
        if (hi - lo) < 0.1:
            mid = (lo + hi) / 2
            lo, hi = mid - 0.05, mid + 0.05
        mask = (ppm_axis >= lo) & (ppm_axis <= hi)
        ppm = ppm_axis[mask]
        y = mean_intensity[mask]
        if len(ppm) < 3:
            return None
        return {'ppm': ppm, 'intensity': y, 'lo': lo, 'hi': hi}

    def get_mean_excluding(self, peak_idx):
        """Compute mean excluding excluded_samples for peak[peak_idx].
        Returns processor.mean_intensity if no exclusions or no raw data."""
        _get_processor_axes_and_mean(self.processor)
        excluded = self.get_excluded_samples(peak_idx)
        if not excluded:
            return self.processor.mean_intensity
        app = getattr(self.processor, 'app', None)
        raw = getattr(app, 'all_intensities', None)
        si = getattr(app, 'samples_info', [])
        if raw is None or raw.ndim < 2:
            return self.processor.mean_intensity
        # Build group mean from non-BLK, non-excluded samples
        from collections import defaultdict
        groups = defaultdict(list)
        for i, s in enumerate(si):
            stype = s.get('type', s.get('label', 'Sample'))
            if str(stype).upper() != 'BLK' \
                    and not s.get('excluded', False) \
                    and i not in excluded \
                    and i < raw.shape[1]:
                groups[s.get('label', 'unknown')].append(i)
        if not groups:
            return self.processor.mean_intensity
        gmeans = np.stack(
            [np.mean(raw[:, idxs], axis=1) for idxs in groups.values()],
            axis=1)
        return np.mean(gmeans, axis=1)

    # ── Deconvolution ─────────────────────────────────────────────────────────

    def deconvolve_peak(self, peak_idx, params):
        """
        Run deconvolution for peak[peak_idx].
        Stores result in self.results — does NOT touch processor.peaks.
        Respects excluded_samples by recomputing mean without them.
        """
        if peak_idx >= len(self.peaks):
            return None
        p = self.peaks[peak_idx]
        ppm_axis, _ = _get_processor_axes_and_mean(self.processor)
        if ppm_axis is None:
            return None

        # Fit window = peak boundaries (tight)
        fit_lo = min(p.get('start_ppm', p['ppm']),
                     p.get('end_ppm', p['ppm']))
        fit_hi = max(p.get('start_ppm', p['ppm']),
                     p.get('end_ppm', p['ppm']))
        if (fit_hi - fit_lo) < 0.01:
            mid = (fit_lo + fit_hi) / 2
            fit_lo, fit_hi = mid - 0.005, mid + 0.005

        # Use mean excluding removed samples
        mean_vec = self.get_mean_excluding(peak_idx)

        # Auto-extend region to stable baseline on both sides
        noise_std = getattr(self.processor, 'noise_std', None)
        if noise_std and noise_std > 0:
            fit_lo, fit_hi = _extend_to_baseline(
                ppm_axis, mean_vec, fit_lo, fit_hi, noise_std)

        mask = (ppm_axis >= fit_lo) & (ppm_axis <= fit_hi)
        x = ppm_axis[mask]
        y = mean_vec[mask]
        if len(x) < 5:
            return None

        baseline_validation = _baseline_edge_metrics(x, y, noise_std)
        if not baseline_validation.get('valid', False):
            reason = baseline_validation.get('reason', '')
            print(f"   ⚠ Baseline edge warning ({reason}) — proceeding with deconvolution anyway.")

        app = getattr(self.processor, 'app', None)
        raw = getattr(app, 'all_intensities', None)

        if raw is not None and raw.ndim >= 2:
            si_list = getattr(app, 'samples_info', [])
            excl = set(self.get_excluded_samples(peak_idx))
            # also exclude samples deselected in the visualization panel or typed BLK
            excl |= {k for k, s in enumerate(si_list)
                     if s.get('excluded', False)
                     or str(s.get('type', '')).upper() == 'BLK'}
            valid_cols = [k for k in range(raw.shape[1]) if k not in excl]
            if not valid_cols:
                valid_cols = list(range(raw.shape[1]))

            y_samples_full = raw[mask, :]

            # Find the best reference sample among valid samples (one with highest peak)
            max_vals = np.max(y_samples_full[:, valid_cols], axis=0)
            best_valid_local_idx = int(np.argmax(max_vals))
            ref_idx = valid_cols[best_valid_local_idx]

            # Align samples for per-sample NNLS only — do NOT change y used for the
            # main fit so that the displayed spectrum (proc.mean_intensity) and the
            # fitted curves share the same amplitude scale.
            y_samples_aligned = _local_align_to_ref(y_samples_full, ref_idx)
            y_samples = y_samples_aligned

            # ── Per-group SNR check ───────────────────────────────────────────
            # Group valid columns by label, measure peak SNR in the fit region.
            # Groups below min_snr are excluded from the fit mean but still
            # included in y_samples — their per-sample areas are zeroed afterwards.
            from collections import defaultdict as _dd
            _min_snr   = float(params.get('min_snr', 2.5))
            _ns        = max(float(noise_std), 1e-10) if noise_std and noise_std > 0 else 1e-10
            _group_cols: dict = _dd(list)
            for k in valid_cols:
                if k < len(si_list):
                    _lbl = si_list[k].get('label', 'Unknown')
                else:
                    _lbl = 'Unknown'
                _group_cols[_lbl].append(k)

            below_snr_groups: dict = {}  # label → list of col indices
            above_snr_cols: list  = []
            for _lbl, _cols in _group_cols.items():
                _gmean = np.mean(y_samples_full[:, _cols], axis=1)
                _peak_val = float(np.max(_gmean))
                _snr_val  = _peak_val / _ns
                if _snr_val < _min_snr:
                    below_snr_groups[_lbl] = _cols
                    print(f"   ℹ Group '{_lbl}': SNR={_snr_val:.2f} < {_min_snr} — will be zeroed")
                else:
                    above_snr_cols.extend(_cols)

            # Use only above-SNR cols for the fit mean; fall back to all valid if none pass
            fit_cols = above_snr_cols if above_snr_cols else valid_cols
            y = np.mean(y_samples_full[:, fit_cols], axis=1)
            # ─────────────────────────────────────────────────────────────────
        else:
            y_samples       = None
            below_snr_groups = {}
            fit_cols         = []

        self.params_used[peak_idx] = params
        p_full = dict(params)
        p_full['noise_std'] = getattr(self.processor, 'noise_std', None)
        p_full['require_baseline_edges'] = False
        p_full['baseline_edges_valid'] = bool(baseline_validation.get('valid', False))
        p_full['baseline_validation'] = baseline_validation
        result = deconvolve_algorithm(
            x, y, p_full,
            progress_callback=params.get('_progress_cb'),
            y_samples=y_samples)

        # Zero out per-sample data for below-SNR groups
        if result and result.get('peaks') and below_snr_groups:
            _zero_cols = [c for cols in below_snr_groups.values() for c in cols]
            for sp in result['peaks']:
                for arr_key in ('area_per_sample', 'height_per_sample'):
                    arr = sp.get(arr_key)
                    if arr is not None:
                        for _c in _zero_cols:
                            if _c < len(arr):
                                arr[_c] = 0.0
            result['below_snr_groups'] = list(below_snr_groups.keys())

        if result and result.get('peaks'):
            ns = max(getattr(self.processor, 'noise_std', 1e-10), 1e-10)
            r2_region = result.get('r_squared', None)
            bic_region = result.get('bic', None)

            excl_list = sorted(self.get_excluded_samples(peak_idx))
            # Deconv R² threshold — from app if available, else default 0.95
            _app = getattr(self.processor, 'app', None)
            _deconv_r2 = float(getattr(_app, '_deconv_r2_min', 0.95))

            for sp in result['peaks']:
                sp['deconvolved'] = False
                sp['is_sub_peak'] = True
                sp['overlapping'] = False
                sp['parent_ppm'] = p['ppm']
                sp['excluded_samples'] = excl_list
                if r2_region is not None: sp.setdefault('r_squared', r2_region)
                if bic_region is not None: sp.setdefault('bic', bic_region)
                # Pre-assign status based on deconv R²:
                # Poor  → R² below threshold or unknown
                # None  → good fit, user reviews in Peak List
                r2_sp = sp.get('r_squared', r2_region)
                if r2_sp is None or float(r2_sp) < _deconv_r2:
                    sp['status'] = 'Poor'
                else:
                    sp['status'] = None  # user assigns Good/Average/Poor
                w = sp.get('width', sp.get('width_ppm', 0.005))
                sp['width_ppm'] = float(w)
                sp['width_at_half'] = float(w)
                sp.setdefault('start_ppm', sp['ppm'] + 2 * w)
                sp.setdefault('end_ppm', sp['ppm'] - 2 * w)
                sp.setdefault('snr', sp['height'] / ns)
                sp.setdefault('prominence', sp['height'] * 0.5)

            # Build lightweight sibling list — stored on each sub-peak so
            # PeakSampleView can reconstruct the full deconvolution in context
            # without needing a reference back to the controller.
            _sub_keys = ('ppm', 'width', 'width_ppm', 'height',
                         'height_per_sample', 'area_per_sample',
                         'model', 'eta', 'gamma',
                         'start_ppm', 'end_ppm')
            region_sub_peaks = [
                {k: sp[k] for k in _sub_keys if k in sp}
                for sp in result['peaks']
            ]
            for sp in result['peaks']:
                sp['region_sub_peaks'] = region_sub_peaks
                sp['region_ppm']       = x       # full ppm axis of the fitted region
                sp['region_intensity'] = y        # real mean spectrum of the region

            self.peaks[peak_idx]['_deconvolved'] = True
            result.update({
                'peak_ppm': p['ppm'],
                'region_ppm': x,
                'region_intensity': y,
                'params_used': params,
                'baseline_validation': baseline_validation,
                'excluded_samples': list(self.get_excluded_samples(peak_idx)),
            })
            self.results[peak_idx] = result
            model_name = result.get('model', params.get('model', 'Lorentzian'))
            self.model_results.setdefault(peak_idx, {})[model_name] = result

        return result

    def restore_peak(self, peak_idx):
        """Undo deconvolution for peak[peak_idx].
        Removes result and clears _deconvolved flag.
        The original peak stays in self.peaks untouched.
        """
        if peak_idx in self.results:
            del self.results[peak_idx]
        self.model_results.pop(peak_idx, None)
        if peak_idx < len(self.peaks):
            self.peaks[peak_idx].pop('_deconvolved', None)
            self.peaks[peak_idx].pop('has_sub_peaks', None)
            self.peaks[peak_idx].pop('deconvolved', None)

    def refit_confirmed_peaks(self, peak_idx):
        """Re-fit the region using ONLY the non-Poor sub-peaks.

        When the user marks some sub-peaks as Poor, the remaining confirmed
        peaks carry amplitudes from an N-component fit where N was larger.
        This method re-optimises the confirmed peaks for a reduced-component
        fit — giving correct amplitudes and areas.

        Updates self.results[peak_idx]['peaks'] in-place (only confirmed ones).
        """
        if peak_idx not in self.results:
            return
        res = self.results[peak_idx]
        all_sub = res.get('peaks', [])
        confirmed = [sp for sp in all_sub if sp.get('status') != 'Poor']
        n_poor = len(all_sub) - len(confirmed)

        if n_poor == 0 or not confirmed:
            return  # nothing to refit

        x = np.asarray(res.get('region_ppm', []), dtype=float)
        y = np.asarray(res.get('region_intensity', []), dtype=float)
        if len(x) < 5:
            return

        from .deconv_algorithm import refit_with_selected_peaks
        model = res.get('model', 'Lorentzian')
        params = res.get('params_used', {})
        min_width = float(params.get('min_width', 0.002))
        noise_std = getattr(self.processor, 'noise_std', None)

        selected_ppms = [sp['ppm'] for sp in confirmed]
        new_peaks_data, r2, bic = refit_with_selected_peaks(
            x, y, selected_ppms,
            model=model, min_width=min_width, noise_std=noise_std)

        if new_peaks_data is None:
            print(f"   refit_confirmed_peaks({peak_idx}): refit failed — keeping original amplitudes")
            return

        print(f"   refit_confirmed_peaks({peak_idx}): {n_poor} Poor removed, "
              f"{len(confirmed)} confirmed re-fitted  R²={r2:.4f}")

        # Update confirmed sub-peaks in-place
        for sp, nd in zip(confirmed, new_peaks_data):
            sp['ppm']       = nd['ppm']
            sp['height']    = nd['height']
            sp['width']     = nd['width']
            sp['width_ppm'] = nd['width_ppm']
            sp['area']      = nd['area']
            for k in ('eta', 'gamma'):
                if k in nd:
                    sp[k] = nd[k]
            sp['r_squared'] = r2

        # Update region-level quality metrics
        res['r_squared'] = r2
        if bic is not None:
            res['bic'] = bic

        # Rebuild region_sub_peaks sibling list in each confirmed sub-peak
        _sub_keys = ('ppm', 'width', 'width_ppm', 'height', 'area', 'model')
        region_subs = [{k: sp[k] for k in _sub_keys if k in sp} for sp in confirmed]
        for sp in confirmed:
            sp['region_sub_peaks'] = region_subs

    def deconvolve_all(self, params):
        done = []
        for i, p in enumerate(self.peaks):
            if not p.get('_deconvolved'):
                r = self.deconvolve_peak(i, params)
                if r and r.get('peaks'):
                    done.append(r)
        return done


    # ── Backward-compat shims ─────────────────────────────────────────────────


    @property
    def regions(self):
        return [(p.get('end_ppm', p['ppm']), p.get('start_ppm', p['ppm']))
                for p in self.peaks]

    @property
    def _region_meta(self):
        out = []
        for i, p in enumerate(self.peaks):
            res = self.results.get(i, {})
            out.append({
                'start': max(p.get('start_ppm', p['ppm']),
                             p.get('end_ppm', p['ppm'])),
                'end': min(p.get('start_ppm', p['ppm']),
                           p.get('end_ppm', p['ppm'])),
                'status': 'deconvolved' if p.get('_deconvolved') else 'pending',
                'n_peaks': len(res.get('peaks', [])) if p.get('_deconvolved') else 1,
                'peaks': [p],
            })
        return out

    def get_regions(self):
        """Shim for DeconvListPanel — returns region-style dicts."""
        out = []
        for i, p in enumerate(self.peaks):
            res = self.results.get(i, {})
            out.append({
                'status': 'deconvolved' if p.get('_deconvolved') else 'pending',
                'n_peaks': len(res.get('peaks', [])),
                'peaks': [p],
            })
        return out

    def get_region_data(self, region_idx, context_ppm=None):
        """Shim for DeconvListPanel — returns spectrum slice for region."""
        return self.get_peak_data(region_idx, context_ppm)

    def set_regions(self, regions):
        """Convert (lo,hi) region tuples into sub-regions of ≤ 3 peaks each.
        Each sub-region gets one representative entry (tallest peak) whose
        start_ppm/end_ppm span the sub-region so deconvolution fits all peaks
        inside it.
        """
        _MAX = 3
        new_peaks = []
        for lo, hi in regions:
            lo, hi = min(lo, hi), max(lo, hi)
            matched = sorted(
                [p for p in (self.processor.peaks or []) if lo <= p['ppm'] <= hi],
                key=lambda p: p['ppm'])
            if matched:
                n = len(matched)
                for i in range(0, n, _MAX):
                    chunk = matched[i:i + _MAX]
                    sub_lo = lo if i == 0 else (matched[i - 1]['ppm'] + matched[i]['ppm']) / 2
                    last   = min(i + _MAX, n) - 1
                    sub_hi = hi if (i + _MAX >= n) else (matched[last]['ppm'] + matched[last + 1]['ppm']) / 2
                    rep = max(chunk, key=lambda p: p.get('height', 0))
                    rp = dict(rep)
                    rp['start_ppm'] = sub_hi
                    rp['end_ppm']   = sub_lo
                    rp['overlapping'] = True
                    rp['_grouped_from'] = [p['ppm'] for p in chunk]
                    new_peaks.append(rp)
            else:
                new_peaks.append({
                    'ppm': (lo + hi) / 2, 'start_ppm': hi, 'end_ppm': lo,
                    'height': 1.0, 'snr': 1.0,
                    'width_ppm': hi - lo, 'width_at_half': hi - lo,
                    'rank': '?', 'overlapping': True,
                    'from_overlap_search': True,
                })
        self.set_peaks(new_peaks)