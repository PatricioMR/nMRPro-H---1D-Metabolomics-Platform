# gui/notation/peak_processing.py
"""
Peak detection using the P-Bin algorithm with prominence + width validation.

References:
  - AlpsNMR (Madrid-Gambin et al., Bioinformatics 2020): SNR.Th=4, CWT scales 1-16
  - Zhang et al., Bioinformatics 2006: true peaks must have characteristic shape
    across multiple scales — implemented here via scipy prominence + width criteria.
  - speaq 2.0 (Beirnaert et al., PLoS Comp Biol 2018): wavelet-based width validation

Key improvement over a simple local-maxima search:
  1. prominence  — peak must stand out from its local neighbourhood by at least
                   (prominence_factor × noise_std). This enforces the
                   shoulder-head-shoulder criterion: the "head" must rise above
                   both shoulders by a meaningful amount.
  2. width       — peak must be at least min_width_ppm wide at half prominence.
                   Eliminates 1-2 point noise spikes that pass the SNR threshold
                   but are too narrow to be real NMR resonances.
  3. SNR         — absolute height must still exceed noise × min_snr.

Consensus overlap detection (AlpsNMR-style, find_overlapping_regions_consensus):
  - Each non-BLK spectrum is analysed individually.
  - A region is reported as overlapping only when detected in >= consensus_threshold
    fraction of spectra (default 0.5 = 50%), making the result robust to noisy or
    atypical samples that would distort a simple mean spectrum.
"""

import numpy as np
from scipy.signal import find_peaks
import re

from processing.peak_alignment import global_peak_rigid_align


def trapz_compat(y, dx=1.0):
    """Compatible trapz for all NumPy versions."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, dx=dx)
    return np.trapz(y, dx=dx)


class PeakProcessor:
    def __init__(self, app):
        self.app           = app
        self.peaks         = []
        self.buckets       = []
        self.mean_intensity = None
        self.noise_std     = None
        self.ppm           = None
        print("[OK] PeakProcessor initialized")

    def set_noise(self, noise_std):
        self.noise_std = noise_std

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect_peaks(self, ppm, intensities, min_snr=3,
                     min_width_ppm=0.003, prominence_factor=1.5,
                     shape_r2_min=0.90):
        """
        Detect NMR peaks using the pointwise maximum of the best group as
        the reference spectrum, combined CWT + find_peaks detection.

        Design philosophy
        -----------------
        Over-filtering loses real peaks and frustrates users.
        Better to detect generously and let the user review in Peak List.
        The only hard filters are:
          1. SNR (height / noise_std >= min_snr)
          2. Prominence (must stand out from local baseline)
          3. Minimum width (eliminates 1-2 pt noise spikes)

        Reference spectrum
        ------------------
        Pointwise max of the highest-intensity group. Using max instead of
        mean ensures every real peak in that group appears at full height.
        A peak that exists only in group X (not Y) is still detected here
        and visible in the Peak List for the user to evaluate.

        References
        ----------
        - Smith et al., Anal Chem 2006 (MassSpecWavelet / CWT detection)
        - Beirnaert et al., PLoS Comp Biol 2018 (speaq 2.0)
        - AlpsNMR (Madrid-Gambin et al., Bioinformatics 2020)
        """
        print(f"\n[INFO] PEAK DETECTION")
        print(f"   min_snr={min_snr}  min_width={min_width_ppm} ppm  "
              f"prominence_factor={prominence_factor}")
        print(f"   noise_std={self.noise_std:.4f}")

        if intensities.ndim == 1:
            intensities = intensities.reshape(-1, 1)
        if intensities.shape[1] > 1:
            intensities, shifts_pts, shifts_ppm, _ = global_peak_rigid_align(
                ppm,
                intensities,
                samples_info=getattr(self.app, 'samples_info', None),
                max_shift_ppm=0.03,
            )
            print(
                "   Global rigid alignment before detection: "
                f"min={np.min(shifts_ppm) * 1000:.2f} mppm  "
                f"max={np.max(shifts_ppm) * 1000:.2f} mppm"
            )
        n_samples = intensities.shape[1]
        self.ppm  = ppm
        ppm_step  = abs(float(ppm[1] - ppm[0]))

        # ── Reference spectrum: pointwise max of the best group ───────────────
        si = getattr(self.app, 'samples_info', [])
        from collections import defaultdict
        groups = defaultdict(list)
        for k in range(n_samples):
            label = (si[k].get('label', 'unknown')
                     if k < len(si) else 'unknown')
            # exclude BLK and user-deselected samples from reference spectrum
            s_info = si[k] if k < len(si) else {}
            stype = s_info.get('type', 'Sample')
            if (str(stype).upper() != 'BLK'
                    and not s_info.get('excluded', False)):
                groups[label].append(k)

        if groups:
            # Pointwise maximum of per-group means.
            # Each group contributes its own mean spectrum; we then take the
            # element-wise max across all group means.
            # Result: a peak present in ANY group is preserved at that group's
            # mean height. Single-sample noise spikes are NOT promoted (we use
            # means, not raw values). Metabolites unique to one group (e.g.
            # Alanine in group A only) are still detected.
            group_means = np.stack(
                [np.mean(intensities[:, cols], axis=1) for cols in groups.values()],
                axis=1)           # shape: (n_points, n_groups)
            ref = np.max(group_means, axis=1).astype(float)
            print(f"   Reference: pointwise-max of {len(groups)} group mean(s) "
                  f"({', '.join(f'{k}({len(v)})' for k, v in groups.items())})")
        else:
            ref = np.mean(intensities, axis=1).astype(float)
            print("   Reference: global mean (no labelled groups found)")

        self.mean_intensity = ref

        # ── Thresholds ────────────────────────────────────────────────────────
        min_height     = self.noise_std * min_snr
        min_prominence = self.noise_std * prominence_factor
        min_width_pts  = max(2, min_width_ppm / ppm_step)

        print(f"   min_height={min_height:.4f}  "
              f"min_prominence={min_prominence:.4f}  "
              f"min_width_pts={min_width_pts:.1f}")

        # ── CWT detection (multi-scale, catches shoulders) ────────────────────
        from scipy.signal import find_peaks_cwt, find_peaks

        min_scale = max(2, int(min_width_ppm / ppm_step))
        max_scale = max(min_scale + 4, int(0.12 / ppm_step))
        scales = np.unique(
            np.round(np.geomspace(min_scale, max_scale, 16)).astype(int))

        def ricker(points, a):
            vec = np.arange(points) - (points - 1) / 2.0
            return (2/(np.sqrt(3*a)*np.pi**0.25) *
                    (1-(vec/a)**2) * np.exp(-vec**2/(2*a**2)))

        try:
            cwt_idx = find_peaks_cwt(
                ref, widths=scales, wavelet=ricker,
                min_snr=max(1.0, min_snr * 0.4),
                noise_perc=10)
        except Exception:
            cwt_idx = np.array([], dtype=int)

        # ── Classic find_peaks (height + prominence + width) ─────────────────
        classic_idx, classic_props = find_peaks(
            ref,
            height=min_height,
            prominence=min_prominence,
            width=min_width_pts,
            rel_height=0.5,
        )

        # ── Union of CWT + classic candidates ─────────────────────────────────
        candidates = set(classic_idx.tolist())
        for ci in cwt_idx:
            if ref[ci] >= min_height:
                lo = max(0, ci - 3)
                hi = min(len(ref) - 1, ci + 3)
                lm = lo + int(np.argmax(ref[lo:hi+1]))
                candidates.add(lm)

        candidates = sorted(c for c in candidates if ref[c] >= min_height)
        print(f"   Candidates (CWT ∪ classic): {len(candidates)}")

        if not candidates:
            self.peaks = []
            self.buckets = []
            print("   No peaks above SNR threshold.")
            return self.peaks

        cand_arr = np.array(candidates)

        # ── Recompute prominence + width for all candidates ───────────────────
        from scipy.signal import peak_prominences, peak_widths
        try:
            proms, _, _ = peak_prominences(ref, cand_arr)
            widths_pts, _, wl, wr = peak_widths(ref, cand_arr, rel_height=0.5)
        except Exception:
            proms      = np.full(len(cand_arr), min_prominence)
            widths_pts = np.full(len(cand_arr), min_width_pts)
            wl = cand_arr - widths_pts / 2
            wr = cand_arr + widths_pts / 2

        # Filter: prominence must meet threshold
        keep = proms >= min_prominence * 0.6
        cand_arr   = cand_arr[keep]
        proms      = proms[keep]
        widths_pts = widths_pts[keep]
        wl         = wl[keep]
        wr         = wr[keep]

        print(f"   After prominence filter: {len(cand_arr)} peaks")

        # ── Skewed-Lorentzian shape filter with neighbour subtraction ───────
        #
        # For each peak candidate, fits a Skewed-Lorentzian to verify it has
        # a real NMR peak shape (R² ≥ 0.90).
        #
        # Skewed-Lorentzian: L(x) = A / (1 + ((x-c)/w_eff)²)
        #   where w_eff = w·(1 + skew·sign(x-c))
        #   skew=0 → symmetric Lorentzian
        #   skew≠0 → asymmetric (common in NMR due to shimming, susceptibility)
        #
        # Handling overlapping peaks — iterative neighbour subtraction:
        #   1. For each peak, detect if neighbours share the base
        #      (valley height > 20% of the smaller apex → overlapping)
        #   2. If overlapping: fit the neighbour quickly, subtract its model
        #      from the spectrum, then fit the current peak on the residual
        #   3. R² is measured on the neighbour-subtracted spectrum
        #
        # This is a simplified 1-iteration version of the BATMAN/NMRProcFlow
        # approach (Hao et al. 2014): fit neighbours first, subtract, then
        # evaluate the peak of interest on a cleaner signal.
        #
        # Threshold: R² ≥ 0.90

        from scipy.optimize import curve_fit

        def _skewed_lorentz(x, amp, cen, wid, skew):
            dx    = x - cen
            w_eff = np.maximum(wid * (1 + skew * np.sign(dx)), wid * 0.05)
            return amp / (1 + (dx / w_eff) ** 2)

        r2_min    = float(shape_r2_min)
        shape_ok  = np.ones(len(cand_arr), dtype=bool)
        shape_r2  = np.full(len(cand_arr), -1.0)   # -1 = not computed / failed

        sorted_c = np.sort(cand_arr)
        val_map  = {}
        for _j in range(len(sorted_c) - 1):
            _a, _b = sorted_c[_j], sorted_c[_j+1]
            val_map[(_a, _b)] = _a + int(np.argmin(ref[_a:_b+1]))

        # Cache fitted models for neighbours (avoids re-fitting)
        _fitted_model = {}   # idx → (popt or None)

        def _fit_sl(x_seg, y_seg, idx_local, apex):
            """Quick Skewed-Lorentz fit. Returns popt or None.
            x_seg is in ppm — may be decreasing (NMR convention).
            All bounds are computed with abs() to handle either direction."""
            try:
                cen0   = float(x_seg[idx_local])
                span   = abs(float(x_seg[-1] - x_seg[0]))
                w0     = max(abs(ppm_step) * 3, span / 6)
                w_max  = max(span / 2, w0 * 2)
                cen_lo = min(float(x_seg[0]), float(x_seg[-1]))
                cen_hi = max(float(x_seg[0]), float(x_seg[-1]))
                p0     = [apex, cen0, w0, 0.0]
                lo_b   = [0,      cen_lo, abs(ppm_step), -1.5]
                hi_b   = [apex * 5, cen_hi, w_max,        1.5]
                popt, _ = curve_fit(
                    _skewed_lorentz, x_seg, y_seg,
                    p0=p0, bounds=(lo_b, hi_b), maxfev=600)
                return popt
            except Exception:
                return None

        ppm_x = ppm.astype(float)  # use ppm as x-axis for physical units

        for k, idx in enumerate(cand_arr):
            try:
                pos_s  = int(np.where(sorted_c == idx)[0][0])
                apex   = float(ref[idx])
                hw_pts = max(6, int(widths_pts[k] * 3))
                lo     = max(0, idx - hw_pts)
                hi     = min(len(ref) - 1, idx + hw_pts)

                # Adjacent valleys
                v_lo = lo
                v_hi = hi
                if pos_s > 0:
                    v_lo = val_map.get((sorted_c[pos_s-1], idx), lo)
                if pos_s < len(sorted_c) - 1:
                    v_hi = val_map.get((idx, sorted_c[pos_s+1]), hi)

                # Detect overlap
                left_apex  = float(ref[sorted_c[pos_s-1]]) if pos_s > 0 else 0.
                right_apex = float(ref[sorted_c[pos_s+1]]) if pos_s < len(sorted_c)-1 else 0.
                ov_left  = (float(ref[v_lo])/min(apex,left_apex)
                            if left_apex > 0 and v_lo != lo else 0.)
                ov_right = (float(ref[v_hi])/min(apex,right_apex)
                            if right_apex > 0 and v_hi != hi else 0.)
                is_overlapping = ov_left > 0.20 or ov_right > 0.20

                # Working spectrum — start with ref, subtract neighbours if needed
                y_work = ref[lo:hi+1].astype(float)
                x_seg  = ppm_x[lo:hi+1]

                if is_overlapping:
                    # Subtract left neighbour if overlapping
                    if ov_left > 0.20 and pos_s > 0:
                        nb_idx = sorted_c[pos_s - 1]
                        if nb_idx not in _fitted_model:
                            hw_nb  = max(6, int(widths_pts[pos_s-1] * 3)
                                         if pos_s-1 < len(widths_pts) else hw_pts)
                            lo_nb  = max(0, nb_idx - hw_nb)
                            hi_nb  = min(len(ref)-1, nb_idx + hw_nb)
                            _fitted_model[nb_idx] = _fit_sl(
                                ppm_x[lo_nb:hi_nb+1],
                                ref[lo_nb:hi_nb+1],
                                nb_idx - lo_nb,
                                float(ref[nb_idx]))
                        popt_nb = _fitted_model[nb_idx]
                        if popt_nb is not None:
                            y_work -= _skewed_lorentz(x_seg, *popt_nb)

                    # Subtract right neighbour if overlapping
                    if ov_right > 0.20 and pos_s < len(sorted_c) - 1:
                        nb_idx = sorted_c[pos_s + 1]
                        if nb_idx not in _fitted_model:
                            hw_nb  = max(6, int(widths_pts[pos_s+1] * 3)
                                         if pos_s+1 < len(widths_pts) else hw_pts)
                            lo_nb  = max(0, nb_idx - hw_nb)
                            hi_nb  = min(len(ref)-1, nb_idx + hw_nb)
                            _fitted_model[nb_idx] = _fit_sl(
                                ppm_x[lo_nb:hi_nb+1],
                                ref[lo_nb:hi_nb+1],
                                nb_idx - lo_nb,
                                float(ref[nb_idx]))
                        popt_nb = _fitted_model[nb_idx]
                        if popt_nb is not None:
                            y_work -= _skewed_lorentz(x_seg, *popt_nb)

                # Fit Skewed-Lorentzian to (possibly neighbour-subtracted) segment
                apex_w = max(float(y_work[idx - lo]), self.noise_std)
                popt = _fit_sl(x_seg, y_work, idx - lo, apex_w)

                if popt is None:
                    # fit failed — keep peak but mark as Poor quality
                    shape_ok[k] = True
                    shape_r2[k] = -1.0   # signals "fit failed"
                    continue

                y_fit  = _skewed_lorentz(x_seg, *popt)
                ss_res = float(np.sum((y_work - y_fit) ** 2))
                ss_tot = float(np.sum((y_work - y_work.mean()) ** 2))
                r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                shape_ok[k]  = True          # always keep — status set below
                shape_r2[k]  = float(r2)

                # Cache this fit for potential use by neighbours
                _fitted_model[idx] = popt

            except Exception:
                shape_ok[k] = True
                shape_r2[k] = -1.0   # exception → Poor

        # All peaks are kept — shape_r2 carries the quality score.
        # Peaks with r2 < r2_min or r2 == -1 (fit failed) → status 'Poor'
        # Peaks with r2 >= r2_min → status left as None (user assigns Good/Undef)
        n_total    = len(shape_r2)
        n_good     = int(np.sum(shape_r2 >= r2_min))
        n_poor_r2  = int(np.sum((shape_r2 >= 0) & (shape_r2 < r2_min)))
        n_failed   = int(np.sum(shape_r2 < 0))
        print(f"   Skewed-Lorentz shape filter (R²≥{r2_min}): "
              f"{n_good} good  {n_poor_r2} poor (low R²)  "
              f"{n_failed} poor (fit failed)  [{n_total} total, all kept]")

        # ── Build peak list with valley-split boundaries ───────────────────────
        sorted_idx = np.sort(cand_arr)
        valley = {}
        for k in range(len(sorted_idx) - 1):
            a, b = sorted_idx[k], sorted_idx[k+1]
            valley[(a, b)] = a + int(np.argmin(ref[a:b+1]))

        self.peaks   = []
        self.buckets = []

        for k, idx in enumerate(cand_arr):
            w_left_pt  = wl[k]
            w_right_pt = wr[k]
            fwhm_pts   = max(1, int(w_right_pt) - int(w_left_pt))
            max_walk   = fwhm_pts * 8
            limit_left  = max(0, idx - max_walk)
            limit_right = min(len(ref) - 1, idx + max_walk)

            pos = int(np.where(sorted_idx == idx)[0][0])
            if pos > 0:
                ln = sorted_idx[pos - 1]
                limit_left = max(limit_left, valley[(ln, idx)])
            if pos < len(sorted_idx) - 1:
                rn = sorted_idx[pos + 1]
                limit_right = min(limit_right, valley[(idx, rn)])

            left = idx
            while left > limit_left and ref[left] > self.noise_std:
                left -= 1
            right = idx
            while right < limit_right and ref[right] > self.noise_std:
                right += 1

            snr  = float(ref[idx] / self.noise_std)
            area = float(abs(trapz_compat(ref[left:right+1], dx=ppm_step)))
            wl_i = max(int(w_left_pt),  0)
            wr_i = min(int(w_right_pt), len(ppm)-1)

            r2_val = float(shape_r2[k]) if k < len(shape_r2) else -1.0
            # Auto-assign status based on Skewed-Lorentz R²:
            #   R² ≥ 0.80  → Good    (clean Lorentzian shape)
            #   0.50–0.80  → Average (partial overlap / slight distortion)
            #   < 0.50     → Poor    (noise artefact / severe overlap)
            #   fit failed → Poor
            if r2_val < 0 or r2_val < 0.50:
                pre_status = 'Poor'
            elif r2_val < 0.80:
                pre_status = 'Average'
            else:
                pre_status = 'Good'
            self.peaks.append({
                'index':         int(idx),
                'ppm':           float(ppm[idx]),
                'height':        float(ref[idx]),
                'snr':           snr,
                'prominence':    float(proms[k]),
                'start_ppm':     float(ppm[left]),
                'end_ppm':       float(ppm[right]),
                'start_idx':     int(left),
                'end_idx':       int(right),
                'width_ppm':     float(abs(ppm[wr_i] - ppm[wl_i])),
                'width_at_half': float(abs(ppm[wr_i] - ppm[wl_i])),
                'area':          area,
                'deconvolved':   False,
                'shape_r2':      round(r2_val, 4) if r2_val >= 0 else None,
                'status':        pre_status,
            })
            self.buckets.append({
                'center_ppm':    float(ppm[idx]),
                'left_boundary': float(ppm[left]),
                'right_boundary':float(ppm[right]),
                'width':         float(abs(ppm[right] - ppm[left])),
                'integral':      area,
                'peak_height':   float(ref[idx]),
                'snr':           snr,
            })

        # Refinar limites usando método robusto (derivada/valley)
        try:
            from processing.peak_detection import separate_overlapping_peaks
            sep_peaks = separate_overlapping_peaks(
                ref, cand_arr, ppm, noise_std=self.noise_std, method="auto")
            # Atualiza os limites start/end dos picos detectados
            sep_map = {p['index']: p for p in sep_peaks}
            for p in self.peaks:
                idx = p['index']
                if idx in sep_map:
                    p['start_idx'] = sep_map[idx]['start_idx']
                    p['end_idx'] = sep_map[idx]['end_idx']
                    p['start_ppm'] = sep_map[idx]['start_ppm']
                    p['end_ppm'] = sep_map[idx]['end_ppm']
                    p['width_ppm'] = sep_map[idx]['width_ppm']
        except Exception as e:
            print(f"[WARN] Falha ao refinar limites robustos: {e}")

        self.peaks.sort(key=lambda x: abs(x['ppm']))
        for rank, p in enumerate(self.peaks, 1):
            p['rank'] = rank

        print(f"[OK] Detected {len(self.peaks)} peaks")
        if self.peaks:
            print(f"   Tallest: {self.peaks[0]['ppm']:.4f} ppm  "
                  f"SNR={self.peaks[0]['snr']:.1f}  "
                  f"width={self.peaks[0]['width_ppm']:.4f} ppm")
        return self.peaks


    # ══════════════════════════════════════════════════════════════════════════
    #  OVERLAP DETECTION — HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_peaks_single(self, ppm, intensity_1d,
                             min_snr=3, min_width_ppm=0.003,
                             prominence_factor=1.5):
        """
        Detect peaks in a single 1-D spectrum without mutating self.peaks.

        Returns a list of lightweight dicts:
            {ppm, height, prominence, start_ppm, end_ppm}
        Used internally by find_overlapping_regions_consensus().
        """
        if self.noise_std is None or self.noise_std <= 0:
            return []

        ppm_step       = abs(ppm[1] - ppm[0])
        min_width_pts  = max(2, min_width_ppm / ppm_step)
        min_height     = self.noise_std * min_snr
        min_prominence = self.noise_std * prominence_factor

        try:
            idx_arr, props = find_peaks(
                intensity_1d,
                height=min_height,
                prominence=min_prominence,
                width=min_width_pts,
                rel_height=0.5,
            )
        except Exception:
            return []

        if len(idx_arr) == 0:
            return []

        # Valley split — same logic as detect_peaks()
        sorted_idx = np.sort(idx_arr)
        valley = {}
        for k in range(len(sorted_idx) - 1):
            a, b = sorted_idx[k], sorted_idx[k + 1]
            seg = intensity_1d[a:b + 1]
            valley[(a, b)] = a + int(np.argmin(seg))

        results = []
        for i, idx in enumerate(idx_arr):
            w_left_pt  = props['left_ips'][i]
            w_right_pt = props['right_ips'][i]
            fwhm_pts   = max(1, int(w_right_pt) - int(w_left_pt))
            max_walk   = fwhm_pts * 8

            limit_left  = max(0, idx - max_walk)
            limit_right = min(len(intensity_1d) - 1, idx + max_walk)

            pos_in_sorted = int(np.where(sorted_idx == idx)[0][0])
            if pos_in_sorted > 0:
                ln = sorted_idx[pos_in_sorted - 1]
                limit_left = max(limit_left, valley[(ln, idx)])
            if pos_in_sorted < len(sorted_idx) - 1:
                rn = sorted_idx[pos_in_sorted + 1]
                limit_right = min(limit_right, valley[(idx, rn)])

            left = idx
            while left > limit_left and intensity_1d[left] > self.noise_std:
                left -= 1
            right = idx
            while right < limit_right and intensity_1d[right] > self.noise_std:
                right += 1

            results.append({
                'index':      int(idx),
                'ppm':        float(ppm[idx]),
                'height':     float(intensity_1d[idx]),
                'prominence': float(props['prominences'][i]),
                'start_ppm':  float(ppm[left]),
                'end_ppm':    float(ppm[right]),
                'width_ppm':  float(abs(ppm[right] - ppm[left])),
                'snr':        float(intensity_1d[idx] / max(self.noise_std, 1e-12)),
            })
        return results

    def _group_peaks_single(self, peaks_1d, distance_threshold, intensity_1d=None):
        """
        Apply the same proximity / base-overlap / shoulder grouping used in
        find_overlapping_regions(), but on a lightweight per-spectrum peak list.
        Returns list of (lo, hi) tuples for overlapping regions in that spectrum.
        """
        if len(peaks_1d) < 2:
            return []

        def _lo(p): return min(p['start_ppm'], p['end_ppm'])
        def _hi(p): return max(p['start_ppm'], p['end_ppm'])

        sorted_peaks = sorted(peaks_1d, key=lambda p: p['ppm'])
        regions = []
        used = set()
        valley_threshold = 0.20     # unresolved when valley >20% of smaller apex
        shoulder_prom_fraction = 0.35
        shoulder_height_fraction = 0.25

        for i, pi in enumerate(sorted_peaks):
            if i in used:
                continue
            group = [pi]
            g_lo  = _lo(pi)
            g_hi  = _hi(pi)
            used.add(i)

            for j in range(i + 1, len(sorted_peaks)):
                if j in used:
                    continue
                pj = sorted_peaks[j]
                dist = min(abs(pj['ppm'] - pk['ppm']) for pk in group)

                pj_lo = _lo(pj)
                pj_hi = _hi(pj)

                close = dist < distance_threshold

                raw_overlap = max(0.0, min(g_hi, pj_hi) - max(g_lo, pj_lo))
                min_width   = min(g_hi - g_lo, pj_hi - pj_lo)
                overlap = (raw_overlap > 0
                           and min_width > 0
                           and raw_overlap / min_width > 0.35)

                unresolved = False
                if intensity_1d is not None:
                    idx_i = int(pi.get('index', -1))
                    idx_j = int(pj.get('index', -1))
                    if idx_i >= 0 and idx_j >= 0 and idx_i != idx_j:
                        a, b = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
                        seg = intensity_1d[a:b + 1]
                        if len(seg) > 0:
                            valley = float(np.min(seg))
                            smaller_apex = min(float(pi.get('height', 0.0)),
                                               float(pj.get('height', 0.0)))
                            if smaller_apex > 0:
                                unresolved = (valley / smaller_apex) > valley_threshold

                # Shoulder model: smaller peak sits inside the feet of taller peak,
                # has reduced prominence, and is not a tiny noise spur.
                smaller, taller = (pi, pj) if pi['height'] <= pj['height'] else (pj, pi)
                smaller_prom = float(smaller.get('prominence', 0.0))
                smaller_h = float(smaller.get('height', 0.0))
                taller_h = float(taller.get('height', 0.0))
                prom_frac = (smaller_prom / smaller_h) if smaller_h > 0 else 0.0
                inside_taller_base = (_lo(taller) <= smaller['ppm'] <= _hi(taller))
                height_ok = (taller_h > 0) and (smaller_h >= shoulder_height_fraction * taller_h)
                is_shoulder = bool(
                    inside_taller_base
                    and height_ok
                    and prom_frac < shoulder_prom_fraction
                    and dist < distance_threshold * 2.5
                )

                # Do not group by proximity alone; require structural evidence.
                if unresolved or overlap or is_shoulder:
                    group.append(pj)
                    g_lo = min(g_lo, pj_lo)
                    g_hi = max(g_hi, pj_hi)
                    used.add(j)
                elif (not close) and pj_lo > g_hi + distance_threshold:
                    break

            if len(group) > 1:
                regions.append((g_lo, g_hi))

        return regions

    def _baseline_edge_metrics(self, ppm, intensity, lo, hi,
                               edge_fraction=0.12, min_edge_points=6):
        """Compute edge-baseline metrics for one ppm region."""
        lo_n, hi_n = min(lo, hi), max(lo, hi)
        mask = (ppm >= lo_n) & (ppm <= hi_n)
        idx = np.flatnonzero(mask)
        if idx.size < (2 * min_edge_points + 2):
            return {
                'valid': False,
                'reason': 'region_too_narrow',
                'lo': float(lo_n),
                'hi': float(hi_n),
            }

        y_reg = np.asarray(intensity[idx], dtype=float)
        n_edge = max(min_edge_points, int(round(idx.size * edge_fraction)))
        n_edge = min(n_edge, max(min_edge_points, idx.size // 3))
        left = y_reg[:n_edge]
        right = y_reg[-n_edge:]

        left_level = float(np.median(np.abs(left)))
        right_level = float(np.median(np.abs(right)))
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        tilt = float(abs(np.median(right) - np.median(left)))

        sigma = float(self.noise_std) if self.noise_std and self.noise_std > 0 else float(
            np.median(np.abs(y_reg - np.median(y_reg))) * 1.4826 + 1e-12)
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
            'lo': float(lo_n),
            'hi': float(hi_n),
            'left_level': left_level,
            'right_level': right_level,
            'left_std': left_std,
            'right_std': right_std,
            'tilt': tilt,
            'edge_limit': float(edge_limit),
            'tilt_limit': float(tilt_limit),
            'noise_std': float(sigma),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  OVERLAP DETECTION — PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def find_overlapping_regions(self, distance_threshold=0.04):
        """
        Find regions that NEED deconvolution — i.e. contain true shoulders
        or unresolved overlapping peaks.

        A peak requires deconvolution when at least one of:

        1. UNRESOLVED OVERLAP — the valley between two adjacent peaks is
           > valley_threshold (20%) of the smaller apex. The peaks share
           significant baseline and cannot be quantified independently.

        2. TRUE SHOULDER — a peak has prominence < shoulder_fraction (35%)
           of its own height, meaning it sits on the flank of a taller
           neighbour and was only detected because of local asymmetry.

        Deliberately excluded:
        - Proximity alone (two resolved peaks at 0.03 ppm with a deep
          valley are perfectly separable — no deconvolution needed).
        - Minor base overlap (< 20% of valley height) — resolved enough
          for independent integration.

        Parameters
        ----------
        distance_threshold : float
            Maximum ppm distance to consider two peaks as neighbours
            when checking shoulder / valley conditions. Default 0.04 ppm.
        """
        if not self.peaks:
            return []

        # Thresholds
        valley_threshold  = 0.20   # valley > 20% of smaller apex → unresolved
        shoulder_fraction = 0.35   # prominence < 35% of height → true shoulder

        print(f"\n🔎 Finding shoulders & unresolved overlaps "
              f"(valley_thr={valley_threshold*100:.0f}%, "
              f"shoulder_prom={shoulder_fraction*100:.0f}%)...")

        def _lo(p): return min(p['start_ppm'], p['end_ppm'])
        def _hi(p): return max(p['start_ppm'], p['end_ppm'])

        # Exclude Poor-quality peaks and sub-peaks (already deconvolved) —
        # only original detected peaks participate in overlap detection.
        candidate_peaks = [p for p in self.peaks
                           if p.get('status') != 'Poor'
                           and not p.get('is_sub_peak', False)]
        if not candidate_peaks:
            print("   No non-Poor peaks to analyse.")
            return []
        print(f"   Using {len(candidate_peaks)}/{len(self.peaks)} peaks "
              f"(Poor excluded)")

        sorted_peaks = sorted(candidate_peaks, key=lambda p: p['ppm'])
        n            = len(sorted_peaks)
        needs_deconv = set()   # indices into sorted_peaks

        # Build ppm array for valley lookup (use self.mean_intensity)
        ref_ppm = self.ppm
        ref_int = self.mean_intensity

        for i in range(n):
            pi = sorted_peaks[i]

            for j in range(i + 1, n):
                pj = sorted_peaks[j]
                dist = abs(pi['ppm'] - pj['ppm'])

                if dist > distance_threshold * 4:
                    break   # too far — stop inner loop

                # Find valley between pi and pj in the spectrum
                idx_i = pi['index']
                idx_j = pj['index']
                if idx_i > idx_j:
                    idx_i, idx_j = idx_j, idx_i
                if idx_j > idx_i:
                    seg    = ref_int[idx_i:idx_j + 1]
                    val_h  = float(np.min(seg))
                else:
                    val_h  = 0.0

                apex_i = float(pi['height'])
                apex_j = float(pj['height'])
                smaller_apex = min(apex_i, apex_j)

                # Condition 1: unresolved overlap — valley > 20% of smaller apex
                valley_ratio = val_h / smaller_apex if smaller_apex > 0 else 0.0
                unresolved   = valley_ratio > valley_threshold

                # Condition 2: shoulder — embedded lower head inside the base
                # of a taller peak with reduced prominence.
                prom_i = float(pi.get('prominence', apex_i))
                prom_j = float(pj.get('prominence', apex_j))

                pi_inside_pj = (_lo(pj) <= pi['ppm'] <= _hi(pj))
                pj_inside_pi = (_lo(pi) <= pj['ppm'] <= _hi(pi))

                shoulder_i = (prom_i < apex_i * shoulder_fraction
                              and apex_i >= 0.25 * apex_j
                              and pi_inside_pj
                              and dist < distance_threshold * 2.5)
                shoulder_j = (prom_j < apex_j * shoulder_fraction
                              and apex_j >= 0.25 * apex_i
                              and pj_inside_pi
                              and dist < distance_threshold * 2.5)

                if unresolved:
                    needs_deconv.add(i)
                    needs_deconv.add(j)
                    print(f"   Unresolved: {pi['ppm']:.4f} ↔ {pj['ppm']:.4f} ppm  "
                          f"valley={valley_ratio*100:.0f}%")
                if shoulder_i:
                    needs_deconv.add(i)
                    print(f"   Shoulder: {pi['ppm']:.4f} ppm  "
                          f"prom/height={prom_i/apex_i*100:.0f}%")
                if shoulder_j:
                    needs_deconv.add(j)
                    print(f"   Shoulder: {pj['ppm']:.4f} ppm  "
                          f"prom/height={prom_j/apex_j*100:.0f}%")

        # Group the flagged peaks into contiguous regions
        flagged_sorted = sorted(needs_deconv)
        for idx in flagged_sorted:
            sorted_peaks[idx]['_in_overlap_group'] = True

        overlapping = []
        if flagged_sorted:
            # Merge flagged peaks that are within distance_threshold of each other
            # into single regions
            current_group = [sorted_peaks[flagged_sorted[0]]]
            for k in range(1, len(flagged_sorted)):
                pk = sorted_peaks[flagged_sorted[k]]
                if abs(pk['ppm'] - current_group[-1]['ppm']) <= distance_threshold * 3:
                    current_group.append(pk)
                else:
                    lo = min(_lo(p) for p in current_group)
                    hi = max(_hi(p) for p in current_group)
                    overlapping.append((lo, hi))
                    current_group = [pk]
            lo = min(_lo(p) for p in current_group)
            hi = max(_hi(p) for p in current_group)
            overlapping.append((lo, hi))

        for lo, hi in overlapping:
            n_pk = sum(1 for p in sorted_peaks
                       if lo <= p['ppm'] <= hi)
            print(f"   -> Region {lo:.4f}-{hi:.4f}  ({n_pk} peaks)")

        print(f"[OK] Found {len(overlapping)} region(s) needing deconvolution "
              f"({len(needs_deconv)} peaks flagged)")
        return overlapping

    def find_overlapping_regions_consensus(
            self,
            all_intensities,
            samples_info=None,
            distance_threshold=0.04,
            consensus_threshold=0.50,
            min_snr=3,
            min_width_ppm=0.003,
            prominence_factor=1.5,
            enforce_baseline_edges=False):
        """
        AlpsNMR-style consensus overlap detection.

        Each non-BLK spectrum is analysed individually.  A ppm region is only
        reported as overlapping when it was flagged in at least
        ``consensus_threshold`` fraction of the spectra (default 0.50 = 50%).
        This avoids the distortion introduced by using a single mean spectrum,
        where shoulders from a subset of samples can be washed out by the
        averaging process.

        Parameters
        ----------
        all_intensities : np.ndarray, shape (n_points, n_samples)
            Raw intensity matrix (all samples, including BLK if present).
        samples_info : list of dict or None
            Each dict should have at least a 'type' or 'label' key.
            Entries whose type/label == 'BLK' are excluded.
            Pass None to include every column.
        distance_threshold : float
            Same meaning as in find_overlapping_regions() (ppm).
        consensus_threshold : float  [0, 1]
            Fraction of spectra that must flag a region for it to be reported.
            0.5 → majority vote; 1.0 → unanimous.
        min_snr, min_width_ppm, prominence_factor
            Forwarded to _detect_peaks_single() for each spectrum.

        Returns
        -------
        list of (lo_ppm, hi_ppm) tuples — same format as
        find_overlapping_regions().
        """
        if all_intensities is None or all_intensities.ndim < 2:
            print("   [consensus] single spectrum — falling back to mean-based method")
            return self.find_overlapping_regions(distance_threshold)

        n_points, n_samples = all_intensities.shape
        ppm = self.ppm
        if ppm is None:
            print("   [consensus] ppm axis not set — falling back")
            return self.find_overlapping_regions(distance_threshold)

        aligned_intensities = all_intensities
        if n_samples > 1:
            aligned_intensities, shifts_pts, shifts_ppm, _ = global_peak_rigid_align(
                ppm,
                all_intensities,
                samples_info=samples_info,
                max_shift_ppm=distance_threshold,
            )
            print(
                "   [consensus] rigid alignment applied: "
                f"min={np.min(shifts_ppm) * 1000:.2f} mppm  "
                f"max={np.max(shifts_ppm) * 1000:.2f} mppm"
            )

        # ── Select non-BLK column indices ─────────────────────────────────────
        if samples_info:
            valid_cols = [
                i for i, s in enumerate(samples_info)
                if str(s.get('type', s.get('label', 'Sample'))).upper() != 'BLK'
                and not s.get('excluded', False)
                and i < n_samples
            ]
        else:
            valid_cols = list(range(n_samples))

        if not valid_cols:
            print("   [consensus] no valid (non-BLK) samples — falling back")
            return self.find_overlapping_regions(distance_threshold)

        n_valid = len(valid_cols)
        print(f"\n[INFO] Consensus overlap detection")
        print(f"   {n_valid} spectra | threshold={consensus_threshold*100:.0f}%"
              f" | distance={distance_threshold} ppm")

        # ── Vote map — accumulate overlap coverage per ppm point ───────────────
        # vote_map[k] = number of spectra that flagged point k as overlapping
        vote_map = np.zeros(n_points, dtype=np.int32)

        spectra_with_overlaps = 0
        for col_i, col in enumerate(valid_cols):
            spectrum = aligned_intensities[:, col]

            peaks_1d = self._detect_peaks_single(
                ppm, spectrum,
                min_snr=min_snr,
                min_width_ppm=min_width_ppm,
                prominence_factor=prominence_factor,
            )

            regions_1d = self._group_peaks_single(
                peaks_1d, distance_threshold, intensity_1d=spectrum)

            if regions_1d:
                spectra_with_overlaps += 1
                spectrum_mask = np.zeros(n_points, dtype=bool)
                for (lo, hi) in regions_1d:
                    # Mark every ppm point inside this region
                    lo_n, hi_n = min(lo, hi), max(lo, hi)
                    mask = (ppm >= lo_n) & (ppm <= hi_n)
                    spectrum_mask |= mask
                vote_map[spectrum_mask] += 1

        print(f"   Spectra reporting >=1 overlap: "
              f"{spectra_with_overlaps}/{n_valid} "
              f"({spectra_with_overlaps/n_valid*100:.0f}%)")

        # ── Apply consensus threshold ──────────────────────────────────────────
        min_votes = max(1, int(np.ceil(consensus_threshold * n_valid)))
        consensus_mask = (vote_map >= min_votes)

        if not np.any(consensus_mask):
            print("[OK] [consensus] No overlapping regions passed the threshold")
            return []

        # ── Merge contiguous True blocks into (lo, hi) regions ────────────────
        # Add padding of False on each side so diff() detects edges cleanly
        padded = np.concatenate(([False], consensus_mask, [False]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]   # indices in padded → same in ppm
        ends   = np.where(diff == -1)[0] - 1

        overlapping = []
        for s, e in zip(starts, ends):
            lo_ppm = min(float(ppm[s]), float(ppm[e]))
            hi_ppm = max(float(ppm[s]), float(ppm[e]))
            # Skip trivially narrow artefacts (< distance_threshold)
            if hi_ppm - lo_ppm < distance_threshold * 0.25:
                continue

            # Report vote fraction for diagnostics
            max_votes = int(vote_map[s:e + 1].max())
            frac      = max_votes / n_valid
            print(f"   Region {lo_ppm:.4f}-{hi_ppm:.4f} ppm  "
                  f"({max_votes}/{n_valid} spectra, {frac*100:.0f}%)")
            overlapping.append((lo_ppm, hi_ppm))

        if enforce_baseline_edges:
            # Optional hard gate. Stage-level workflow may enforce baseline after
            # splitting broad consensus regions into smaller deconvolution chunks.
            ref_intensity = np.mean(all_intensities[:, valid_cols], axis=1)
            filtered = []
            rejected = 0
            for lo_ppm, hi_ppm in overlapping:
                m = self._baseline_edge_metrics(ppm, ref_intensity, lo_ppm, hi_ppm)
                if m['valid']:
                    filtered.append((m['lo'], m['hi']))
                else:
                    rejected += 1
                    print(f"   Reject {lo_ppm:.4f}-{hi_ppm:.4f} ppm: {m['reason']}")
            overlapping = filtered
            if rejected:
                print(f"   Baseline gate removed {rejected} region(s)")

        # ── Flag self.peaks that fall inside consensus regions ─────────────────
        for peak in self.peaks:
            peak_ppm = peak['ppm']
            in_region = any(
                lo <= peak_ppm <= hi
                for (lo, hi) in overlapping
            )
            if in_region:
                peak['_in_overlap_group'] = True

        print(f"[OK] [consensus] Found {len(overlapping)} overlapping region(s)")
        return overlapping

    # ══════════════════════════════════════════════════════════════════════════
    #  ACCESSORS
    # ══════════════════════════════════════════════════════════════════════════

    def get_peaks(self):
        import copy
        return copy.deepcopy(self.peaks) if self.peaks else []

    def get_buckets(self):
        return self.buckets

    def parse_region_from_text(self, region_text):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", region_text)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        return None, None