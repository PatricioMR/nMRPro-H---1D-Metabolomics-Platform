# gui/stages/peak_detection_stage.py
"""
Peak Detection Stage — Step 1 (noise) + Step 2 (peaks + deconvolution).
The matplotlib canvas is shared with all other stages and is never
recreated here — only the left-panel menu changes when switching tabs.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import time
import threading
import queue
import certifi

os.environ['SSL_CERT_FILE']      = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from processing.noise_estimation import estimate_noise_from_region
from .base_stage import BaseStage



def _no_blk_group_mean(all_intensities, samples_info):
    """Group mean of non-BLK, non-excluded samples (equal group weight)."""
    import numpy as np
    from collections import defaultdict
    if all_intensities is None: return None
    if all_intensities.ndim == 1: return all_intensities
    groups = defaultdict(list)
    for i, s in enumerate(samples_info or []):
        stype = s.get('type', s.get('label', 'Sample'))
        if (str(stype).upper() != 'BLK'
                and not s.get('excluded', False)
                and i < all_intensities.shape[1]):
            groups[s.get('label', 'unknown')].append(i)
    if not groups:
        return np.mean(all_intensities, axis=1)
    gmeans = np.stack([np.mean(all_intensities[:, idxs], axis=1)
                       for idxs in groups.values()], axis=1)
    n_blk = all_intensities.shape[1] - sum(len(v) for v in groups.values())
    print(f'   _no_blk_group_mean: {len(groups)} group(s), '
          f'{sum(len(v) for v in groups.values())} samples, {n_blk} BLK excl')
    return np.mean(gmeans, axis=1)


def _reference_group_mean(all_intensities, samples_info):
    """Mean of the non-BLK group with highest total intensity."""
    import numpy as np
    from collections import defaultdict
    if all_intensities is None: return None
    groups = defaultdict(list)
    for i, s in enumerate(samples_info or []):
        stype = s.get('type', s.get('label', 'Sample'))
        if str(stype).upper() != 'BLK' and i < all_intensities.shape[1]:
            groups[s.get('label', 'unknown')].append(i)
    if not groups:
        return np.mean(all_intensities, axis=1)
    best = max(groups, key=lambda g: float(
        np.sum(np.mean(all_intensities[:, groups[g]], axis=1))))
    print(f'   _reference_group_mean: group "{best}" '
          f'({len(groups[best])} sample(s)) — highest intensity')
    return np.mean(all_intensities[:, groups[best]], axis=1)


def _validate_region_baseline_edges(
        ppm, intensity, lo, hi, noise_std,
        edge_fraction=0.12, min_edge_points=6,
        max_edge_ratio=2.5, max_tilt_ratio=1.5):
    """Return baseline edge metrics and validity for one overlap region."""
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

    # Use robust magnitude (abs) to keep the rule stable if baseline oscillates around 0.
    left_level = float(np.median(np.abs(left)))
    right_level = float(np.median(np.abs(right)))
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    tilt = float(abs(np.median(right) - np.median(left)))

    sigma = float(noise_std) if noise_std and noise_std > 0 else float(
        np.median(np.abs(y_reg - np.median(y_reg))) * 1.4826 + 1e-12)
    edge_limit = max_edge_ratio * sigma
    tilt_limit = max_tilt_ratio * sigma

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
        'left_ok': bool(left_ok),
        'right_ok': bool(right_ok),
        'tilt_ok': bool(tilt_ok),
        'left_level': left_level,
        'right_level': right_level,
        'left_std': left_std,
        'right_std': right_std,
        'tilt': tilt,
        'edge_limit': float(edge_limit),
        'tilt_limit': float(tilt_limit),
        'noise_std': float(sigma),
    }


def _expand_region_to_baseline_edges(
        ppm, intensity, lo, hi, noise_std,
        max_expand_factor=2.5, step_points=3):
    """Expand region until both edges satisfy baseline criteria, if possible."""
    lo_n, hi_n = float(min(lo, hi)), float(max(lo, hi))
    idx_lo = int(np.argmin(np.abs(ppm - lo_n)))
    idx_hi = int(np.argmin(np.abs(ppm - hi_n)))
    if idx_lo > idx_hi:
        idx_lo, idx_hi = idx_hi, idx_lo

    base_pts = max(1, idx_hi - idx_lo + 1)
    max_extra = max(4, int(base_pts * max_expand_factor))

    best = _validate_region_baseline_edges(ppm, intensity, lo_n, hi_n, noise_std)
    best_lo, best_hi = lo_n, hi_n

    if best.get('valid', False):
        best['validation_mode'] = 'strict'
        best['expanded'] = False
        best['original_lo'] = lo_n
        best['original_hi'] = hi_n
        return best_lo, best_hi, best

    cur_lo, cur_hi = idx_lo, idx_hi
    expanded = 0
    while expanded < max_extra:
        metric = _validate_region_baseline_edges(
            ppm, intensity, float(min(ppm[cur_lo], ppm[cur_hi])),
            float(max(ppm[cur_lo], ppm[cur_hi])), noise_std)

        if metric.get('valid', False):
            metric['validation_mode'] = 'strict'
            metric['expanded'] = True
            metric['original_lo'] = lo_n
            metric['original_hi'] = hi_n
            return float(min(ppm[cur_lo], ppm[cur_hi])), float(max(ppm[cur_lo], ppm[cur_hi])), metric

        # Expand to both sides to capture full foot-head-foot envelope.
        moved = False
        if cur_lo > 0:
            new_lo = max(0, cur_lo - step_points)
            expanded += (cur_lo - new_lo)
            cur_lo = new_lo
            moved = True
        if cur_hi < (len(ppm) - 1):
            new_hi = min(len(ppm) - 1, cur_hi + step_points)
            expanded += (new_hi - cur_hi)
            cur_hi = new_hi
            moved = True
        if not moved:
            break

        best = metric
        best_lo = float(min(ppm[cur_lo], ppm[cur_hi]))
        best_hi = float(max(ppm[cur_lo], ppm[cur_hi]))

    # Relaxed fallback after max expansion.
    relaxed = _validate_region_baseline_edges(
        ppm, intensity, best_lo, best_hi, noise_std,
        max_edge_ratio=4.0, max_tilt_ratio=3.0)
    relaxed['validation_mode'] = 'relaxed' if relaxed.get('valid', False) else 'failed'
    relaxed['expanded'] = True
    relaxed['original_lo'] = lo_n
    relaxed['original_hi'] = hi_n
    return best_lo, best_hi, relaxed

class PeakDetectionStage(BaseStage):

    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.step1_completed = False
        self.noise_std       = None
        self.noise_source    = None
        self.peaks           = []
        self.default_noise_region = (12.0, 10.0)
        self._deconv_controller = None   # persists across tool opens
        self._overlaps = []              # last found overlap regions
        self._overlap_metrics = []       # baseline-edge validation per region

        # Graph STOCSY — pairwise peak correlation matrix built after detection.
        # Keys: (rank_i, rank_j) with rank_i < rank_j.  Values: Pearson r.
        # Empty when n_samples < 3 (insufficient for meaningful correlation).
        self._peak_corr    = {}
        self._n_samples_pk = 0

        # Peak processor
        try:
            from gui.notation.peak_processing import PeakProcessor
            self.processor = PeakProcessor(app)
            print("✅ PeakProcessor initialized")
        except Exception as e:
            print(f"❌ PeakProcessor: {e}")
            self.processor = None

        self.create_widgets()

    # ── Status bar ────────────────────────────────────────────────────────────

    def _status(self, msg, color="blue"):
        try:
            self.app.status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    def _status_ok(self, msg):    self._status(msg, "green")
    def _status_err(self, msg):   self._status(msg, "red")
    def _status_ready(self):      self._status("Ready", "gray")

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text="Peak Detection",
                  font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        # ── STEP 1: Noise region (optional override) ──────────────────────────
        # Noise is estimated automatically as part of Detect Peaks. These entries
        # only let the user pin a specific signal-free region; left as-is, the
        # detector estimates noise from the quietest segment of the spectrum.
        s1 = ttk.LabelFrame(self.frame,
                             text="📏 Noise Region (optional)", padding=5)
        s1.pack(fill=tk.X, padx=5, pady=5)

        rf = ttk.Frame(s1)
        rf.pack(fill=tk.X, pady=2)
        ttk.Label(rf, text="Region (ppm):").pack(side=tk.LEFT, padx=2)
        self.noise_start_var = tk.StringVar(
            value=str(self.default_noise_region[0]))
        self.noise_end_var = tk.StringVar(
            value=str(self.default_noise_region[1]))
        ttk.Entry(rf, textvariable=self.noise_start_var,
                  width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(rf, text="to").pack(side=tk.LEFT)
        ttk.Entry(rf, textvariable=self.noise_end_var,
                  width=8).pack(side=tk.LEFT, padx=2)

        self.noise_value_label = ttk.Label(s1, text="Noise Std Dev: --")
        self.noise_value_label.pack(anchor=tk.W, pady=2)
        # Kept for project-load compatibility (no longer a separate UI step).
        self.step1_status = ttk.Label(s1, text="")

        # ── STEP 2: Peak Detection ────────────────────────────────────────────
        s2 = ttk.LabelFrame(
            self.frame,
            text="🔍 STEP 2: Peak Detection & Deconvolution", padding=5)
        s2.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: Baseline Correction Toggle
        bf1 = ttk.Frame(s2)
        bf1.pack(fill=tk.X, pady=2)
        self.baseline_correct_var = tk.BooleanVar(value=True)
        self._baseline_chk = ttk.Checkbutton(bf1, text="Apply Baseline Correction",
                                              variable=self.baseline_correct_var)
        self._baseline_chk.pack(side=tk.LEFT, padx=2)
        self.baseline_method_var = tk.StringVar(value='airPLS')
        cb = ttk.Combobox(bf1, textvariable=self.baseline_method_var,
                          values=['airPLS', 'snip', 'rolling', 'constant'],
                          state='readonly', width=8)
        cb.pack(side=tk.LEFT, padx=2)
        ttk.Label(bf1, text="λ (airPLS):", foreground='gray').pack(side=tk.LEFT, padx=(8, 2))
        self.baseline_lam_var = tk.DoubleVar(value=1e5)
        ttk.Entry(bf1, textvariable=self.baseline_lam_var, width=6).pack(side=tk.LEFT, padx=2)
        self._baseline_applied_lbl = ttk.Label(
            s2, text="⚠ Baseline already corrected — 'Apply Baseline Correction' disabled.",
            foreground='orange', font=('Arial', 8))
        self._alignment_applied_lbl = ttk.Label(
            s2, text="⚠ Spectrum already aligned — pipeline alignment skipped.",
            foreground='orange', font=('Arial', 8))
        # shown/hidden dynamically; not packed yet

        # Row 2: Min SNR + per-sample detection toggle
        pf1 = ttk.Frame(s2)
        pf1.pack(fill=tk.X, pady=2)
        ttk.Label(pf1, text="Min SNR:").pack(side=tk.LEFT, padx=2)
        self.min_snr_var = tk.DoubleVar(value=6.0)
        ttk.Entry(pf1, textvariable=self.min_snr_var,
                  width=6).pack(side=tk.LEFT, padx=2)
        self.per_sample_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pf1, text="Detect per sample (union)",
            variable=self.per_sample_var).pack(side=tk.LEFT, padx=(10, 2))

        # Row 3: Min Width + Prominence
        pf2 = ttk.Frame(s2)
        pf2.pack(fill=tk.X, pady=2)
        ttk.Label(pf2, text="Min Width (ppm):").pack(side=tk.LEFT, padx=2)
        self.min_width_var = tk.DoubleVar(value=0.01)
        ttk.Entry(pf2, textvariable=self.min_width_var,
                  width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(pf2, text="Prominence:").pack(side=tk.LEFT, padx=(8, 2))
        self.prominence_var = tk.DoubleVar(value=2.0)
        ttk.Entry(pf2, textvariable=self.prominence_var,
                  width=5).pack(side=tk.LEFT, padx=2)

        self.btn_detect = ttk.Button(
            s2, text="🎯 Detect Peaks",
            command=self.detect_peaks, state="normal")
        self.btn_detect.pack(fill=tk.X, pady=5)

        self.peaks_count_label = ttk.Label(s2, text="Peaks found: 0")
        self.peaks_count_label.pack(anchor=tk.W, pady=2)
        self.step2_status = ttk.Label(
            s2, text="⏳ Load data, then detect peaks", foreground="gray")
        self.step2_status.pack(anchor=tk.W, pady=2)

        self.btn_toggle_peaks = ttk.Button(
            s2, text="👁 Show/Hide Peak Areas",
            command=self.toggle_peaks, state="disabled")
        self.btn_toggle_peaks.pack(fill=tk.X, pady=2)

        self.btn_show_baseline = ttk.Button(
            s2, text="📈 Show/Hide Estimated Baseline",
            command=self.toggle_baseline, state="disabled")
        self.btn_show_baseline.pack(fill=tk.X, pady=2)
        self._baseline_drawn = False
        self._baseline_line = None

        self.btn_search_overlaps = ttk.Button(
            s2, text="🔍 Search for Overlaps",
            command=self.search_for_overlaps, state="disabled")
        self.btn_search_overlaps.pack(fill=tk.X, pady=2)

        self.btn_auto_deconv_asym = ttk.Button(
            s2, text="↔ Auto-deconvolve Asymmetric Peaks",
            command=self.auto_deconvolve_asymmetric, state="disabled")
        self.btn_auto_deconv_asym.pack(fill=tk.X, pady=2)

        self.overlap_status = ttk.Label(
            s2, text="", foreground="gray", font=("Arial", 8))
        self.overlap_status.pack(anchor=tk.W, pady=2)

        ttk.Separator(s2, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        self.btn_open_deconv = ttk.Button(
            s2, text="🧅 Open Deconvolution Tool",
            command=self.open_deconv_tool, state="disabled")
        self.btn_open_deconv.pack(fill=tk.X, pady=2)

        self.deconv_status = ttk.Label(
            s2, text="Run peak detection first",
            foreground="gray", font=("Arial", 8))
        self.deconv_status.pack(anchor=tk.W, pady=2)

        # Internal consensus threshold (not exposed in UI)
        self.consensus_var = tk.DoubleVar(value=50.0)

        # ── Mouse Controls ────────────────────────────────────────────────────
        mf = ttk.LabelFrame(self.frame, text="🐭 Mouse Controls", padding=5)
        mf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mf,
                  text="• Scroll: Zoom in/out\n"
                       "• Drag: Pan\n"
                       "• R: Reset view",
                  font=("Arial", 8), foreground="blue",
                  justify=tk.LEFT).pack(anchor=tk.W, padx=5)

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _estimate_noise(self):
        """Estimate σ from the (optional) noise region; return σ or None.

        Embedded in Detect Peaks — no separate button. If the region entries are
        valid, noise is taken there (per-sample std → median across samples);
        otherwise returns None and the detector falls back to its own quiet-
        segment estimate.
        """
        try:
            r_start = float(self.noise_start_var.get())
            r_end   = float(self.noise_end_var.get())
        except (ValueError, TypeError):
            return None
        if r_start <= r_end:
            return None
        try:
            sigma = estimate_noise_from_region(
                self.app.ppm, self.app.all_intensities, r_start, r_end)
        except Exception as e:
            print(f"[noise] region estimate failed ({e}); using auto fallback")
            return None
        self.noise_std    = float(sigma)
        self.noise_source = 'ppm'
        self.step1_completed = True
        self.noise_value_label.config(text=f"Noise Std Dev: {self.noise_std:.4f}")
        return self.noise_std

    def _sync_baseline_corrected_ui(self):
        """Reflect baseline_corrected and alignment_corrected flags in Step 2 UI."""
        try:
            if getattr(self.app, 'baseline_corrected', False):
                self.baseline_correct_var.set(False)
                self._baseline_chk.config(state='disabled')
                self._baseline_applied_lbl.pack(anchor=tk.W, pady=1)
            else:
                self._baseline_chk.config(state='normal')
                self._baseline_applied_lbl.pack_forget()
        except Exception:
            pass
        try:
            if getattr(self.app, 'alignment_corrected', False):
                self._alignment_applied_lbl.pack(anchor=tk.W, pady=1)
            else:
                self._alignment_applied_lbl.pack_forget()
        except Exception:
            pass

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def detect_peaks(self):
        if self.app.all_intensities is None or self.app.ppm is None:
            messagebox.showinfo("Info", "Please load data first.")
            return
        if self.processor is None:
            messagebox.showerror("Error", "Peak processor not initialized.")
            return

        # Noise is now estimated here (no separate button). Use the optional
        # region if set; otherwise detect_peaks() estimates it internally from
        # the quietest segment (noise_std=None).
        self._estimate_noise()

        # ── Loading overlay ───────────────────────────────────────────────────
        overlay = tk.Toplevel(self.app.root)
        overlay.title("Peak Detection")
        overlay.resizable(False, False)
        overlay.transient(self.app.root)
        overlay.grab_set()
        overlay.attributes('-topmost', True)

        # Centre over the main window
        self.app.root.update_idletasks()
        rw = self.app.root.winfo_width()
        rh = self.app.root.winfo_height()
        rx = self.app.root.winfo_rootx()
        ry = self.app.root.winfo_rooty()
        w, h = 360, 150
        x = rx + (rw - w) // 2
        y = ry + (rh - h) // 2
        overlay.geometry(f"{w}x{h}+{x}+{y}")
        overlay.deiconify()
        overlay.lift()
        try:
            overlay.focus_force()
        except Exception:
            pass
        overlay.update_idletasks()
        overlay.after(1200, lambda: overlay.winfo_exists() and overlay.attributes('-topmost', False))

        # Style — native ttk (matches all other progress dialogs)
        outer = ttk.Frame(overlay, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="🔍  Peak Detection",
                  font=('Arial', 12, 'bold')).pack(pady=(4, 4))

        _lbl = ttk.Label(outer, text="Initialising…",
                         font=('Arial', 9), foreground='gray')
        _lbl.pack(pady=(0, 4))

        _pct_lbl = ttk.Label(outer, text="0%", font=('Arial', 9, 'bold'))
        _pct_lbl.pack(pady=(0, 4))

        pb = ttk.Progressbar(outer, mode='determinate', length=280, maximum=100)
        pb.pack(pady=(0, 12))
        pb['value'] = 0

        # ── Shared result container ───────────────────────────────────────────
        _result = {'details': None, 'error': None, 'elapsed': 0.0}
        _progress_queue = queue.Queue()

        def _push_progress(progress, message):
            _progress_queue.put((float(progress), str(message)))

        # ── Worker thread ─────────────────────────────────────────────────────
        def _worker():
            t0 = time.monotonic()
            try:
                # Use new baseline-aware peak detection
                from processing.peak_detection import detect_peaks

                # Parse baseline lambda (handle scientific notation)
                try:
                    baseline_lam = float(self.baseline_lam_var.get())
                except (ValueError, TypeError):
                    baseline_lam = 1e5

                # Skip pipeline alignment if already done via Alignment dialog
                _run_align = not getattr(self.app, 'alignment_corrected', False)

                details = detect_peaks(
                    self.app.ppm, self.app.all_intensities,
                    noise_std=self.noise_std,
                    min_snr=self.min_snr_var.get(),
                    min_distance=self.min_width_var.get(),
                    baseline_correct=self.baseline_correct_var.get(),
                    baseline_method=self.baseline_method_var.get(),
                    baseline_lam=baseline_lam,
                    global_align=_run_align,
                    samples_info=self.app.samples_info,
                    progress_callback=_push_progress,
                    return_details=True,
                    per_sample_detection=bool(self.per_sample_var.get()),
                )
                _result['details'] = details
                _result['elapsed'] = time.monotonic() - t0
            except Exception as exc:
                _result['error']   = exc
                _result['elapsed'] = time.monotonic() - t0

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        # ── Poll until done, then update UI on main thread ────────────────────
        def _poll():
            while True:
                try:
                    progress, message = _progress_queue.get_nowait()
                except queue.Empty:
                    break
                pb['value'] = max(0, min(100, progress * 100.0))
                _lbl.config(text=message)
                _pct_lbl.config(text=f"{int(round(progress * 100.0))}%")

            if thread.is_alive():
                overlay.after(100, _poll)
                return

            # Detection done — keep the overlay up while we finalise (peak
            # format conversion, area drawing, STOCSY matrix). Destroying it
            # here would leave that heavy post-processing running with no
            # progress indicator.
            try:
                pb['value'] = 100
                _pct_lbl.config(text="100%")
                _lbl.config(text="Finalising…")
                overlay.update_idletasks()
            except Exception:
                pass

            def _destroy_overlay():
                try:
                    overlay.grab_release()
                    overlay.destroy()
                except Exception:
                    pass

            def _overlay_step(msg):
                try:
                    _lbl.config(text=msg)
                    overlay.update_idletasks()
                except Exception:
                    pass

            if _result['error'] is not None:
                _destroy_overlay()
                self._status_err(f"Peak detection failed: {_result['error']}")
                messagebox.showerror("Error",
                                     f"Peak detection failed:\n{_result['error']}")
                import traceback; traceback.print_exc()
                return

            details = _result['details'] or {}
            raw = details.get('peaks', [])
            elapsed = _result['elapsed']

            self.processor.ppm = np.asarray(self.app.ppm, dtype=float)
            self.processor.mean_intensity = details.get('reference_intensity')
            self.processor.noise_std = float(details.get('noise_std', self.noise_std or 0.0))
            # Reflect the σ actually used (region or internal fallback) in the UI.
            self.noise_std = self.processor.noise_std
            self.step1_completed = True
            try:
                self.noise_value_label.config(
                    text=f"Noise Std Dev: {self.noise_std:.4f}")
            except Exception:
                pass
            self.processor.peaks = self.peaks
            self._store_baseline(details.get('baseline'))

            # Convert new peak format to internal format
            self.peaks = []
            for p in raw:
                peak_dict = {
                    'rank': p.get('rank', 0),
                    'ppm': p.get('ppm', 0),
                    'height': p.get('height', 0),
                    'snr': p.get('snr', 0),
                    'start_ppm': p.get('start_ppm', 0),
                    'end_ppm': p.get('end_ppm', 0),
                    'width_ppm': p.get('width_ppm', 0),
                    'start_idx': p.get('start_idx', 0),
                    'end_idx': p.get('end_idx', 0),
                    'index': p.get('index', 0),
                    'area': p.get('area', 0),
                    'area_per_sample': p.get('area_per_sample', np.array([])),
                    # Per-sample apex alignment (same analyte across samples)
                    'ppm_per_sample':       p.get('ppm_per_sample'),
                    'shift_ppm_per_sample': p.get('shift_ppm_per_sample'),
                    'aligned_per_sample':   p.get('aligned_per_sample'),
                    'overlapping': False,  # Will be set by overlap search
                    # Quality fields
                    'gaussian_r2':       p.get('gaussian_r2',      float('nan')),
                    'symmetry_score':    p.get('symmetry_score',    float('nan')),
                    'gaussian_quality':  p.get('gaussian_quality',  'unknown'),
                    'status': {'good': 'Good', 'poor': 'Average', 'overlap': 'Poor'}.get(
                        p.get('gaussian_quality', ''), None),
                    'left_grounded':     p.get('left_grounded',     False),
                    'right_grounded':    p.get('right_grounded',    False),
                    'baseline_grounded': p.get('baseline_grounded', False),
                    # Cluster fields (used by annotation for multiplicity)
                    'n_cluster_peaks':  p.get('n_cluster_peaks',  1),
                    'cluster_rank':     p.get('cluster_rank',     0),
                    'cluster_lo':       p.get('cluster_lo',       p.get('start_idx', 0)),
                    'cluster_hi':       p.get('cluster_hi',       p.get('end_idx',   0)),
                    'cluster_lo_ppm':   p.get('cluster_lo_ppm',   p.get('start_ppm', 0.0)),
                    'cluster_hi_ppm':   p.get('cluster_hi_ppm',   p.get('end_ppm',   0.0)),
                }
                self.peaks.append(peak_dict)

            self.processor.peaks = self.peaks

            for p in self.peaks:
                p['overlapping'] = False
            self._deconv_controller = None
            self._overlaps = []
            self.btn_open_deconv.config(state='disabled')
            self.peaks_count_label.config(text=f"Found {len(self.peaks)} peaks")
            try:
                self.app.btn_peak_list.config(state="normal")
            except Exception:
                pass
            try:
                plw = getattr(self.app, '_peak_list_win', None)
                if plw is not None and plw.win.winfo_exists():
                    plw.peaks = self.peaks
                    plw._populate()
            except Exception:
                pass
            self.btn_toggle_peaks.config(state="normal")
            self.btn_show_baseline.config(state="normal")
            self.btn_search_overlaps.config(state="normal")
            self.btn_auto_deconv_asym.config(state="normal")
            _overlay_step("Drawing peak areas…")
            self._status("Drawing peak areas…")
            self.color_peaks(force=True)
            self._status_ok(f"Found {len(self.peaks)} peaks  ({elapsed:.2f}s)")

            # Build graph STOCSY correlation matrix from area_per_sample vectors.
            _overlay_step("Computing peak correlations…")
            try:
                from processing.stocsy import compute_peak_correlation_matrix
                _aint = getattr(self.app, 'all_intensities', None)
                _n_s  = (int(_aint.shape[1])
                         if _aint is not None and _aint.ndim == 2 else 0)
                self._n_samples_pk = _n_s
                self._peak_corr = (compute_peak_correlation_matrix(self.peaks)
                                   if _n_s >= 3 else {})
            except Exception:
                self._peak_corr = {}

            self.overlap_status.config(text="", foreground="gray")
            self.deconv_status.config(
                text="Run 'Search for Overlaps' to find regions needing deconvolution",
                foreground="gray")
            _destroy_overlay()
            messagebox.showinfo(
                "Success",
                f"Detected {len(self.peaks)} peaks\n"
                f"Click 'Search for Overlaps' to find regions needing deconvolution.")

        overlay.after(100, _poll)

    def toggle_peaks(self):
        if self.peaks:
            self.color_peaks(force=False)

    def toggle_baseline(self):
        """Toggle display of estimated baseline on spectrum."""
        if not hasattr(self, '_last_baseline') or self._last_baseline is None:
            messagebox.showinfo("Info", "Run peak detection first to estimate baseline.")
            return

        ax = self.app.ax
        if self._baseline_drawn and self._baseline_line is not None:
            # Remove baseline
            try:
                self._baseline_line.remove()
            except Exception:
                pass
            self._baseline_line = None
            self._baseline_drawn = False
            self.app.canvas.draw_idle()
            self._status("Baseline hidden")
        else:
            # Draw baseline
            ppm = self.app.ppm
            baseline = self._last_baseline

            if ppm is not None and baseline is not None:
                self._baseline_line, = ax.plot(
                    ppm, baseline, 'r--', linewidth=1.5, alpha=0.7,
                    label='Estimated Baseline', zorder=5)
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
                self.app.canvas.draw_idle()
                self._baseline_drawn = True
                self._status("Baseline shown")

    def _store_baseline(self, baseline):
        """Store baseline for later visualization."""
        self._last_baseline = baseline

    # ── Search for Overlaps ───────────────────────────────────────────────────

    def search_for_overlaps(self):
        """
        For each detected peak region (start_ppm to end_ppm), scan the mean
        spectrum looking for:
          - multiple local maxima (peaks)
          - shoulders (inflection points detected via 2nd derivative)
        Regions with >1 peak OR 1 peak + at least one shoulder are marked as
        'needs deconvolution' and loaded into the Deconvolution Tool list.
        """
        if not self.peaks:
            messagebox.showinfo("Info", "Please detect peaks first.")
            return
        if self.processor is None:
            messagebox.showerror("Error", "Peak processor not initialized.")
            return

        mean_vec = _no_blk_group_mean(self.app.all_intensities, self.app.samples_info)
        if mean_vec is None:
            mean_vec = getattr(self.processor, 'mean_intensity', None)
        if mean_vec is None:
            messagebox.showerror("Error", "No reference spectrum available.")
            return

        ppm = self.app.ppm
        noise_std = self.noise_std or 0.0

        self._status("Searching for overlaps and shoulders...")
        import scipy.signal as _sig

        # work only on original (non-sub) peaks
        candidate_peaks = [p for p in self.peaks if not p.get('is_sub_peak', False)]

        # Reset all overlapping flags first
        for p in self.peaks:
            p['overlapping'] = False

        # ── Two-path overlap detection ────────────────────────────────────────
        #
        # Path A  (cluster-based, fast)
        #   Peaks already identified as part of a multi-apex cluster during
        #   separate_overlapping_peaks() use their pre-computed cluster outer
        #   boundaries directly — no expensive scan required.
        #
        # Path B  (scan-based, single-apex peaks only)
        #   Hidden shoulders not resolved as separate apices are detected via
        #   second-derivative sign-change analysis on the peak region.
        # ─────────────────────────────────────────────────────────────────────
        region_peaks   = []
        accepted_ranges = []
        seen_clusters  = set()   # avoid duplicate entries for the same cluster
        min_prom       = max(3.0 * noise_std, 1e-10)

        for p in candidate_peaks:
            n_clust = int(p.get('n_cluster_peaks', 1))

            # ── Path A: multi-apex cluster ────────────────────────────────────
            if n_clust > 1:
                c_lo = float(min(p.get('cluster_lo_ppm', p['ppm']),
                                 p.get('cluster_hi_ppm', p['ppm'])))
                c_hi = float(max(p.get('cluster_lo_ppm', p['ppm']),
                                 p.get('cluster_hi_ppm', p['ppm'])))
                c_key = (round(c_lo, 5), round(c_hi, 5))
                p['overlapping'] = True
                if c_key in seen_clusters:
                    continue   # this cluster was already added
                seen_clusters.add(c_key)
                accepted_ranges.append((c_lo, c_hi))
                rp = dict(p)
                rp['start_ppm']          = c_hi    # NMR convention: start > end
                rp['end_ppm']            = c_lo
                rp['overlapping']        = True
                rp['from_overlap_search'] = True
                rp['_n_maxima']          = n_clust
                rp['_n_shoulders']       = 0
                rp['_foot_not_grounded'] = False
                rp['_poor_gaussian']     = False
                rp['baseline_validation'] = {'valid': True}
                print(f"   Cluster {c_lo:.4f}–{c_hi:.4f}: {n_clust} peaks → DECONV")
                region_peaks.append(rp)
                continue

            # ── Path B: single-apex — scan for hidden shoulders ───────────────
            plo = float(min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
            phi = float(max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
            if (phi - plo) < 1e-4:
                mid = p['ppm']
                hw  = max(0.02, float(p.get('width_ppm', 0.01)) * 3)
                plo, phi = mid - hw, mid + hw

            mask = (ppm >= plo) & (ppm <= phi)
            if np.count_nonzero(mask) < 5:
                continue

            x_reg       = ppm[mask]
            y_reg       = mean_vec[mask]
            region_apex = float(np.max(y_reg)) if len(y_reg) else 0.0
            if region_apex <= min_prom:
                continue

            n_shoulders = 0
            if len(y_reg) >= 7:
                from scipy.ndimage import gaussian_filter1d
                smooth_pts = max(3, int(0.005 / abs(ppm[1] - ppm[0])) if len(ppm) > 1 else 3)
                y_smooth   = gaussian_filter1d(y_reg, sigma=smooth_pts)
                d2         = np.gradient(np.gradient(y_smooth))
                edge_pts   = max(2, len(x_reg) // 10)
                # A shoulder is a SECONDARY curvature maximum (a peak in -d²) that
                # is not the main apex. A clean single peak has exactly one
                # curvature maximum (at its apex) → 0 shoulders. Counting raw
                # d² sign-changes (previous approach) flagged every clean peak,
                # because a lone Gaussian already has a centre→wing transition.
                neg_d2  = -d2
                d2_mad  = float(np.median(np.abs(d2 - np.median(d2)))) * 1.4826
                cmax, _ = _sig.find_peaks(neg_d2, prominence=max(3.0 * d2_mad, 1e-12))
                apex_local = int(np.argmax(y_smooth))
                n_shoulders = sum(
                    1 for c in cmax
                    if abs(int(c) - apex_local) > edge_pts
                    and edge_pts < int(c) < (len(x_reg) - edge_pts)
                    and y_smooth[int(c)] > max(0.10 * region_apex, min_prom)
                )

            foot_not_grounded = not p.get('baseline_grounded', True)
            poor_gaussian     = p.get('gaussian_quality', 'good') in ('poor', 'overlap')
            needs_deconv      = n_shoulders > 0 or foot_not_grounded or poor_gaussian

            _r2 = f"{p.get('gaussian_r2', float('nan')):.2f}"
            print(f"   Peak {plo:.4f}–{phi:.4f}: {n_shoulders} shoulders, "
                  f"grounded={p.get('baseline_grounded', True)}, R²={_r2} "
                  f"→ {'DECONV' if needs_deconv else 'ok'}")

            if not needs_deconv:
                continue

            sigma = float(noise_std) if noise_std > 0 else 1e-10
            adj_lo, adj_hi, chosen_m = _expand_region_to_baseline_edges(
                ppm, mean_vec, plo, phi, sigma)
            if not chosen_m.get('valid', False):
                adj_lo, adj_hi = plo, phi

            p['overlapping'] = True
            accepted_ranges.append((float(min(adj_lo, adj_hi)),
                                    float(max(adj_lo, adj_hi))))
            rp = dict(p)
            rp['start_ppm']          = float(max(adj_lo, adj_hi))
            rp['end_ppm']            = float(min(adj_lo, adj_hi))
            rp['overlapping']        = True
            rp['from_overlap_search'] = True
            rp['_n_maxima']          = 1
            rp['_n_shoulders']       = n_shoulders
            rp['_foot_not_grounded'] = bool(foot_not_grounded)
            rp['_poor_gaussian']     = bool(poor_gaussian)
            rp['baseline_validation'] = chosen_m
            region_peaks.append(rp)

        # Merge adjacent / overlapping regions that belong to groups of peaks
        # (two peaks whose adjusted regions touch or overlap → merge them)
        if region_peaks:
            region_peaks.sort(key=lambda r: min(r['start_ppm'], r['end_ppm']))
            merged = []
            for rp in region_peaks:
                r_lo = float(min(rp['start_ppm'], rp['end_ppm']))
                r_hi = float(max(rp['start_ppm'], rp['end_ppm']))
                if merged and r_lo <= merged[-1]['_hi'] + 0.005:
                    # Extend the previous merged region
                    new_lo = float(min(merged[-1]['start_ppm'], merged[-1]['end_ppm']))
                    new_hi = float(max(r_hi, merged[-1]['_hi']))
                    merged[-1]['start_ppm'] = new_hi
                    merged[-1]['end_ppm']   = new_lo
                    merged[-1]['_hi']       = new_hi
                    merged[-1]['_n_maxima']   = merged[-1].get('_n_maxima', 1) + rp.get('_n_maxima', 1)
                    merged[-1]['_n_shoulders'] = merged[-1].get('_n_shoulders', 0) + rp.get('_n_shoulders', 0)
                    # Mark all peaks in merged range as overlapping
                    for orig_p in candidate_peaks:
                        orig_ppm = orig_p['ppm']
                        if new_lo <= orig_ppm <= new_hi:
                            orig_p['overlapping'] = True
                else:
                    rp['_hi'] = r_hi
                    merged.append(rp)
            region_peaks = merged

        # ── Update controller ─────────────────────────────────────────────────
        self.processor.peaks = self.peaks
        self._overlaps = accepted_ranges
        self._deconv_controller = None

        # Redraw with updated overlap colours
        if self.app.plot_manager and self.app.plot_manager._peaks_drawn:
            self.color_peaks(force=True)

        n = len(region_peaks)
        if not region_peaks:
            self.btn_open_deconv.config(state="disabled")
            try:
                self.app.btn_deconv_shortcut.config(state="disabled")
            except Exception:
                pass
            self.overlap_status.config(
                text="No regions needing deconvolution found", foreground="green")
            self.deconv_status.config(
                text="All peaks are well separated", foreground="gray")
            self._status_ok("No overlapping/shoulder regions found")
            messagebox.showinfo("Result",
                "No regions needing deconvolution were found.\n"
                "All peaks appear to be well separated.")
            return

        from gui.notation.deconvolution import DeconvController
        self._deconv_controller = DeconvController(self.processor)
        self._deconv_controller.set_peaks(region_peaks)

        self.btn_open_deconv.config(state="normal")
        try:
            self.app.btn_deconv_shortcut.config(state="normal")
        except Exception:
            pass

        multi_max_count   = sum(1 for r in region_peaks if r.get('_n_maxima', 0) > 1)
        shoulder_count    = sum(1 for r in region_peaks
                                if r.get('_n_shoulders', 0) > 0
                                and r.get('_n_maxima', 0) < 2
                                and not r.get('_foot_not_grounded', False)
                                and not r.get('_poor_gaussian', False))
        foot_count        = sum(1 for r in region_peaks
                                if r.get('_foot_not_grounded', False)
                                and r.get('_n_maxima', 0) < 2)
        gauss_count       = sum(1 for r in region_peaks
                                if r.get('_poor_gaussian', False)
                                and r.get('_n_maxima', 0) < 2
                                and not r.get('_foot_not_grounded', False))
        self.overlap_status.config(
            text=f"{n} region(s) need deconvolution "
                 f"({multi_max_count} multi-peak, {shoulder_count} shoulder, "
                 f"{foot_count} foot∉baseline, {gauss_count} poor R²)",
            foreground="orange")
        self.deconv_status.config(
            text=f"{n} region(s) queued — click 'Open Deconvolution Tool'",
            foreground="blue")
        self._status_ok(f"Found {n} region(s) needing deconvolution")
        messagebox.showinfo("Result",
            f"{n} region(s) need deconvolution:\n"
            f"  • {multi_max_count} with multiple peaks\n"
            f"  • {shoulder_count} with shoulder(s)\n"
            f"  • {foot_count} with foot not at baseline\n"
            f"  • {gauss_count} with poor Gaussian fit (R² < 0.90)\n\n"
            f"Open the Deconvolution Tool to proceed.")

    def color_peaks(self, force=False):
        if not self.peaks or self.app.all_intensities is None:
            return
        mean = _no_blk_group_mean(
            self.app.all_intensities, self.app.samples_info)
        if self.app.plot_manager is not None:
            if force:
                self.app.plot_manager.force_redraw_peaks(
                    self.peaks, mean, self.app.ppm)
            else:
                self.app.plot_manager.draw_peak_overlay(
                    self.peaks, mean, self.app.ppm)

    def find_overlaps(self, _silent=False):
        if not self.peaks:
            if not _silent:
                messagebox.showinfo("Info", "Please detect peaks first.")
            return
        self._status("Searching for overlapping regions (consensus)...")
        t0 = time.monotonic()

        # Read consensus threshold from UI (convert % → fraction)
        try:
            consensus_pct = float(self.consensus_var.get())
            consensus_thr = max(0.0, min(1.0, consensus_pct / 100.0))
        except Exception:
            consensus_thr = 0.50

        # Consensus-based detection: each spectrum votes independently.
        # Store a copy of the current mean_intensity so downstream code
        # (colour_peaks, DeconvController) still has a reference spectrum.
        self.processor._original_mean_intensity = (
            self.processor.mean_intensity.copy()
            if self.processor.mean_intensity is not None else None)

        # Sync processor.peaks to stage.peaks BEFORE consensus detection
        # so that _in_overlap_group flags land on the same list we iterate below.
        self.processor.peaks = self.peaks

        overlaps = self.processor.find_overlapping_regions_consensus(
            all_intensities=self.app.all_intensities,
            samples_info=self.app.samples_info,
            distance_threshold=0.04,
            consensus_threshold=consensus_thr,
            min_snr=self.min_snr_var.get(),
            min_width_ppm=self.min_width_var.get(),
            prominence_factor=self.prominence_var.get(),
            enforce_baseline_edges=False,
        )

        # Baseline validation is enforced at chunk level to avoid rejecting broad
        # consensus regions that still contain baseline-safe sub-regions.
        mean_vec = _no_blk_group_mean(self.app.all_intensities, self.app.samples_info)
        if mean_vec is None:
            mean_vec = getattr(self.processor, 'mean_intensity', None)
        sigma = float(getattr(self.processor, 'noise_std', 0.0) or self.noise_std or 0.0)
        metrics = []

        elapsed = time.monotonic() - t0

        # Build controller entries from overlap-event subgroups (variable number
        # of peaks), then enforce/expand to baseline on both edges.
        region_peaks = []
        accepted_ranges = []
        rejected_chunks = 0
        expanded_chunks = 0

        if mean_vec is not None:
            for (lo, hi) in overlaps:
                lo_n, hi_n = min(lo, hi), max(lo, hi)
                peaks_in = sorted(
                    [p for p in self.peaks if lo_n <= p.get('ppm', 0.0) <= hi_n],
                    key=lambda p: p['ppm'])
                if not peaks_in:
                    continue

                # Build lightweight peaks and split by true overlap events
                # (shoulder/saddle/base sharing), not by fixed peak count.
                lite_peaks = []
                for pk in peaks_in:
                    lite_peaks.append({
                        'ppm': float(pk['ppm']),
                        'start_ppm': float(min(pk.get('start_ppm', pk['ppm']),
                                               pk.get('end_ppm', pk['ppm']))),
                        'end_ppm': float(max(pk.get('start_ppm', pk['ppm']),
                                             pk.get('end_ppm', pk['ppm']))),
                        'height': float(pk.get('height', 0.0)),
                        'prominence': float(pk.get('prominence', pk.get('height', 0.0))),
                        'index': int(pk.get('index', np.argmin(np.abs(self.app.ppm - pk['ppm'])))),
                    })

                sub_regions = self.processor._group_peaks_single(
                    lite_peaks, distance_threshold=0.04, intensity_1d=mean_vec)
                if not sub_regions:
                    sub_regions = [(lo_n, hi_n)]

                for rg_lo, rg_hi in sub_regions:
                    rg_lo_n, rg_hi_n = min(rg_lo, rg_hi), max(rg_lo, rg_hi)
                    subgroup = [p for p in peaks_in if rg_lo_n <= p['ppm'] <= rg_hi_n]
                    if not subgroup:
                        continue

                    adj_lo, adj_hi, chosen_m = _expand_region_to_baseline_edges(
                        self.app.ppm, mean_vec, rg_lo_n, rg_hi_n, sigma)
                    if not chosen_m.get('valid', False):
                        rejected_chunks += 1
                        metrics.append(chosen_m)
                        continue

                    if chosen_m.get('expanded', False):
                        expanded_chunks += 1

                    metrics.append(chosen_m)
                    accepted_ranges.append((float(min(adj_lo, adj_hi)),
                                            float(max(adj_lo, adj_hi))))

                    rep = max(subgroup, key=lambda p: p.get('height', 0))
                    rp = dict(rep)
                    rp['start_ppm'] = float(max(adj_lo, adj_hi))
                    rp['end_ppm'] = float(min(adj_lo, adj_hi))
                    rp['overlapping'] = True
                    rp['from_overlap_search'] = True
                    rp['_grouped_from'] = [p['ppm'] for p in subgroup]
                    rp['baseline_validation'] = chosen_m
                    region_peaks.append(rp)
        else:
            for lo, hi in overlaps:
                metrics.append({
                    'valid': False,
                    'reason': 'missing_reference_spectrum',
                    'lo': float(min(lo, hi)),
                    'hi': float(max(lo, hi)),
                })
            rejected_chunks = len(overlaps)

        # Update overlapping flags from accepted (baseline-safe) chunks.
        for peak in self.peaks:
            peak_ppm = float(peak.get('ppm', np.nan))
            in_group = any(lo <= peak_ppm <= hi for (lo, hi) in accepted_ranges)
            peak.pop('_in_overlap_group', None)
            peak['overlapping'] = in_group
            peak.pop('_deconvolved', None)
            if in_group:
                peak.setdefault('from_overlap_search', True)
        self.processor.peaks = self.peaks
        self._overlap_metrics = metrics
        self._overlaps = accepted_ranges
        self._deconv_controller = None

        if not region_peaks:
            rejected = max(rejected_chunks, len(overlaps))
            self.btn_open_deconv.config(state="disabled")
            try:
                self.app.btn_deconv_shortcut.config(state="disabled")
            except Exception:
                pass
            self._status_ok(f"No overlapping regions  ({elapsed:.2f}s)")
            if rejected > 0:
                self.deconv_status.config(
                    text=(f"Rejected {rejected} overlap region(s): "
                          "missing baseline on both edges"),
                    foreground="orange")
            else:
                self.deconv_status.config(
                    text="No overlapping regions found",
                    foreground="gray")
            if not _silent:
                if rejected > 0:
                    messagebox.showwarning(
                        "Baseline Validation",
                        "Overlapping regions were detected, but all were rejected "
                        "because baseline was not stable on both PPM edges.")
                else:
                    messagebox.showinfo("Info",
                                        "No overlapping regions found.\n"
                                        "All peaks are well separated.")
            return

        from gui.notation.deconvolution import DeconvController
        self._deconv_controller = DeconvController(self.processor)
        self._deconv_controller.set_peaks(region_peaks)

        # Redraw overlay with updated overlap flags
        if self.app.plot_manager and self.app.plot_manager._peaks_drawn:
            self.color_peaks(force=True)

        n = len(region_peaks)
        rejected = rejected_chunks
        self.btn_open_deconv.config(state="normal")
        if rejected > 0:
            self.deconv_status.config(
                text=(f"{n} accepted, {rejected} rejected "
                      f"(baseline rule; expanded {expanded_chunks})"),
                foreground="orange")
        else:
            self.deconv_status.config(
                text=f"{n} overlapping region{'s' if n != 1 else ''} found "
                     f"(expanded {expanded_chunks})",
                foreground="orange")
        self._status_ok(f"Found {n} overlapping regions  ({elapsed:.2f}s)")
        # Enable bottom-bar shortcut
        try:
            self.app.btn_deconv_shortcut.config(state="normal")
        except Exception:
            pass

    def auto_deconvolve_asymmetric(self, sym_threshold=0.70):
        """Queue asymmetric peaks (symmetry_score < threshold) for deconvolution
        and open the tool immediately.

        Intended for peaks with a hidden shoulder that the Gaussian/Lorentzian
        fit may still score well but that are detectably asymmetric.
        """
        if not self.peaks:
            messagebox.showinfo("Info", "No peaks detected.")
            return

        mean_vec = _no_blk_group_mean(self.app.all_intensities, self.app.samples_info)
        if mean_vec is None:
            mean_vec = getattr(self.processor, 'mean_intensity', None)
        if mean_vec is None:
            messagebox.showerror("Error", "No reference spectrum available.")
            return

        ppm      = self.app.ppm
        noise_std = self.noise_std or 0.0
        sigma     = float(noise_std) if noise_std > 0 else 1e-10

        # Candidates: single-apex Good/Average peaks with measured low symmetry.
        # Multi-apex clusters are already handled by Search for Overlaps.
        candidate_peaks = [
            p for p in self.peaks
            if not p.get('is_sub_peak', False)
            and int(p.get('n_cluster_peaks', 1)) == 1
            and p.get('status') in ('Good', 'Average')
            and float(p.get('symmetry_score', 1.0)) < sym_threshold
        ]

        if not candidate_peaks:
            messagebox.showinfo(
                "Auto-deconvolve",
                f"No asymmetric peaks found (symmetry < {sym_threshold:.0%}).\n"
                "All single-apex peaks appear symmetric.")
            return

        region_peaks    = []
        accepted_ranges = []

        for p in candidate_peaks:
            plo = float(min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
            phi = float(max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
            if (phi - plo) < 1e-4:
                hw  = max(0.02, float(p.get('width_ppm', 0.01)) * 3)
                plo, phi = p['ppm'] - hw, p['ppm'] + hw

            adj_lo, adj_hi, chosen_m = _expand_region_to_baseline_edges(
                ppm, mean_vec, plo, phi, sigma)
            if not chosen_m.get('valid', False):
                adj_lo, adj_hi = plo, phi

            p['overlapping'] = True
            accepted_ranges.append((float(min(adj_lo, adj_hi)),
                                    float(max(adj_lo, adj_hi))))
            rp = dict(p)
            rp['start_ppm']           = float(max(adj_lo, adj_hi))
            rp['end_ppm']             = float(min(adj_lo, adj_hi))
            rp['overlapping']         = True
            rp['from_overlap_search'] = True
            rp['_n_maxima']           = 1
            rp['_n_shoulders']        = 1   # suspected shoulder
            rp['_foot_not_grounded']  = not p.get('baseline_grounded', True)
            rp['_poor_gaussian']      = p.get('gaussian_quality') in ('poor', 'overlap')
            rp['_sym_score']          = float(p.get('symmetry_score', 0.0))
            rp['baseline_validation'] = chosen_m
            region_peaks.append(rp)

        # Merge with any existing overlap regions from Search for Overlaps
        existing = list(self._overlaps or [])
        self._overlaps = existing + accepted_ranges

        from gui.notation.deconvolution import DeconvController, DeconvWindow
        if self._deconv_controller is None:
            self._deconv_controller = DeconvController(self.processor)

        # Merge with peaks already in controller (if any)
        existing_peaks = list(getattr(self._deconv_controller, 'peaks', None) or [])
        existing_ppms  = {round(ep['ppm'], 4) for ep in existing_peaks}
        new_peaks      = [rp for rp in region_peaks
                          if round(rp['ppm'], 4) not in existing_ppms]
        self._deconv_controller.set_peaks(existing_peaks + new_peaks)

        self.btn_open_deconv.config(state="normal")
        self.processor.peaks = self.peaks
        if self.app.plot_manager and self.app.plot_manager._peaks_drawn:
            self.color_peaks(force=True)

        n = len(new_peaks)
        self.overlap_status.config(
            text=f"{n} asymmetric peak(s) queued for deconvolution",
            foreground="orange")
        self.deconv_status.config(
            text=f"{n} peak(s) queued — opening tool…",
            foreground="blue")
        self._status_ok(f"Auto-deconv: {n} asymmetric peak(s) queued")

        # Open tool immediately
        dw = DeconvWindow(self.app.root, self._deconv_controller, self)
        try:
            self.app._deconv_win = dw
            dw.window.protocol('WM_DELETE_WINDOW',
                lambda: (dw.window.destroy(),
                         setattr(self.app, '_deconv_win', None)))
        except Exception:
            pass

    def open_deconv_tool(self):
        """Open the deconvolution tool, reusing the persistent controller."""
        if not self._overlaps:
            messagebox.showinfo("Info",
                                "No overlapping regions found.\n"
                                "All peaks are well separated.")
            return

        # Reuse the existing controller so all deconvolved results are preserved.
        # If the processor changed (new detection run), _deconv_controller was
        # reset in detect_peaks and recreated in find_overlaps.
        if self._deconv_controller is None:
            from gui.notation.deconvolution import DeconvController
            self._deconv_controller = DeconvController(self.processor)
            pass  # set_peaks already called in find_overlaps

        from gui.notation.deconvolution import DeconvWindow
        dw = DeconvWindow(self.app.root, self._deconv_controller, self)
        # Keep singleton reference in app
        try:
            self.app._deconv_win = dw
            dw.window.protocol('WM_DELETE_WINDOW',
                lambda: (dw.window.destroy(),
                         setattr(self.app, '_deconv_win', None)))
        except Exception:
            pass

    def _show_deconv_results(self, results):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self.app.root)
        win.title(f"Deconvolution — {len(results['peaks'])} peaks")
        win.geometry("900x700")
        win.transient(self.app.root)
        mf = ttk.Frame(win, padding=10)
        mf.pack(fill=tk.BOTH, expand=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        def gaussian(x, a, c, w):
            return a * np.exp(-(x - c) ** 2 / (2 * w ** 2))

        ax1.plot(results['region_ppm'], results['region_intensity'],
                 'b-', lw=1.5, label='Original', alpha=0.8)
        for i, p in enumerate(results['peaks']):
            ax1.axvline(p['ppm'], color='r', ls='--', alpha=0.5, lw=1)
            ax1.text(p['ppm'], p['height'], f' {i+1}',
                     fontsize=9, fontweight='bold', color='darkred')
        ax1.invert_xaxis()
        ax1.set_xlabel('δ (ppm)'); ax1.set_ylabel('Intensity')
        ax1.set_title('Detected Peak Positions')
        ax1.grid(True, alpha=0.3, ls='--'); ax1.legend()

        ax2.plot(results['region_ppm'], results['region_intensity'],
                 'b-', lw=1.5, alpha=0.4, label='Original')
        xd = np.linspace(
            results['region_ppm'][0], results['region_ppm'][-1], 500)
        cols = plt.cm.tab10(np.linspace(0, 1, len(results['peaks'])))
        cum = np.zeros_like(xd)
        for i, p in enumerate(results['peaks']):
            yp = gaussian(xd, p['height'], p['ppm'], p['width'])
            cum += yp
            ax2.plot(xd, yp, '--', color=cols[i], lw=2,
                     label=f"Peak {i+1}: {p['ppm']:.4f}")
            ax2.fill_between(xd, 0, yp, alpha=0.2, color=cols[i])
        ax2.plot(xd, cum, 'r-', lw=2, label='Fit', alpha=0.9)
        ax2.invert_xaxis()
        ax2.set_xlabel('δ (ppm)'); ax2.set_ylabel('Intensity')
        ax2.set_title(f"Deconvolution — BIC: {results['bic']:.1f}")
        ax2.grid(True, alpha=0.3, ls='--')
        ax2.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        cv = FigureCanvasTkAgg(fig, master=mf)
        cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(mf, text="Close",
                   command=win.destroy).pack(side=tk.RIGHT, pady=5)
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry(
            f'{w}x{h}+{win.winfo_screenwidth()//2-w//2}'
            f'+{win.winfo_screenheight()//2-h//2}')

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        print("🔄 PeakDetectionStage activated")
        if self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)
        if self.app.plot_manager and self.app.plot_manager.original_xlim is None:
            self.app.plot_manager.original_xlim = self.app.ax.get_xlim()
            self.app.plot_manager.original_ylim = self.app.ax.get_ylim()
        self._status_ready()
        self.app.root.bind('<Key-r>', lambda e: self.reset_zoom())
        self.app.root.bind('<Key-R>', lambda e: self.reset_zoom())

    def on_deactivate(self):
        print("PeakDetectionStage deactivated")
        self.app.root.unbind('<Key-r>')
        self.app.root.unbind('<Key-R>')
        if self.frame:
            self.frame.pack_forget()

    def reset_zoom(self):
        if self.app.plot_manager:
            self.app.plot_manager.reset_zoom()