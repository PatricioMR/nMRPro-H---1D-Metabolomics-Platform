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
        self.per_sample_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            pf1, text="Detect per sample (union)",
            variable=self.per_sample_var).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Label(pf1, text="Min samples/group:").pack(side=tk.LEFT, padx=(10, 2))
        self.min_group_presence_var = tk.IntVar(value=3)
        ttk.Spinbox(pf1, from_=1, to=50, width=4,
                    textvariable=self.min_group_presence_var).pack(side=tk.LEFT)
        ttk.Label(pf1, text="(keep peak only if seen in ≥N samples of a group)",
                  font=('Arial', 7), foreground='#666').pack(side=tk.LEFT, padx=(4, 0))

        # Row 2b: shoulder recovery (2nd-derivative) — recovers overlapping
        # lines that have no intensity maximum of their own.
        pf1b = ttk.Frame(s2)
        pf1b.pack(fill=tk.X, pady=2)
        self.detect_shoulders_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            pf1b, text="Recover shoulders (2nd-deriv)",
            variable=self.detect_shoulders_var).pack(side=tk.LEFT, padx=2)

        # Row 2c: shoulder sensitivity on its own line (kept off the crowded row
        # above so the entry is always visible).
        pf1c = ttk.Frame(s2)
        pf1c.pack(fill=tk.X, pady=2)
        ttk.Label(pf1c, text="Sensitivity (k·σ):").pack(side=tk.LEFT, padx=2)
        self.shoulder_k_var = tk.DoubleVar(value=3.0)
        ttk.Entry(pf1c, textvariable=self.shoulder_k_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(pf1c, text="(lower = more sensitive)",
                  font=('Arial', 7), foreground='#666').pack(side=tk.LEFT)

        # Row 3: Min Width + Prominence
        pf2 = ttk.Frame(s2)
        pf2.pack(fill=tk.X, pady=2)
        ttk.Label(pf2, text="Min Width (ppm):").pack(side=tk.LEFT, padx=2)
        self.min_width_var = tk.DoubleVar(value=0.005)
        ttk.Entry(pf2, textvariable=self.min_width_var,
                  width=6).pack(side=tk.LEFT, padx=2)

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
                    detect_shoulders=bool(self.detect_shoulders_var.get()),
                    shoulder_curvature_k=float(self.shoulder_k_var.get()),
                    min_group_presence=int(self.min_group_presence_var.get()),
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
                    # Shape-based 3-tier quality (green/yellow/red + score).
                    'peak_quality':       p.get('peak_quality', 'yellow'),
                    'peak_quality_score': p.get('peak_quality_score', float('nan')),
                    # Manual annotation status starts unset — the user marks
                    # Good/Average/Poor; it is NOT derived from the shape quality.
                    'status': None,
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

            _destroy_overlay()
            messagebox.showinfo(
                "Success",
                f"Detected {len(self.peaks)} peaks")

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