# gui/stages/peak_detection_stage.py
"""
Signal Detection Stage — noise region + detection parameters.
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
try:
    import certifi
except ImportError:
    certifi = None

if certifi is not None:
    os.environ['SSL_CERT_FILE']      = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from processing.noise_estimation import estimate_noise_from_region
from processing.sample_selection import included_sample_indices
from processing.spectral_regions import (
    DEFAULT_EXCLUDED_REGIONS, format_regions, measure_linewidth_ppm,
    parse_regions, unresolvable_split_warning)
from processing.signal_quality import SIGNAL_AVERAGE
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

        ttk.Label(self.frame, text="Signal Detection",
                  font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        # ── Noise region ──────────────────────────────────────────────────────
        # σ is measured when signals are detected. These entries pin a specific
        # signal-free region; left blank, the detector falls back to the
        # quietest segment it can find. Which of the two was used is reported
        # below the entries — a silent fallback would hide a mistyped region and
        # change every SNR downstream.
        s1 = ttk.LabelFrame(self.frame, text="📏 Noise Region", padding=5)
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
        ttk.Label(rf, text="(high → low δ)",
                  font=('Arial', 7), foreground='#666').pack(side=tk.LEFT, padx=(4, 0))

        self.noise_value_label = ttk.Label(s1, text="Noise Std Dev: --")
        self.noise_value_label.pack(anchor=tk.W, pady=2)
        # Reports where σ came from, or why the region was rejected.
        self.step1_status = ttk.Label(s1, text="", font=('Arial', 8),
                                      foreground='gray')
        self.step1_status.pack(anchor=tk.W)

        # ── Parameters ────────────────────────────────────────────────────────
        s2 = ttk.LabelFrame(self.frame, text="🔍 Parameters", padding=5)
        s2.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: Baseline Correction Toggle
        bf1 = ttk.Frame(s2)
        bf1.pack(fill=tk.X, pady=2)
        self.baseline_correct_var = tk.BooleanVar(value=True)
        self._baseline_chk = ttk.Checkbutton(bf1, text="Apply Baseline Correction",
                                              variable=self.baseline_correct_var)
        self._baseline_chk.pack(side=tk.LEFT, padx=2)
        # SNIP is the default: it preserves ~99% of a sharp apex against ~92%
        # for arPLS at λ=1e5, which eats the peak skirts and biases integrals
        # low by roughly 20% of the area.
        self.baseline_method_var = tk.StringVar(value='snip')
        cb = ttk.Combobox(bf1, textvariable=self.baseline_method_var,
                          values=['snip', 'arPLS', 'rolling', 'constant'],
                          state='readonly', width=8)
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind('<<ComboboxSelected>>', self._on_baseline_method_change)

        # One entry, retargeted per method. Each method's width cutoff has its
        # own parameter and its own units; a fixed "λ (arPLS)" box would sit
        # there doing nothing whenever another method is selected.
        self.baseline_lam_var   = tk.DoubleVar(value=1e5)   # arPLS
        self.baseline_niter_var = tk.IntVar(value=40)       # snip
        self._bl_param_lbl = ttk.Label(bf1, text="", foreground='gray')
        self._bl_param_lbl.pack(side=tk.LEFT, padx=(8, 2))
        self._bl_param_entry = ttk.Entry(bf1, width=6)
        self._bl_param_entry.pack(side=tk.LEFT, padx=2)
        self._on_baseline_method_change()
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
        ttk.Label(pf1, text="(keep signal only if seen in ≥N samples of a group)",
                  font=('Arial', 7), foreground='#666').pack(side=tk.LEFT, padx=(4, 0))

        # Row 2b: shoulder recovery (2nd-derivative) — recovers overlapping
        # lines that have no intensity maximum of their own. Off by default: a
        # recovered shoulder has no apex of its own, so its position and area
        # carry more uncertainty than a resolved line; it is opt-in.
        pf1b = ttk.Frame(s2)
        pf1b.pack(fill=tk.X, pady=2)
        self.detect_shoulders_var = tk.BooleanVar(value=False)
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

        # Row 3: minimum separation between apices
        pf2 = ttk.Frame(s2)
        pf2.pack(fill=tk.X, pady=2)
        ttk.Label(pf2, text="Min Distance (ppm):").pack(side=tk.LEFT, padx=2)

        # 0.004 ppm = 2.4 Hz at 600 MHz. Measured on a 37-sample mouse
        # serum set (600.71 MHz, BMRB): going from 0.008 to 0.004 raised
        # the multiplets with a usable ΔJ from 59 to 86 of 214 and halved
        # the median ΔJ (0.74 -> 0.52 Hz), at the cost of one Good singlet.
        # 0.003 bought one more multiplet for 141 extra peaks, so the knee
        # is here. Despite the label this is find_peaks(distance=...) — the
        # minimum SEPARATION between apices, not a linewidth.
        self.min_width_var = tk.DoubleVar(value=0.004)
        ttk.Entry(pf2, textvariable=self.min_width_var,
                  width=6).pack(side=tk.LEFT, padx=2)

        # Excluded regions. The residual water resonance is orders of magnitude
        # above any metabolite and presaturation flattens whatever sits under
        # it, so peaks there are artefact — but the reference database has 82
        # signals between 4.5 and 4.9 ppm that will happily match one. See
        # processing/spectral_regions.py for the protocol references.
        pf3 = ttk.Frame(s2)
        pf3.pack(fill=tk.X, pady=1)
        ttk.Label(pf3, text="Exclude regions (ppm):").pack(side=tk.LEFT)
        self.excluded_regions_var = tk.StringVar(
            value=format_regions(DEFAULT_EXCLUDED_REGIONS))
        ttk.Entry(pf3, textvariable=self.excluded_regions_var,
                  width=14).pack(side=tk.LEFT, padx=2)

        self.btn_detect = ttk.Button(
            s2, text="🎯 Detect Signals",
            command=self.detect_peaks, state="normal")
        self.btn_detect.pack(fill=tk.X, pady=5)

        self.peaks_count_label = ttk.Label(s2, text="Signals found: 0")
        self.peaks_count_label.pack(anchor=tk.W, pady=2)
        self.step2_status = ttk.Label(
            s2, text="⏳ Load data, then detect signals", foreground="gray")
        self.step2_status.pack(anchor=tk.W, pady=2)

        self.btn_toggle_peaks = ttk.Button(
            s2, text="👁 Show/Hide Signal Areas",
            command=self.toggle_peaks, state="disabled")
        self.btn_toggle_peaks.pack(fill=tk.X, pady=2)

        self.btn_show_baseline = ttk.Button(
            s2, text="📈 Show/Hide Estimated Baseline",
            command=self.toggle_baseline, state="disabled")
        self.btn_show_baseline.pack(fill=tk.X, pady=2)
        self._baseline_drawn = False
        self._baseline_line = None

        # ── Mouse Controls ────────────────────────────────────────────────────
        mf = ttk.LabelFrame(self.frame, text="🐭 Mouse Controls", padding=5)
        mf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mf,
                  text="• Scroll: Zoom in/out\n"
                       "• Drag: Pan\n"
                       "• R: Reset view",
                  font=("Arial", 8), foreground="blue",
                  justify=tk.LEFT).pack(anchor=tk.W, padx=5)

    # ── Noise ─────────────────────────────────────────────────────────────────

    def _estimate_noise(self):
        """Measure σ from the noise region; return σ, or None to fall back.

        Runs as part of signal detection — there is no separate button.  When
        the region is usable, σ is its per-sample std reduced by the median
        across samples; otherwise the detector estimates σ from the quietest
        segment it can find.

        Every rejection path reports itself in the panel: σ scales every SNR in
        the run and gates the quality classifier, so a mistyped region that
        silently reverted to the automatic estimate would move the whole
        detection without the analyst ever seeing it.
        """
        raw_start = str(self.noise_start_var.get()).strip()
        raw_end   = str(self.noise_end_var.get()).strip()
        if not raw_start and not raw_end:
            self._set_noise_status(
                'σ measured automatically (no region set).', 'gray')
            return None
        try:
            r_start = float(raw_start)
            r_end   = float(raw_end)
        except (ValueError, TypeError):
            self._set_noise_status(
                '⚠ Region is not numeric — σ measured automatically.', 'orange')
            return None
        if r_start <= r_end:
            self._set_noise_status(
                f'⚠ Region must run high → low δ (got {r_start:g} to '
                f'{r_end:g}) — σ measured automatically.', 'orange')
            return None
        try:
            sigma = estimate_noise_from_region(
                self.app.ppm, self.app.all_intensities, r_start, r_end)
        except Exception as e:
            self._set_noise_status(
                f'⚠ {e} — σ measured automatically.', 'orange')
            return None
        if not np.isfinite(sigma) or sigma <= 0:
            self._set_noise_status(
                '⚠ Region gave σ ≤ 0 (flat or empty) — '
                'σ measured automatically.', 'orange')
            return None

        self.noise_std    = float(sigma)
        self.noise_source = 'ppm'
        self.step1_completed = True
        self.noise_value_label.config(text=f"Noise Std Dev: {self.noise_std:.4f}")
        self._set_noise_status(f'✅ σ from {r_start:g} → {r_end:g} ppm.', 'green')
        return self.noise_std

    def _set_noise_status(self, msg, colour):
        """Report where σ came from — or why the typed region was rejected."""
        try:
            self.step1_status.config(text=msg, foreground=colour)
        except Exception:
            pass

    # Parameter that sets each method's width cutoff — what counts as "broad
    # enough to be baseline". Label, units and default all differ, so the shared
    # entry is retargeted rather than mislabelled.
    _BASELINE_PARAMS = {
        'arPLS':    ('λ (smoothness):', 'baseline_lam_var'),
        'snip':     ('Iterations:',     'baseline_niter_var'),
        'rolling':  ('',                None),
        'constant': ('',                None),
    }

    def push_baseline_model(self):
        """Write the panel's choice into the project's single baseline model.

        The model is what every step reads, so editing it here changes
        detection, quantification, statistics and MCR together instead of only
        this panel's own working copy.
        """
        try:
            from processing.baseline_model import BaselineModel, METHOD_PARAMS
            method = self.baseline_method_var.get()
            name, _ = METHOD_PARAMS.get(method, (None, None))
            params = {}
            if name == 'lam':
                params = {'lam': float(self.baseline_lam_var.get())}
            elif name == 'niter':
                params = {'niter': int(self.baseline_niter_var.get())}
            self.app.baseline_model = BaselineModel(
                enabled=bool(self.baseline_correct_var.get()),
                method=method, params=params)
            self.app.invalidate_baseline_cache()
        except Exception as e:
            print(f'[baseline] model not updated: {e}')

    def pull_baseline_model(self):
        """Load the project's baseline model into the panel's widgets."""
        model = getattr(self.app, 'baseline_model', None)
        if model is None:
            return
        try:
            self.baseline_correct_var.set(model.enabled)
            self.baseline_method_var.set(model.method)
            if 'lam' in model.params:
                self.baseline_lam_var.set(float(model.params['lam']))
            if 'niter' in model.params:
                self.baseline_niter_var.set(int(model.params['niter']))
            self._on_baseline_method_change(push=False)
        except Exception as e:
            print(f'[baseline] panel not synced: {e}')

    def _on_baseline_method_change(self, _event=None, push=True):
        """Point the parameter entry at the selected method's variable."""
        method = self.baseline_method_var.get()
        label, var_name = self._BASELINE_PARAMS.get(method, ('', None))
        self._bl_param_lbl.config(text=label)
        if var_name is None:
            # Detaching the variable leaves the previous method's value on
            # screen; clear it so a greyed box never shows a number that is not
            # being used.
            self._bl_param_entry.config(state='normal', textvariable='')
            self._bl_param_entry.delete(0, tk.END)
            self._bl_param_entry.config(state='disabled')
        else:
            self._bl_param_entry.config(
                state='normal', textvariable=getattr(self, var_name))
        if push:
            self.push_baseline_model()

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

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_peaks(self):
        if self.app.all_intensities is None or self.app.ppm is None:
            messagebox.showinfo("Info", "Please load data first.")
            return
        if self.processor is None:
            messagebox.showerror("Error", "Signal processor not initialized.")
            return

        # Noise is now estimated here (no separate button). Use the optional
        # region if set; otherwise detect_peaks() estimates it internally from
        # the quietest segment (noise_std=None).
        # Parsed here, before anything runs: a malformed region has to stop
        # the detection with a message. Skipping it would leave the analyst
        # believing the water is excluded while it is being detected.
        try:
            _excluded = parse_regions(self.excluded_regions_var.get())
        except ValueError as exc:
            messagebox.showerror('Excluded regions', str(exc))
            return

        self.push_baseline_model()
        self._estimate_noise()

        # ── Loading overlay ───────────────────────────────────────────────────
        overlay = tk.Toplevel(self.app.root)
        overlay.title("Signal Detection")
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

        ttk.Label(outer, text="🔍  Signal Detection",
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
                try:
                    baseline_niter = int(self.baseline_niter_var.get())
                except (ValueError, TypeError):
                    baseline_niter = 40

                # Skip pipeline alignment if already done via Alignment dialog
                _run_align = not getattr(self.app, 'alignment_corrected', False)

                # One correction for the whole project: take the already
                # corrected matrix from the app and hand detect_peaks the
                # baselines that were removed, so it does not redo the work and
                # the audit trail still knows what was subtracted.
                _raw  = np.asarray(self.app.all_intensities, dtype=float)
                _corr = self.app.corrected_intensities()
                _corr = (np.asarray(_corr, dtype=float)
                         if _corr is not None else _raw)
                _bl2d = (_raw - _corr) if _corr is not _raw else None

                details = detect_peaks(
                    self.app.ppm, _corr,
                    noise_std=self.noise_std,
                    min_snr=self.min_snr_var.get(),
                    min_distance=self.min_width_var.get(),
                    baseline_correct=False,   # already applied above
                    baseline_2d=_bl2d,
                    baseline_method=self.baseline_method_var.get(),
                    baseline_lam=baseline_lam,
                    baseline_niter=baseline_niter,
                    global_align=_run_align,
                    samples_info=self.app.samples_info,
                    progress_callback=_push_progress,
                    return_details=True,
                    per_sample_detection=bool(self.per_sample_var.get()),
                    detect_shoulders=bool(self.detect_shoulders_var.get()),
                    shoulder_curvature_k=float(self.shoulder_k_var.get()),
                    min_group_presence=int(self.min_group_presence_var.get()),
                    excluded_regions=_excluded,
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
                                     f"Signal detection failed:\n{_result['error']}")
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
                    # Automatic signal quality: Good / Average / Poor, plus the
                    # score and the per-component breakdown that justifies it.
                    'signal_quality':        p.get('signal_quality', SIGNAL_AVERAGE),
                    'signal_quality_score':  p.get('signal_quality_score', float('nan')),
                    'signal_quality_detail': p.get('signal_quality_detail', {}),
                    'fwhm_pts':              p.get('fwhm_pts', float('nan')),
                    # Fraction of this window's raw area that was broad
                    # component (macromolecule envelope) removed by the
                    # baseline model — real NMR signal, so its removal is
                    # recorded rather than assumed.
                    'broad_removed_frac':    p.get('broad_removed_frac', 0.0),
                    'broad_removed_frac_per_sample':
                        p.get('broad_removed_frac_per_sample', np.array([])),
                    # Manual annotation status starts unset — the user marks
                    # Good/Average/Poor; it is NOT derived from the signal quality.
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
            # An exclusion and a detection failure call for opposite responses,
            # so the count says which happened rather than leaving the analyst
            # to infer it from a smaller number.
            self._excluded_regions = _excluded
            _n_excl = len(details.get('excluded_peaks') or ()) if isinstance(
                details, dict) else 0
            # Not named _lbl: that is the overlay's progress label, and
            # rebinding it here would make it a local of _poll, so the read at
            # the top of the polling loop would raise before this line ever ran.
            _count_txt = f"Found {len(self.peaks)} signals"
            if _n_excl:
                _count_txt += (f"  ({_n_excl} excluded: "
                               f"{format_regions(_excluded)} ppm)")
            self.peaks_count_label.config(text=_count_txt)

            # A splitting narrower than the linewidth cannot be resolved, and
            # apices that close are more likely two readings of one line. Said
            # out loud rather than corrected: which splittings a study can
            # resolve is the analyst's call.
            #
            # The linewidth is measured on the trace over the analytical
            # samples. Every width the detector records moves with the setting
            # being judged -- median width_ppm ran 4.40 to 7.81 Hz across this
            # sweep on one dataset and FWHM 1.70 to 2.77, while the trace held
            # at 3.22 -- a threshold built on either of the first two would
            # advise a value that moves the measurement again.
            _all_lw = np.asarray(self.app.all_intensities, dtype=float)
            _lw_incl = included_sample_indices(
                self.app.samples_info,
                _all_lw.shape[1] if _all_lw.ndim == 2 else 1)
            _corr_for_lw = (_all_lw[:, _lw_incl]
                            if _lw_incl and _all_lw.ndim == 2 else _all_lw)
            _split_warn = unresolvable_split_warning(
                measure_linewidth_ppm(self.app.ppm, _corr_for_lw, self.peaks),
                self.min_width_var.get(),
                float(getattr(self.annotation_stage, 'spectrometer_var',
                              None).get())
                if getattr(self, 'annotation_stage', None) is not None
                and getattr(self.annotation_stage, 'spectrometer_var', None)
                is not None else 600.0)
            if _split_warn:
                print(f'[Detection] {_split_warn}')
                self._status(_split_warn, 'orange')
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
            self._status("Drawing signal areas…")
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
                # Correlated over the analytical samples only. A blank drags
                # every peak to nothing at once, and that shared descent reads
                # as co-variation between molecules that have nothing to do
                # with each other.
                _incl_corr = included_sample_indices(self.app.samples_info,
                                                     _n_s)
                self._peak_corr = (
                    compute_peak_correlation_matrix(self.peaks,
                                                    included=_incl_corr)
                    if len(_incl_corr) >= 3 else {})
            except Exception:
                self._peak_corr = {}

            _destroy_overlay()
            messagebox.showinfo(
                "Success",
                f"Detected {len(self.peaks)} signals")

        overlay.after(100, _poll)

    def toggle_peaks(self):
        if self.peaks:
            self.color_peaks(force=False)

    def toggle_baseline(self):
        """Toggle display of estimated baseline on spectrum."""
        if not hasattr(self, '_last_baseline') or self._last_baseline is None:
            messagebox.showinfo("Info", "Run signal detection first to estimate baseline.")
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