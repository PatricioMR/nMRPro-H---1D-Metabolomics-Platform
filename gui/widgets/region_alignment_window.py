# gui/widgets/region_alignment_window.py
"""
Regional Alignment Window.

Opened by right-clicking the peak visualizer in PeakListWindow.
Lets the user align samples within a specific ppm region using
icoshift (segment-wise cross-correlation) or a single global shift.

Features:
  - Detects peaks in the region and marks them as alignment anchors
  - Preview: shows before (dashed) vs after (solid) overlay
  - Apply: patches app.all_intensities for the region only. The spectrum before
    any processing is kept (Revert), the shifts applied are recorded in
    ``app.region_alignments`` (saved with the project, listed in the report),
    and the vacated edges of a shifted piece hold the edge value rather than
    stepping to zero.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class RegionAlignmentWindow:

    def __init__(self, parent, ppm, all_intensities, peak, app, on_apply=None):
        """
        Parameters
        ----------
        parent          : tk parent widget
        ppm             : 1-D ndarray (n_pts,)
        all_intensities : 2-D ndarray (n_pts, n_samples)
        peak            : peak dict with start_ppm / end_ppm / ppm / rank
        app             : main App instance
        on_apply        : optional callback(aligned_array, updated_peak) called on Apply
        """
        self.ppm      = ppm
        self.all_int  = all_intensities.copy().astype(float)
        self.peak     = peak
        self.app      = app
        self._on_apply_cb = on_apply  # callback → PeakListWindow

        # Context window: 2× peak width on each side
        width = abs(peak.get('end_ppm', peak['ppm']) - peak.get('start_ppm', peak['ppm']))
        ctx   = max(0.04, width * 2.0)
        self._default_lo = peak.get('start_ppm', peak['ppm']) - ctx
        self._default_hi = peak.get('end_ppm',   peak['ppm']) + ctx

        self._detected_peaks = []   # list of ppm floats
        self._state = {'aligned': None}

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"Regional Alignment — Signal #{peak['rank']}  "
            f"{peak['ppm']:.4f} ppm")
        self.win.geometry('960x540')
        self._build_ui()
        self._draw_original()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=0)
        self._build_controls(left)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=1)
        self._build_plot(right)

    def _build_controls(self, parent):
        ttk.Label(parent, text='Regional Alignment',
                  font=('Arial', 11, 'bold')).pack(pady=(0, 6))

        # Region bounds
        rf = ttk.LabelFrame(parent, text='Region (ppm)', padding=4)
        rf.pack(fill=tk.X, pady=3)
        for lbl, attr, default in [
            ('From:', '_lo_var', f'{self._default_lo:.4f}'),
            ('To:',   '_hi_var', f'{self._default_hi:.4f}'),
        ]:
            fr = ttk.Frame(rf); fr.pack(fill=tk.X, pady=1)
            ttk.Label(fr, text=lbl, width=6).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(fr, textvariable=var, width=9).pack(side=tk.LEFT, padx=2)

        # Method
        mf = ttk.LabelFrame(parent, text='Method', padding=4)
        mf.pack(fill=tk.X, pady=3)
        self._mode_var = tk.StringVar(value='icoshift')
        ttk.Radiobutton(mf, text='icoshift  (segment-wise)',
                        variable=self._mode_var, value='icoshift').pack(anchor=tk.W)
        ttk.Radiobutton(mf, text='Single shift  (global cross-corr)',
                        variable=self._mode_var, value='single').pack(anchor=tk.W)

        # icoshift params
        ic = ttk.LabelFrame(parent, text='icoshift Parameters', padding=4)
        ic.pack(fill=tk.X, pady=3)
        for lbl, attr, default in [
            ('Max shift ± ppm:', '_max_shift_var', '0.010'),
            ('Segments:',        '_n_seg_var',     '3'),
        ]:
            fr = ttk.Frame(ic); fr.pack(fill=tk.X, pady=1)
            ttk.Label(fr, text=lbl, width=16).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(fr, textvariable=var, width=7).pack(side=tk.LEFT)

        # Peak detection
        df = ttk.LabelFrame(parent, text='Signal Detection', padding=4)
        df.pack(fill=tk.X, pady=3)
        ttk.Button(df, text='🔍 Detect signals in region',
                   command=self._detect_peaks).pack(fill=tk.X)
        self._detect_lbl = ttk.Label(df, text='—', foreground='gray',
                                     font=('Arial', 8), wraplength=200)
        self._detect_lbl.pack(anchor=tk.W, pady=2)

        # Status
        self._status_var = tk.StringVar(value='')
        ttk.Label(parent, textvariable=self._status_var,
                  foreground='blue', font=('Arial', 8),
                  wraplength=210).pack(anchor=tk.W, pady=4)

        # Action buttons
        bf = ttk.Frame(parent); bf.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))
        ttk.Button(bf, text='👁  Preview',  command=self._preview).pack(fill=tk.X, pady=2)
        ttk.Button(bf, text='✓  Apply',     command=self._apply).pack(fill=tk.X, pady=2)
        ttk.Button(bf, text='Cancel',       command=self.win.destroy).pack(fill=tk.X, pady=2)

    def _build_plot(self, parent):
        self._fig    = Figure(figsize=(6, 4), dpi=95, facecolor='#f8f8f8')
        self._ax     = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        tb.update()
        tb.pack(fill=tk.X)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _region_mask(self):
        try:
            lo = float(self._lo_var.get())
            hi = float(self._hi_var.get())
        except ValueError:
            lo, hi = self._default_lo, self._default_hi
        lo, hi = min(lo, hi), max(lo, hi)
        return (self.ppm >= lo) & (self.ppm <= hi)

    def _sample_colors(self, n):
        colors = ['#888888'] * n
        if self.app and hasattr(self.app, 'samples_info'):
            for i, si in enumerate(self.app.samples_info[:n]):
                colors[i] = si.get('color', '#888888')
        return colors

    def _draw_spectra(self, ppm_r, int_r, title, orig_r=None):
        """Draw region spectra.  If orig_r supplied, dashes = original, solid = aligned."""
        ax = self._ax
        ax.clear()
        n  = int_r.shape[1]
        colors = self._sample_colors(n)

        for i in range(min(n, 30)):
            if orig_r is not None:
                ax.plot(ppm_r, orig_r[:, i],  color=colors[i], lw=0.6, alpha=0.25, ls='--')
            ax.plot(ppm_r, int_r[:, i], color=colors[i], lw=0.9, alpha=0.75)

        # Peak boundary markers
        ax.axvline(self.peak.get('start_ppm', self.peak['ppm']),
                   color='#2e7d32', lw=1.0, ls='--', alpha=0.7, label='boundary')
        ax.axvline(self.peak.get('end_ppm', self.peak['ppm']),
                   color='#2e7d32', lw=1.0, ls='--', alpha=0.7)
        ax.axvline(self.peak['ppm'],
                   color='#c62828', lw=1.0, ls=':', alpha=0.6, label='center')

        # Detected peak markers
        for pp in self._detected_peaks:
            ax.axvline(pp, color='#e65100', lw=1.5, alpha=0.70)

        ax.invert_xaxis()
        ax.set_xlabel('ppm', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.set_title(title, fontsize=8)
        if orig_r is not None:
            ax.text(0.02, 0.97, 'dashed = original   solid = aligned',
                    transform=ax.transAxes, fontsize=7,
                    va='top', color='#555555')
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _draw_original(self):
        mask  = self._region_mask()
        ppm_r = self.ppm[mask]
        int_r = self.all_int[mask, :]
        self._draw_spectra(ppm_r, int_r,
                           f'Peak #{self.peak["rank"]}  {self.peak["ppm"]:.4f} ppm — original')

    # ── Peak detection ────────────────────────────────────────────────────────

    def _detect_peaks(self):
        try:
            from scipy.signal import find_peaks
        except ImportError:
            messagebox.showerror('Error', 'scipy is required for signal detection.')
            return

        mask     = self._region_mask()
        ppm_r    = self.ppm[mask]
        mean_int = self.all_int[mask, :].mean(axis=1)

        noise = float(np.median(np.abs(mean_int - np.median(mean_int)))) * 1.4826 + 1e-12
        idx, _ = find_peaks(mean_int, height=noise * 3.0, distance=3)

        self._detected_peaks = [float(ppm_r[i]) for i in idx]

        if self._detected_peaks:
            shown = ', '.join(f'{p:.4f}' for p in self._detected_peaks[:6])
            if len(self._detected_peaks) > 6:
                shown += f'… (+{len(self._detected_peaks)-6})'
            self._detect_lbl.config(
                text=f'{len(self._detected_peaks)} signal(s):\n{shown}',
                foreground='black')
        else:
            self._detect_lbl.config(text='No signals found.', foreground='orange')

        self._draw_original()

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _preview(self):
        mask  = self._region_mask()
        ppm_r = self.ppm[mask]
        int_r = self.all_int[mask, :].copy()

        try:
            mode = self._mode_var.get()
            if mode == 'icoshift':
                from processing.peak_alignment import icoshift_align
                n_seg  = max(1, int(self._n_seg_var.get()))
                max_sh = float(self._max_shift_var.get())
                aligned_r, shifts, _ = icoshift_align(
                    ppm_r, int_r,
                    n_segments=n_seg,
                    max_shift_ppm=max_sh)
                n_moved = int(np.sum(np.any(shifts != 0, axis=1)))
                self._state['shifts_pts'] = shifts.astype(int).tolist()
                self._status_var.set(
                    f'icoshift: {n_moved}/{shifts.shape[0]} segment(s) shifted.')
            else:
                # Single global cross-correlation shift per sample
                from processing.peak_alignment import _shift_1d
                ref       = int_r.mean(axis=1)
                aligned_r = int_r.copy()
                ppm_span  = abs(float(ppm_r[-1]) - float(ppm_r[0])) + 1e-9
                max_pts   = max(1, int(float(self._max_shift_var.get()) /
                                       ppm_span * len(ppm_r)))
                shifts_applied = []
                for i in range(int_r.shape[1]):
                    corr = np.correlate(ref, int_r[:, i], mode='full')
                    lag0 = len(int_r[:, i]) - 1
                    lo_s = max(0, lag0 - max_pts)
                    hi_s = min(len(corr), lag0 + max_pts + 1)
                    best = int(np.argmax(corr[lo_s:hi_s])) + lo_s - lag0
                    if best != 0:
                        aligned_r[:, i] = _shift_1d(int_r[:, i], best, edge=True)
                    shifts_applied.append(best)
                self._state['shifts_pts'] = [int(v) for v in shifts_applied]
                ppm_per_pt = ppm_span / max(1, len(ppm_r) - 1)
                max_sh_ppm = max(abs(s) * ppm_per_pt for s in shifts_applied)
                self._status_var.set(
                    f'Single shift: max={max_sh_ppm:.4f} ppm applied.')

            # Patch the full-spectrum array
            aligned_full = self.all_int.copy()
            aligned_full[mask, :] = aligned_r
            self._state['aligned'] = aligned_full
            self._state['record'] = {
                'region_ppm': (float(ppm_r.min()), float(ppm_r.max())),
                'mode': mode,
                'n_segments': (int(self._n_seg_var.get())
                               if mode == 'icoshift' else None),
                'max_shift_ppm': float(self._max_shift_var.get()),
                'signal_rank': self.peak.get('rank'),
                'signal_ppm': float(self.peak.get('ppm')),
                'shifts_pts': self._state.get('shifts_pts'),
                'ppm_per_point': float(abs(self.ppm[1] - self.ppm[0])),
            }

            self._draw_spectra(ppm_r, aligned_r,
                               'Preview — dashed: original  |  solid: aligned',
                               orig_r=int_r)

        except Exception as e:
            self._status_var.set(f'Error: {e}')
            import traceback; traceback.print_exc()

    def _apply(self):
        if self._state['aligned'] is None:
            messagebox.showinfo('Info', 'Run Preview first.')
            return

        aligned = self._state['aligned']

        # ── Update peak dict to reflect new peak position ─────────────────────
        # Re-locate the peak maximum in the aligned mean spectrum within the
        # original boundary window, and update ppm / start_ppm / end_ppm.
        mask  = self._region_mask()
        ppm_r = self.ppm[mask]
        mean_r = aligned[mask, :].mean(axis=1)
        # Restrict search to original peak window
        pk_lo = self.peak.get('start_ppm', self.peak['ppm'])
        pk_hi = self.peak.get('end_ppm',   self.peak['ppm'])
        inner = (ppm_r >= min(pk_lo, pk_hi)) & (ppm_r <= max(pk_lo, pk_hi))
        if inner.sum() > 0:
            new_center = float(ppm_r[inner][np.argmax(mean_r[inner])])
            shift = new_center - self.peak['ppm']
            self.peak['ppm']       = new_center
            self.peak['start_ppm'] = self.peak.get('start_ppm', new_center) + shift
            self.peak['end_ppm']   = self.peak.get('end_ppm',   new_center) + shift
            self.peak['aligned_region'] = True   # flag for annotation re-search

        # ── Persist: update app (so future windows use aligned data) ──────────
        # What is about to change is kept first, so Revert can put it back:
        # the spectrum as it was before any processing (once), and a record
        # of what was shifted where, which the project saves.
        import datetime
        vis = (getattr(self.app, 'stages', {}) or {}).get('visualization')
        if vis is not None and getattr(vis, '_original_intensities', None) is None:
            vis._original_intensities = np.array(self.app.all_intensities,
                                                 dtype=float, copy=True)
        record = dict(self._state.get('record') or {})
        record['applied'] = datetime.datetime.now().isoformat(timespec='seconds')
        if not hasattr(self.app, 'region_alignments'):
            self.app.region_alignments = []
        self.app.region_alignments.append(record)
        self.app.all_intensities = aligned
        try:
            self.app.invalidate_baseline_cache()
        except Exception:
            pass
        if hasattr(self.app, 'update_plot'):
            self.app.update_plot()
        try:
            vis.btn_revert.config(state='normal')
        except Exception:
            pass
        messagebox.showinfo(
            'Region aligned',
            f"Samples were shifted between {record['region_ppm'][0]:.4f} and "
            f"{record['region_ppm'][1]:.4f} ppm.\n\nThe signals and the "
            "quantification were measured on the spectrum as it was: detect "
            "the signals again to measure on this one. Revert in the "
            "visualization stage undoes it; the report lists it.",
            parent=self.win)

        # ── Notify PeakListWindow via callback (updates its own array + redraw)
        if self._on_apply_cb is not None:
            try:
                self._on_apply_cb(aligned, self.peak)
            except Exception as e:
                print(f'RegionAlignmentWindow callback error: {e}')

        self.win.destroy()
