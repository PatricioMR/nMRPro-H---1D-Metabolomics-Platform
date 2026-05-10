# gui/stages/notation_stage.py
"""
Metabolite annotation stage.
Step 1 → noise estimation
Step 2 → peak detection & deconvolution
Step 3 → database search
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import sys
import time
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from processing.noise_estimation import estimate_noise_from_region
from data.metabolite_db import MetaboliteDatabase
from .base_stage import BaseStage


class NotationStage(BaseStage):

    def __init__(self, parent, app):
        super().__init__(parent, app)

        self.step1_completed = False
        self.step2_completed = False

        self.noise_std    = None
        self.noise_source = None
        self.annotations  = {}
        self.peaks        = []

        self.default_noise_region = (12.0, 10.0)
        self.plot_manager = None

        try:
            from gui.notation.peak_processing import PeakProcessor
            self.processor = PeakProcessor(app)
            print("✅ PeakProcessor initialized successfully")
        except ImportError as e:
            print(f"❌ Failed to import PeakProcessor: {e}")
            self.processor = None
        except Exception as e:
            print(f"❌ Failed to initialize PeakProcessor: {e}")
            self.processor = None

        try:
            from controllers.bmrb_controller import BMRBController
            self.bmrb = BMRBController()
            print("✅ BMRBController initialized successfully")
        except ImportError as e:
            print(f"❌ Failed to import BMRBController: {e}")
            self.bmrb = None

        self.create_widgets()

    # ══════════════════════════════════════════════════════════════════════════
    #  STATUS BAR HELPER
    # ══════════════════════════════════════════════════════════════════════════

    def _status(self, msg: str, color: str = "blue"):
        """Update the app-wide status bar at the bottom of the window."""
        try:
            self.app.status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    def _status_ok(self, msg: str):
        self._status(msg, color="green")

    def _status_error(self, msg: str):
        self._status(msg, color="red")

    def _status_ready(self):
        self._status("Ready", color="gray")

    # ══════════════════════════════════════════════════════════════════════════
    #  WIDGETS
    # ══════════════════════════════════════════════════════════════════════════

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text="Metabolite Annotation",
                  font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Separator(self.frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)

        # ── STEP 1 ────────────────────────────────────────────────────────────
        step1_frame = ttk.LabelFrame(self.frame, text="📏 STEP 1: Noise Estimation", padding=5)
        step1_frame.pack(fill=tk.X, padx=5, pady=5)

        region_frame = ttk.Frame(step1_frame)
        region_frame.pack(fill=tk.X, pady=2)
        ttk.Label(region_frame, text="Noise Region (ppm):").pack(side=tk.LEFT, padx=2)
        self.noise_start_var = tk.StringVar(value=str(self.default_noise_region[0]))
        self.noise_end_var   = tk.StringVar(value=str(self.default_noise_region[1]))
        ttk.Entry(region_frame, textvariable=self.noise_start_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(region_frame, text="to").pack(side=tk.LEFT)
        ttk.Entry(region_frame, textvariable=self.noise_end_var, width=8).pack(side=tk.LEFT, padx=2)

        self.btn_calculate_noise = ttk.Button(step1_frame, text="🧮 Calculate Noise",
                                              command=self.calculate_noise)
        self.btn_calculate_noise.pack(fill=tk.X, pady=5)

        self.noise_value_label = ttk.Label(step1_frame, text="Noise Std Dev: --")
        self.noise_value_label.pack(anchor=tk.W, pady=2)
        self.step1_status = ttk.Label(step1_frame, text="⏳ Not started", foreground="gray")
        self.step1_status.pack(anchor=tk.W, pady=2)

        # ── STEP 2 ────────────────────────────────────────────────────────────
        step2_frame = ttk.LabelFrame(self.frame,
                                     text="🔍 STEP 2: Peak Detection & Deconvolution", padding=5)
        step2_frame.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: Min SNR
        param_frame = ttk.Frame(step2_frame)
        param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(param_frame, text="Min SNR:").pack(side=tk.LEFT, padx=2)
        self.min_snr_var = tk.DoubleVar(value=3.0)
        ttk.Entry(param_frame, textvariable=self.min_snr_var, width=6).pack(side=tk.LEFT, padx=2)

        # Row 2: Min Width + Prominence
        param_frame2 = ttk.Frame(step2_frame)
        param_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(param_frame2, text="Min Width (ppm):").pack(side=tk.LEFT, padx=2)
        self.min_width_var = tk.DoubleVar(value=0.003)
        ttk.Entry(param_frame2, textvariable=self.min_width_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(param_frame2, text="Prominence:").pack(side=tk.LEFT, padx=(8, 2))
        self.prominence_var = tk.DoubleVar(value=1.5)
        ttk.Entry(param_frame2, textvariable=self.prominence_var, width=5).pack(side=tk.LEFT, padx=2)

        self.btn_detect_peaks = ttk.Button(step2_frame, text="🎯 Detect Peaks",
                                           command=self.detect_peaks, state="disabled")
        self.btn_detect_peaks.pack(fill=tk.X, pady=5)

        self.peaks_count_label = ttk.Label(step2_frame, text="Peaks found: 0")
        self.peaks_count_label.pack(anchor=tk.W, pady=2)
        self.step2_status = ttk.Label(step2_frame,
                                      text="⏳ Complete Step 1 first", foreground="gray")
        self.step2_status.pack(anchor=tk.W, pady=2)

        self.btn_toggle_peaks = ttk.Button(step2_frame, text="👁 Show/Hide Peak Areas",
                                           command=self.toggle_peaks, state="disabled")
        self.btn_toggle_peaks.pack(fill=tk.X, pady=2)

        ttk.Separator(step2_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)

        deconv_header = ttk.Frame(step2_frame)
        deconv_header.pack(fill=tk.X, pady=2)
        ttk.Label(deconv_header, text="Deconvolution:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.btn_find_overlaps = ttk.Button(deconv_header, text="🔍 Find Overlaps",
                                            command=self.find_overlaps, state="disabled")
        self.btn_find_overlaps.pack(side=tk.RIGHT, padx=2)

        listbox_frame = ttk.Frame(step2_frame)
        listbox_frame.pack(fill=tk.X, pady=2)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.overlap_listbox = tk.Listbox(listbox_frame, height=3,
                                          yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.overlap_listbox.yview)
        self.overlap_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        deconv_param_frame = ttk.Frame(step2_frame)
        deconv_param_frame.pack(fill=tk.X, pady=2)
        ttk.Label(deconv_param_frame, text="Min SNR:").pack(side=tk.LEFT, padx=2)
        self.deconv_snr_var = tk.DoubleVar(value=3.0)
        ttk.Entry(deconv_param_frame, textvariable=self.deconv_snr_var,
                  width=5).pack(side=tk.LEFT, padx=2)

        self.btn_run_deconv = ttk.Button(step2_frame, text="🧅 Deconvolve Selected",
                                         command=self.deconvolve_selected, state="disabled")
        self.btn_run_deconv.pack(fill=tk.X, pady=2)
        self.deconv_status = ttk.Label(step2_frame, text="Run peak detection first",
                                       foreground="gray", font=("Arial", 8))
        self.deconv_status.pack(anchor=tk.W, pady=2)

        # ── STEP 3 ────────────────────────────────────────────────────────────
        step3_frame = ttk.LabelFrame(self.frame, text="📡 STEP 3: Database Search", padding=5)
        step3_frame.pack(fill=tk.X, padx=5, pady=5)

        tol_frame = ttk.Frame(step3_frame)
        tol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tol_frame, text="Tolerance (ppm):").pack(side=tk.LEFT, padx=2)
        self.tolerance_var = tk.DoubleVar(value=0.05)
        ttk.Entry(tol_frame, textvariable=self.tolerance_var, width=8).pack(side=tk.LEFT, padx=2)

        mode_frame = ttk.Frame(step3_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mode_frame, text="Internal library:").pack(side=tk.LEFT, padx=2)
        self.internal_library_mode_var = tk.StringVar(value='Metabolomics (default)')
        self.internal_library_mode = ttk.Combobox(
            mode_frame,
            textvariable=self.internal_library_mode_var,
            state='readonly',
            width=24,
            values=['Metabolomics (default)', 'Lipidomics'])
        self.internal_library_mode.pack(side=tk.LEFT, padx=2)

        btn_frame = ttk.Frame(step3_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_search_internal = ttk.Button(btn_frame, text="🔍 Search Internal DB",
                                              command=self.search_internal_db, state="disabled")
        self.btn_search_internal.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_search_hmdb_online = ttk.Button(btn_frame, text="📚 Use Internal DB",
                                                 command=self.search_builtin_library, state="disabled")
        self.btn_search_hmdb_online.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.internal_status = ttk.Label(step3_frame, text="Internal DB: ready",
                                         foreground="gray")
        self.internal_status.pack(anchor=tk.W, pady=2)
        self.hmdb_status = ttk.Label(step3_frame, text="Internal DB: ready", foreground="gray")
        self.hmdb_status.pack(anchor=tk.W, pady=2)

        # ── Mouse Controls ────────────────────────────────────────────────────
        mouse_frame = ttk.LabelFrame(self.frame, text="🐭 Mouse Controls", padding=5)
        mouse_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mouse_frame,
                  text="• Scroll up/down: Zoom in/out\n"
                       "• Click and drag: Pan (move view)\n"
                       "• Press R: Reset view",
                  font=("Arial", 8), foreground="blue",
                  justify=tk.LEFT).pack(anchor=tk.W, padx=5)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — NOISE ESTIMATION
    # ══════════════════════════════════════════════════════════════════════════

    def calculate_noise(self):
        if self.app.ppm is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        try:
            region_start = float(self.noise_start_var.get())
            region_end   = float(self.noise_end_var.get())
            if region_start <= region_end:
                messagebox.showerror(
                    "Error", "For NMR, the left value must be greater than the right value.")
                return

            self._status("Estimating noise...")
            t0 = time.monotonic()

            self.noise_std    = estimate_noise_from_region(
                self.app.ppm, self.app.all_intensities, region_start, region_end)
            self.noise_source = 'ppm'

            elapsed = time.monotonic() - t0
            self.noise_value_label.config(text=f"Noise Std Dev: {self.noise_std:.4f}")
            self.step1_completed = True
            self.step1_status.config(text="✅ Completed", foreground="green")
            self.btn_detect_peaks.config(state="normal")
            self.step2_status.config(text="⏳ Ready to detect peaks", foreground="blue")
            self._status_ok(f"Noise estimated: σ = {self.noise_std:.4f}  ({elapsed:.2f}s)")
            messagebox.showinfo("Success", f"Noise estimated: σ = {self.noise_std:.4f}")
        except ValueError as e:
            self._status_error(f"Noise estimation failed: {e}")
            messagebox.showerror("Error", f"Invalid region values: {e}")
        except Exception as e:
            self._status_error(f"Noise estimation failed: {e}")
            messagebox.showerror("Error", f"Could not estimate noise: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — PEAK DETECTION & DECONVOLUTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect_peaks(self):
        if not self.step1_completed or self.noise_std is None:
            messagebox.showinfo("Info", "Please estimate noise first.")
            return
        if self.processor is None:
            messagebox.showerror("Error", "Peak processor not initialized.")
            return
        try:
            self._status("Detecting peaks...")
            t0 = time.monotonic()

            self.processor.set_noise(self.noise_std)
            processor_peaks = self.processor.detect_peaks(
                self.app.ppm, self.app.all_intensities,
                min_snr=self.min_snr_var.get(),
                min_width_ppm=self.min_width_var.get(),
                prominence_factor=self.prominence_var.get(),
            )
            self.peaks = processor_peaks.copy()

            self.peaks_count_label.config(text=f"Found {len(self.peaks)} peaks")
            self.btn_find_overlaps.config(state="normal")
            self.btn_toggle_peaks.config(state="normal")
            self.deconv_status.config(
                text="Click 'Find Overlaps' to identify regions for deconvolution",
                foreground="blue")
            self.btn_search_internal.config(state="normal")
            self.btn_search_hmdb_online.config(state="normal")

            self._status("Drawing peak areas...")
            self.color_peaks(force=True)

            elapsed = time.monotonic() - t0
            self._status_ok(f"Found {len(self.peaks)} peaks  ({elapsed:.2f}s)")
            messagebox.showinfo("Success", f"Detected {len(self.peaks)} peaks")
        except Exception as e:
            self._status_error(f"Peak detection failed: {e}")
            messagebox.showerror("Error", f"Peak detection failed: {e}")
            import traceback; traceback.print_exc()

    def toggle_peaks(self):
        if not self.peaks:
            return
        self.color_peaks(force=False)

    def color_peaks(self, force: bool = False):
        if not self.peaks or self.app.all_intensities is None:
            return
        mean_intensity = np.mean(self.app.all_intensities, axis=1)
        if self.plot_manager is not None:
            if force:
                self.plot_manager.force_redraw_peaks(
                    self.peaks, mean_intensity, self.app.ppm)
            else:
                self.plot_manager.draw_peak_overlay(
                    self.peaks, mean_intensity, self.app.ppm)
        else:
            self._draw_peaks_fallback(mean_intensity)

    def _draw_peaks_fallback(self, mean_intensity):
        import matplotlib.pyplot as _plt
        for coll in list(self.app.ax.collections):
            if getattr(coll, 'get_label', lambda: '')() == '_peak_fills':
                coll.remove()
        for txt in list(self.app.ax.texts):
            if getattr(txt, 'get_gid', lambda: '')() == '_peak_label':
                txt.remove()
        base_colors = _plt.cm.tab20(np.linspace(0, 1, 20))
        for i, peak in enumerate(self.peaks):
            if 'start_idx' in peak and 'end_idx' in peak:
                l, r = int(peak['start_idx']), int(peak['end_idx'])
            else:
                l = int(np.argmin(np.abs(self.app.ppm - peak['start_ppm'])))
                r = int(np.argmin(np.abs(self.app.ppm - peak['end_ppm'])))
            if l > r:
                l, r = r, l
            self.app.ax.fill_between(
                self.app.ppm[l:r + 1], 0, mean_intensity[l:r + 1],
                alpha=0.25, color=base_colors[i % 20], label='_peak_fills')
        for p in sorted(self.peaks, key=lambda x: x['height'], reverse=True)[:20]:
            idx = p.get('index')
            h = (float(mean_intensity[idx]) if idx is not None
                 else float(np.interp(p['ppm'], self.app.ppm[::-1], mean_intensity[::-1])))
            self.app.ax.text(p['ppm'], h * 1.03, str(p['rank']),
                              ha='center', va='bottom', fontsize=7,
                              color='#222', fontweight='bold',
                              gid='_peak_label', clip_on=True)
        self.app.canvas.draw_idle()

    def find_overlaps(self):
        if not self.peaks:
            messagebox.showinfo("Info", "Please detect peaks first.")
            return

        self._status("Searching for overlapping regions (consensus)...")
        t0 = time.monotonic()

        # AlpsNMR-style consensus: each non-BLK spectrum votes independently.
        # A region is reported only when >= 50% of spectra agree.
        overlaps = self.processor.find_overlapping_regions_consensus(
            all_intensities=self.app.all_intensities,
            samples_info=getattr(self.app, 'samples_info', None),
            distance_threshold=0.02,
            consensus_threshold=0.50,
        )

        elapsed = time.monotonic() - t0

        if not overlaps:
            self._status_ok(f"No overlapping regions found  ({elapsed:.2f}s)")
            messagebox.showinfo("Info",
                                "No overlapping regions found.\n"
                                "All peaks are well separated. Proceed to Step 3.")
            return

        self._status_ok(f"Found {len(overlaps)} overlapping regions  ({elapsed:.2f}s)")
        if messagebox.askyesno(
            "Deconvolution Required",
            f"Found {len(overlaps)} regions with overlapping peaks.\n\n"
            "Open deconvolution tool now?",
            icon='warning'
        ):
            # Singleton: raise existing window if already open
            _dw = getattr(self.app, '_deconv_win', None)
            if _dw is not None:
                try:
                    if _dw.window.winfo_exists():
                        _dw.window.deiconify(); _dw.window.lift(); _dw.window.focus_force()
                        return
                except Exception:
                    pass
            from gui.notation.deconvolution import DeconvController, DeconvWindow
            controller = DeconvController(self.processor)
            controller.set_regions(overlaps)
            dw = DeconvWindow(self.app.root, controller, self)
            self.app._deconv_win = dw
            try:
                dw.window.protocol('WM_DELETE_WINDOW',
                                   lambda: (dw.window.destroy(),
                                            setattr(self.app, '_deconv_win', None)))
            except Exception:
                pass

    def deconvolve_selected(self):
        selection = self.overlap_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a region to deconvolve.")
            return
        if not self.processor:
            messagebox.showerror("Error", "Peak processor not initialized.")
            return
        if not self.peaks:
            messagebox.showerror("Error",
                                 "No peaks detected. Please run peak detection first.")
            return

        region_text = self.overlap_listbox.get(selection[0])
        region_start, region_end = self.processor.parse_region_from_text(region_text)
        if region_start is None:
            messagebox.showerror("Error", f"Could not parse region: {region_text}")
            return

        self._status(f"Deconvolving region {region_start:.3f}–{region_end:.3f} ppm...")
        self.deconv_status.config(text="Running deconvolution...", foreground="blue")
        self.app.root.update()
        t0 = time.monotonic()

        try:
            results = self.processor.deconvolve_region(
                region_start, region_end,
                min_snr=self.deconv_snr_var.get()
            )
            elapsed = time.monotonic() - t0

            if results and results.get('peaks'):
                self.peaks = self.processor.peaks.copy()
                self.color_peaks(force=True)
                self.overlap_listbox.delete(selection[0])
                n = len(results['peaks'])
                bic = results.get('bic', 0)
                self.deconv_status.config(
                    text=f"✓ {n} peaks found (BIC: {bic:.1f})",
                    foreground="green")
                self._status_ok(
                    f"Deconvolution complete: {n} peaks found  ({elapsed:.2f}s)")
                self.show_deconv_results(results)
            else:
                self.deconv_status.config(text="No peaks found in region",
                                          foreground="orange")
                self._status("No peaks found in selected region", color="orange")
                messagebox.showinfo("Deconvolution",
                                    "No peaks were found in the selected region.")
        except Exception as e:
            self._status_error(f"Deconvolution failed: {e}")
            self.deconv_status.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Deconvolution failed:\n{str(e)}")
            import traceback; traceback.print_exc()

    def show_deconv_results(self, results):
        if not results or not results.get('peaks'):
            messagebox.showinfo("Deconvolution",
                                "No peaks were found in the selected region.")
            return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        win = tk.Toplevel(self.app.root)
        win.title(f"Deconvolution Results - {len(results['peaks'])} Peaks Found")
        win.geometry("900x700")
        win.transient(self.app.root)

        main_frame = ttk.Frame(win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        ax1.plot(results['region_ppm'], results['region_intensity'],
                 'b-', linewidth=1.5, label='Original Spectrum', alpha=0.8)
        for i, peak in enumerate(results['peaks']):
            ax1.axvline(x=peak['ppm'], color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(peak['ppm'], peak['height'], f' {i + 1}', fontsize=9,
                     fontweight='bold', color='darkred')
        ax1.set_xlabel('δ (ppm)', fontsize=10)
        ax1.set_ylabel('Intensity', fontsize=10)
        ax1.invert_xaxis()
        ax1.set_title('Original Spectrum with Detected Peak Positions',
                      fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right')

        ax2.plot(results['region_ppm'], results['region_intensity'],
                 'b-', linewidth=1.5, alpha=0.4, label='Original Spectrum')
        x_dense    = np.linspace(results['region_ppm'][0], results['region_ppm'][-1], 500)
        colors     = plt.cm.tab10(np.linspace(0, 1, len(results['peaks'])))
        cumulative = np.zeros_like(x_dense)
        for i, peak in enumerate(results['peaks']):
            y_peak = gaussian(x_dense, peak['height'], peak['ppm'], peak['width'])
            cumulative += y_peak
            ax2.plot(x_dense, y_peak, '--', color=colors[i], linewidth=2,
                     label=f"Peak {i + 1}: {peak['ppm']:.4f} ppm")
            ax2.fill_between(x_dense, 0, y_peak, alpha=0.2, color=colors[i])
        ax2.plot(x_dense, cumulative, 'r-', linewidth=2,
                 label='Cumulative Fit', alpha=0.9)
        ax2.set_xlabel('δ (ppm)', fontsize=10)
        ax2.set_ylabel('Intensity', fontsize=10)
        ax2.invert_xaxis()
        ax2.set_title(f"Deconvolution Result - BIC: {results['bic']:.1f}",
                      fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Export to CSV",
                   command=lambda: self.export_deconv_results(results)
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close",
                   command=win.destroy).pack(side=tk.RIGHT, padx=5)

        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry(f'{w}x{h}+{(win.winfo_screenwidth()//2)-(w//2)}'
                     f'+{(win.winfo_screenheight()//2)-(h//2)}')

    def export_deconv_results(self, results):
        from tkinter import filedialog
        import csv
        filename = filedialog.asksaveasfilename(
            title="Export Deconvolution Results", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filename:
            return
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Peak #', 'PPM', 'Height', 'Width', 'SNR', 'BIC'])
                for i, p in enumerate(results['peaks'], 1):
                    writer.writerow([i, f"{p.get('ppm', 0):.4f}",
                                     f"{p.get('height', 0):.2f}",
                                     f"{p.get('width', 0):.4f}",
                                     f"{p.get('snr', 0):.1f}",
                                     f"{p.get('bic', 0):.1f}"])
            messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — DATABASE SEARCH
    # ══════════════════════════════════════════════════════════════════════════

    def search_internal_db(self):
        if not self.peaks:
            messagebox.showinfo("Info",
                                "No peaks detected. Please run peak detection first.")
            return

        from tkinter import filedialog
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _db_dir = os.path.join(_proj_root, 'databases')
        _init_dir = _db_dir if os.path.isdir(_db_dir) else os.path.expanduser('~')
        db_path = filedialog.askopenfilename(
            title="Select Database File",
            filetypes=[("Database files", "*.db *.csv"),
                       ("SQLite database", "*.db"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            initialdir=_init_dir
        )
        if not db_path:
            return

        ext = os.path.splitext(db_path)[1].lower()
        if ext == '.db':
            db_type = 'hmdb'
        elif ext == '.csv':
            db_type = 'csv'
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            return

        self.current_db_path = db_path
        self.current_db_type = db_type
        self.internal_status.config(
            text=f"Selected: {os.path.basename(db_path)}", foreground="blue")
        self._status(f"Searching {os.path.basename(db_path)} "
                     f"— {len(self.peaks)} peaks...")

        tolerance = self.tolerance_var.get()

        # Progress window
        progress_win = tk.Toplevel(self.app.root)
        progress_win.title("Searching Database")
        progress_win.geometry("500x250")
        progress_win.transient(self.app.root)
        progress_win.grab_set()

        ttk.Label(progress_win,
                  text=f"Database: {os.path.basename(db_path)}",
                  font=("Arial", 9)).pack(pady=5)
        ttk.Label(progress_win, text="Searching peaks:",
                  font=("Arial", 10)).pack(pady=5)
        self.progress_bar = ttk.Progressbar(progress_win, length=400,
                                             mode='determinate')
        self.progress_bar.pack(pady=5)

        counter_frame = ttk.Frame(progress_win)
        counter_frame.pack(pady=5)
        for row, (label, attr) in enumerate([
            ("Peaks processed:",    "progress_peaks_label"),
            ("Matches found:",      "progress_matches_label"),
            ("Peaks with matches:", "progress_peaks_matched_label"),
        ]):
            ttk.Label(counter_frame, text=label,
                      font=("Arial", 9)).grid(row=row, column=0, padx=5)
            lbl = ttk.Label(counter_frame, text="0", font=("Arial", 9, "bold"))
            lbl.grid(row=row, column=1, padx=5)
            setattr(self, attr, lbl)

        time_frame = ttk.Frame(progress_win)
        time_frame.pack(pady=5)
        ttk.Label(time_frame, text="Elapsed time:",
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.progress_time_label = ttk.Label(time_frame, text="0.0 s",
                                              font=("Arial", 9, "bold"))
        self.progress_time_label.pack(side=tk.LEFT, padx=5)
        self.progress_operation_label = ttk.Label(
            progress_win, text="Initializing...",
            font=("Arial", 8), foreground="blue")
        self.progress_operation_label.pack(pady=5)
        progress_win.update()

        import threading
        import sqlite3

        def search_task():
            start_time = time.monotonic()
            all_results, peaks_with_matches = [], set()
            processed   = 0
            total_peaks = len(self.peaks)
            self.progress_bar['maximum'] = total_peaks
            conn = None
            try:
                if db_type == 'hmdb':
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    tables = {r[0] for r in cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()}
                    # Support both schema versions:
                    # v1 (legacy): multiplets + peaks
                    # v2 (current): shifts_1h + transitions
                    if 'shifts_1h' in tables:
                        db_schema = 'v2'
                    elif 'multiplets' in tables:
                        db_schema = 'v1'
                    else:
                        conn.close()
                        self.app.root.after(0, lambda: messagebox.showerror(
                            "Incompatible database",
                            "Could not identify database schema.\n"
                            "Expected tables: shifts_1h (or multiplets) + metabolites."))
                        self.app.root.after(0, progress_win.destroy)
                        self.app.root.after(
                            0, lambda: self._status_error("Incompatible database selected"))
                        return

                    for peak in self.peaks:
                        p_center = peak['ppm']
                        p_lo = min(peak.get('start_ppm', p_center),
                                   peak.get('end_ppm',   p_center)) - tolerance
                        p_hi = max(peak.get('start_ppm', p_center),
                                   peak.get('end_ppm',   p_center)) + tolerance

                        if db_schema == 'v2':
                            # ── Schema v2: shifts_1h + transitions ────────────
                            cursor.execute("""
                                SELECT s.id, s.ppm, s.multiplicity, s.j_coupling,
                                       s.atom_name as atom,
                                       met.bmrb_id, met.hmdb_id, met.name,
                                       ? as peak_ppm, ? as peak_rank,
                                       ABS(s.ppm - ?) as ppm_diff,
                                       (SELECT COUNT(DISTINCT s2.bmrb_id)
                                        FROM shifts_1h s2
                                        WHERE s2.bmrb_id != s.bmrb_id
                                          AND ABS(s2.ppm - s.ppm) < 0.1
                                       ) as overlap_count
                                FROM shifts_1h s
                                JOIN metabolites met ON s.bmrb_id = met.bmrb_id
                                WHERE ABS(s.ppm - ?) <= ?
                                ORDER BY ABS(s.ppm - ?)
                                LIMIT 5
                            """, (p_center, peak['rank'], p_center,
                                  p_center, tolerance + (p_hi - p_lo) / 2,
                                  p_center))
                            for row in cursor.fetchall():
                                r = dict(zip([d[0] for d in cursor.description], row))
                                r['match_type'] = r.pop('multiplicity', 'shift')
                                r['database']   = 'HMDB'
                                r['protons']    = r.get('atom', '')
                                all_results.append(r)
                                peaks_with_matches.add(peak['rank'])

                            # transitions: individual lines within multiplets
                            cursor.execute("""
                                SELECT t.ppm, t.intensity,
                                       met.bmrb_id, met.hmdb_id, met.name,
                                       ? as peak_ppm, ? as peak_rank,
                                       ABS(t.ppm - ?) as ppm_diff,
                                       0 as overlap_count
                                FROM transitions t
                                JOIN metabolites met ON t.bmrb_id = met.bmrb_id
                                WHERE t.ppm BETWEEN ? AND ?
                                  AND ABS(t.ppm - ?) < 0.03
                                ORDER BY ABS(t.ppm - ?)
                                LIMIT 3
                            """, (p_center, peak['rank'], p_center,
                                  p_lo, p_hi, p_center, p_center))
                            for row in cursor.fetchall():
                                r = dict(zip([d[0] for d in cursor.description], row))
                                r['match_type'] = 'transition'
                                r['database']   = 'HMDB'
                                r['protons']    = ''
                                all_results.append(r)
                                peaks_with_matches.add(peak['rank'])

                        else:
                            # ── Schema v1 (legacy): multiplets + peaks ────────
                            s_start = p_lo
                            s_end   = p_hi
                            cursor.execute("""
                                SELECT m.*, met.name,
                                       ? as peak_ppm, ? as peak_rank,
                                       ABS(m.ppm_center - ?) as ppm_diff,
                                       (SELECT COUNT(DISTINCT m2.hmdb_id)
                                        FROM multiplets m2
                                        WHERE m2.hmdb_id != m.hmdb_id
                                          AND ABS(m2.ppm_center - m.ppm_center) < 0.1
                                       ) as overlap_count
                                FROM multiplets m
                                JOIN metabolites met ON m.hmdb_id = met.hmdb_id
                                WHERE m.ppm_start <= ? AND m.ppm_end >= ?
                                  AND ABS(m.ppm_center - ?) < 0.2
                                ORDER BY ABS(m.ppm_center - ?)
                                LIMIT 3
                            """, (peak['ppm'], peak['rank'], peak['ppm'],
                                  s_end, s_start, peak['ppm'], peak['ppm']))
                            for row in cursor.fetchall():
                                r = dict(row)
                                r['match_type'] = 'multiplet'
                                r['database']   = 'HMDB'
                                all_results.append(r)
                                peaks_with_matches.add(peak['rank'])

                            cursor.execute("""
                                SELECT p.*, met.name,
                                       ? as peak_ppm, ? as peak_rank,
                                       ABS(p.ppm - ?) as ppm_diff
                                FROM peaks p
                                JOIN metabolites met ON p.hmdb_id = met.hmdb_id
                                WHERE p.ppm BETWEEN ? AND ?
                                  AND ABS(p.ppm - ?) < 0.05
                                LIMIT 2
                            """, (peak['ppm'], peak['rank'], peak['ppm'],
                                  s_start, s_end, peak['ppm']))
                            for row in cursor.fetchall():
                                r = dict(row)
                                r['match_type'] = 'peak'
                                r['database']   = 'HMDB'
                                all_results.append(r)
                                peaks_with_matches.add(peak['rank'])

                        processed += 1
                        if processed % 5 == 0 or processed == total_peaks:
                            elapsed = time.monotonic() - start_time
                            self.app.root.after(
                                0, lambda p=processed, t=total_peaks,
                                          m=len(all_results),
                                          pm=len(peaks_with_matches),
                                          e=elapsed:
                                self.update_progress_display(p, t, m, pm, e))
                            # Also update status bar
                            pct = int(processed / total_peaks * 100)
                            self.app.root.after(
                                0, lambda p=pct, e=elapsed:
                                self._status(f"Searching database... {p}%"
                                             f"  ({e:.1f}s)"))

                elif db_type == 'csv':
                    import pandas as pd
                    df = pd.read_csv(db_path)
                    for peak in self.peaks:
                        s_start = min(peak['start_ppm'], peak['end_ppm']) - tolerance
                        s_end   = max(peak['start_ppm'], peak['end_ppm']) + tolerance
                        mask = (df['ppm'] >= s_start) & (df['ppm'] <= s_end)
                        for _, row in df[mask].iterrows():
                            r = dict(row)
                            r.update({'peak_ppm':     peak['ppm'],
                                      'peak_rank':    peak['rank'],
                                      'match_type':   'csv_match',
                                      'database':     'CSV',
                                      'overlap_count': 0})
                            all_results.append(r)
                            peaks_with_matches.add(peak['rank'])
                        processed += 1
                        if processed % 5 == 0 or processed == total_peaks:
                            elapsed = time.monotonic() - start_time
                            self.app.root.after(
                                0, lambda p=processed, t=total_peaks,
                                          m=len(all_results),
                                          pm=len(peaks_with_matches),
                                          e=elapsed:
                                self.update_progress_display(p, t, m, pm, e))

                elapsed = time.monotonic() - start_time
                n_matches = len(all_results)
                n_peaks_m = len(peaks_with_matches)
                self.app.root.after(
                    0, lambda: self._status_ok(
                        f"Search complete — {n_matches} matches "
                        f"for {n_peaks_m}/{total_peaks} peaks  ({elapsed:.2f}s)"))
                self.app.root.after(0, lambda: self.show_internal_results(
                    all_results, progress_win, n_peaks_m, total_peaks,
                    os.path.basename(db_path), elapsed))
            except Exception as e:
                self.app.root.after(
                    0, lambda: self._status_error(f"Database search failed: {e}"))
                self.app.root.after(
                    0, lambda: self.handle_search_error(progress_win, e))
                import traceback; traceback.print_exc()
            finally:
                if conn:
                    conn.close()

        threading.Thread(target=search_task, daemon=True).start()

    def search_builtin_library(self):
        """Search built-in library (Metabolomics or Lipidomics mode)."""
        import threading
        import pandas as pd

        mode = 'Metabolomics (default)'
        try:
            mode = str(getattr(self, 'internal_library_mode_var').get())
        except Exception:
            pass
        is_lipidomics = (mode == 'Lipidomics')

        if not self.peaks and not is_lipidomics:
            messagebox.showinfo("Info", "Please detect peaks first.")
            return

        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if is_lipidomics:
            lib_name = 'lipidomics_simple_library.csv'
            title = 'Searching Lipidomics Library'
            info_text = 'Searching built-in lipidomics library…'
            db_label = 'Internal Lipidomics DB'
        else:
            lib_name = 'nmr_metabolite_library.csv'
            title = 'Searching Internal Library'
            info_text = 'Searching built-in NMR metabolite library…'
            db_label = 'Internal Library'

        lib_path = os.path.join(_root, 'data', lib_name)
        if not os.path.isfile(lib_path):
            messagebox.showerror("Error", f"Built-in library not found:\n{lib_path}")
            return

        tolerance = self.tolerance_var.get()
        self.hmdb_status.config(text="Internal DB: searching…", foreground="blue")
        self.app.root.update()

        progress_win = tk.Toplevel(self.app.root)
        progress_win.title(title)
        progress_win.geometry("420x120")
        progress_win.transient(self.app.root)
        progress_win.grab_set()
        ttk.Label(progress_win, text=info_text,
                  font=("Arial", 10)).pack(pady=10)
        pb = ttk.Progressbar(progress_win, length=320, mode='indeterminate')
        pb.pack(pady=10)
        pb.start()
        progress_win.update()

        def _search():
            try:
                df = pd.read_csv(lib_path)
                all_results, peaks_with_matches = [], set()
                start_time = time.monotonic()

                if is_lipidomics:
                    if self.peaks:
                        for _, row in df.iterrows():
                            r_start = float(row['start_ppm'])
                            r_end = float(row['end_ppm'])
                            r_lo = min(r_start, r_end)
                            r_hi = max(r_start, r_end)
                            center = (r_start + r_end) / 2.0
                            overlapped = []
                            for peak in self.peaks:
                                p_lo = min(peak.get('start_ppm', peak['ppm']),
                                           peak.get('end_ppm', peak['ppm'])) - tolerance
                                p_hi = max(peak.get('start_ppm', peak['ppm']),
                                           peak.get('end_ppm', peak['ppm'])) + tolerance
                                if r_lo <= p_hi and r_hi >= p_lo:
                                    overlapped.append(peak)
                            if not overlapped:
                                continue
                            peak = min(overlapped, key=lambda p: abs(float(p.get('ppm', 0)) - center))
                            all_results.append({
                                'name': str(row.get('name', 'Unknown')),
                                'id': str(row.get('id', '')),
                                'ppm': center,
                                'start_ppm': r_start,
                                'end_ppm': r_end,
                                'match_type': str(row.get('signal', row.get('multiplicity', '')) or ''),
                                'database': db_label,
                                'protons': 1,
                                'peak_ppm': peak['ppm'],
                                'peak_rank': peak['rank'],
                                'overlap_count': 0,
                            })
                            peaks_with_matches.add(peak['rank'])
                    else:
                        for idx, row in df.iterrows():
                            r_start = float(row['start_ppm'])
                            r_end = float(row['end_ppm'])
                            center = (r_start + r_end) / 2.0
                            all_results.append({
                                'name': str(row.get('name', 'Unknown')),
                                'id': str(row.get('id', '')),
                                'ppm': center,
                                'start_ppm': r_start,
                                'end_ppm': r_end,
                                'match_type': str(row.get('signal', row.get('multiplicity', '')) or ''),
                                'database': db_label,
                                'protons': 1,
                                'peak_ppm': center,
                                'peak_rank': -(idx + 1),
                                'overlap_count': 0,
                            })
                else:
                    for peak in self.peaks:
                        p_lo = min(peak.get('start_ppm', peak['ppm']),
                                   peak.get('end_ppm',   peak['ppm'])) - tolerance
                        p_hi = max(peak.get('start_ppm', peak['ppm']),
                                   peak.get('end_ppm',   peak['ppm'])) + tolerance
                        mask = (df['start_ppm'] <= p_hi) & (df['end_ppm'] >= p_lo)
                        for _, row in df[mask].iterrows():
                            all_results.append({
                                'name':        row['name'],
                                'id':          str(int(row['id'])),
                                'ppm':         (row['start_ppm'] + row['end_ppm']) / 2.0,
                                'start_ppm':   row['start_ppm'],
                                'end_ppm':     row['end_ppm'],
                                'match_type':  row.get('multiplicity', ''),
                                'database':    db_label,
                                'protons':     row.get('multiplicity', ''),
                                'peak_ppm':    peak['ppm'],
                                'peak_rank':   peak['rank'],
                                'overlap_count': 0,
                            })
                            peaks_with_matches.add(peak['rank'])

                elapsed = time.monotonic() - start_time

                def _show():
                    self.show_internal_results(
                        all_results, progress_win, len(peaks_with_matches),
                        len(self.peaks), lib_name, elapsed)
                    if is_lipidomics:
                        self.hmdb_status.config(
                            text=f"Internal DB: {len(all_results)} lipid regions loaded",
                            foreground='green')
                    else:
                        self.hmdb_status.config(
                            text=f"Internal DB: {len(all_results)} matches "
                                 f"({len(peaks_with_matches)}/{len(self.peaks)} peaks)",
                            foreground='green')

                self.app.root.after(0, _show)
            except Exception as exc:
                self.app.root.after(0, progress_win.destroy)
                self.app.root.after(
                    0, lambda: messagebox.showerror("Error", f"Library search failed:\n{exc}"))
                import traceback; traceback.print_exc()

        threading.Thread(target=_search, daemon=True).start()

    def search_hmdb_online(self):
        if not self.peaks:
            messagebox.showinfo("Info", "Please detect peaks first.")
            return

        self._status("Connecting to BMRB online database...")
        tolerance = self.tolerance_var.get()
        self.hmdb_status.config(text="BMRB: searching online...", foreground="blue")
        self.app.root.update()

        progress_win = tk.Toplevel(self.app.root)
        progress_win.title("Searching BMRB Database")
        progress_win.geometry("400x150")
        progress_win.transient(self.app.root)
        progress_win.grab_set()
        ttk.Label(progress_win, text="Searching BMRB online database...",
                  font=("Arial", 10)).pack(pady=10)
        pb = ttk.Progressbar(progress_win, length=300, mode='indeterminate')
        pb.pack(pady=10)
        pb.start()
        progress_win.update()

        import threading

        def search_task():
            from controllers.bmrb_controller import BMRBController
            bmrb = BMRBController()
            all_results, peaks_with_matches = [], set()
            for i, peak in enumerate(self.peaks):
                self.app.root.after(
                    0, lambda i=i: self._status(
                        f"BMRB online search — peak {i + 1}/{len(self.peaks)}..."))
                results = bmrb.search_by_ppm(peak['start_ppm'], peak['end_ppm'], tolerance)
                for r in results:
                    r['peak_ppm']  = peak['ppm']
                    r['peak_rank'] = peak['rank']
                    all_results.append(r)
                    peaks_with_matches.add(peak['rank'])
                time.sleep(0.2)
            n = len(all_results)
            pm = len(peaks_with_matches)
            self.app.root.after(
                0, lambda: self._status_ok(
                    f"BMRB online — {n} matches for {pm}/{len(self.peaks)} peaks"))
            self.app.root.after(0, lambda: self.show_bmrb_results(
                all_results, progress_win, pm, len(self.peaks)))

        threading.Thread(target=search_task, daemon=True).start()

    def show_bmrb_results(self, results, progress_win, peaks_matched, total_peaks):
        progress_win.destroy()
        if not results:
            self.hmdb_status.config(text="BMRB: no matches found", foreground="orange")
            messagebox.showinfo("BMRB Search", "No matching metabolites found.")
            return
        self.hmdb_status.config(
            text=f"BMRB: {len(results)} matches ({peaks_matched}/{total_peaks} peaks)",
            foreground="green")
        win = tk.Toplevel(self.app.root)
        win.title(f"BMRB Online Results - {len(results)} matches")
        win.geometry("1000x500")

    def show_internal_results(self, results, progress_win, peaks_with_matches,
                               total_peaks, db_name, elapsed_time):
        if progress_win:
            progress_win.destroy()
        if not results:
            self.internal_status.config(text="No matches found", foreground="orange")
            messagebox.showinfo("Search Results", "No matching metabolites found.")
            return

        unique = len(set(r.get('hmdb_id', r.get('bmrb_id', '')) for r in results))
        self.internal_status.config(
            text=f"Found {len(results)} matches from {unique} metabolites "
                 f"({peaks_with_matches}/{total_peaks} peaks) in {elapsed_time:.2f}s",
            foreground="green")

        res_win = tk.Toplevel(self.app.root)
        res_win.title(
            f"Database Results - {len(results)} matches ({elapsed_time:.2f}s)")
        res_win.geometry("1000x500")
        res_win.transient(self.app.root)

        main_frame = ttk.Frame(res_win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame,
                  text=f"Found {len(results)} matches from {unique} unique metabolites. "
                       f"Matched {peaks_with_matches}/{total_peaks} peaks "
                       f"in {elapsed_time:.2f}s.",
                  font=("Arial", 9, "italic")).pack(anchor=tk.W, pady=5)

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        columns = ('id', 'name', 'type', 'ppm', 'protons',
                   'source', 'peak_ppm', 'peak_rank')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                            height=15)
        for col, heading, width in [
            ('id', 'ID', 100), ('name', 'Metabolite', 200),
            ('type', 'Type', 80), ('ppm', 'PPM', 80), ('protons', 'H', 40),
            ('source', 'Source', 120), ('peak_ppm', 'Peak PPM', 80),
            ('peak_rank', 'Peak #', 60),
        ]:
            tree.heading(col, text=heading)
            tree.column(col, width=width)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        for r in results:
            if 'hmdb_id' in r:
                source, met_id = 'HMDB', r.get('hmdb_id', '')
            elif 'bmrb_id' in r:
                source, met_id = 'BMRB', r.get('bmrb_id', '')
            else:
                source, met_id = r.get('database', 'Unknown'), r.get('id', '')
            ppm = (f"{r.get('ppm_center', r.get('ppm', 0)):.3f}"
                   if r.get('match_type') == 'multiplet'
                   else f"{r.get('ppm', 0):.3f}")
            tree.insert('', tk.END, values=(
                met_id, r.get('name', 'Unknown'), r.get('match_type', 'unknown'),
                ppm, r.get('protons', r.get('atom', '')),
                source, f"{r.get('peak_ppm', 0):.3f}", r.get('peak_rank', '')))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Export to CSV",
                   command=lambda: self.export_results(results)
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close",
                   command=res_win.destroy).pack(side=tk.RIGHT, padx=5)

        res_win.update_idletasks()
        w, h = res_win.winfo_width(), res_win.winfo_height()
        res_win.geometry(f'{w}x{h}+{(res_win.winfo_screenwidth()//2)-(w//2)}'
                         f'+{(res_win.winfo_screenheight()//2)-(h//2)}')

    def export_results(self, results):
        from tkinter import filedialog
        import csv
        filename = filedialog.asksaveasfilename(
            title="Save Results As", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="search_results.csv")
        if not filename:
            return
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['HMDB ID', 'Metabolite Name', 'Match Type',
                                 'Peak PPM', 'Peak Rank', 'DB PPM', 'Protons',
                                 'Multiplet Type', 'DB Interval', 'Overlaps',
                                 'Confidence'])
                for r in results:
                    if r.get('match_type') == 'multiplet':
                        db_ppm   = f"{r.get('ppm_center', 0):.3f}"
                        protons  = r.get('protons', '')
                        mult     = r.get('type', '')
                        interval = (f"{r.get('ppm_start', 0):.3f} - "
                                    f"{r.get('ppm_end', 0):.3f}")
                    else:
                        db_ppm, protons, mult, interval = (
                            f"{r.get('ppm', 0):.3f}", '', 'peak', '')
                    overlaps   = r.get('overlap_count', 0)
                    confidence = "Unique" if overlaps == 0 else "Overlap"
                    writer.writerow([
                        r.get('hmdb_id', ''), r.get('name', ''),
                        r.get('match_type', ''),
                        f"{r.get('peak_ppm', 0):.3f}", r.get('peak_rank', ''),
                        db_ppm, protons, mult, interval, overlaps, confidence])
            messagebox.showinfo("Success",
                                f"Exported {len(results)} results to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export: {str(e)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  MISC HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def draw_annotation_on_plot(self, peak, name):
        self.app.ax.annotate(
            name, xy=(peak['ppm'], peak['height']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        self.app.canvas.draw_idle()

    def reset_zoom(self):
        if self.plot_manager:
            self.plot_manager.reset_zoom()

    def update_progress_display(self, processed, total, matches,
                                 peaks_matched, elapsed):
        try:
            if self.progress_bar:
                self.progress_bar['value'] = processed
            if self.progress_peaks_label:
                self.progress_peaks_label.config(text=f"{processed}/{total}")
            if self.progress_matches_label:
                self.progress_matches_label.config(text=str(matches))
            if self.progress_peaks_matched_label:
                self.progress_peaks_matched_label.config(text=str(peaks_matched))
            if self.progress_time_label:
                self.progress_time_label.config(text=f"{elapsed:.1f} s")
            if self.progress_operation_label:
                pct = (processed / total * 100) if total > 0 else 0
                op  = ("Initializing..." if processed == 0
                       else f"Processing peaks... {pct:.1f}%"
                       if processed < total else "Finalizing results...")
                self.progress_operation_label.config(text=op)
            self.app.root.update_idletasks()
        except Exception as e:
            print(f"Progress update error: {e}")

    def handle_search_error(self, progress_win, error):
        for attr in ['progress_bar', 'progress_peaks_label',
                     'progress_matches_label', 'progress_peaks_matched_label',
                     'progress_time_label', 'progress_operation_label']:
            setattr(self, attr, None)
        progress_win.destroy()
        self.internal_status.config(text="Internal DB: search failed",
                                     foreground="red")
        messagebox.showerror("Error", f"Database search failed:\n{str(error)}")
        import traceback; traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════════

    def on_activate(self):
        print("🔄 Notation stage activated")
        if hasattr(self, 'frame') and self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)
        if hasattr(self.app, 'ax') and self.plot_manager is None:
            try:
                from plotting.interactive import InteractivePlotManager
                self.plot_manager = InteractivePlotManager(
                    self.app.figure, self.app.ax, self.app.canvas, self.app)
                print("✅ Interactive plot manager initialized")
            except Exception as e:
                print(f"❌ Error initializing plot manager: {e}")
        if hasattr(self.app, 'ax') and self.plot_manager is not None:
            if self.plot_manager.original_xlim is None:
                self.plot_manager.original_xlim = self.app.ax.get_xlim()
                self.plot_manager.original_ylim = self.app.ax.get_ylim()
        self._status_ready()
        self.app.root.bind('<Key-r>', lambda e: self.reset_zoom())
        self.app.root.bind('<Key-R>', lambda e: self.reset_zoom())

    def on_deactivate(self):
        print("Notation stage deactivated")
        self.app.root.unbind('<Key-r>')
        self.app.root.unbind('<Key-R>')
        if hasattr(self, 'frame') and self.frame:
            self.frame.pack_forget()