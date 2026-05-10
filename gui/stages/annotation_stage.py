# gui/stages/annotation_stage.py
"""
Metabolites Annotation Stage — Step 3.

Algorithm
---------
1. Align spectrum to 0 ppm (DSS/TSP) using the same anchor as peak_alignment.
2. For each detected peak, query DB candidates within tolerance.
3. Score each candidate:
   a. PPM distance score (Gaussian, σ = tolerance/2)
   b. Multiplet pattern score: find surrounding peaks that form the expected
      multiplet pattern using J from DB, score position + relative intensity.
4. Rank by composite score = 0.4×ppm_score + 0.6×multiplet_score.
5. Open results window showing ranked candidates with fingerprint overlay.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import certifi

os.environ['SSL_CERT_FILE']      = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from .base_stage import BaseStage
from ._annotation_search import _AnnotationSearchMixin


class AnnotationStage(BaseStage, _AnnotationSearchMixin):

    def __init__(self, parent, app, peak_stage):
        super().__init__(parent, app)
        self.peak_stage         = peak_stage
        self.annotations        = {}
        self._last_results_args = None
        self._metabolites       = None   # live reference set by AnnotationResultsWindow
        self._quality_marks       = {}   # {(met_id, peak_rank): 'good'|'bad'} — persisted
        self._child_quality_marks = {}   # {(met_id, round(dp_ppm,3)): 'good'|'bad'} — persisted
        self._anchor_lines        = {}   # {(met_id, peak_rank): line_idx} — anchor line for quantification
        self._subline_quality_marks = {} # {(met_id, peak_rank, line_idx): 'good'|'bad'} — persisted
        self.mcr_r2_threshold   = 0.95   # used by annotation MCR deconvolution
        self.create_widgets()

    # ── Status helpers ────────────────────────────────────────────────────────

    def _status(self, msg, color='blue'):
        try:
            self.app.status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    def _status_ok(self,  m): self._status(m, 'green')
    def _status_err(self, m): self._status(m, 'red')
    def _status_ready(self):  self._status('Ready', 'gray')

    @property
    def peaks(self):
        return self.peak_stage.peaks

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text='Metabolites Annotation',
                  font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        s3 = ttk.LabelFrame(self.frame, text='📡 Database Search', padding=5)
        s3.pack(fill=tk.X, padx=5, pady=5)

        tf = ttk.Frame(s3); tf.pack(fill=tk.X, pady=2)
        ttk.Label(tf, text='Tolerance (ppm):', width=18).pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.05)
        ttk.Entry(tf, textvariable=self.tolerance_var, width=7).pack(
            side=tk.LEFT, padx=2)

        sf = ttk.Frame(s3); sf.pack(fill=tk.X, pady=2)
        ttk.Label(sf, text='Spectrometer (MHz):', width=18).pack(side=tk.LEFT)
        self.spectrometer_var = tk.StringVar(value='600')
        ttk.Entry(sf, textvariable=self.spectrometer_var, width=7).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(sf, text='for J Hz→ppm',
                  font=('Arial', 7), foreground='gray').pack(
                      side=tk.LEFT, padx=4)

        mf = ttk.Frame(s3); mf.pack(fill=tk.X, pady=2)
        ttk.Label(mf, text='Internal library:', width=18).pack(side=tk.LEFT)
        self.internal_library_mode_var = tk.StringVar(value='Metabolomics (default)')
        self.internal_library_mode = ttk.Combobox(
            mf,
            textvariable=self.internal_library_mode_var,
            state='readonly',
            width=24,
            values=['Metabolomics (default)', 'Lipidomics'])
        self.internal_library_mode.pack(side=tk.LEFT, padx=2)

        bf = ttk.Frame(s3); bf.pack(fill=tk.X, pady=5)
        self.btn_search_internal = ttk.Button(
            bf, text='🔍 Search in External DB',
            command=self.search_internal_db)
        self.btn_search_internal.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_search_online = ttk.Button(
            bf, text='📚 Use Internal DB',
            command=self.search_builtin_library)
        self.btn_search_online.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        mcr_bf = ttk.Frame(s3); mcr_bf.pack(fill=tk.X, pady=(0, 5))
        self.mcr_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            mcr_bf, text='🔬 Run MCR Deconvolution',
            variable=self.mcr_run_var).pack(side=tk.LEFT, padx=2)

        self.internal_status = ttk.Label(
            s3, text='Internal DB: ready', foreground='gray')
        self.internal_status.pack(anchor=tk.W, pady=2)
        self.hmdb_status = ttk.Label(
            s3, text='Internal DB: ready', foreground='gray')
        self.hmdb_status.pack(anchor=tk.W, pady=2)

        info_frame = ttk.LabelFrame(
            self.frame, text='📊 Current Peaks', padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.peaks_info_label = ttk.Label(
            info_frame, text='No peaks detected yet.', foreground='gray')
        self.peaks_info_label.pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text='Only peaks marked Good in Peak List are annotated.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        mf = ttk.LabelFrame(self.frame, text='🐭 Mouse Controls', padding=5)
        mf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mf,
                  text='• Scroll: Zoom in/out\n'
                       '• Drag: Pan\n'
                       '• R: Reset view',
                  font=('Arial', 8), foreground='blue',
                  justify=tk.LEFT).pack(anchor=tk.W, padx=5)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        n      = len(self.peaks)
        n_good = sum(1 for p in self.peaks
                     if p.get('status', 'Undefined') == 'Good')
        if n == 0:
            self.peaks_info_label.config(
                text='No peaks detected yet.', foreground='gray')
        elif n_good == 0:
            self.peaks_info_label.config(
                text=f'{n} peaks — 0 marked Good. Mark in Peak List first.',
                foreground='orange')
        else:
            self.peaks_info_label.config(
                text=f'{n} peaks loaded — {n_good} Good (will annotate).',
                foreground='blue')

    def on_deactivate(self):
        self.frame.pack_forget()

