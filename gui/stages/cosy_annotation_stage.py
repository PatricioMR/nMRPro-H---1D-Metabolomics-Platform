# gui/stages/cosy_annotation_stage.py
"""
COSY Annotation Stage — rota alternativa de anotação para espectros 2D COSY.

Substitui a anotação 1D quando um ficheiro COSY ¹H-¹H está carregado.
Painel esquerdo com os parâmetros de qualidade usados na literatura COSY
(COLMAR — Bingol & Brüschweiler), mantendo a escolha de DB. MCR não se aplica
a 2D e por isso não é exposto aqui.

Fluxo:
  1. Picking de cross-peaks no contour (processing.cosy_peaks).
  2. Matching por co-ocorrência de shifts ¹H contra a DB (processing.cosy_annotation).
  3. Abre a janela de resultados COSY (lista de analitos + visualização 2D).
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os

from .base_stage import BaseStage
from ._annotation_helpers import _pick_database
from processing.spectral_regions import WATER_REGION


class CosyAnnotationStage(BaseStage):

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._db_path = None
        self._results_win = None
        self.create_widgets()

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text='COSY Annotation',
                  font=('Arial', 12, 'bold')).pack(pady=(10, 0))
        ttk.Label(self.frame,
                  text='¹H-¹H cross-peak matching\n(shift co-occurrence — COLMAR)',
                  font=('Arial', 8), foreground='gray',
                  justify=tk.CENTER).pack(pady=(0, 4))
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=4)

        # ── Database ──────────────────────────────────────────────────────────
        dbf = ttk.LabelFrame(self.frame, text='📡 Reference Database', padding=5)
        dbf.pack(fill=tk.X, padx=5, pady=4)
        self.db_label = ttk.Label(dbf, text='Default: HMDB_General.db',
                                  font=('Arial', 8), foreground='gray',
                                  wraplength=260)
        self.db_label.pack(anchor=tk.W)
        ttk.Button(dbf, text='Choose Database…',
                   command=self._choose_db).pack(fill=tk.X, pady=(3, 0))

        # ── Cross-peak picking ────────────────────────────────────────────────
        pf = ttk.LabelFrame(self.frame, text='🔍 Cross-peak Picking', padding=5)
        pf.pack(fill=tk.X, padx=5, pady=4)

        self._row(pf, 'Threshold ×noise:', 'thr_var', tk.DoubleVar(value=8.0))
        self._row(pf, 'Diagonal excl. (ppm):', 'diag_var', tk.DoubleVar(value=0.05))
        self._row(pf, 'Min separation (ppm):', 'sep_var', tk.DoubleVar(value=0.02))

        self.sym_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pf, text='Require symmetric partner',
                        variable=self.sym_var).pack(anchor=tk.W, pady=1)
        ttk.Label(pf, text='Real cross-peaks are symmetric; rejects t1-noise.',
                  font=('Arial', 7), foreground='gray',
                  wraplength=255).pack(anchor=tk.W)

        # ── Matching ──────────────────────────────────────────────────────────
        mf = ttk.LabelFrame(self.frame, text='🧪 Matching', padding=5)
        mf.pack(fill=tk.X, padx=5, pady=4)
        self._row(mf, 'Tolerance F2 (ppm):', 'tol_f2_var', tk.DoubleVar(value=0.03))
        self._row(mf, 'Tolerance F1 (ppm):', 'tol_f1_var', tk.DoubleVar(value=0.03))
        self._row(mf, 'J tolerance (Hz):', 'jtol_var', tk.DoubleVar(value=0.5))
        self._row(mf, 'Min cross-peaks:', 'mincp_var', tk.IntVar(value=1))

        self.requirej_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text='Require J-coupling confirmation',
                        variable=self.requirej_var).pack(anchor=tk.W, pady=1)
        ttk.Label(mf,
                  text='Stricter: only pairs whose shifts share a J value.',
                  font=('Arial', 7), foreground='gray',
                  wraplength=255).pack(anchor=tk.W)

        # Water exclusion
        wf = ttk.Frame(mf); wf.pack(fill=tk.X, pady=(4, 0))
        self.exclude_water_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(wf, text='Exclude water',
                        variable=self.exclude_water_var).pack(side=tk.LEFT)
        self.water_lo_var = tk.DoubleVar(value=WATER_REGION[0])
        self.water_hi_var = tk.DoubleVar(value=WATER_REGION[1])
        ttk.Entry(wf, textvariable=self.water_lo_var, width=5).pack(
            side=tk.LEFT, padx=(6, 1))
        ttk.Label(wf, text='–').pack(side=tk.LEFT)
        ttk.Entry(wf, textvariable=self.water_hi_var, width=5).pack(
            side=tk.LEFT, padx=1)
        ttk.Label(wf, text='ppm', font=('Arial', 7),
                  foreground='gray').pack(side=tk.LEFT)

        # ── Run ───────────────────────────────────────────────────────────────
        self.btn_run = ttk.Button(self.frame, text='🧬 Annotate COSY',
                                  command=self._run)
        self.btn_run.pack(fill=tk.X, padx=5, pady=(6, 2))
        self.status = ttk.Label(self.frame, text='', font=('Arial', 8),
                                foreground='blue', wraplength=260)
        self.status.pack(anchor=tk.W, padx=6, pady=2)

    def _row(self, parent, label, attr, var):
        """Cria uma linha label + entry e guarda a var em self.<attr>."""
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        setattr(self, attr, var)

    # ── DB selection ──────────────────────────────────────────────────────────

    def _default_db(self):
        from app_paths import databases_dir
        p = os.path.join(databases_dir(), 'HMDB_General.db')
        return p if os.path.isfile(p) else None

    def _choose_db(self):
        path = _pick_database(self.app.root)
        if path:
            self._db_path = path
            self.db_label.config(text=os.path.basename(path), foreground='black')

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        # Mostrar o visualizador 2D na área direita
        if getattr(self.app, 'cosy_data', None) is not None:
            try:
                self.app._show_2d_viewer()
            except Exception as e:
                print(f'[CosyAnnotation] viewer: {e}')

    def on_deactivate(self):
        self.frame.pack_forget()
        if getattr(self.app, '_2d_canvas', None) is not None:
            try:
                self.app._hide_2d_viewer()
            except Exception:
                pass

    # ── Run annotation ────────────────────────────────────────────────────────

    def _run(self):
        cd = getattr(self.app, 'cosy_data', None)
        if cd is None:
            messagebox.showinfo('COSY', 'No COSY data loaded.')
            return

        db_path = self._db_path or self._default_db()
        if not db_path or not os.path.isfile(db_path):
            messagebox.showerror(
                'COSY', 'No reference database found.\n'
                'Choose a database (HMDB_General.db).')
            return

        try:
            thr   = float(self.thr_var.get())
            diag  = float(self.diag_var.get())
            sep   = float(self.sep_var.get())
            tol2  = float(self.tol_f2_var.get())
            tol1  = float(self.tol_f1_var.get())
            jtol  = float(self.jtol_var.get())
            mincp = int(self.mincp_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror('COSY', 'Invalid parameter value.')
            return

        water = None
        if self.exclude_water_var.get():
            water = (float(self.water_lo_var.get()),
                     float(self.water_hi_var.get()))

        self.status.config(text='Picking cross-peaks…', foreground='blue')
        self.app.root.update_idletasks()

        from processing.cosy_peaks import pick_cosy_crosspeaks
        from processing.cosy_annotation import annotate_cosy

        try:
            crosspeaks = pick_cosy_crosspeaks(
                cd['f2_ppm'], cd['f1_ppm'], cd['data'],
                noise_mult=thr, diag_width=diag, min_sep=sep,
                symmetric=self.sym_var.get(), exclude_water=water)
        except Exception as e:
            messagebox.showerror('COSY picking', str(e))
            return

        if not crosspeaks:
            self.status.config(
                text='No cross-peaks above threshold. Lower the threshold.',
                foreground='red')
            return

        self.status.config(
            text=f'{len(crosspeaks)} cross-peaks → matching…',
            foreground='blue')
        self.app.root.update_idletasks()

        try:
            results = annotate_cosy(
                crosspeaks, db_path, tol_f2=tol2, tol_f1=tol1,
                j_tol=jtol, require_j=self.requirej_var.get(),
                min_crosspeaks=mincp)
        except Exception as e:
            messagebox.showerror('COSY matching', str(e))
            return

        self.status.config(
            text=f'{len(crosspeaks)} cross-peaks · {len(results)} candidates',
            foreground='green')

        # Guardar para projecto/reabertura
        self.app.cosy_annotation = {
            'crosspeaks': crosspeaks, 'results': results,
            'db_path': db_path,
        }

        # Habilitar o atalho "Annotation Results" no rodapé
        try:
            self.app.btn_annot_shortcut.config(state='normal')
        except Exception:
            pass

        from gui.widgets.cosy_results_window import CosyResultsWindow
        self._results_win = CosyResultsWindow(
            self.app, cd, crosspeaks, results)
