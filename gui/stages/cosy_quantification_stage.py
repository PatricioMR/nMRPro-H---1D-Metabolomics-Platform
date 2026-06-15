# gui/stages/cosy_quantification_stage.py
"""
COSY Quantification Stage — quantificação relativa só com COSY.

Não depende de anotação 1D nem de MCR. Usa os metabolitos anotados na
CosyAnnotationStage e integra o VOLUME dos seus cross-peaks (processing.
cosy_quant) para dar abundância relativa por metabolito.

Semi-quantitativo (volume COSY depende de J/aquisição) — ver aviso em
processing/cosy_quant.py.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv

from .base_stage import BaseStage


class CosyQuantificationStage(BaseStage):

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._quant = []
        self.create_widgets()

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text='COSY Quantification',
                  font=('Arial', 12, 'bold')).pack(pady=(10, 0))
        ttk.Label(self.frame,
                  text='Relative abundance from\ncross-peak volumes',
                  font=('Arial', 8), foreground='gray',
                  justify=tk.CENTER).pack(pady=(0, 4))
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=4)

        # ── Integration params ────────────────────────────────────────────────
        pf = ttk.LabelFrame(self.frame, text='📐 Integration', padding=5)
        pf.pack(fill=tk.X, padx=5, pady=4)
        self._row(pf, 'Box ½-width F2 (ppm):', 'box_f2_var', tk.DoubleVar(value=0.02))
        self._row(pf, 'Box ½-width F1 (ppm):', 'box_f1_var', tk.DoubleVar(value=0.02))
        self._row(pf, 'Noise floor ×:', 'floor_var', tk.DoubleVar(value=3.0))

        af = ttk.Frame(pf); af.pack(fill=tk.X, pady=2)
        ttk.Label(af, text='Aggregate:', width=18).pack(side=tk.LEFT)
        self.agg_var = tk.StringVar(value='median')
        ttk.Combobox(af, textvariable=self.agg_var, state='readonly', width=9,
                     values=['median', 'sum', 'mean']).pack(side=tk.LEFT, padx=2)

        ttk.Label(pf,
                  text='Volume per metabolite = aggregate of its cross-peak '
                       'volumes. Semi-quantitative.',
                  font=('Arial', 7), foreground='gray',
                  wraplength=255).pack(anchor=tk.W, pady=(2, 0))

        bf = ttk.Frame(self.frame); bf.pack(fill=tk.X, padx=5, pady=4)
        self.btn_run = ttk.Button(bf, text='⚗ Quantify', command=self._run)
        self.btn_run.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_export = ttk.Button(bf, text='⬇ CSV', width=8,
                                     command=self._export, state='disabled')
        self.btn_export.pack(side=tk.LEFT, padx=2)

        self.status = ttk.Label(self.frame, text='', font=('Arial', 8),
                                foreground='blue', wraplength=260)
        self.status.pack(anchor=tk.W, padx=6, pady=2)

        # ── Results table ─────────────────────────────────────────────────────
        rf = ttk.LabelFrame(self.frame, text='📊 Relative Abundance', padding=2)
        rf.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        cols = ('name', 'rel', 'vol', 'cp')
        self.tree = ttk.Treeview(rf, columns=cols, show='headings',
                                 selectmode='browse')
        for c, txt, w in [('name', 'Analyte', 120), ('rel', 'Rel%', 50),
                          ('vol', 'Volume', 70), ('cp', '#CP', 36)]:
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w,
                             anchor=tk.W if c == 'name' else tk.CENTER)
        vsb = ttk.Scrollbar(rf, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

    def _row(self, parent, label, attr, var):
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var, width=8).pack(side=tk.LEFT, padx=2)
        setattr(self, attr, var)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        if getattr(self.app, 'cosy_data', None) is not None:
            try:
                self.app._show_2d_viewer()
            except Exception as e:
                print(f'[CosyQuant] viewer: {e}')
        # Reflectir estado da anotação
        if getattr(self.app, 'cosy_annotation', None):
            self.status.config(
                text='Annotation ready — click Quantify.', foreground='blue')
        else:
            self.status.config(
                text='Run COSY annotation first.', foreground='orange')

    def on_deactivate(self):
        self.frame.pack_forget()
        if getattr(self.app, '_2d_canvas', None) is not None:
            try:
                self.app._hide_2d_viewer()
            except Exception:
                pass

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        ca = getattr(self.app, 'cosy_annotation', None)
        if not ca or not ca.get('results'):
            messagebox.showinfo(
                'COSY Quantification',
                'No COSY annotation found.\nRun Annotation (COSY) first.')
            return

        try:
            hf2 = float(self.box_f2_var.get())
            hf1 = float(self.box_f1_var.get())
            floor_mult = float(self.floor_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror('COSY Quantification', 'Invalid parameter.')
            return

        from processing.cosy_quant import quantify_metabolites
        try:
            self._quant = quantify_metabolites(
                ca['results'], self.app.cosy_data,
                half_f2=hf2, half_f1=hf1, noise_mult=floor_mult,
                aggregate=self.agg_var.get())
        except Exception as e:
            messagebox.showerror('COSY Quantification', str(e))
            return

        self._populate()
        self.app.cosy_annotation['quant'] = self._quant
        self.btn_export.config(state='normal')
        self.status.config(
            text=f'{len(self._quant)} metabolites quantified (relative).',
            foreground='green')

    def _populate(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for q in self._quant:
            rel = q.get('rel_abundance', 0.0) * 100
            self.tree.insert('', 'end', values=(
                q['name'], f'{rel:.1f}', f"{q['volume']:.3g}",
                q.get('n_matched', len(q.get('matches', [])))))

    def _on_select(self, event=None):
        """Realça os cross-peaks do metabolito seleccionado no viewer 2D."""
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx >= len(self._quant):
            return
        ax = getattr(self.app, '_2d_ax', None)
        canvas = getattr(self.app, '_2d_canvas', None)
        if ax is None or canvas is None:
            return
        # Limpar realce anterior
        for art in getattr(self, '_hl', []):
            try:
                art.remove()
            except Exception:
                pass
        self._hl = []
        for m in self._quant[idx].get('matches', []):
            for (x, y) in ((m['f2'], m['f1']), (m['f1'], m['f2'])):
                ln, = ax.plot(x, y, 'o', mfc='#e91e63', mec='black',
                              ms=9, mew=0.8, zorder=12)
                self._hl.append(ln)
        canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if not self._quant:
            return
        path = filedialog.asksaveasfilename(
            title='Export COSY Quantification',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')])
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Analyte', 'HMDB_ID', 'Rel_abundance_%', 'Volume',
                        'N_crosspeaks', 'Coverage', 'N_J_confirmed', 'Score'])
            for q in self._quant:
                w.writerow([
                    q['name'], q.get('hmdb_id', ''),
                    f"{q.get('rel_abundance', 0)*100:.3f}",
                    f"{q['volume']:.6g}", q.get('n_matched', ''),
                    f"{q.get('coverage', 0):.3f}", q.get('n_j_confirmed', ''),
                    f"{q.get('score', 0):.4f}"])
        self.status.config(text=f'Exported → {path}', foreground='green')
