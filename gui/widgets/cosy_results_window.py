# gui/widgets/cosy_results_window.py
"""
Janela de resultados da anotação COSY.

Layout (estilo TopSpin / COLMAR):
  • Esquerda  — lista de analitos candidatos, ordenada por score
                (nome, coverage, nº cross-peaks, J-confirmados).
  • Direita   — visualização COSY: contour + todos os cross-peaks detectados.
                Seleccionar um analito realça os cross-peaks que ele explica.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.colors import LogNorm


class CosyResultsWindow:

    def __init__(self, app, cosy_data, crosspeaks, results):
        self.app        = app
        self.cd         = cosy_data
        self.crosspeaks = crosspeaks
        # Quantificação relativa por volume de cross-peaks (semi-quantitativa)
        try:
            from processing.cosy_quant import quantify_metabolites
            quant = quantify_metabolites(results, cosy_data)
            # Manter ordem por score (confiança de identificação), não por volume
            self.results = sorted(quant, key=lambda r: -r.get('score', 0))
        except Exception as e:
            print(f'[CosyResults] quant failed: {e}')
            self.results = results
        self._highlight = None   # artistas do realce actual

        sw = app.root.winfo_screenwidth()
        sh = app.root.winfo_screenheight()
        W, H = min(1180, sw - 60), min(720, sh - 80)

        self.win = tk.Toplevel(app.root)
        self.win.title('COSY Annotation Results')
        self.win.geometry(f"{W}x{H}+{(sw - W)//2}+{(sh - H)//2}")
        self.win.minsize(820, 500)

        pw = tk.PanedWindow(self.win, orient=tk.HORIZONTAL,
                            sashwidth=5, sashrelief=tk.RAISED)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw, width=360)
        right = ttk.Frame(pw)
        pw.add(left, minsize=300, stretch='never')
        pw.add(right, minsize=460, stretch='always')
        left.pack_propagate(False)

        self._build_list(left)
        self._build_plot(right)
        self._draw_base()
        self._populate()

    # ── Painel esquerdo: lista de analitos ────────────────────────────────────

    def _build_list(self, parent):
        ttk.Label(parent, text='Candidate Analytes',
                  font=('Arial', 11, 'bold')).pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Label(parent,
                  text=f'{len(self.crosspeaks)} cross-peaks detected · '
                       f'{len(self.results)} candidates',
                  font=('Arial', 8), foreground='gray').pack(
                      anchor=tk.W, padx=8)

        cols = ('name', 'cov', 'n', 'j', 'rel', 'score')
        self.tree = ttk.Treeview(parent, columns=cols, show='headings',
                                 selectmode='browse')
        headers = [('name', 'Analyte', 138), ('cov', 'Cov', 46),
                   ('n', '#CP', 42), ('j', 'J✓', 36),
                   ('rel', 'Rel%', 50), ('score', 'Score', 54)]
        for c, txt, w in headers:
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w,
                             anchor=tk.W if c == 'name' else tk.CENTER)
        vsb = ttk.Scrollbar(parent, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=6)
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=6)
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

    def _populate(self):
        for r in self.results:
            rel = r.get('rel_abundance')
            rel_txt = f"{rel*100:.1f}" if rel is not None else '—'
            self.tree.insert('', 'end', values=(
                r['name'], f"{r['coverage']*100:.0f}%",
                f"{r['n_matched']}/{r['n_total']}",
                r['n_j_confirmed'], rel_txt, f"{r['score']:.3f}"))
        if self.results:
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)

    # ── Painel direito: visualização COSY ─────────────────────────────────────

    def _build_plot(self, parent):
        self.fig = plt.Figure(facecolor='#fafafa')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        tb = NavigationToolbar2Tk(self.canvas, parent)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _draw_base(self):
        """Contour + todos os cross-peaks detectados (cinza)."""
        f2, f1, data = self.cd['f2_ppm'], self.cd['f1_ppm'], self.cd['data']
        ax = self.ax
        ax.clear()
        ax.set_facecolor('#ffffff')

        dmax = float(np.percentile(np.abs(data), 99.9))
        noise = float(np.median(np.abs(data)))
        base = max(noise * 8.0, dmax * 1e-3)
        if base < dmax:
            levels = [base * (dmax / base) ** (i / 11) for i in range(12)]
            try:
                ax.contour(f2, f1, data, levels=levels, cmap='jet',
                           norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                           linewidths=0.5, alpha=0.7)
            except Exception:
                pass

        # Todos os cross-peaks (ambos os lados da diagonal) a cinza
        for cp in self.crosspeaks:
            for (x, y) in ((cp['f2'], cp['f1']), (cp['f1'], cp['f2'])):
                ax.plot(x, y, 'o', mfc='none', mec='#999999', ms=7, mew=0.8,
                        zorder=5)

        ax.set_xlim(float(f2[0]), float(f2[-1]))
        ax.set_ylim(float(f1[-1]), float(f1[0]))
        ax.set_xlabel('δ F2 (ppm)')
        ax.set_ylabel('δ F1 (ppm)')
        # Diagonal
        hi = min(float(f2[0]), float(f1[0]))
        lo = max(float(f2[-1]), float(f1[-1]))
        ax.plot([hi, lo], [hi, lo], color='#888', lw=0.6, ls='--', alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx >= len(self.results):
            return
        self._highlight_analyte(self.results[idx])

    def _highlight_analyte(self, result):
        """Realça os cross-peaks explicados pelo analito seleccionado."""
        # Remover realce anterior
        if self._highlight:
            for art in self._highlight:
                try:
                    art.remove()
                except Exception:
                    pass
        self._highlight = []

        for m in result['matches']:
            color = '#2e7d32' if m['j_confirmed'] else '#e53935'
            for (x, y) in ((m['f2'], m['f1']), (m['f1'], m['f2'])):
                ln, = self.ax.plot(x, y, 'o', mfc=color, mec='black',
                                   ms=9, mew=0.8, alpha=0.9, zorder=10)
                self._highlight.append(ln)

        self.ax.set_title(
            f"{result['name']}  —  {result['n_matched']}/{result['n_total']} "
            f"cross-peaks  ·  {result['n_j_confirmed']} J-confirmed  "
            f"(green = J ✓, red = shift-only)", fontsize=9)
        self.canvas.draw_idle()
