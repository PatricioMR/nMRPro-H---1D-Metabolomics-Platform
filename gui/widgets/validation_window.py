# gui/widgets/validation_window.py
"""
Validation Window — Method Performance.

Compares quantified concentrations against user-supplied theoretical values.
Theoretical values are editable (double-click) and persist in memory
(quant_stage._validation_theoretical).  Import/export via CSV.

Color coding (absolute |Dev%|):
  green  < 10 %
  yellow 10 – 15 %
  red    > 15 %
  gray   not identified / no data
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import csv
import os


# Default theoretical concentrations (mM) for the validation mixture.
# Keys are matched against metabolite name (case-insensitive, partial).
_DEFAULT_THEORETICAL = {
    'tsp':                5.81,
    'taurine':            3.10,
    'taurina':            3.10,
    'glutamic acid':      2.41,
    'glutamico':          2.41,
    'l-glutamic':         2.41,
    'glucose':            1.86,
    'glicose':            1.86,
    'd-glucose':          1.86,
    'mannitol':           1.79,
    'manitol':            1.79,
    'd-mannitol':         1.79,
    'arabinose':          2.19,
    'l-arabinose':        2.19,
    'ascorbic':           1.90,
    'ascorbico':          1.90,
    'l-ascorbic':         1.90,
    'cysteine':           2.74,
    'cisteina':           2.74,
    'l-cysteine':         2.74,
    'caffeine':           2.11,
    'cafeina':            2.11,
    'itaconic':           2.86,
    'itaconico':          2.86,
    'itaconic acid':      2.86,
    'urea':               5.53,
    'ureia':              5.53,
}


def _lookup_default(name: str) -> float | None:
    """Return default theoretical mM for a metabolite name, or None."""
    n = name.lower().strip()
    for key, val in _DEFAULT_THEORETICAL.items():
        if key in n or n in key:
            return val
    return None


class ValidationWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    quant_stage : QuantificationStage instance
    app         : main App instance
    """

    _COLUMNS = [
        ('metabolite',   'Metabolite',          180, 'w'),
        ('theoretical',  'Theoretical (mM)',     120, 'center'),
        ('mean_mm',      'Mean mM',               90, 'center'),
        ('sd_mm',        'SD (mM)',               75, 'center'),
        ('cv_pct',       'CV%',                   60, 'center'),
        ('dev_pct',      'Dev% from theoretical', 130, 'center'),
    ]

    def __init__(self, parent, quant_stage, app):
        self.qs  = quant_stage
        self.app = app

        self.win = tk.Toplevel(parent)
        self.win.title('Validation — Method Performance')
        self.win.geometry('800x520')
        self.win.resizable(True, True)

        # {hmdb_id: float} — loaded from quant_stage or empty
        self._theoretical = dict(
            getattr(quant_stage, '_validation_theoretical', {}) or {})

        # Pre-fill defaults for any metabolite not yet assigned a theoretical value
        for r in getattr(quant_stage, '_quant_results', []) or []:
            key = str(r.get('hmdb_id', r['metabolite']))
            if key not in self._theoretical:
                dflt = _lookup_default(r['metabolite'])
                if dflt is not None:
                    self._theoretical[key] = dflt

        self._groups  = self._get_groups()
        self._sel_grp = tk.StringVar(value='All')

        self._build_ui()
        self._populate()

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _get_groups(self):
        si   = getattr(self.app, 'samples_info', None) or []
        seen, out = set(), []
        for s in si:
            lbl = s.get('label', 'Unknown')
            if str(s.get('type', lbl)).upper() == 'BLK':
                continue
            if lbl not in seen:
                seen.add(lbl)
                out.append(lbl)
        return out

    def _concs_for_result(self, r):
        """Return concentration list for the selected group (or all non-BLK)."""
        grp = self._sel_grp.get()
        cbg = r.get('_concs_by_group', {})
        if grp == 'All':
            vals = [v for g, vv in cbg.items() if g != 'BLK'
                    for v in vv if np.isfinite(v) and v > 0]
        else:
            vals = [v for v in cbg.get(grp, []) if np.isfinite(v) and v > 0]
        return vals

    def _row_data(self, r):
        """Return (mean, sd, cv_pct, dev_pct) for one result dict."""
        vals = self._concs_for_result(r)
        if not vals:
            return None, None, None, None

        mean = float(np.mean(vals))
        sd   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv   = (sd / mean * 100) if mean > 0 else float('nan')

        key  = str(r.get('hmdb_id', r['metabolite']))
        theo = self._theoretical.get(key)
        if theo is not None and theo > 0:
            dev = (mean - theo) / theo * 100
        else:
            dev = None

        return mean, sd, cv, dev

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        win = self.win

        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = ttk.Frame(win)
        tb.pack(fill='x', padx=8, pady=(8, 2))

        ttk.Label(tb, text='Group:').pack(side='left')
        grp_opts = ['All'] + self._groups
        cb = ttk.Combobox(tb, textvariable=self._sel_grp,
                          values=grp_opts, state='readonly', width=14)
        cb.pack(side='left', padx=(4, 16))
        cb.bind('<<ComboboxSelected>>', lambda _: self._populate())

        ttk.Button(tb, text='Import theoretical CSV…',
                   command=self._import_csv).pack(side='left', padx=2)
        ttk.Button(tb, text='Export table CSV…',
                   command=self._export_csv).pack(side='left', padx=2)

        ttk.Label(tb, text='Double-click Theoretical cell to edit',
                  foreground='#666666', font=('Arial', 8)).pack(
            side='right', padx=4)

        # ── Tree ─────────────────────────────────────────────────────────────
        frm = ttk.Frame(win)
        frm.pack(fill='both', expand=True, padx=8, pady=4)

        col_ids = [c[0] for c in self._COLUMNS]
        self._tree = ttk.Treeview(frm, columns=col_ids, show='headings',
                                  selectmode='browse', height=16)

        for cid, hdr, w, anc in self._COLUMNS:
            self._tree.heading(cid, text=hdr)
            self._tree.column(cid, width=w, minwidth=w, anchor=anc, stretch=(cid == 'metabolite'))

        vsb = ttk.Scrollbar(frm, orient='vertical',   command=self._tree.yview)
        hsb = ttk.Scrollbar(frm, orient='horizontal', command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        self._tree.tag_configure('green',  background='#c8e6c9')
        self._tree.tag_configure('yellow', background='#fff9c4')
        self._tree.tag_configure('red',    background='#ffcdd2')
        self._tree.tag_configure('gray',   background='#f5f5f5', foreground='#9e9e9e')
        self._tree.tag_configure('no_theo', background='#ffffff')

        self._tree.bind('<Double-1>', self._on_double_click)

        # ── Bottom bar ───────────────────────────────────────────────────────
        bot = ttk.Frame(win)
        bot.pack(fill='x', padx=8, pady=(2, 8))

        self._summary_lbl = ttk.Label(bot, text='', foreground='#333333',
                                      font=('Arial', 8))
        self._summary_lbl.pack(side='left')
        ttk.Button(bot, text='Close', command=self.win.destroy).pack(side='right')
        ttk.Button(bot, text='Save theoretical values',
                   command=self._save_theoretical).pack(side='right', padx=4)

    def _populate(self):
        tree = self._tree
        for row in tree.get_children():
            tree.delete(row)

        results = [r for r in (getattr(self.qs, '_quant_results', []) or [])
                   if not r.get('_is_is')]

        n_green = n_yellow = n_red = n_gray = 0

        for r in results:
            name  = r['metabolite']
            key   = str(r.get('hmdb_id', name))
            theo  = self._theoretical.get(key)

            mean, sd, cv, dev = self._row_data(r)

            theo_s = f'{theo:.4f}' if theo is not None else '—'
            mean_s = f'{mean:.4f}' if mean is not None else 'N/D'
            sd_s   = f'{sd:.4f}'   if sd   is not None else '—'
            cv_s   = f'{cv:.2f}'   if cv   is not None and np.isfinite(cv) else '—'

            if dev is not None and np.isfinite(dev):
                dev_s = f'{dev:+.1f}%'
                adev  = abs(dev)
                if adev < 10:
                    tag = 'green';  n_green  += 1
                elif adev < 15:
                    tag = 'yellow'; n_yellow += 1
                else:
                    tag = 'red';    n_red    += 1
            elif mean is None:
                dev_s = '—'
                tag   = 'gray';    n_gray += 1
            else:
                dev_s = '—'        # no theoretical yet
                tag   = 'no_theo'

            tree.insert('', 'end', iid=key, tags=(tag,), values=(
                name, theo_s, mean_s, sd_s, cv_s, dev_s))

        total = n_green + n_yellow + n_red
        if total:
            self._summary_lbl.config(
                text=f'✓ {n_green} (<10%)   ~ {n_yellow} (10–15%)   ✗ {n_red} (>15%)   '
                     f'N/D {n_gray}')
        else:
            self._summary_lbl.config(text='Run quantification first.')

    # ── Inline edit for Theoretical column ───────────────────────────────────

    def _on_double_click(self, event):
        region = self._tree.identify_region(event.x, event.y)
        if region != 'cell':
            return
        col = self._tree.identify_column(event.x)
        if col != '#2':   # theoretical column index
            return
        iid = self._tree.identify_row(event.y)
        if not iid:
            return

        # Get bounding box of cell
        bbox = self._tree.bbox(iid, column='theoretical')
        if not bbox:
            return
        x, y, w, h = bbox

        cur_val = self._tree.set(iid, 'theoretical')
        if cur_val == '—':
            cur_val = ''

        var = tk.StringVar(value=cur_val)
        entry = ttk.Entry(self._tree, textvariable=var, width=14)
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, 'end')

        def _commit(_event=None):
            entry.destroy()
            txt = var.get().strip().replace(',', '.')
            try:
                val = float(txt)
                self._theoretical[iid] = val
            except ValueError:
                if txt == '' and iid in self._theoretical:
                    del self._theoretical[iid]
            self._populate()

        entry.bind('<Return>',  _commit)
        entry.bind('<Tab>',     _commit)
        entry.bind('<Escape>',  lambda _: entry.destroy())
        entry.bind('<FocusOut>', _commit)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_theoretical(self):
        """Persist theoretical values on the quant_stage object."""
        self.qs._validation_theoretical = dict(self._theoretical)
        messagebox.showinfo('Saved',
                            'Theoretical values saved to session.\n'
                            'Use "Export table CSV…" for a permanent record.')

    def _import_csv(self):
        """Load theoretical values from a 2-column CSV: name/id, mM."""
        path = filedialog.askopenfilename(
            title='Import theoretical values',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if not path:
            return
        loaded = 0
        try:
            with open(path, newline='', encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue
                    key = row[0].strip()
                    try:
                        val = float(row[1].strip().replace(',', '.'))
                        self._theoretical[key] = val
                        loaded += 1
                    except ValueError:
                        pass
        except Exception as exc:
            messagebox.showerror('Error', str(exc))
            return
        self._populate()
        messagebox.showinfo('Imported', f'Loaded {loaded} theoretical value(s).')

    def _export_csv(self):
        """Export the full validation table as CSV."""
        path = filedialog.asksaveasfilename(
            title='Export validation table',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Metabolite', 'Theoretical (mM)',
                             'Mean mM', 'SD (mM)', 'CV%', 'Dev% from theoretical'])
                for iid in self._tree.get_children():
                    w.writerow(self._tree.item(iid, 'values'))
            messagebox.showinfo('Exported', f'Saved to {os.path.basename(path)}')
        except Exception as exc:
            messagebox.showerror('Error', str(exc))
