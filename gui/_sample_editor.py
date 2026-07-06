# gui/_sample_editor.py
"""
Sample Editor — spreadsheet Entry grid with rectangular cell selection.
Columns: Sample (read-only) | Group | Dil. Factor | Corr. Factor | Type
Click column header  → select entire column.
Shift+Click          → extend rectangular selection.
Ctrl+C               → copy selected cells as TSV (Excel compatible).
Ctrl+V               → paste TSV; if > 1 cell selected table-paste, else normal.
"""
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import colorsys


_GROUP_TYPES  = ['Sample', 'QC', 'IS', 'BLK', 'CTRL']
_TYPE_SET     = set(_GROUP_TYPES)

_DEFAULT_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#17BEBB', '#C73E1D', '#593C8F',
    '#E84855', '#3BB273', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F0B27A', '#82E0AA', '#F1948A', '#85929E', '#73C6B6', '#F8C471',
]

# (key, header, col_width_px, kind)  kind: 'text'|'float'|'combo'
_COLS = [
    ('name',  'Sample',       170, 'text'),
    ('group', 'Group',        120, 'text'),
    ('corr',  'Corr. Factor', 110, 'float'),
    ('type',  'Type',          80, 'combo'),
]
_N_EDIT    = len(_COLS)
_KEY_COL   = {k: i for i, (k, _, _, _) in enumerate(_COLS)}
_GROUP_COL = _KEY_COL['group']

_BG_DEFAULT = '#ffffff'
_BG_SELECT  = '#cce8ff'
_BG_ANCHOR  = '#66b3ff'
_BG_HDR     = '#e0e0e0'
_BG_RDONLY  = '#f5f5f5'

_HEADER_MAP = {
    'sample': 'name', 'name': 'name', 'amostra': 'name',
    'group': 'group', 'grupo': 'group',
    'corr': 'corr', 'corr.': 'corr', 'corrfactor': 'corr',
    'correctionfactor': 'corr', 'correction': 'corr',
    'type': 'type', 'tipo': 'type',
}


def _lighten(hex_color: str, factor: float = 0.60) -> str:
    try:
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
        hh, ll, s = colorsys.rgb_to_hls(r, g, b)
        ll = min(1.0, ll + (1 - ll) * factor)
        r2, g2, b2 = colorsys.hls_to_rgb(hh, ll, s)
        return '#{:02x}{:02x}{:02x}'.format(int(r2*255), int(g2*255), int(b2*255))
    except Exception:
        return '#f5f5f5'


class SampleEditorDialog:

    def __init__(self, parent, samples_info, group_colors,
                 group_dilution_factors=None):
        self.samples_info           = samples_info
        self.group_colors           = group_colors
        self.group_dilution_factors = group_dilution_factors or {}
        self._next_color_idx        = len(group_colors)

        self._n_rows   = len(samples_info)
        self._anchor   = None       # (row, col) — int col index into _COLS
        self._selected = set()      # set of (row, col)

        # [row][col] → StringVar / [row][col] → widget
        self._vars:    list[list[tk.StringVar]] = []
        self._widgets: list[list[tk.Widget]]    = []

        dlg = tk.Toplevel(parent)
        self._dlg = dlg
        dlg.title('Sample Editor')
        dlg.transient(parent)
        dlg.grab_set()
        dlg.focus_set()

        sw, sh = parent.winfo_screenwidth(), parent.winfo_screenheight()
        w, h   = min(880, sw - 80), min(680, sh - 80)
        dlg.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')
        dlg.minsize(600, 320)

        self._build_ui(dlg)
        self._populate()

        # Escape clears selection
        dlg.bind('<Escape>', lambda e: self._clear_selection())
        dlg.bind('<Control-a>', lambda e: self._select_all())
        dlg.bind('<Control-A>', lambda e: self._select_all())

        dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)
        parent.wait_window(dlg)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self, dlg):
        ttk.Label(dlg, text='Sample Editor',
                  font=('Arial', 11, 'bold')).pack(pady=(10, 1))
        ttk.Label(dlg,
                  text='Click column header to select column  ·  '
                       'Shift+Click for range  ·  Ctrl+C copy  ·  Ctrl+V paste from Excel',
                  font=('Arial', 8), foreground='gray').pack()
        ttk.Separator(dlg, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        outer = ttk.Frame(dlg)
        outer.pack(fill=tk.BOTH, expand=True, padx=10)

        self._canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL,   command=self._canvas.yview)
        hsb = ttk.Scrollbar(outer, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas, bg=_BG_DEFAULT)
        self._cwin  = self._canvas.create_window((0, 0), window=self._inner, anchor='nw')

        self._inner.bind('<Configure>', lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox('all')))
        self._canvas.bind('<Configure>', lambda e: self._canvas.itemconfig(
            self._cwin, width=e.width))
        def _on_wheel(e):
            # bind_all is global; the binding outlives this dialog, so guard
            # against the canvas already being destroyed (TclError spam).
            if self._canvas.winfo_exists():
                self._canvas.yview_scroll(-1 * (e.delta // 120), 'units')

        self._canvas.bind_all('<MouseWheel>', _on_wheel)
        dlg.bind('<Destroy>', lambda e: self._canvas.unbind_all('<MouseWheel>'),
                 add='+')

        btn = ttk.Frame(dlg)
        btn.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btn, text='Save',
                   command=lambda: self._save(dlg)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text='Cancel',
                   command=dlg.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Separator(btn, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(btn, text='Edit Group Colors',
                   command=self._open_color_editor).pack(side=tk.LEFT, padx=5)

    # ── Grid ──────────────────────────────────────────────────────────────────

    def _populate(self):
        p   = self._inner
        pad = {'padx': 1, 'pady': 1}

        # Header row
        for c, (key, hdr, w_px, _) in enumerate(_COLS):
            anchor = 'w' if key == 'name' else 'center'
            lbl = tk.Label(p, text=hdr, bg=_BG_HDR, font=('Arial', 9, 'bold'),
                           width=w_px // 8, anchor=anchor,
                           relief='groove', cursor='hand2')
            lbl.grid(row=0, column=c, sticky='nsew', **pad)
            lbl.bind('<Button-1>', lambda e, c=c: self._select_column(c))

        for c, (_, _, w_px, _) in enumerate(_COLS):
            p.columnconfigure(c, minsize=w_px)

        # Data rows
        for r, sample in enumerate(self.samples_info):
            grp = sample.get('label', '')
            if grp and grp not in self.group_colors:
                self.group_colors[grp] = _DEFAULT_COLORS[
                    self._next_color_idx % len(_DEFAULT_COLORS)]
                self._next_color_idx += 1

            row_bg = _lighten(self.group_colors.get(grp, '#cccccc'), 0.65)

            row_vars    = []
            row_widgets = []

            for c, (key, _, w_px, kind) in enumerate(_COLS):
                var = tk.StringVar()
                if key == 'name':
                    var.set(sample.get('name', ''))
                elif key == 'group':
                    var.set(grp)
                elif key == 'corr':
                    var.set(str(sample.get('correction_factor', 1.0)))
                elif key == 'type':
                    var.set(sample.get('type', 'Sample'))

                if kind == 'combo':
                    w = tk.OptionMenu(p, var, *_GROUP_TYPES)
                    w.configure(
                        bg=row_bg, activebackground=_BG_SELECT,
                        relief='flat', borderwidth=1,
                        highlightthickness=1,
                        highlightbackground='#cccccc',
                        anchor='w', width=w_px // 8,
                        indicatoron=True)
                    w['menu'].configure(bg='white')
                else:
                    w = tk.Entry(p, textvariable=var, bg=row_bg,
                                 relief='flat', borderwidth=0,
                                 highlightthickness=1,
                                 highlightbackground='#cccccc',
                                 highlightcolor='#66b3ff')

                w.grid(row=r + 1, column=c, sticky='nsew', **pad)

                # Selection bindings — do NOT return 'break' on plain click
                w.bind('<Button-1>',
                       lambda ev, r=r, c=c: self._on_click(r, c))
                w.bind('<Shift-Button-1>',
                       lambda ev, r=r, c=c: (self._on_shift_click(r, c), 'break')[1])
                w.bind('<Control-Button-1>',
                       lambda ev, r=r, c=c: (self._on_ctrl_click(r, c), 'break')[1])

                # Keyboard navigation (Entry only)
                if kind != 'combo':
                    w.bind('<Tab>',    lambda ev, r=r, c=c: self._move(r, c, 0, 1))
                    w.bind('<Return>', lambda ev, r=r, c=c: self._move(r, c, 1, 0))
                    w.bind('<Up>',     lambda ev, r=r, c=c: self._move(r, c, -1, 0))
                    w.bind('<Down>',   lambda ev, r=r, c=c: self._move(r, c, 1, 0))

                # Ctrl+C/V — fires at widget level (before class binding)
                w.bind('<Control-c>', self._ctrl_c)
                w.bind('<Control-C>', self._ctrl_c)
                w.bind('<Control-v>', self._ctrl_v)
                w.bind('<Control-V>', self._ctrl_v)

                if key == 'group':
                    var.trace_add('write', lambda *a, r=r:
                                  self._dlg.after(50, lambda: self._update_row_bg(r)))

                row_vars.append(var)
                row_widgets.append(w)

            self._vars.append(row_vars)
            self._widgets.append(row_widgets)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_click(self, row, col):
        self._anchor   = (row, col)
        self._selected = {(row, col)}
        self._refresh_selection()
        # do NOT return 'break' — Entry must stay editable

    def _on_shift_click(self, row, col):
        if self._anchor is None:
            self._anchor = (row, col)
        r0, c0 = self._anchor
        self._selected = {
            (r, c)
            for r in range(min(r0, row), max(r0, row) + 1)
            for c in range(min(c0, col), max(c0, col) + 1)
        }
        self._refresh_selection()

    def _on_ctrl_click(self, row, col):
        cell = (row, col)
        if cell in self._selected:
            self._selected.discard(cell)
        else:
            self._selected.add(cell)
            self._anchor = cell
        self._refresh_selection()

    def _select_column(self, col):
        self._anchor   = (0, col)
        self._selected = {(r, col) for r in range(self._n_rows)}
        self._refresh_selection()
        if self._n_rows:
            self._widgets[0][col].focus_set()

    def _select_all(self):
        self._anchor   = (0, 0)
        self._selected = {(r, c)
                          for r in range(self._n_rows)
                          for c in range(_N_EDIT)}
        self._refresh_selection()

    def _clear_selection(self):
        self._selected = set()
        self._anchor   = None
        self._refresh_selection()

    def _refresh_selection(self):
        for r in range(self._n_rows):
            row_bg = self._get_row_bg(r)
            for c in range(_N_EDIT):
                w = self._widgets[r][c]
                if (r, c) == self._anchor:
                    bg = _BG_ANCHOR
                elif (r, c) in self._selected:
                    bg = _BG_SELECT
                else:
                    bg = row_bg
                try:
                    w.configure(background=bg)
                except tk.TclError:
                    pass

    def _get_row_bg(self, row: int) -> str:
        grp = self._vars[row][_GROUP_COL].get()
        return _lighten(self.group_colors.get(grp, '#cccccc'), 0.65)

    def _update_row_bg(self, row: int):
        bg = self._get_row_bg(row)
        for c in range(_N_EDIT):
            if (row, c) not in self._selected and (row, c) != self._anchor:
                try:
                    self._widgets[row][c].configure(background=bg)
                except tk.TclError:
                    pass

    # ── Navigation ────────────────────────────────────────────────────────────

    def _move(self, row, col, dr, dc):
        nr = max(0, min(self._n_rows - 1, row + dr))
        nc = max(0, min(_N_EDIT - 1, col + dc))
        self._on_click(nr, nc)
        self._widgets[nr][nc].focus_set()
        return 'break'

    # ── Clipboard ─────────────────────────────────────────────────────────────

    def _ctrl_c(self, event):
        if len(self._selected) > 1:
            self._copy_selection()
            return 'break'
        elif len(self._selected) == 1:
            # Copy full cell value even without text selection in Entry
            r, c = next(iter(self._selected))
            self._dlg.clipboard_clear()
            self._dlg.clipboard_append(self._vars[r][c].get())
            return 'break'
        # 0 selected: let widget default handle

    def _ctrl_v(self, event):
        if len(self._selected) > 1:
            # Multi-cell selection: always table-paste
            self._paste_clipboard()
            return 'break'
        elif self._anchor is not None:
            # Single anchor: table-paste from anchor position
            self._paste_clipboard()
            return 'break'
        # No anchor: let widget default paste

    def _copy_selection(self):
        if not self._selected:
            return
        rows = sorted({r for r, c in self._selected})
        cols = sorted({c for r, c in self._selected})
        lines = []
        for r in rows:
            lines.append('\t'.join(
                self._vars[r][c].get() if (r, c) in self._selected else ''
                for c in cols))
        self._dlg.clipboard_clear()
        self._dlg.clipboard_append('\n'.join(lines))

    def _paste_clipboard(self):
        try:
            raw = self._dlg.clipboard_get()
        except tk.TclError:
            messagebox.showwarning('Paste', 'Clipboard empty.', parent=self._dlg)
            return

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return
        rows_tsv = [ln.split('\t') for ln in lines]

        # Detect header row
        first_norm = [cell.strip().lower().replace(' ', '').replace('.', '')
                      for cell in rows_tsv[0]]
        has_header = any(f in _HEADER_MAP for f in first_norm)

        if has_header:
            col_map   = {i: _HEADER_MAP[f]
                         for i, f in enumerate(first_norm) if f in _HEADER_MAP}
            data_rows = rows_tsv[1:]
        else:
            # Positional starting from anchor column
            anchor_col = self._anchor[1] if self._anchor else 0
            positional = [k for k, _, _, _ in _COLS]
            col_map    = {}
            for i in range(len(rows_tsv[0])):
                gi = anchor_col + i
                if gi < _N_EDIT:
                    col_map[i] = positional[gi]
            data_rows = rows_tsv

        if not col_map:
            messagebox.showwarning(
                'Paste',
                'No recognisable columns.\n'
                'Accepted headers: Group, Dil. Factor, Corr. Factor, Type.',
                parent=self._dlg)
            return

        start_row    = self._anchor[0] if self._anchor else 0
        key_to_col   = {k: i for i, (k, _, _, _) in enumerate(_COLS)}
        errors       = []
        applied      = 0

        for offset, tsv_row in enumerate(data_rows):
            r = start_row + offset
            if r >= self._n_rows:
                break
            for ci, field in col_map.items():
                if ci >= len(tsv_row):
                    continue
                val = tsv_row[ci].strip()
                if not val:
                    continue
                col_idx = key_to_col.get(field)
                if col_idx is None:
                    continue
                try:
                    if field == 'corr':
                        val = val.replace(',', '.')
                        float(val)
                    elif field == 'type' and val not in _TYPE_SET:
                        errors.append(
                            f'Row {r+1}: unknown type "{val}" '
                            f'(accepted: {", ".join(_GROUP_TYPES)})')
                        continue
                    self._vars[r][col_idx].set(val)
                except ValueError:
                    errors.append(f'Row {r+1}: invalid number "{val}"')
            applied += 1

        self._refresh_selection()

        # Surface parse errors only — successful pastes apply silently.
        if errors:
            messagebox.showwarning(
                'Paste', 'Warnings:\n' + '\n'.join(errors[:10]),
                parent=self._dlg)

    # ── Color editor ──────────────────────────────────────────────────────────

    def _open_color_editor(self):
        groups, seen = [], set()
        for row_vars in self._vars:
            g = row_vars[_GROUP_COL].get().strip()
            if g and g not in seen:
                groups.append(g)
                seen.add(g)

        dlg = tk.Toplevel(self._dlg)
        dlg.title('Group Colors')
        dlg.transient(self._dlg)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text='Click a button to change group color.',
                  font=('Arial', 9)).grid(row=0, column=0, columnspan=2,
                                          padx=12, pady=(10, 6))
        for i, grp in enumerate(groups, start=1):
            color = self.group_colors.get(grp, '#cccccc')
            ttk.Label(dlg, text=grp, width=20, anchor=tk.W).grid(
                row=i, column=0, padx=12, pady=3, sticky=tk.W)
            btn = tk.Button(dlg, bg=color, width=10, height=1,
                            relief=tk.RAISED, bd=2)
            btn.configure(command=lambda g=grp, b=btn: self._pick_color(g, b))
            btn.grid(row=i, column=1, padx=12, pady=3)

        ttk.Button(dlg, text='Done',
                   command=lambda: (self._refresh_selection(), dlg.destroy())
                   ).grid(row=len(groups) + 1, column=0, columnspan=2, pady=10)

    def _pick_color(self, group, btn):
        result = colorchooser.askcolor(
            color=self.group_colors.get(group, '#2E86AB'),
            title=f"Color for '{group}'",
            parent=self._dlg)
        if result[1]:
            self.group_colors[group] = result[1]
            btn.configure(bg=result[1])

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self, dlg):
        group_type = {}
        for row_vars in self._vars:
            grp = row_vars[_GROUP_COL].get().strip()
            typ = row_vars[_KEY_COL['type']].get().strip()
            if typ not in _TYPE_SET:
                typ = 'Sample'
            group_type[grp] = typ

        for r, sample in enumerate(self.samples_info):
            row_vars = self._vars[r]
            name = row_vars[_KEY_COL['name']].get().strip() or sample.get('name', '')
            grp = row_vars[_GROUP_COL].get().strip() or sample.get('label', '')
            try:
                corr = float(row_vars[_KEY_COL['corr']].get().replace(',', '.'))
                if corr <= 0:
                    corr = 1.0
            except ValueError:
                corr = 1.0

            sample['name']              = name
            sample['label']             = grp
            sample['type']              = group_type.get(grp, 'Sample')
            sample['correction_factor'] = corr
            sample['display']           = f"{name} ({grp})"

            if grp and grp not in self.group_colors:
                self.group_colors[grp] = _DEFAULT_COLORS[
                    self._next_color_idx % len(_DEFAULT_COLORS)]
                self._next_color_idx += 1

            # dil removed from UI; keep backend key at 1.0 so quant is unaffected
            self.group_dilution_factors[grp] = 1.0

        dlg.destroy()
