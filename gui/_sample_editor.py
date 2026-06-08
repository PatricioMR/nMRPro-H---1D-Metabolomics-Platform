# gui/_sample_editor.py
"""
Diálogo de edição de amostras após importação.
Colunas: Sample | Group | Color | Dil. Factor | Type
Color e Type são partilhados por grupo — mudar numa linha afecta todas do mesmo grupo.
"""
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox


_GROUP_TYPES = ['Sample', 'QC', 'IS', 'BLK', 'CTRL']

_DEFAULT_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#17BEBB', '#C73E1D', '#593C8F',
    '#E84855', '#3BB273', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F0B27A', '#82E0AA', '#F1948A', '#85929E', '#73C6B6', '#F8C471',
]


class SampleEditorDialog:
    """
    Diálogo modal per-sample.

    Color e Type são indexados por nome de grupo.
    Mudar o campo Group numa linha imediatamente actualiza o botão de cor
    e o combobox de tipo dessa linha para os valores do grupo destino.
    """

    def __init__(self, parent, samples_info, group_colors,
                 group_dilution_factors=None):
        self.samples_info           = samples_info
        self.group_colors           = group_colors
        self.group_dilution_factors = group_dilution_factors or {}

        # Estado partilhado por grupo  {group_name: StringVar}
        self._grp_color : dict[str, tk.StringVar] = {}
        self._grp_type  : dict[str, tk.StringVar] = {}

        # Widgets que precisam de ser actualizados quando a cor/tipo muda
        # {group_name: [color_btn, ...]}
        self._grp_color_btns: dict[str, list] = {}

        # Linhas da tabela
        self._rows: list[dict] = []

        dlg = tk.Toplevel(parent)
        dlg.title("Sample Editor")
        dlg.transient(parent)
        dlg.grab_set()
        dlg.focus_set()

        sw, sh = parent.winfo_screenwidth(), parent.winfo_screenheight()
        w, h   = min(820, sw - 80), min(600, sh - 80)
        dlg.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        dlg.minsize(640, 300)

        ttk.Label(dlg, text="Configure samples — group, color, dilution, correction and type",
                  font=("Arial", 10, "bold")).pack(pady=(10, 2))
        ttk.Label(dlg,
                  text="Color and Type are shared per group. Correction Factor is per sample (multiplies quantification result).",
                  font=("Arial", 8), foreground="gray").pack()
        ttk.Separator(dlg, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        # ── Área com scroll ───────────────────────────────────────────────────
        outer    = ttk.Frame(dlg)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        canvas   = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        v_scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL,   command=canvas.yview)
        self._inner = ttk.Frame(canvas)
        self._inner.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._inner, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT,   fill=tk.BOTH, expand=True)
        def _on_mousewheel(e):
            if canvas.winfo_exists():
                canvas.yview_scroll(-1 * (e.delta // 120), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        dlg.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>") if e.widget is dlg else None)

        # ── Cabeçalho ─────────────────────────────────────────────────────────
        headers = [("Sample", 26), ("Group", 14), ("Color", 10),
                   ("Dil. Factor", 11), ("Corr. Factor", 11), ("Type", 11)]
        for col, (hdr, w_ch) in enumerate(headers):
            ttk.Label(self._inner, text=hdr, font=('Arial', 9, 'bold'),
                      width=w_ch, anchor=tk.W).grid(
                row=0, column=col, padx=6, pady=(0, 4), sticky=tk.W)

        # ── Linhas ────────────────────────────────────────────────────────────
        for i, sample in enumerate(samples_info, start=1):
            self._add_row(i, sample)

        # ── Botões ────────────────────────────────────────────────────────────
        btn_frm = ttk.Frame(dlg)
        btn_frm.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btn_frm, text="Save",
                   command=lambda: self._save(dlg)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frm, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT, padx=5)

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        parent.wait_window(dlg)

    # ── Estado partilhado por grupo ───────────────────────────────────────────

    def _ensure_group(self, label: str):
        """Garante que existem StringVars para o grupo e devolve (color_var, type_var)."""
        if label not in self._grp_color:
            color = self.group_colors.get(
                label,
                _DEFAULT_COLORS[len(self._grp_color) % len(_DEFAULT_COLORS)])
            self._grp_color[label]      = tk.StringVar(value=color)
            self._grp_type[label]       = tk.StringVar(value='Sample')
            self._grp_color_btns[label] = []
        return self._grp_color[label], self._grp_type[label]

    # ── Construção de uma linha ───────────────────────────────────────────────

    def _add_row(self, row_idx: int, sample: dict):
        label    = sample.get('label', 'Sample')
        cur_type = sample.get('type', 'Sample')

        color_var, type_var = self._ensure_group(label)
        # Inicializar type com o valor da amostra (última amostra do grupo vence,
        # mas normalmente são consistentes)
        type_var.set(cur_type)

        # Col 0 — nome (read-only)
        ttk.Label(self._inner, text=sample.get('name', ''), width=26,
                  anchor=tk.W).grid(row=row_idx, column=0, padx=6, pady=2, sticky=tk.W)

        # Col 1 — Group (Entry)
        group_var = tk.StringVar(value=label)
        entry = ttk.Entry(self._inner, textvariable=group_var, width=14)
        entry.grid(row=row_idx, column=1, padx=6, pady=2, sticky=tk.W)

        # Col 2 — Color (botão)
        btn = tk.Button(self._inner, bg=color_var.get(), width=10, height=1,
                        relief=tk.RAISED, bd=2)
        btn.grid(row=row_idx, column=2, padx=6, pady=2)
        self._grp_color_btns[label].append(btn)

        # sincronizar botão quando a StringVar de cor muda
        color_var.trace_add('write',
                            lambda *_, cv=color_var, b=btn: b.config(bg=cv.get()))
        btn.config(command=lambda cv=color_var, lbl_ref=[label]:
                   self._pick_color(cv, lbl_ref[0]))

        # Col 3 — Dil. Factor (partilhado por grupo)
        dil_var = tk.DoubleVar(value=self.group_dilution_factors.get(label, 1.0))
        ttk.Entry(self._inner, textvariable=dil_var, width=11).grid(
            row=row_idx, column=3, padx=6, pady=2, sticky=tk.W)

        # Col 4 — Corr. Factor (por amostra)
        corr_var = tk.StringVar(value=str(sample.get('correction_factor', 1.0)))
        ttk.Entry(self._inner, textvariable=corr_var, width=11).grid(
            row=row_idx, column=4, padx=6, pady=2, sticky=tk.W)

        # Col 5 — Type (Combobox partilhado pelo grupo via StringVar)
        cb = ttk.Combobox(self._inner, textvariable=type_var,
                          values=_GROUP_TYPES, state='readonly', width=11)
        cb.grid(row=row_idx, column=5, padx=6, pady=2, sticky=tk.W)

        row_data = {
            'sample':    sample,
            'group_var': group_var,
            'color_var': color_var,   # aponta para o grupo actual
            'dil_var':   dil_var,
            'corr_var':  corr_var,    # por amostra
            'type_var':  type_var,    # aponta para o grupo actual
            'btn':       btn,
            'cb':        cb,
            'lbl_ref':   [label],     # lista mutável — actualizada ao mudar grupo
        }
        self._rows.append(row_data)

        # Reagir a mudanças no campo Group
        group_var.trace_add('write',
                            lambda *_, rv=row_data: self._on_group_change(rv))

    # ── Mudança de grupo ──────────────────────────────────────────────────────

    def _on_group_change(self, rv: dict):
        new_label = rv['group_var'].get().strip()
        if not new_label:
            return
        old_label = rv['lbl_ref'][0]
        if new_label == old_label:
            return

        # Remover botão do grupo antigo
        old_btns = self._grp_color_btns.get(old_label, [])
        if rv['btn'] in old_btns:
            old_btns.remove(rv['btn'])

        # Garantir que o novo grupo existe
        new_color_var, new_type_var = self._ensure_group(new_label)

        # Actualizar referências na linha
        rv['color_var'] = new_color_var
        rv['type_var']  = new_type_var
        rv['lbl_ref'][0] = new_label

        # Actualizar botão de cor e combobox para os valores do novo grupo
        rv['btn'].config(bg=new_color_var.get(),
                         command=lambda cv=new_color_var, lbl=new_label:
                         self._pick_color(cv, lbl))
        rv['cb'].config(textvariable=new_type_var)
        self._grp_color_btns[new_label].append(rv['btn'])

        # Re-ligar trace de cor ao novo color_var
        new_color_var.trace_add('write',
                                lambda *_, cv=new_color_var, b=rv['btn']:
                                b.config(bg=cv.get()))

    # ── Escolha de cor ────────────────────────────────────────────────────────

    def _pick_color(self, color_var: tk.StringVar, group_label: str):
        result = colorchooser.askcolor(
            color=color_var.get(),
            title=f"Color for group '{group_label}'")
        if result[1]:
            color_var.set(result[1])   # trace actualiza todos os botões do grupo

    # ── Guardar ───────────────────────────────────────────────────────────────

    def _save(self, dlg):
        # Cores finais por grupo
        for grp, cv in self._grp_color.items():
            self.group_colors[grp] = cv.get()

        # Actualizar cada amostra
        for rv in self._rows:
            sample    = rv['sample']
            new_label = rv['group_var'].get().strip() or sample['label']
            new_color = rv['color_var'].get()
            new_type  = rv['type_var'].get()
            try:
                new_dil = float(rv['dil_var'].get())
            except (ValueError, tk.TclError):
                new_dil = 1.0
            try:
                new_corr = float(rv['corr_var'].get())
                if new_corr <= 0:
                    new_corr = 1.0
            except (ValueError, tk.TclError):
                new_corr = 1.0

            sample['label']             = new_label
            sample['type']              = new_type
            sample['correction_factor'] = new_corr
            sample['display']           = f"{sample['name']} ({new_label})"

            self.group_colors[new_label]              = new_color
            self.group_dilution_factors[new_label]    = new_dil

        dlg.destroy()
