# gui/_group_editor.py
"""Mixin with group editor methods for App."""
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser


class _GroupEditorMixin:

    def open_group_editor(self):
        if not self.samples_info:
            messagebox.showinfo("Info", "Please load data first to see available groups.")
            return

        _GROUP_TYPES = ['Sample', 'QC', 'IS', 'BLK', 'CTRL']

        groups = sorted(set([s['label'] for s in self.samples_info]))
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Groups")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_set()

        # Center on screen
        _dw, _dh = 780, 420
        _sw = self.root.winfo_screenwidth()
        _sh = self.root.winfo_screenheight()
        dialog.geometry(f"{_dw}x{_dh}+{(_sw-_dw)//2}+{(_sh-_dh)//2}")

        ttk.Label(dialog, text="Configure groups — colors, type and dilution factor:",
                  font=("Arial", 10, "bold")).pack(pady=(10, 4))

        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=6)

        canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.group_entries    = {}
        self.group_color_vars = {}
        self.group_dil_vars   = {}
        self.group_type_vars  = {}
        self._color_buttons   = {}

        grid = ttk.Frame(scrollable_frame)
        grid.pack(fill=tk.X, padx=4, pady=4)

        COL_W = [14, 14, 8, 6, 10, 10]
        for col, (hdr, anchor) in enumerate([
            ('Original Group', tk.W), ('New Name', tk.W),
            ('Color',   tk.CENTER), ('Preview', tk.CENTER),
            ('Dil. Factor', tk.CENTER), ('Type', tk.W),
        ]):
            ttk.Label(grid, text=hdr, font=('Arial', 9, 'bold'),
                      width=COL_W[col], anchor=anchor).grid(
                row=0, column=col, padx=6, pady=(0, 4), sticky=tk.W)

        for i, group in enumerate(groups, start=1):
            color = self.group_colors.get(group, '#000000')
            color_var = tk.StringVar(value=color)
            self.group_color_vars[group] = color_var

            ttk.Label(grid, text=group, width=COL_W[0], anchor=tk.W).grid(
                row=i, column=0, padx=6, pady=2, sticky=tk.W)

            name_var = tk.StringVar(value=group)
            ttk.Entry(grid, textvariable=name_var, width=COL_W[1]).grid(
                row=i, column=1, padx=6, pady=2, sticky=tk.W)
            self.group_entries[group] = name_var

            btn = tk.Button(grid, bg=color, width=COL_W[2], height=1,
                            relief=tk.RAISED, bd=2)
            btn.config(command=lambda g=group, v=color_var, b=btn:
                       self.choose_color(g, v, b))
            btn.grid(row=i, column=2, padx=6, pady=2)
            self._color_buttons[group] = btn

            preview = tk.Label(grid, bg=color, width=COL_W[3],
                               relief=tk.SUNKEN, bd=1)
            preview.grid(row=i, column=3, padx=6, pady=2, ipady=8, sticky=tk.EW)
            color_var._preview = preview

            dil_var = tk.StringVar(
                value=str(self.group_dilution_factors.get(group, 1.0)))
            ttk.Entry(grid, textvariable=dil_var, width=COL_W[4]).grid(
                row=i, column=4, padx=6, pady=2, sticky=tk.W)
            self.group_dil_vars[group] = dil_var

            # Infer current type from samples (majority vote within group)
            group_samples = [s for s in self.samples_info if s['label'] == group]
            _cur_type = 'BLK' if group.upper() == 'BLK' else 'Sample'
            if group_samples:
                from collections import Counter
                _cur_type = Counter(
                    s.get('type', 'Sample') for s in group_samples).most_common(1)[0][0]
            type_var = tk.StringVar(value=_cur_type)
            ttk.Combobox(grid, textvariable=type_var, values=_GROUP_TYPES,
                         state='readonly', width=COL_W[5]).grid(
                row=i, column=5, padx=6, pady=2, sticky=tk.W)
            self.group_type_vars[group] = type_var

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Button(btn_frame, text="Save",
                   command=lambda: self.save_group_changes(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Default",
                   command=self.reset_group_colors).pack(side=tk.RIGHT, padx=5)

    def choose_color(self, group, color_var, btn=None):
        current = color_var.get()
        color = colorchooser.askcolor(color=current, title=f"Choose color for {group}")
        if color[1]:
            color_var.set(color[1])
            if btn:
                btn.config(bg=color[1])
            preview = getattr(color_var, '_preview', None)
            if preview:
                preview.config(bg=color[1])

    def save_group_changes(self, dialog):
        group_mapping = {}
        for old, name_var in self.group_entries.items():
            new = name_var.get().strip()
            if new and new != old:
                group_mapping[old] = new

        for old, color_var in self.group_color_vars.items():
            new_color = color_var.get()
            self.group_colors[old] = new_color
            if old in group_mapping:
                self.group_colors[group_mapping[old]] = new_color

        for old, dil_var in getattr(self, 'group_dil_vars', {}).items():
            try:
                factor = float(dil_var.get())
            except ValueError:
                factor = 1.0
            new_name = group_mapping.get(old, old)
            self.group_dilution_factors[new_name] = factor

        # Apply chosen type to every sample belonging to each group
        for old, type_var in getattr(self, 'group_type_vars', {}).items():
            chosen_type = type_var.get()
            for sample in self.samples_info:
                if sample['label'] == old:
                    sample['type'] = chosen_type

        if group_mapping:
            for sample in self.samples_info:
                old_label = sample['label']
                if old_label in group_mapping:
                    sample['label'] = group_mapping[old_label]
                    sample['display'] = f"{sample['name']} ({sample['label']})"

        if 'visualization' in self.stages:
            self.stages['visualization'].refresh_sample_list()

        self.update_plot()
        dialog.destroy()

    def _assign_group_colors(self):
        """Assign colors to all groups present in samples_info."""
        # Paleta base com cores distintas e saturadas
        _PALETTE = [
            '#2E86AB', '#A23B72', '#F18F01', '#17BEBB', '#C73E1D', '#593C8F',
            '#E84855', '#3BB273', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F0B27A', '#82E0AA', '#F1948A', '#85929E', '#73C6B6', '#F8C471',
        ]
        groups = sorted(set(s['label'] for s in self.samples_info))
        palette_idx = len([g for g in self.group_colors
                           if g not in ('CTRL', 'G1', 'G2', 'G3', 'BLK', 'Unknown')])
        for group in groups:
            if group not in self.group_colors:
                color = _PALETTE[palette_idx % len(_PALETTE)]
                used = set(self.group_colors.values())
                offset = 0
                while color in used and offset < len(_PALETTE):
                    offset += 1
                    color = _PALETTE[(palette_idx + offset) % len(_PALETTE)]
                self.group_colors[group] = color
                palette_idx += 1

    def reset_group_colors(self):
        self.group_colors = {
            'CTRL': '#2E86AB',
            'G1': '#A23B72',
            'G2': '#F18F01',
            'G3': '#17BEBB',
            'BLK': '#C73E1D',
            'Unknown': '#593C8F'
        }
        if 'visualization' in self.stages:
            self.stages['visualization'].refresh_sample_list()
        if self.samples_info:
            self.update_plot()
