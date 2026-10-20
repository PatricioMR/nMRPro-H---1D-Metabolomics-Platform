# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/_group_editor.py
"""Mixin with group editor methods for App."""
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser


class _GroupEditorMixin:


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
