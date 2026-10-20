# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/_bruker_import_dialog.py
"""Diálogo de opções para importação de dados Bruker."""
import tkinter as tk
from tkinter import ttk, messagebox


class _BrukerImportDialog:
    """
    Opções de importação Bruker: expno, ppm range, nº de pontos.

    result = {'expno': int, 'procno': int,
              'ppm_min': float, 'ppm_max': float, 'n_points': int}
    ou None se cancelado.
    """

    def __init__(self, parent):
        self.result = None

        dlg = tk.Toplevel(parent)
        dlg.title("Bruker Import Options")
        dlg.resizable(False, False)
        dlg.transient(parent)
        dlg.grab_set()

        w, h = 360, 270
        px = parent.winfo_rootx() + (parent.winfo_width()  - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{px}+{py}")

        pad = dict(padx=14, pady=5)

        ttk.Label(dlg, text="Bruker Import Options",
                  font=("Arial", 10, "bold")).pack(pady=(14, 4))
        ttk.Separator(dlg, orient='horizontal').pack(fill=tk.X, padx=10)

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.X)

        # ── Experiment / Processing number ────────────────────────────────────
        ttk.Label(frm, text="Experiment No. (expno):",
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, **pad)
        self._expno = tk.IntVar(value=1)
        ttk.Spinbox(frm, textvariable=self._expno, from_=1, to=999,
                    width=6).grid(row=0, column=1, sticky=tk.W, **pad)

        ttk.Label(frm, text="Processing No. (procno):").grid(
            row=1, column=0, sticky=tk.W, **pad)
        self._procno = tk.IntVar(value=1)
        ttk.Spinbox(frm, textvariable=self._procno, from_=1, to=99,
                    width=6).grid(row=1, column=1, sticky=tk.W, **pad)

        ttk.Separator(frm, orient='horizontal').grid(
            row=2, column=0, columnspan=2, sticky=tk.EW, pady=6)

        # ── Eixo ppm ─────────────────────────────────────────────────────────
        ttk.Label(frm, text="ppm min (right):").grid(row=3, column=0, sticky=tk.W, **pad)
        self._ppm_min = tk.DoubleVar(value=-0.5)
        ttk.Entry(frm, textvariable=self._ppm_min, width=10).grid(
            row=3, column=1, sticky=tk.W, **pad)

        ttk.Label(frm, text="ppm max (left):").grid(row=4, column=0, sticky=tk.W, **pad)
        self._ppm_max = tk.DoubleVar(value=15.0)
        ttk.Entry(frm, textvariable=self._ppm_max, width=10).grid(
            row=4, column=1, sticky=tk.W, **pad)

        ttk.Label(frm, text="Points (interpolation):").grid(
            row=5, column=0, sticky=tk.W, **pad)
        self._n_points = tk.IntVar(value=32768)
        ttk.Combobox(frm, textvariable=self._n_points, width=8,
                     values=[8192, 16384, 32768, 65536],
                     state='readonly').grid(row=5, column=1, sticky=tk.W, **pad)

        btn_frm = ttk.Frame(dlg)
        btn_frm.pack(pady=10)
        ttk.Button(btn_frm, text="Import",
                   command=lambda: self._ok(dlg)).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frm, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT, padx=6)

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        parent.wait_window(dlg)

    def _ok(self, dlg):
        try:
            expno   = int(self._expno.get())
            procno  = int(self._procno.get())
            ppm_min = float(self._ppm_min.get())
            ppm_max = float(self._ppm_max.get())
            n_pts   = int(self._n_points.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Error", "Valores inválidos.", parent=dlg)
            return
        if ppm_min >= ppm_max:
            messagebox.showerror("Error", "ppm min deve ser menor que ppm max.",
                                 parent=dlg)
            return
        self.result = {
            'expno': expno, 'procno': procno,
            'ppm_min': ppm_min, 'ppm_max': ppm_max,
            'n_points': n_pts,
        }
        dlg.destroy()
