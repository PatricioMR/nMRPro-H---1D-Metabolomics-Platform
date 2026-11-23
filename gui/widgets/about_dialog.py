# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Help → About: version, where to report bugs and suggestions, and the
GNU AGPL notice an interactive program has to show."""
import tkinter as tk
import webbrowser
from tkinter import ttk

from app_paths import (APP_VERSION, CONTACT_EMAIL, COPYRIGHT, ISSUES_URL,
                       LICENSE_TERMS, LINKEDIN_URL, SOURCE_URL, asset_path)
from gui.utils.window_place import center_on

_LOGO_PX = 120
_TEXT_PX = 620
_LINK_FG = '#1a5fb4'

CHANNELS = (
    ('E-mail',        CONTACT_EMAIL,  f'mailto:{CONTACT_EMAIL}'),
    ('GitHub issues', ISSUES_URL,     ISSUES_URL),
    ('GitHub',        SOURCE_URL,     SOURCE_URL),
    ('LinkedIn',      LINKEDIN_URL,   LINKEDIN_URL),
)


class AboutDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title('About NMRPro-H')
        self.transient(parent)
        self.resizable(False, False)

        body = ttk.Frame(self, padding=(22, 18))
        body.pack(fill=tk.BOTH, expand=True)

        head = ttk.Frame(body)
        head.pack(fill=tk.X)
        self._logo = self._load_logo()
        if self._logo is not None:
            ttk.Label(head, image=self._logo).pack(side=tk.LEFT, padx=(0, 16))
        title = ttk.Frame(head)
        title.pack(side=tk.LEFT, anchor='center')
        ttk.Label(title, text=APP_VERSION,
                  font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        ttk.Label(title, text='1D ¹H NMR metabolomics platform',
                  font=('Segoe UI', 10)).pack(anchor='w', pady=(2, 0))
        ttk.Label(title, text=COPYRIGHT, foreground='#555',
                  font=('Segoe UI', 9)).pack(anchor='w', pady=(6, 0))

        ttk.Separator(body).pack(fill=tk.X, pady=14)

        ttk.Label(body, text='Bugs and suggestions',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(body, wraplength=_TEXT_PX, justify='left',
                  text='Bug reports and suggestions can be sent through any of '
                       'the channels below.').pack(anchor='w', pady=(2, 8))

        links = ttk.Frame(body)
        links.pack(fill=tk.X)
        for row, (label, text, target) in enumerate(CHANNELS):
            ttk.Label(links, text=f'{label}:', font=('Segoe UI', 9, 'bold')
                      ).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=2)
            self._link(links, text, target).grid(row=row, column=1, sticky='w')

        ttk.Label(body, text='Contributions and collaborations',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(14, 0))
        ttk.Label(body, wraplength=_TEXT_PX, justify='left',
                  text='Contributions and collaborations are welcome: get in '
                       'touch through any of the channels above or open a pull '
                       'request on GitHub.').pack(anchor='w', pady=(2, 0))

        ttk.Separator(body).pack(fill=tk.X, pady=14)

        ttk.Label(body, text=LICENSE_TERMS, wraplength=_TEXT_PX, justify='left',
                  foreground='#555', font=('Segoe UI', 8)).pack(anchor='w')

        close = ttk.Button(body, text='Close', command=self.destroy)
        close.pack(anchor='e', pady=(14, 0))

        self.bind('<Escape>', lambda _e: self.destroy())
        center_on(self, parent)
        close.focus_set()

    @staticmethod
    def _load_logo():
        try:
            from PIL import Image, ImageTk
            img = Image.open(asset_path('utils', 'Logo_NMRPro-H.png')).convert('RGBA')
            img.thumbnail((_LOGO_PX, _LOGO_PX), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    @staticmethod
    def _link(parent, text, target):
        link = tk.Label(parent, text=text, fg=_LINK_FG, cursor='hand2',
                        font=('Segoe UI', 9, 'underline'),
                        bg=ttk.Style().lookup('TFrame', 'background') or None)
        link.bind('<Button-1>', lambda _e: webbrowser.open(target))
        return link
