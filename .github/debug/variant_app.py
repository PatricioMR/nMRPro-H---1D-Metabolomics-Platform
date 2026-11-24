"""Temporary: run the full app with one suspect switched off.

    python variant_app.py normal|no_pil_tk|no_maximize
"""
import base64
import io
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ['NMRPRO_NO_UPDATE_CHECK'] = '1'
variant = sys.argv[1]

if variant == 'no_pil_tk':
    # Images reach Tk as PNG data through Tk's own photo loader, never
    # through Pillow's direct writes into Tk photo memory.
    import tkinter as tk
    from PIL import ImageTk

    def _photo(image=None, size=None, **kw):
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return tk.PhotoImage(data=base64.b64encode(buf.getvalue()), **kw)

    ImageTk.PhotoImage = _photo

import gui.app as A  # noqa: E402

if variant == 'no_maximize':
    A.maximize = lambda win: None

app = A.App()
app.run()
