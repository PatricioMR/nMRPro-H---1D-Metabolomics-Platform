"""Temporary: the database picker opened from the real application (every
global hook of v0.89.1 installed), double-clicked with xdotool."""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
import tkinter as tk

hooks = sys.argv[1] if len(sys.argv) > 1 else 'all'
if hooks == 'nofit':
    import gui.utils.window_place as wp
    wp.fit_fixed_windows = lambda root: None
    import gui.app as A
    A.fit_fixed_windows = wp.fit_fixed_windows

from gui.app import App
app = App()
app._build_ui()
root = app.root
root.deiconify(); root.geometry('1100x700+0+0'); root.update()
errors = []
root.report_callback_exception = lambda *a: errors.append(''.join(traceback.format_exception(*a)))
from gui.stages._annotation_helpers import _pick_database


def announce():
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)
            and w.winfo_exists() and w.title() == 'Select Database']
    if not tops:
        print('no Select Database dialog', flush=True); return
    dlg = tops[-1]; dlg.update()
    kids = [w for c in dlg.winfo_children() for w in [c] + list(c.winfo_children())]
    lb = next((w for w in kids if isinstance(w, tk.Listbox)), None)
    print('dialog', dlg.winfo_width(), 'x', dlg.winfo_height(), 'mapped', dlg.winfo_ismapped(),
          'children', [c.winfo_class() for c in dlg.winfo_children()], flush=True)
    if lb is None:
        print('BLANK: no listbox', flush=True); return
    print('listbox mapped', lb.winfo_ismapped(), 'size', lb.winfo_width(), 'x', lb.winfo_height(), flush=True)
    open('/tmp/lb_xy', 'w').write(f'{lb.winfo_rootx() + 40} {lb.winfo_rooty() + 8}')


root.after(2500, announce)
try:
    path = _pick_database(root)
    print('picker returned:', path, flush=True)
except Exception as e:
    print('picker raised:', type(e).__name__, e, flush=True)
for e in errors:
    print('callback error:', e, flush=True)
os._exit(0)
