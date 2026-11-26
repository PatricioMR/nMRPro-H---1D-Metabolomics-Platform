"""Temporary: the database picker with a real double-click (xdotool)."""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
import tkinter as tk

root = tk.Tk()
root.geometry('900x600+0+0')
errors = []
root.report_callback_exception = lambda *a: errors.append(''.join(traceback.format_exception(*a)))
from gui.stages._annotation_helpers import _pick_database

def announce():
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    dlg = tops[-1]; dlg.update()
    lb = next(w for c in dlg.winfo_children() for w in [c] + list(c.winfo_children()) if isinstance(w, tk.Listbox))
    x, y = lb.winfo_rootx() + 40, lb.winfo_rooty() + 8
    open('/tmp/lb_xy', 'w').write(f'{x} {y}')
    print('picker shown; listbox item 0 at', x, y, 'dialog', dlg.winfo_width(), 'x', dlg.winfo_height(), flush=True)

root.after(1500, announce)
try:
    path = _pick_database(root)
    print('picker returned:', path, flush=True)
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel) and w.winfo_exists()]
    print('toplevels left:', len(tops), flush=True)
except Exception as e:
    print('picker raised:', type(e).__name__, e, flush=True)
for e in errors:
    print('callback error:', e, flush=True)
root.after(1500, root.destroy); root.mainloop()
