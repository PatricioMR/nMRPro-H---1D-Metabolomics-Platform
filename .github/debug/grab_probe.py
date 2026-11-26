"""Temporary: does grab_set on a just-created Toplevel fail on X11, and does
the database picker work with a double-click?"""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
import tkinter as tk

root = tk.Tk()
print('Tk', root.tk.call('info', 'patchlevel'), root.tk.call('tk', 'windowingsystem'), flush=True)

t = tk.Toplevel(root)
try:
    t.grab_set()
    print('immediate grab_set: OK', flush=True)
except tk.TclError as e:
    print('immediate grab_set: TclError:', e, flush=True)
t.destroy()

errors = []
root.report_callback_exception = lambda *a: errors.append(''.join(traceback.format_exception(*a)))

from gui.stages._annotation_helpers import _pick_database

def auto():
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    if not tops:
        print('picker: no dialog appeared', flush=True); return
    dlg = tops[-1]
    lbs = [w for w in dlg.winfo_children() for w in [w] + list(w.winfo_children()) if isinstance(w, tk.Listbox)]
    print('picker: dialog children', [c.winfo_class() for c in dlg.winfo_children()], flush=True)
    if not lbs:
        print('picker: blank dialog (no listbox)', flush=True); dlg.destroy(); return
    lb = lbs[0]; lb.selection_clear(0, 'end'); lb.selection_set(0)
    lb.event_generate('<Double-Button-1>', x=10, y=5)

root.after(1500, auto)
try:
    path = _pick_database(root)
    print('picker returned:', path, flush=True)
except Exception as e:
    print('picker raised:', type(e).__name__, e, flush=True)
for e in errors:
    print('callback error:', e, flush=True)
