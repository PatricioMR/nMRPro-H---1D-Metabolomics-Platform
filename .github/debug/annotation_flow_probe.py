"""Temporary: the annotation search as the analyst runs it on Linux --
import a CSV, detect, search, double-click the database (xdotool, outside)."""
import os, sys, time, traceback
sys.path.insert(0, os.getcwd())
import numpy as np
import tkinter as tk
from tkinter import filedialog

CSV = '/tmp/synthetic.csv'


def _write_csv():
    ppm = np.linspace(10.0, -0.5, 32768)
    rng = np.random.default_rng(0)
    lw = 1.2 / 600.0                                    # 1.2 Hz at 600 MHz

    def lor(c, h):
        return h / (1 + ((ppm - c) / (lw / 2)) ** 2)

    def mult(c, j_hz, n, h):
        j = j_hz / 600.0
        from math import comb
        return sum(lor(c + (k - (n - 1) / 2) * j, h * comb(n - 1, k))
                   for k in range(n))
    cols = []
    for s in range(4):
        y = (mult(1.318, 7.0, 2, 900) + mult(4.101, 7.0, 4, 200)
             + mult(1.466, 7.2, 2, 500) + lor(3.024, 1200) + lor(3.920, 800)
             + lor(3.545, 600) + rng.normal(0, 5, ppm.size))
        cols.append(y * (1 + 0.05 * s))
    with open(CSV, 'w') as f:
        f.write('ppm,' + ','.join(f'S{i}' for i in range(4)) + '\n')
        f.write('label,A,A,B,B\n')
        for i in range(ppm.size):
            f.write(f'{ppm[i]:.6f},' + ','.join(f'{c[i]:.3f}' for c in cols) + '\n')


_write_csv()
filedialog.askopenfilename = lambda *a, **k: CSV

from gui.app import App
app = App()
app._build_ui()
root = app.root
root.deiconify(); root.geometry('1200x760+0+0')
errors = []
root.report_callback_exception = lambda *a: errors.append(''.join(traceback.format_exception(*a)))
t0 = time.monotonic()


def log(*a):
    print(f'[{time.monotonic() - t0:6.1f}s]', *a, flush=True)


def tops():
    return [(w.title(), w.winfo_width(), w.winfo_height(), w.winfo_ismapped(),
             len(w.winfo_children()))
            for w in root.winfo_children()
            if isinstance(w, tk.Toplevel) and w.winfo_exists()]


def step_import():
    app._import_csv()
    log('imported', None if app.all_intensities is None else app.all_intensities.shape)
    root.after(500, step_detect)


def step_detect():
    app.stages['peak_detection'].detect_peaks()
    root.after(500, wait_peaks)


def wait_peaks(n=[0]):
    peaks = app.stages['peak_detection'].peaks
    n[0] += 1
    if peaks:
        log('peaks', len(peaks)); root.after(500, step_search)
    elif n[0] < 120:
        root.after(500, wait_peaks)
    else:
        log('no peaks after 60 s', tops())


def announce(n=[0]):
    n[0] += 1
    dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)
           and w.winfo_exists() and w.title() == 'Select Database']
    if not dlg:
        if n[0] < 40:
            root.after(250, announce)
        else:
            log('no Select Database dialog', tops())
        return
    dlg = dlg[0]; dlg.update()
    lb = next(w for c in dlg.winfo_children()
              for w in [c] + list(c.winfo_children()) if isinstance(w, tk.Listbox))
    open('/tmp/lb_xy', 'w').write(f'{lb.winfo_rootx() + 40} {lb.winfo_rooty() + 8}')
    log('picker up')


def step_search():
    app.switch_stage('annotation')
    root.after(300, announce)
    log('search starts')
    app.stages['annotation'].search_internal_db()
    log('search_internal_db returned; windows:', tops())
    for k in (2, 5, 10, 20, 40):
        root.after(k * 1000, lambda k=k: log(f'+{k}s windows:', tops()))
    root.after(45000, finish)


def finish():
    for e in errors:
        print('CALLBACK ERROR:\n' + e, flush=True)
    log('done; results window?', any('esult' in t[0] for t in tops()))
    os._exit(0)


root.after(1500, step_import)
root.mainloop()
