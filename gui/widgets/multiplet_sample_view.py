# gui/widgets/multiplet_sample_view.py
"""
Per-MULTIPLET across-samples view (MS-DIAL style) for NMR.

Opened from the annotation results window by right-clicking a multiplet line
(e.g. a triplet of Leucine). Shows that chemical shift's multiplet crop in every
sample overlaid (coloured by group), plus a per-sample table with the centre
δppm deviation, the observed J, the δJ vs the database J, the area, and whether
the multiplet was detected in that sample. A header reports in how many samples
the metabolite was found at this multiplicity.

The per-sample measurements are RE-EXTRACTED from each sample's crop (independent
of the detection mode), mirroring how MS-DIAL re-reads each sample for a feature.

The math is kept in module-level functions so it can be unit-tested without Tk.
"""
import re
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from processing.multiplet_line_selector import multiplet_line_peaks
from processing.peak_detection import estimate_noise_minimum_segment
from processing.per_sample_signal import alignment_shifts
from processing.sample_selection import included_sample_indices


# ── Column definitions ────────────────────────────────────────────────────────
COLS = [
    ('sample',  'Sample',       150, tk.W),
    ('group',   'Group',         80, tk.CENTER),
    ('present', 'Present',       70, tk.CENTER),
    ('dppm',    'δppm (mppm)',   95, tk.CENTER),
    ('jobs',    'J_obs (Hz)',    85, tk.CENTER),
    ('dj',      'δJ (Hz)',       75, tk.CENTER),
    ('area',    'Area',          95, tk.CENTER),
]


# ── Module-level math (Tk-free, unit-testable) ────────────────────────────────

def _trapz(y, x):
    try:
        return float(np.trapezoid(y, x=x))
    except AttributeError:
        return float(np.trapz(y, x=x))


def parse_j_hz(val):
    """Parse a J value like '7.1 Hz', 7.1, '7' → float Hz (or None)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else None
    m = re.search(r'[-+]?\d*\.?\d+', str(val))
    return float(m.group()) if m else None


def multiplet_window(mp_ref, peak, dp, field_mhz, default_margin=0.01):
    """Return (lo_ppm, hi_ppm, centre_ppm, db_j_hz, mult) for the multiplet.

    Window spans all predicted lines (or the matched peak's cluster), padded by a
    margin; falls back to centre ± n_lines·J/MHz when no span is available.
    """
    mp_ref = mp_ref or {}
    dp = dp or {}
    centre = mp_ref.get('peak_ppm') or mp_ref.get('db_ppm')
    if centre is None:
        centre = dp.get('ppm_center', dp.get('ppm', 0.0))
    centre = float(centre)

    db_j = parse_j_hz(dp.get('j_coupling') or dp.get('j_hz') or dp.get('J'))
    mult = str(dp.get('type') or mp_ref.get('type') or '?')

    lo = hi = None
    lines = mp_ref.get('predicted_lines') or []
    ps = [float(l['ppm']) for l in lines if isinstance(l, dict) and 'ppm' in l]
    if ps:
        lo, hi = min(ps), max(ps)
    if lo is None and peak:
        clo, chi = peak.get('cluster_lo_ppm'), peak.get('cluster_hi_ppm')
        if clo is not None and chi is not None:
            lo, hi = min(float(clo), float(chi)), max(float(clo), float(chi))
    if lo is None:
        n_lines = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quint': 5,
                   'dd': 4, 'ddd': 8, 'm': 3}.get(mult, 3)
        half = (db_j / field_mhz) * n_lines if db_j else default_margin * 3
        lo, hi = centre - half, centre + half

    margin = max((hi - lo) * 0.6, default_margin)
    return lo - margin, hi + margin, centre, db_j, mult


def measure_j(x, y, field_mhz):
    """Observed J (Hz) = median spacing of local maxima in the crop, or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 6:
        return None
    yy = np.clip(y - y.min(), 0, None)
    if yy.max() < 1e-12:
        return None
    yy = yy / yy.max()
    try:
        from scipy.signal import find_peaks
    except Exception:
        return None
    min_dist = max(2, int(x.size * 0.06))
    pks, _ = find_peaks(yy, height=0.15, prominence=0.10, distance=min_dist)
    if pks.size < 2:
        return None
    xo = np.sort(x[pks])
    seps = np.abs(np.diff(xo)) * field_mhz
    return float(np.median(seps)) if seps.size else None


def measure_sample(ppm, col, lo, hi, centre_ppm, field_mhz, db_j, noise,
                   min_snr, centre_win_ppm=None):
    """Per-sample multiplet measurement on the crop [lo, hi].

    Returns dict(present, dppm_mppm, j_obs, dj, area, height). 'present' is True
    when the central line clears min_snr·noise above the local baseline.
    """
    ppm = np.asarray(ppm, dtype=float)
    col = np.asarray(col, dtype=float)
    mask = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 3:
        return dict(present=False, dppm_mppm=float('nan'),
                    j_obs=None, dj=None, area=0.0, height=0.0)
    x, y = ppm[mask], col[mask]
    base = float(np.median(y))                    # robust local baseline
    y0 = np.clip(y - base, 0, None)

    cw = centre_win_ppm if centre_win_ppm else max((hi - lo) * 0.25, 1e-4)
    cmask = np.abs(x - centre_ppm) <= cw
    if cmask.sum() < 1:
        cmask = np.ones_like(x, dtype=bool)
    # Smooth before measuring the central-line height so a lone noise spike does
    # not pass the SNR gate (max over many points biases noise upward).
    y_meas = y0
    if y0.size >= 5:
        try:
            from scipy.ndimage import gaussian_filter1d
            y_meas = gaussian_filter1d(y0, 2.0)
        except Exception:
            pass
    yc = y_meas[cmask]
    height = float(yc.max()) if yc.size else 0.0
    apex_ppm = float(x[cmask][int(np.argmax(yc))]) if yc.size else centre_ppm

    present = bool(noise and noise > 0 and height >= float(min_snr) * float(noise))
    dppm = (apex_ppm - centre_ppm) * 1000.0
    j_obs = measure_j(x, y0, field_mhz)
    dj = (j_obs - db_j) if (j_obs is not None and db_j) else None
    area = abs(_trapz(y0, x))
    return dict(present=present, dppm_mppm=dppm, j_obs=j_obs, dj=dj,
                area=area, height=height)


# ── Widget ────────────────────────────────────────────────────────────────────

class MultipletSampleView:

    def __init__(self, parent, met, dp, mp_ref, ppm, all_intensities,
                 samples_info, app):
        self.met = met or {}
        self.dp = dp or {}
        self.mp_ref = mp_ref or {}
        self.ppm = np.asarray(ppm, dtype=float)
        self.all_intensities = all_intensities
        self.samples_info = samples_info or []
        self.app = app
        self._rows = []

        self.field_mhz = self._field_mhz()
        self.min_snr = self._min_snr()
        self.peak = self._resolve_peak()
        (self.lo, self.hi, self.centre, self.db_j,
         self.mult) = multiplet_window(self.mp_ref, self.peak, self.dp,
                                       self.field_mhz)

        self._precompute()
        self._shifts = self._alignment_shifts()

        self.win = tk.Toplevel(parent)
        name = self.met.get('name', '?')
        self.win.title(f"Multiplet {name} @ {self.centre:.4f} ppm "
                       f"[{self.mult}] — per sample")
        self.win.geometry('1040x720')
        self.win.minsize(760, 480)
        self._build()
        self._populate()
        self._draw_overlay()
        self._center()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _field_mhz(self):
        try:
            ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            return float(ann.spectrometer_var.get())
        except Exception:
            return 600.0

    def _min_snr(self):
        try:
            pd = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
            return float(pd.min_snr_var.get())
        except Exception:
            return 3.0

    def _resolve_peak(self):
        rank = self.mp_ref.get('peak_rank')
        pd = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
        if pd and rank is not None:
            return next((p for p in getattr(pd, 'peaks', [])
                         if p.get('rank') == rank), None)
        return None

    def _group_color(self, group, idx):
        gc = getattr(self.app, 'group_colors', {}) or {}
        return gc.get(group, f'C{idx % 10}')

    def _precompute(self):
        arr = self.all_intensities
        if arr is None:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[1]
        for s in range(n):
            inf = self.samples_info[s] if s < len(self.samples_info) else {}
            stype = str(inf.get('type', 'Sample')).upper()
            excluded = bool(inf.get('excluded', False))
            col = arr[:, s].astype(float)
            noise = estimate_noise_minimum_segment(col)
            m = measure_sample(self.ppm, col, self.lo, self.hi, self.centre,
                               self.field_mhz, self.db_j, noise, self.min_snr)
            m.update(
                idx=s,
                name=inf.get('name', f'S{s}'),
                group=inf.get('label', inf.get('group', '—')),
                type=stype,
                excluded=excluded,
                col=col,
                noise=noise,
            )
            self._rows.append(m)

    def _alignment_shifts(self):
        """Per-sample shift that stacks this multiplet on top of itself.

        Each sample moves by how far the centre of ITS OWN lines sits from the
        reference centre, so the pattern travels whole and the spacing that
        carries the coupling is untouched — a doublet that drifted 20 Hz still
        reads J = 7 Hz after aligning. Used only to draw; the spectra and the
        quantification windows are never touched.
        """
        pd = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
        peaks = getattr(pd, 'peaks', None) or []
        if not peaks:
            return {}
        try:
            line_pks = multiplet_line_peaks(self.mp_ref,
                                            {p.get('rank'): p for p in peaks})
            if not line_pks:
                return {}
            incl = included_sample_indices(self.samples_info, len(self._rows))
            return alignment_shifts(line_pks, included=incl,
                                    reference=self.centre)
        except Exception:
            return {}

    def _counted_rows(self):
        return [r for r in self._rows
                if r['type'] != 'BLK' and not r['excluded']]

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _build(self):
        name = self.met.get('name', '?')
        counted = self._counted_rows()
        n_present = sum(1 for r in counted if r['present'])
        jtxt = f"{self.db_j:.2f} Hz" if self.db_j else "—"

        head = ttk.Frame(self.win, padding=6)
        head.pack(fill=tk.X)
        ttk.Label(head, text=f"{name}  ·  {self.mult}  ·  {self.centre:.4f} ppm  ·  "
                             f"DB J = {jtxt}",
                  font=('Arial', 11, 'bold')).pack(anchor=tk.W)
        self._count_lbl = ttk.Label(
            head, foreground='#1a7a1a', font=('Arial', 10, 'bold'),
            text=f"Detected in {n_present} of {len(counted)} samples")
        self._count_lbl.pack(anchor=tk.W)
        ttk.Label(head, text="Per-sample crop re-extracted from each spectrum "
                             "(presence = central line ≥ min_snr·noise).",
                  font=('Arial', 8), foreground='#666').pack(anchor=tk.W)

        self._align_var = tk.BooleanVar(value=bool(self._shifts))
        _row = ttk.Frame(head)
        _row.pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(
            _row, text='Align samples on the multiplet',
            variable=self._align_var,
            command=self._draw_overlay,
            state=(tk.NORMAL if self._shifts else tk.DISABLED)).pack(side=tk.LEFT)
        if self._shifts:
            _sp = (max(self._shifts.values()) - min(self._shifts.values()))
            ttk.Label(_row,
                      text=f'   drift {_sp * self.field_mhz:.1f} Hz across samples '
                           f'· display only, spectra unchanged',
                      font=('Arial', 8), foreground='#666').pack(side=tk.LEFT)

        paned = ttk.PanedWindow(self.win, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Plot (top)
        plot_fr = ttk.Frame(paned)
        paned.add(plot_fr, weight=3)
        self._fig = Figure(figsize=(8, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_fr)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, plot_fr)

        # Table (bottom)
        tbl_fr = ttk.Frame(paned)
        paned.add(tbl_fr, weight=2)
        cols = [c[0] for c in COLS]
        self._tree = ttk.Treeview(tbl_fr, columns=cols, show='headings',
                                  selectmode='browse')
        for key, head_txt, width, anchor in COLS:
            self._tree.heading(key, text=head_txt)
            self._tree.column(key, width=width, anchor=anchor, stretch=False)
        vsb = ttk.Scrollbar(tbl_fr, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.tag_configure('present', background='#eafaea')
        self._tree.tag_configure('absent', background='#fdeaea')
        self._tree.tag_configure('blk', background='#f0f0f0', foreground='#888')

    def _populate(self):
        for r in self._rows:
            jtxt = f"{r['j_obs']:.2f}" if r['j_obs'] is not None else '—'
            djtxt = f"{r['dj']:+.2f}" if r['dj'] is not None else '—'
            dpp = r['dppm_mppm']
            dptxt = f"{dpp:+.1f}" if np.isfinite(dpp) else '—'
            if r['type'] == 'BLK' or r['excluded']:
                present_txt, tag = '—', 'blk'
            else:
                present_txt = '✓' if r['present'] else '✗'
                tag = 'present' if r['present'] else 'absent'
            self._tree.insert(
                '', 'end',
                values=(r['name'], r['group'], present_txt, dptxt, jtxt,
                        djtxt, f"{r['area']:.3e}"),
                tags=(tag,))

    def _draw_overlay(self):
        ax = self._ax
        ax.clear()
        mask = (self.ppm >= self.lo) & (self.ppm <= self.hi)
        x = self.ppm[mask]
        if x.size < 2:
            ax.text(0.5, 0.5, 'No data in window', ha='center', va='center')
            self._canvas.draw_idle()
            return

        aligned = bool(getattr(self, '_align_var', None)
                       and self._align_var.get() and self._shifts)
        present_cols = []
        for r in self._rows:
            if r['type'] == 'BLK' or r['excluded']:
                continue
            y = r['col'][mask]
            # Shifting the axis, not the data: the sample is drawn where its own
            # lines say it sits. Samples with no shift (absent here) stay put.
            xs = x - self._shifts.get(r['idx'], 0.0) if aligned else x
            ax.plot(xs, y, color=self._group_color(r['group'], r['idx']),
                    lw=0.8, alpha=0.45)
            if r['present']:
                # Averaging needs one grid, so the aligned curve is resampled
                # onto the reference axis. np.interp wants ascending x.
                if aligned and xs is not x:
                    o = np.argsort(xs)
                    present_cols.append(np.interp(x, xs[o], y[o]))
                else:
                    present_cols.append(y)

        if present_cols:
            mean_y = np.mean(np.vstack(present_cols), axis=0)
            ax.plot(x, mean_y, color='#1a1a6e', lw=2.2,
                    label='Mean (aligned)' if aligned else 'Mean (present)')

        # DB predicted lines (sticks) + reference centre
        ymax = ax.get_ylim()[1]
        for l in (self.mp_ref.get('predicted_lines') or []):
            if isinstance(l, dict) and 'ppm' in l:
                lp = float(l['ppm'])
                rel = float(l.get('rel_intensity', 1.0))
                ax.plot([lp, lp], [0, rel * ymax], color='#1a7a1a',
                        lw=1.6, alpha=0.7)
        ax.axvline(self.centre, color='red', ls=':', lw=0.9, alpha=0.6)

        ax.set_xlim(self.hi, self.lo)            # inverted ppm axis
        ax.set_xlabel('δ (ppm)')
        ax.set_ylabel('Intensity (a.u.)')
        if present_cols:
            ax.legend(fontsize=8, loc='upper right')
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _center(self):
        self.win.update_idletasks()
        try:
            w, h = self.win.winfo_width(), self.win.winfo_height()
            sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
            self.win.geometry(f'+{(sw - w)//2}+{(sh - h)//2}')
        except Exception:
            pass
