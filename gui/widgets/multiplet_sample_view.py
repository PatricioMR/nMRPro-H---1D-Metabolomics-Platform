# gui/widgets/multiplet_sample_view.py
"""
Per-MULTIPLET across-samples view (MS-DIAL style) for NMR.

Opened from the annotation results window by right-clicking a multiplet line
(e.g. a triplet of Leucine). Shows that chemical shift's multiplet crop in every
sample overlaid (coloured by group), plus a per-sample table with the centre
δppm deviation, the observed J, the δJ vs the database J, the area, and whether
the multiplet was detected in that sample. A header reports in how many samples
the metabolite was found at this multiplicity.

The per-sample measurements are NOT made here. The annotation search settles
them — reading each sample from what detection recorded — and this window
renders what it stored. It used to re-extract them from a crop of the spectrum
and the two disagreed, because the crop is padded for context and the estimate
picked up whatever neighbouring resonances fell inside it.

The crop geometry is kept in module-level functions so it can be unit-tested
without Tk.
"""
import re
import tkinter as tk
from tkinter import ttk

from processing.annotation_scoring import SCORING_DEFAULTS
from processing.per_sample_signal import (
    group_means, percent_deviation, relative_sd_percent)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)



# ── Column definitions ────────────────────────────────────────────────────────
COLS = [
    ('sample',   'Sample',       150, tk.W),
    ('group',    'Group',         80, tk.CENTER),
    ('present',  'Annotated',    110, tk.CENTER),
    ('dppm',     'Δppm',          85, tk.CENTER),
    ('dppm_sd',  'DP %',          65, tk.CENTER),
    ('jobs',     'J_obs (Hz)',    85, tk.CENTER),
    ('dj',       'ΔJ (Hz)',       75, tk.CENTER),
    ('dj_sd',    'DP %',          65, tk.CENTER),
    ('area',     'Area',          95, tk.CENTER),
]

# The plot is pinned to the table's width so the two read as one figure: a
# sample's trace sits above its own row's numbers. Derived from the columns
# themselves, so adding one moves both.
TABLE_WIDTH_PX = sum(c[2] for c in COLS) + 18     # + vertical scrollbar


# ── Module-level math (Tk-free, unit-testable) ────────────────────────────────

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
        self.win.geometry(f'{TABLE_WIDTH_PX + 40}x760')
        self.win.minsize(TABLE_WIDTH_PX + 20, 520)
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
        """Assemble the table rows. Reads; measures nothing.

        Every number here — presence, δppm, observed J, δJ, area — was settled
        by the annotation search, which reads each sample from what detection
        recorded. This window used to re-derive them from a crop of the
        spectrum, and the two disagreed: the crop is padded by 60% per side, a
        50 Hz window for a 7 Hz doublet, so the J it reported was the spacing of
        whichever neighbouring resonances fell inside it. Cytidine, hippuric
        acid and 2'-deoxyuridine, three signals sharing a region, all read
        4.76 Hz. Median |ΔJ| was 1.46 Hz that way against 0.77 Hz from the
        annotated lines.

        A project annotated before the per-sample pass carries no rows: the
        table then shows dashes and the header says to run the search again,
        rather than this window quietly inventing a second set of numbers.
        """
        arr = self.all_intensities
        if arr is None:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stored = {int(r['sample_idx']): r
                  for r in (self.mp_ref.get('per_sample') or [])
                  if r.get('sample_idx') is not None}
        self.has_readings = bool(stored)
        for s in range(arr.shape[1]):
            inf = self.samples_info[s] if s < len(self.samples_info) else {}
            r = stored.get(s, {})
            centre = r.get('ppm_centre')
            j_obs = r.get('j_obs_hz')
            self._rows.append(dict(
                idx=s,
                name=inf.get('name', f'S{s}'),
                group=inf.get('label', inf.get('group', '—')),
                type=str(inf.get('type', 'Sample')).upper(),
                excluded=bool(inf.get('excluded', False)),
                col=arr[:, s].astype(float),
                present=bool(r.get('present')),
                ppm_centre=centre,
                dppm=(centre - self.centre if centre is not None else None),
                j_obs=j_obs,
                dj=(j_obs - self.db_j
                    if (j_obs is not None and self.db_j) else None),
                area=r.get('area'),
                collapsed=bool(r.get('collapsed')),
                shift=r.get('align_shift_ppm'),
            ))

    def _alignment_shifts(self):
        """The per-sample display shift the annotation search already computed.

        Each sample moves by how far the centre of its own lines sits from the
        reference, so the pattern travels whole and the coupling is untouched.
        """
        return {r['idx']: r['shift'] for r in self._rows
                if r.get('shift') is not None}

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
        if getattr(self, 'has_readings', False):
            _note, _fg = ("Values as measured by the annotation search — this "
                          "window displays, it does not re-measure."), '#666'
        else:
            _note, _fg = ("This annotation predates the per-sample pass, so "
                          "there is nothing to show per sample. Run the "
                          "annotation search again."), '#a33'
        ttk.Label(head, text=_note, font=('Arial', 8),
                  foreground=_fg).pack(anchor=tk.W)

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
        holder = ttk.Frame(plot_fr, width=TABLE_WIDTH_PX)
        holder.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        holder.pack_propagate(False)      # keep the width, let height follow
        self._fig = Figure(figsize=(TABLE_WIDTH_PX / 100.0, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=holder)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, holder)

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
        self._tree.tag_configure('unverified', background='#fff6e0')
        self._tree.tag_configure('summary', background='#e6e9f5',
                                 font=('Arial', 9, 'bold'))

    def _populate(self):
        # The deviation is read against the samples a sample belongs with: a
        # chemical shift moves with pH and ionic strength, which is exactly what
        # the groups differ in, so a study-wide mean would call a whole group an
        # outlier. Means come from the present samples of each group.
        present_rows = [r for r in self._counted_rows() if r['present']]
        ppm_mean = group_means(present_rows, 'ppm_centre')
        j_mean = group_means(present_rows, 'j_obs')

        for r in self._rows:
            jtxt = f"{r['j_obs']:.2f}" if r['j_obs'] is not None else '—'
            djtxt = f"{r['dj']:+.2f}" if r['dj'] is not None else '—'
            dptxt = f"{r['dppm']:+.4f}" if r['dppm'] is not None else '—'
            if r['type'] == 'BLK' or r['excluded']:
                ann_txt, tag = '—', 'blk'
            else:
                ann_txt, tag = self._sample_verdict(r)
            if r['present']:
                d_ppm = percent_deviation(r['ppm_centre'], ppm_mean.get(r['group']))
                d_j = percent_deviation(r['j_obs'], j_mean.get(r['group']))
            else:
                d_ppm = d_j = None
            self._tree.insert(
                '', 'end',
                values=(r['name'], r['group'], ann_txt,
                        dptxt, f'{d_ppm:+.3f}' if d_ppm is not None else '—',
                        jtxt, djtxt,
                        f'{d_j:+.2f}' if d_j is not None else '—',
                        f"{r['area']:.3e}" if r['area'] is not None else '—'),
                tags=(tag,))
        self._populate_summary()

    def _sample_verdict(self, r):
        """What the annotation concluded about this signal IN THIS SAMPLE.

        Exploratory. The search reaches one verdict for the study; this reports
        what its per-sample reading supports, using the pipeline's own ΔJ
        thresholds rather than any invented here:

          absent                → the lines were not all detected here
          no ΔJ                 → present, but nothing to verify it with
                                  (a singlet, or lines the apex search folded)
          |ΔJ| <= dj_gold_hz    → confirmed in this sample
          |ΔJ| <= dj_bad_hz     → present, coupling off
          beyond                → present, coupling wrong

        Returns ``(text, row_tag)``.
        """
        if not r['present']:
            return '✗ absent', 'absent'
        dj = r['dj']
        if dj is None:
            return ('~ folded', 'unverified') if r.get('collapsed') else                    ('~ no ΔJ', 'unverified')
        a = abs(dj)
        if a <= SCORING_DEFAULTS['dj_gold_hz']:
            return '✓ confirmed', 'present'
        if a <= SCORING_DEFAULTS['dj_bad_hz']:
            return '✓ ΔJ off', 'unverified'
        return '✗ ΔJ wrong', 'absent'

    def _populate_summary(self):
        """One closing row per group: the mean each sample above is measured
        against, and how tightly that group holds together.

        The DP % cells carry the group's own spread here — the coefficient of
        variation of the measured δ and J — while the same column above carries
        each sample's distance from this mean. A mean's distance from itself is
        zero and would say nothing.
        """
        rows = [r for r in self._counted_rows() if r['present']]
        if not rows:
            return
        by_group: dict = {}
        for r in rows:
            by_group.setdefault(r['group'], []).append(r)

        def _mean(v, fmt):
            return format(float(np.mean(v)), fmt) if v else '—'

        def _cv(v):
            c = relative_sd_percent(v)
            return f'{c:.2f}' if c is not None else '—'

        for grp, grows in by_group.items():
            centres = [r['ppm_centre'] for r in grows if r['ppm_centre'] is not None]
            dppm = [r['dppm'] for r in grows if r['dppm'] is not None]
            jobs = [r['j_obs'] for r in grows if r['j_obs'] is not None]
            djs = [r['dj'] for r in grows if r['dj'] is not None]
            areas = [r['area'] for r in grows if r['area'] is not None]
            self._tree.insert(
                '', 'end',
                values=(f'Mean · n = {len(grows)}', grp, '',
                        _mean(dppm, '+.4f'), _cv(centres),
                        _mean(jobs, '.2f'), _mean(djs, '+.2f'), _cv(jobs),
                        _mean(areas, '.3e')),
                tags=('summary',))

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
        # One legend entry per group, not per sample: with 32 traces a
        # per-sample legend covers the plot it is meant to explain.
        labelled = set()
        for r in self._rows:
            if r['type'] == 'BLK' or r['excluded']:
                continue
            y = r['col'][mask]
            grp = r['group']
            lbl = None if grp in labelled else str(grp)
            labelled.add(grp)
            # Shifting the axis, not the data: the sample is drawn where its own
            # lines say it sits. Samples with no shift (absent here) stay put.
            xs = x - self._shifts.get(r['idx'], 0.0) if aligned else x
            ax.plot(xs, y, color=self._group_color(grp, r['idx']),
                    lw=0.8, alpha=0.45, label=lbl)
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
        if labelled or present_cols:
            ax.legend(fontsize=8, loc='upper right', frameon=True,
                      framealpha=0.85,
                      ncol=(2 if len(labelled) > 4 else 1))
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
