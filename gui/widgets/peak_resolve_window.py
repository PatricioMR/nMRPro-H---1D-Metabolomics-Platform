# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/widgets/peak_resolve_window.py
"""
Peak Overlap Resolver.

Opens when right-clicking a peak that is claimed by multiple metabolites
("🔀 Resolve Peak Overlap").

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│  Header: Peak #N  @  X.XXXX ppm  —  N candidates                │
├───────────────────────────────┬──────────────────────────────────┤
│  Spectrum (matplotlib)        │  Candidates table                │
│  • grey fill  = experimental  │                                  │
│  • grey dashed│= DB position  │  Score columns (see below)       │
│  • coloured   = DB shifted to │                                  │
│    experimental position      │  [★ Assign Selected]  [Cancel]   │
└───────────────────────────────┴──────────────────────────────────┘

Table columns
-------------
  #          rank by resolve score
  Metabolite name
  Δ PPM      distance DB→exp (lower = better)
  DB Mult.   expected multiplicity from DB (s / d / t / q / dd / m …)
  Obs lines  number of visible peaks in the experimental region
  ✓ Mult     whether observed lines match DB expectation
  Met peaks  n_matched / n_total_DB  (how many of this metabolite's
             other DB peaks were also found experimentally)
  Fit        per-peak cosine similarity (exp shape vs DB simulation)
  Assigned   whether this metabolite is already marked Good

Resolve score (0–1):
  0.40 × (1 − Δppm / 0.05)        PPM proximity (penalty caps at 50 mDa)
  0.35 × (n_matched / n_db_total)  fraction of metabolite's DB peaks found
  0.25 × fit                       per-peak spectral shape similarity
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch


_CAND_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990',
]

_MULT_NLINES = {
    's': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6,
    'dd': 4, 'dt': 6, 'td': 6, 'ddd': 8,
}


def _j_score(delta_j_hz, db_type):
    """
    J-coupling compatibility score (0–1).
    - Singlets: always 0.15 (low priority — cannot differentiate by J)
    - Non-singlet with known ΔJ: linear from 1.0 (ΔJ=0) to 0.0 (ΔJ≥3 Hz)
    - Non-singlet with unknown ΔJ: neutral 0.55
    """
    db_type = (db_type or '').lower().strip()
    is_singlet = db_type in ('s', 'singlet', '')
    if is_singlet:
        return 0.15
    if delta_j_hz is None:
        return 0.55
    return float(max(0.0, 1.0 - delta_j_hz / 3.0))


def _resolve_score(delta_ppm, n_matched, n_total, fit, delta_j_hz=None, db_type=''):
    s_delta = max(0.0, 1.0 - delta_ppm / 0.05)
    s_cov   = n_matched / max(n_total, 1)
    s_j     = _j_score(delta_j_hz, db_type)
    return round(0.35 * s_delta + 0.25 * s_cov + 0.20 * float(fit or 0) + 0.20 * s_j, 4)


class PeakResolveWindow:
    """
    Parameters
    ----------
    parent         : tk widget
    candidates     : list of (met_dict, mp_dict) – all metabolites that
                     claimed peak_rank, with their specific matched-peak entry
    peak_rank      : int
    peak_ppm       : float  experimental centre
    ppm            : np.ndarray  full ppm axis
    mean_intensity : np.ndarray  mean spectrum over all non-BLK samples
    on_assign      : callable(chosen_met_id, quality_map)
    peaks_list     : list of peak dicts from PeakDetectionStage (optional)
    """

    def __init__(self, parent, candidates, peak_rank, peak_ppm,
                 ppm, mean_intensity, on_assign, peaks_list=None,
                 field_mhz=600.0, eligible_overrides=None):
        self.candidates        = candidates
        self.peak_rank         = peak_rank
        self.peak_ppm          = peak_ppm
        self.ppm               = ppm
        self.mean_intensity    = mean_intensity
        self.on_assign         = on_assign
        self.peaks_list        = peaks_list or []
        self._field_mhz        = float(field_mhz or 600.0)
        self._selected_idx     = None
        self._eligible_overrides = eligible_overrides if eligible_overrides is not None else {}

        self._build_rows()

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"Resolve Signal #{peak_rank}  @  {peak_ppm:.4f} ppm  "
            f"—  {len(candidates)} candidates")
        self.win.geometry('1100x520')
        self.win.minsize(800, 400)
        self.win.grab_set()

        self._build_ui()
        self._populate()
        self._select(0)
        self._center()

    # ── Row data preparation ──────────────────────────────────────────────────

    def _build_rows(self):
        rows = []
        for met, mp in self.candidates:
            db_entry  = self._find_db_entry(met, mp)
            db_type   = (db_entry.get('type', '') or
                         mp.get('type', '') or '?').lower().strip()

            # Skip unresolved multiplets: J unknown, cannot verify J match.
            if db_type in ('m', 'multiplet'):
                continue

            n_db      = len(met.get('db_peaks', []))
            n_matched = len(set(round(x['db_ppm'], 3)
                                for x in met.get('matched_peaks', [])))
            delta     = float(mp.get('ppm_diff', 0))
            fit       = self._peak_cosine(mp, db_entry)

            pred      = mp.get('predicted_lines', [])
            n_db_exp  = len(pred) if pred else _MULT_NLINES.get(db_type, -1)
            n_obs     = self._count_obs_lines(float(mp.get('peak_ppm',
                                                            self.peak_ppm)))
            mult_ok   = self._mult_match(n_db_exp, n_obs)

            # ΔJ: measure J directly from the experimental spectrum.
            # This is the ground truth — actual line separations in the data,
            # not predicted sticks or annotation-algorithm estimates.
            import re as _re_j
            db_j_raw = str(db_entry.get('j_coupling') or '').strip()
            try:
                db_j_nums = [float(x) for x in _re_j.findall(r'\d+\.?\d*', db_j_raw)
                             if float(x) > 0]
                db_j = db_j_nums[0] if db_j_nums else None
            except (ValueError, AttributeError):
                db_j = None

            j_obs_exp = self._measure_j_from_spectrum(mp, db_entry)
            mp['_j_obs_exp'] = j_obs_exp   # store for display

            if db_type in ('s', 'singlet', ''):
                delta_j = None   # singlets have no J
            elif j_obs_exp is not None and db_j is not None:
                delta_j = abs(j_obs_exp - db_j)
            elif j_obs_exp is not None:
                delta_j = None   # observed J known but DB J missing → show obs only
            else:
                delta_j = mp.get('delta_j_hz')  # fallback to annotation value

            score = _resolve_score(delta, n_matched, n_db, fit,
                                   delta_j_hz=delta_j, db_type=db_type)
            mid_val = met.get('id', '')
            pk_rank = mp.get('peak_rank', '')
            elig_key = f'{mid_val}__{pk_rank}'

            # Other peaks (different rank) where this metabolite is already Good
            _other_good = sorted(set(
                mp2.get('peak_rank') for mp2 in met.get('matched_peaks', [])
                if mp2.get('annotation_quality') == 'good'
                and mp2.get('peak_rank') is not None
                and mp2.get('peak_rank') != self.peak_rank
            ))

            rows.append({
                'met':         met,
                'mp':          mp,
                'db_entry':    db_entry,
                'delta':       delta,
                'n_matched':   n_matched,
                'n_db':        n_db,
                'fit':         fit,
                'db_type':     db_type,
                'n_db_exp':    n_db_exp,
                'n_obs':       n_obs,
                'mult_ok':     mult_ok,
                'delta_j':     delta_j,   # Hz or None
                'score':       score,
                'color':       _CAND_COLORS[len(rows) % len(_CAND_COLORS)],
                'current':     mp.get('annotation_quality') == 'good',
                'assigned_at': _other_good,
                'elig_key':    elig_key,
            })
        rows.sort(key=lambda r: r['score'], reverse=True)

        # Mark each row as a valid candidate (good quality) or not.
        # Valid = annotation_quality != 'bad'  AND
        #         (singlet OR ΔJ < 0.5 Hz OR ΔJ unknown)  AND
        #         delta_ppm within tolerance (< 0.05 ppm)
        for r in rows:
            dj        = r.get('delta_j')
            is_singlet = r['db_type'] in ('s', 'singlet', '')
            dj_ok     = is_singlet or dj is None or float(dj) < 0.5
            ppm_ok    = abs(r['delta']) < 0.05
            ann_bad   = r['mp'].get('annotation_quality') == 'bad'
            r['valid_candidate'] = (not ann_bad) and dj_ok and ppm_ok
            # Eligible: user-overridable; defaults to valid_candidate
            r['eligible'] = self._eligible_overrides.get(
                r['elig_key'], r['valid_candidate'])

        self._rows = rows

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_db_entry(self, met, mp):
        db_ppm = float(mp.get('db_ppm', self.peak_ppm))
        return next(
            (p for p in met.get('db_peaks', [])
             if abs(p.get('ppm_center', p.get('ppm', 0)) - db_ppm) < 0.02),
            {'ppm_center': db_ppm, 'ppm_start': db_ppm, 'ppm_end': db_ppm,
             'type': mp.get('type', '?'), 'sticks': []})

    def _peak_cosine(self, mp, db_entry):
        """
        Cosine similarity between the experimental spectrum slice and
        the DB-simulated Lorentzian(s), both aligned to the experimental ppm.
        """
        if self.mean_intensity is None or self.ppm is None:
            return 0.0
        peak_ppm = float(mp.get('peak_ppm', self.peak_ppm))
        db_ppm   = float(mp.get('db_ppm', peak_ppm))
        db_w     = abs(db_entry.get('ppm_end', db_ppm) -
                       db_entry.get('ppm_start', db_ppm))

        # For multiplets: use centroid of predicted_lines as window centre so
        # the fit covers ALL lines, not just the nearest single peak.
        pred = mp.get('predicted_lines', [])
        if pred and len(pred) >= 2:
            peak_ppm = float(np.mean([l['ppm'] for l in pred]))
            # Window wide enough to contain the full multiplet span + padding
            span = max(l['ppm'] for l in pred) - min(l['ppm'] for l in pred)
            win  = max(span * 2.0, db_w * 3.5, 0.025)
        else:
            win = max(db_w * 3.5, 0.025)

        mask = (self.ppm >= peak_ppm - win) & (self.ppm <= peak_ppm + win)
        if mask.sum() < 5:
            return 0.0
        x     = self.ppm[mask]
        y_exp = self.mean_intensity[mask].astype(float)

        # Build simulated shape aligned to experimental peak
        pred = mp.get('predicted_lines', [])
        if pred:
            sticks = [(float(l['ppm']), float(l['rel_intensity']))
                      for l in pred]
        elif db_entry.get('sticks'):
            offset = peak_ppm - db_ppm
            sticks = [(float(sp) + offset, float(si))
                      for sp, si in db_entry['sticks']]
        else:
            sticks = [(peak_ppm, 1.0)]

        # Baseline-correct experimental slice before cosine comparison.
        n_e = len(y_exp)
        en  = max(2, int(n_e * 0.15))
        xl  = float(np.mean(x[:en]));   yl = float(np.mean(y_exp[:en]))
        xr  = float(np.mean(x[-en:]));  yr = float(np.mean(y_exp[-en:]))
        if abs(xr - xl) > 1e-9:
            slope = (yr - yl) / (xr - xl)
            y_exp = y_exp - (yl + slope * (x - xl))
        y_exp = np.clip(y_exp, 0, None)

        hwhm  = max(db_w / 4.0, 0.0015)
        y_db  = np.zeros_like(x, dtype=float)
        for sp, sr in sticks:
            y_db += sr * hwhm**2 / ((x - sp)**2 + hwhm**2)

        ne = np.linalg.norm(y_exp)
        nd = np.linalg.norm(y_db)
        if ne == 0 or nd == 0:
            return 0.0
        return round(float(np.dot(y_exp / ne, y_db / nd)), 4)

    def _count_obs_lines(self, peak_ppm):
        """Count visible local maxima in experimental spectrum near peak_ppm."""
        if self.mean_intensity is None or self.ppm is None:
            return 0
        win  = 0.05
        mask = (self.ppm >= peak_ppm - win) & (self.ppm <= peak_ppm + win)
        if mask.sum() < 5:
            return 0
        y     = self.mean_intensity[mask]
        y_max = float(np.max(y))
        if y_max <= 0:
            return 0
        thresh = y_max * 0.12  # 12 % of peak max
        # local maxima: point higher than both neighbours
        n = 0
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= thresh:
                n += 1
        return max(n, 1)

    def _measure_j_from_spectrum(self, mp, db_entry):
        """
        Measure observed J (Hz) by snapping detected experimental peaks
        (from self.peaks_list) to the predicted_lines positions.

        This mirrors the fingerprint-comparison approach exactly:
        for each predicted line position, find the nearest detected peak
        within a snap radius, then compute median inter-peak separation in Hz.

        Returns observed J in Hz, or None if fewer than 2 lines are found.
        """
        pred = mp.get('predicted_lines', [])
        if not pred or len(pred) < 2:
            return None

        peaks_list = getattr(self, 'peaks_list', None) or []
        if not peaks_list:
            return None

        pl_ppms = sorted(float(l.get('ppm', 0)) for l in pred)
        min_sep = min(abs(pl_ppms[k+1] - pl_ppms[k])
                      for k in range(len(pl_ppms) - 1))
        snap_r  = max(min_sep * 0.55, 0.005)

        used_ranks  = set()
        snapped_ppm = []
        for lp in pl_ppms:
            best_pk, best_d = None, snap_r
            for pk in peaks_list:
                rk = pk.get('rank')
                if rk in used_ranks:
                    continue
                d = abs(float(pk.get('ppm', 0)) - lp)
                if d <= best_d:
                    best_d, best_pk = d, pk
            if best_pk is not None:
                used_ranks.add(best_pk.get('rank'))
                snapped_ppm.append(float(best_pk['ppm']))

        if len(snapped_ppm) < 2:
            return None

        snapped_ppm.sort()
        seps_hz = [abs(snapped_ppm[k+1] - snapped_ppm[k]) * self._field_mhz
                   for k in range(len(snapped_ppm) - 1)]
        return float(np.median(seps_hz))

    @staticmethod
    def _mult_match(n_db_exp, n_obs):
        """✓ match  ~  close  ✗ mismatch  ? unknown"""
        if n_db_exp < 0 or n_obs == 0:
            return '?'
        if n_obs == n_db_exp:
            return '✓'
        if abs(n_obs - n_db_exp) == 1:
            return '~'
        return '✗'

    def _sticks_for_plot(self, r, at_db_position=False):
        """Return (sticks, hwhm) for the DB candidate r.
        at_db_position=True → use raw DB ppm (no shift).
        """
        mp       = r['mp']
        db_entry = r['db_entry']
        peak_ppm = float(mp.get('peak_ppm', self.peak_ppm))
        db_ppm   = float(mp.get('db_ppm', self.peak_ppm))
        db_w     = abs(db_entry.get('ppm_end', db_ppm) -
                       db_entry.get('ppm_start', db_ppm))
        hwhm = max(db_w / 4.0, 0.0015)

        pred = mp.get('predicted_lines', [])
        if pred:
            shift = 0.0 if not at_db_position else (db_ppm - peak_ppm)
            sticks = [(float(l['ppm']) + shift, float(l['rel_intensity']))
                      for l in pred]
        elif db_entry.get('sticks'):
            offset = 0.0 if at_db_position else (peak_ppm - db_ppm)
            sticks = [(float(sp) + offset, float(si))
                      for sp, si in db_entry['sticks']]
        else:
            ppm_pos = db_ppm if at_db_position else peak_ppm
            sticks = [(ppm_pos, 1.0)]
        return sticks, hwhm

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = ttk.Frame(self.win, padding=(8, 4))
        hdr.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(hdr,
                  text=(f"Signal #{self.peak_rank}  ·  "
                        f"{self.peak_ppm:.4f} ppm  ·  "
                        f"{len(self._rows)} candidates"),
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Label(hdr,
                  text='Select candidate → plot updates  ·  Double-click or ★ Assign to resolve',
                  font=('Arial', 8), foreground='#555').pack(side=tk.RIGHT, padx=6)

        ttk.Separator(self.win, orient='horizontal').pack(fill=tk.X, padx=4)

        split = tk.PanedWindow(self.win, orient=tk.HORIZONTAL,
                               sashwidth=5, sashrelief=tk.RAISED)
        split.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        spec_frame = ttk.LabelFrame(split, text='Signal region', padding=2)
        split.add(spec_frame, minsize=360, stretch='always')
        self._build_plot(spec_frame)

        right = ttk.Frame(split)
        split.add(right, minsize=420, stretch='always')
        self._build_table(right)

    def _build_plot(self, parent):
        self._fig = plt.Figure(figsize=(4.8, 4.2), dpi=100)
        self._fig.patch.set_facecolor('#fafafa')
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor('#fafafa')
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_table(self, parent):
        COLS = [
            ('rank',     '#',          28,  tk.CENTER),
            ('name',     'Metabolite', 150, tk.W),
            ('delta',    'Δ PPM',       60, tk.CENTER),
            ('dbtype',   'DB Mult.',    62, tk.CENTER),
            ('deltaj',   'ΔJ (Hz)',     62, tk.CENTER),
            ('obslines', 'Obs lines',   65, tk.CENTER),
            ('multok',   '✓ Mult',      52, tk.CENTER),
            ('metpks',   'Met peaks',   68, tk.CENTER),
            ('fit',      'Fit',         52, tk.CENTER),
            ('score',    'Score',       58, tk.CENTER),
            ('cur',      'Assigned',    62, tk.CENTER),
            ('eligible', 'Eligible',    58, tk.CENTER),
        ]
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        self.tree = ttk.Treeview(
            parent,
            columns=[c[0] for c in COLS],
            show='headings',
            selectmode='browse',
            yscrollcommand=vsb.set)
        vsb.config(command=self.tree.yview)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        for cid, heading, width, anchor in COLS:
            self.tree.heading(cid, text=heading)
            self.tree.column(cid, width=width, anchor=anchor,
                             minwidth=24, stretch=(cid == 'name'))

        self.tree.tag_configure('best',     background='#e0f0e0', font=('Arial', 9, 'bold'))
        self.tree.tag_configure('normal',   background='white')
        self.tree.tag_configure('current',  background='#fff3cd')
        self.tree.tag_configure('bad_mult', foreground='#b30000')
        self.tree.tag_configure('invalid',  background='#f0f0f0',
                                foreground='#aaaaaa', font=('Arial', 9, 'italic'))
        self._col_ids = [c[0] for c in COLS]
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_sel)
        self.tree.bind('<Double-Button-1>',  self._on_double_click)
        self.tree.bind('<Button-1>',         self._on_click)

        # Legend label
        leg = ttk.Frame(parent, padding=(4, 2))
        leg.pack(side=tk.BOTTOM, fill=tk.X, before=self.tree)
        ttk.Label(leg,
                  text=('Δ PPM: shift DB→exp  |  '
                        'ΔJ: |J_obs − J_db| in Hz (green <1, yellow <2.5, red ≥2.5)  |  '
                        '✓ Mult: obs matches DB multiplicity  |  '
                        'Met signals: other signals of this metabolite found  |  '
                        'Fit: shape cosine  |  '
                        'Grey (italic): poor quality — ΔJ ≥ 0.5 Hz, Δppm ≥ 0.05, or marked Bad'),
                  font=('Arial', 7), foreground='#888',
                  wraplength=420, justify=tk.LEFT).pack(
            side=tk.TOP, anchor=tk.W)

        # Bottom bar
        bf = ttk.Frame(parent, padding=(4, 4))
        bf.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bf,
                  text='Score = 0.35×(1−Δppm/50mDa) + 0.25×(met signals) + 0.20×fit + 0.20×J_compat  |  singlets penalised',
                  font=('Arial', 7), foreground='#888').pack(side=tk.LEFT)
        ttk.Button(bf, text='Cancel',
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text='★ Assign Selected',
                   command=self._assign).pack(side=tk.RIGHT, padx=4)

    # ── Populate ─────────────────────────────────────────────────────────────

    def _populate(self):
        self.tree.tag_configure('good_j',  foreground='#1a7a1a')
        self.tree.tag_configure('warn_j',  foreground='#7a6000')
        self.tree.tag_configure('bad_j',   foreground='#9a0000')
        # Index of the first valid candidate (best valid = highlighted green)
        _first_valid = next((j for j, row in enumerate(self._rows)
                             if row.get('valid_candidate', True)), None)
        for i, r in enumerate(self._rows):
            valid = r.get('valid_candidate', True)

            if not valid:
                base_tag   = 'invalid'
                extra_tags = []
            else:
                base_tag   = ('best' if i == _first_valid else
                              'current' if r['current'] else 'normal')
                extra_tags = []
                if r['mult_ok'] == '✗':
                    extra_tags.append('bad_mult')

            dj        = r.get('delta_j')
            is_singlet = r['db_type'] in ('s', 'singlet', '')
            j_obs_exp  = r['mp'].get('_j_obs_exp')
            if dj is not None:
                dj_str = f"{dj:.2f}"
                if valid:
                    if dj < 1.0:
                        extra_tags.append('good_j')
                    elif dj < 2.5:
                        extra_tags.append('warn_j')
                    else:
                        extra_tags.append('bad_j')
            elif is_singlet:
                dj_str = '—'
            elif j_obs_exp is not None:
                dj_str = f"~{j_obs_exp:.1f}"
            else:
                dj_str = '?'

            n_db_exp_str = (str(r['n_db_exp']) if r['n_db_exp'] > 0 else '?')
            elig_sym = '✓' if r['eligible'] else '✗'
            self.tree.insert('', tk.END, iid=str(i),
                             tags=(base_tag, *extra_tags),
                             values=(
                                 i + 1,
                                 r['met']['name'],
                                 f"{r['delta']:.4f}",
                                 r['db_type'].upper() or '?',
                                 dj_str,
                                 f"{r['n_obs']} (exp {n_db_exp_str})",
                                 r['mult_ok'],
                                 f"{r['n_matched']}/{r['n_db']}",
                                 f"{r['fit']:.3f}",
                                 f"{r['score']:.3f}",
                                 (f'✓ #{self.peak_rank}' if r['current']
                                  else (', '.join(f'#{rk}' for rk in r['assigned_at'])
                                        if r.get('assigned_at') else '—')),
                                 elig_sym,
                             ))

    # ── Plot ─────────────────────────────────────────────────────────────────

    def _draw_plot(self, hl_idx):
        ax = self._ax
        ax.clear()
        ax.set_facecolor('#fafafa')

        # Centre view on matched sticks midpoint for multiplets
        _view_centre = self.peak_ppm
        if 0 <= hl_idx < len(self._rows):
            _pred = self._rows[hl_idx]['mp'].get('predicted_lines', [])
            if _pred and len(_pred) >= 2:
                _view_centre = float(np.mean([l['ppm'] for l in _pred]))
        margin = 0.14
        vlo = _view_centre - margin
        vhi = _view_centre + margin
        x_sim = np.linspace(vlo, vhi, 800)

        # ── Deconvolved sub-peak display (mirrors fingerprint comparison) ─────
        # If the highlighted candidate matched a deconvolved sub-peak, show
        # individual components instead of raw spectrum.
        _use_deconv  = False
        _deconv_comp = []   # (x_arr, y_arr) per component
        _comp_peak_y = {}   # peak_ppm → norm height for marker placement
        _ref_max     = 1.0

        if 0 <= hl_idx < len(self._rows):
            _hl_mp = self._rows[hl_idx]['mp']
            if _hl_mp.get('is_sub_peak') and self.peaks_list:
                _parent_rank = _hl_mp.get('peak_rank')
                _siblings = [p for p in self.peaks_list
                             if p.get('is_sub_peak')
                             and int(p.get('rank', -1)) == _parent_rank]
                if _siblings:
                    _reg_ppm_raw = _siblings[0].get('region_ppm')
                    if _reg_ppm_raw is not None and len(_reg_ppm_raw) >= 4:
                        _reg_ppm = np.asarray(_reg_ppm_raw, dtype=float)
                        _rmask   = (_reg_ppm >= vlo) & (_reg_ppm <= vhi)
                        _x_reg   = _reg_ppm[_rmask]
                        if len(_x_reg) >= 4:
                            _y_sum = np.zeros(len(_x_reg))
                            for _sp in _siblings:
                                _cen = float(_sp.get('ppm', 0))
                                _hw  = float(_sp.get('width_ppm',
                                             _sp.get('width_at_half', 0.004))) / 2.0
                                _hw  = max(_hw, 1e-5)
                                _ht  = float(_sp.get('height', 1.0))
                                _yc  = _ht * _hw**2 / ((_x_reg - _cen)**2 + _hw**2)
                                _deconv_comp.append((_x_reg, _yc))
                                _y_sum += _yc
                            if _y_sum.max() > 0:
                                _use_deconv = True
                                _ref_max    = float(_y_sum.max())

        # ── Experimental spectrum ─────────────────────────────────────────────
        y_max = 1.0
        if self.mean_intensity is not None and self.ppm is not None:
            vmask  = (self.ppm >= vlo) & (self.ppm <= vhi)
            x_view = self.ppm[vmask]
            y_view = self.mean_intensity[vmask].astype(float)
            if len(x_view) >= 4 and np.max(y_view) > 0:
                if _use_deconv:
                    # Faded raw behind components (no baseline correction)
                    _raw_ref = float(y_view.max()) if y_view.max() > 0 else 1.0
                    y_raw_n  = y_view / _raw_ref
                    # Scale raw to sit behind components
                    _raw_scale = _ref_max / _raw_ref if _raw_ref > 0 else 1.0
                    y_raw_n_scaled = y_view * _raw_scale / _ref_max
                    ax.fill_between(x_view, 0, y_raw_n_scaled,
                                    color='#aaaaaa', alpha=0.12, zorder=1)
                    ax.plot(x_view, y_raw_n_scaled,
                            color='#aaaaaa', lw=0.8, ls=':', zorder=1,
                            label='Experimental (raw)')
                    # Individual components
                    _comp_colors = ['#1a6eb5', '#e07000', '#8a2be2',
                                    '#009966', '#cc2200']
                    for _ki, (_xc, _yc) in enumerate(_deconv_comp):
                        _yc_n = _yc / _ref_max
                        _cc   = _comp_colors[_ki % len(_comp_colors)]
                        ax.fill_between(_xc, 0, _yc_n,
                                        color=_cc, alpha=0.18, zorder=2)
                        ax.plot(_xc, _yc_n, color=_cc, lw=0.9,
                                alpha=0.70, zorder=3)
                        if len(_yc_n) > 0:
                            _pk_i = int(np.argmax(_yc_n))
                            _comp_peak_y[float(_xc[_pk_i])] = float(_yc_n[_pk_i])
                    # Sum fill
                    _y_sum_n = np.zeros(len(x_view))
                    _sum_mask = (x_view >= _deconv_comp[0][0].min()) & \
                                (x_view <= _deconv_comp[0][0].max())
                    for _xc, _yc in _deconv_comp:
                        _idx_c = np.searchsorted(x_view, _xc)
                        pass  # already plotted individually
                    y_max = 1.0
                else:
                    # Normal raw display — no baseline correction
                    y_max  = float(np.max(y_view))
                    y_n    = y_view / y_max
                    ax.fill_between(x_view, 0, y_n,
                                    color='#aaaaaa', alpha=0.30, zorder=1,
                                    label='Experimental')
                    ax.plot(x_view, y_n,
                            color='#444', lw=0.9, alpha=0.75, zorder=2)

        # ── Non-highlighted candidates (faint) ────────────────────────────────
        for i, r in enumerate(self._rows):
            if i == hl_idx:
                continue
            sticks, hwhm = self._sticks_for_plot(r, at_db_position=False)
            y_db = np.zeros_like(x_sim)
            for sp, sr in sticks:
                y_db += sr * hwhm**2 / ((x_sim - sp)**2 + hwhm**2)
            if y_db.max() > 0:
                y_db /= y_db.max()
            ax.plot(x_sim, y_db,
                    color=r['color'], lw=0.7, alpha=0.22,
                    zorder=3, ls='-')

        # ── Highlighted candidate ─────────────────────────────────────────────
        if 0 <= hl_idx < len(self._rows):
            r   = self._rows[hl_idx]
            col = r['color']
            mp  = r['mp']
            peak_ppm_r = float(mp.get('peak_ppm', self.peak_ppm))
            db_ppm_r   = float(mp.get('db_ppm',   self.peak_ppm))

            sticks_db, hwhm = self._sticks_for_plot(r, at_db_position=True)

            # Build DB aligned curve and scale to component/spectrum height
            sticks_exp, _ = self._sticks_for_plot(r, at_db_position=False)
            y_at_exp = np.zeros_like(x_sim)
            for sp, sr in sticks_exp:
                y_at_exp += sr * hwhm**2 / ((x_sim - sp)**2 + hwhm**2)
            if y_at_exp.max() > 0:
                y_at_exp /= y_at_exp.max()
                # Scale to match component peak height when deconv display active
                if _use_deconv and _comp_peak_y:
                    _mcp = min(_comp_peak_y.keys(),
                               key=lambda cp: abs(cp - peak_ppm_r))
                    y_at_exp *= _comp_peak_y[_mcp]

            ax.fill_between(x_sim, 0, y_at_exp,
                            color=col, alpha=0.28, zorder=6)
            ax.plot(x_sim, y_at_exp,
                    color=col, lw=2.0, alpha=0.90, zorder=7,
                    label=f"{r['met']['name'][:24]} (matched)")

            # DB position dashed overlay (where DB says it should be)
            y_at_db = np.zeros_like(x_sim)
            for sp, sr in sticks_db:
                y_at_db += sr * hwhm**2 / ((x_sim - sp)**2 + hwhm**2)
            if y_at_db.max() > 0:
                y_at_db /= y_at_db.max()
                if _use_deconv and _comp_peak_y:
                    y_at_db *= _comp_peak_y[min(_comp_peak_y.keys(),
                                                key=lambda cp: abs(cp - peak_ppm_r))]
            ax.fill_between(x_sim, 0, y_at_db,
                            color='#999999', alpha=0.10, zorder=4)
            ax.plot(x_sim, y_at_db,
                    color='#888888', lw=1.0, alpha=0.45, zorder=5,
                    ls='--', label=f"DB ({db_ppm_r:.4f})")

            # Orange markers on component peaks (or spectrum peak)
            _snap_r = max(hwhm * 3, 0.008)
            _used   = set()
            for st_ppm, _ in sticks_exp:
                if _use_deconv and _comp_peak_y:
                    _best = min(_comp_peak_y.keys(),
                                key=lambda cp: abs(cp - st_ppm))
                    if abs(_best - st_ppm) <= _snap_r and _best not in _used:
                        _used.add(_best)
                        ax.plot([_best], [_comp_peak_y[_best] + 0.04],
                                'v', color='#cc5500', ms=8, zorder=9,
                                alpha=0.90, markeredgewidth=0)
                elif self.mean_intensity is not None and self.ppm is not None:
                    _vmask2 = (self.ppm >= vlo) & (self.ppm <= vhi)
                    _xv2 = self.ppm[_vmask2]
                    _yv2 = self.mean_intensity[_vmask2].astype(float)
                    if len(_xv2) > 0 and _yv2.max() > 0:
                        _idx2 = int(np.argmin(np.abs(_xv2 - st_ppm)))
                        _h2   = float(_yv2[_idx2]) / float(_yv2.max())
                        ax.plot([st_ppm], [_h2 + 0.04],
                                'v', color='#cc5500', ms=8, zorder=9,
                                alpha=0.90, markeredgewidth=0)

            # ── Vertical markers ──────────────────────────────────────────────
            ax.axvline(db_ppm_r, color='#888888', lw=1.0, ls='--',
                       alpha=0.55, zorder=8)
            ax.axvline(peak_ppm_r, color=col, lw=1.2, ls=':',
                       alpha=0.70, zorder=8)

            # ── Δppm annotation arrow ─────────────────────────────────────────
            if abs(peak_ppm_r - db_ppm_r) > 0.0005:
                y_arrow = 1.05
                ax.annotate(
                    '', xy=(peak_ppm_r, y_arrow),
                    xytext=(db_ppm_r,   y_arrow),
                    arrowprops=dict(arrowstyle='<->',
                                    color='#555555', lw=1.2))
                mid = (peak_ppm_r + db_ppm_r) / 2
                ax.text(mid, y_arrow + 0.06,
                        f"Δ {r['delta']:.4f} ppm",
                        ha='center', va='bottom',
                        fontsize=7, color='#333333')

            # ── Multiplicity + J annotation ───────────────────────────────────
            db_type_str = r['db_type'].upper() or '?'
            n_exp_str   = (str(r['n_db_exp']) if r['n_db_exp'] > 0 else '?')
            mult_color  = ({'✓': '#1a7a1a', '~': '#7a6000',
                            '✗': '#9a0000', '?': '#777777'}
                           .get(r['mult_ok'], '#777777'))
            # J coupling info line
            j_obs_exp = r['mp'].get('_j_obs_exp')
            dj_val    = r.get('delta_j')
            import re as _re_j2
            db_j_raw2 = str(r['db_entry'].get('j_coupling') or '').strip()
            try:
                _db_j2 = float(_re_j2.findall(r'\d+\.?\d*', db_j_raw2)[0]) \
                         if db_j_raw2 else None
            except Exception:
                _db_j2 = None
            if r['db_type'] in ('s', 'singlet', ''):
                j_line = 'J: singlet'
                j_col  = '#777777'
            elif j_obs_exp is not None and _db_j2 is not None:
                j_line = (f"J obs: {j_obs_exp:.2f} Hz  |  DB: {_db_j2:.2f} Hz  |  "
                          f"ΔJ: {dj_val:.2f} Hz" if dj_val is not None
                          else f"J obs: {j_obs_exp:.2f} Hz  |  DB: {_db_j2:.2f} Hz")
                j_col  = ('#1a7a1a' if (dj_val or 99) < 1.0 else
                          '#7a6000' if (dj_val or 99) < 2.5 else '#9a0000')
            elif j_obs_exp is not None:
                j_line = f"J obs: {j_obs_exp:.2f} Hz  |  DB J: unknown"
                j_col  = '#7a6000'
            elif _db_j2 is not None:
                j_line = f"DB J: {_db_j2:.2f} Hz  |  obs: not resolved"
                j_col  = '#777777'
            else:
                j_line = 'J: unknown'
                j_col  = '#777777'
            ax.text(0.02, 0.97,
                    f"DB: {db_type_str} ({n_exp_str} lines)  "
                    f"obs: {r['n_obs']} lines  {r['mult_ok']}\n{j_line}",
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=7.5,
                    color=mult_color,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.80, edgecolor=j_col,
                              linewidth=0.8))

        ax.axvline(self.peak_ppm, color='#333', lw=0.7,
                   ls=':', alpha=0.4, zorder=2)
        ax.invert_xaxis()
        ax.set_xlim(vhi + 0.01, vlo - 0.01)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel('δ (ppm)', fontsize=8)
        ax.set_ylabel('Intensity (norm.)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, ls=':')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select(self, idx):
        if idx < 0 or idx >= len(self._rows):
            return
        self._selected_idx = idx
        self.tree.selection_set(str(idx))
        self.tree.see(str(idx))
        self._draw_plot(idx)

    def _on_tree_sel(self, e=None):
        sel = self.tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        if idx != self._selected_idx:
            self._select(idx)

    def _on_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != 'cell':
            return
        col_id = self.tree.identify_column(event.x)
        # col_id is '#N' (1-based); map to column name
        try:
            col_idx = int(col_id.lstrip('#')) - 1
            col_name = self._col_ids[col_idx]
        except (ValueError, IndexError):
            return
        if col_name != 'eligible':
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        try:
            idx = int(iid)
        except ValueError:
            return
        if idx < 0 or idx >= len(self._rows):
            return
        r = self._rows[idx]
        new_val = not r['eligible']
        r['eligible'] = new_val
        self._eligible_overrides[r['elig_key']] = new_val
        # Refresh just this row's eligible cell
        vals = list(self.tree.item(iid, 'values'))
        elig_col_idx = self._col_ids.index('eligible')
        vals[elig_col_idx] = '✓' if new_val else '✗'
        self.tree.item(iid, values=vals)

    def _on_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        col_id = self.tree.identify_column(event.x)
        try:
            col_idx = int(col_id.lstrip('#')) - 1
            col_name = self._col_ids[col_idx]
        except (ValueError, IndexError):
            col_name = ''
        if region == 'cell' and col_name != 'eligible':
            self._assign()

    # ── Assign ────────────────────────────────────────────────────────────────

    def _assign(self):
        idx = self._selected_idx
        if idx is None:
            messagebox.showwarning('No selection',
                                   'Select a metabolite first.',
                                   parent=self.win)
            return
        chosen_id = self._rows[idx]['met']['id']
        quality_map = {r['met']['id']: ('good' if r['met']['id'] == chosen_id
                                        else 'bad')
                       for r in self._rows}
        self.on_assign(chosen_id, quality_map)
        self.win.destroy()

    def _center(self):
        self.win.update_idletasks()
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f'{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}')
