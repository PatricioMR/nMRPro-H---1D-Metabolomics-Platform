# gui/widgets/resolving_method_window.py
"""
Overlap Resolving Method — J-Ratio-Corrected Cascade.

Algorithm
─────────
Phase 1  – Anchor estimation (user-marked 'Good' peaks)
    For every metabolite that has at least one "Good" peak with no overlap,
    estimate a relative concentration proxy using the effective nH:
        eff_nH    = nH × (line_intensity_fraction)  [Pascal's triangle ratio]
        conc_est  = median( area_anchor_i / eff_nH_i )

    This corrects for non-uniform multiplet line intensities: the centre line
    of a triplet (1:2:1) carries twice the area of an outer line, so using
    raw nH would over/under-estimate concentration depending on which line
    was used as anchor.

Phase 1b – Auto-anchor (non-overlapping matched peaks, no user marking needed)
    For metabolites with no 'Good' peaks, any matched peak that belongs to
    a single-metabolite region is promoted as an automatic anchor.

Phase 2  – Contribution subtraction (J-ratio corrected)
    For every overlapping peak:
        contrib_M  = conc_est_M × eff_nH_M_at_this_peak
        residual   = observed_area − Σ contrib_M  (for all M with known conc_est)
        %_M        = contrib_M / observed_area × 100
        %_resid    = residual  / observed_area × 100

Phase 3  (optional, checkbox)  – Iterative cascade refinement
    After Phase 2, any overlapping peak whose ALL known contributions have
    been subtracted now reduces to a single-metabolite residual → new anchor.
    The cascade correctly propagates the J-ratio correction at each step.
    Repeat until no new anchors emerge.

Phase 4  – Mark unresolvable
    Metabolites that still have no conc_est after all phases are flagged
    as "✗ Unresolvable".

Status per metabolite
─────────────────────
  ✓ Quantifiable   – user-marked 'Good' unambiguous anchor
  ⊙ Auto-anchored  – auto-detected clean anchor (no user marking needed)
  ⚠ Resolved       – no direct anchor, resolved via Phase 3 cascade
  ✗ Unresolvable   – no concentration estimate possible
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ── Colour constants ──────────────────────────────────────────────────────────
_COL_ANCHOR    = '#2196F3'   # blue   – anchor peak cell (user-marked good)
_COL_AUTO      = '#00BCD4'   # cyan   – auto-anchored peak cell (Phase 1b)
_COL_RESOLVED  = '#4CAF50'   # green  – known contribution
_COL_RESIDUAL  = '#FF9800'   # orange – unresolved residual
_COL_UNKNOWN   = '#BDBDBD'   # grey   – unresolvable


class ResolvingMethodWindow:

    def __init__(self, parent, metabolites, peaks, ppm,
                 all_intensities, samples_info, app=None):
        self._metabolites   = metabolites
        self._peaks         = peaks
        self._ppm           = ppm
        self._all_int       = all_intensities  # (n_pts, n_samples)
        self._samples_info  = samples_info or []
        self.app            = app

        # Mean spectrum (BLK excluded)
        self._mean_int = self._build_mean()

        # Results – populated by _run()
        self._peak_overlap   = {}   # rank → [met_name, …]
        self._conc_est       = {}   # met_id → float (area/proton proxy)
        self._contributions  = {}   # rank → {met_id: pct, '_residual_pct': pct,
                                    #          '_observed': float}
        self._status         = {}   # met_id → 'quantifiable'|'resolved'|
                                    #           'auto_anchored'|'unresolvable'
        self._anchor_peaks   = {}   # met_id → [peak_rank, …]
        self._auto_anchored  = set()  # met_ids anchored via Phase 1b (not user-marked)

        self.win = tk.Toplevel(parent)
        self.win.title('Resolving Method — Overlap Decomposition')
        self.win.geometry('1200x620')
        self.win.minsize(900, 460)

        self._iterate_var   = tk.BooleanVar(value=True)  # Phase 3 on by default
        self._anchor_tol    = tk.DoubleVar(value=0.02)   # PPM tolerance for anchor check
        self._build_ui()
        self._run()
        self._center()

    # ── Mean spectrum ─────────────────────────────────────────────────────────

    def _build_mean(self):
        if self._all_int is None:
            return None
        si  = self._samples_info
        blk = {i for i, s in enumerate(si)
               if str(s.get('type', '')).upper() == 'BLK'}
        idxs = [i for i in range(self._all_int.shape[1]) if i not in blk]
        if not idxs:
            idxs = list(range(self._all_int.shape[1]))
        return np.mean(self._all_int[:, idxs], axis=1)

    # ── Overlap map ───────────────────────────────────────────────────────────

    def _build_overlap(self):
        om = {}
        for met in self._metabolites:
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is not None:
                    om.setdefault(rank, []).append(met['name'])
        self._peak_overlap = om

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _met_by_id(self, mid):
        return next((m for m in self._metabolites if m['id'] == mid), None)

    def _met_by_name(self, name):
        return next((m for m in self._metabolites if m['name'] == name), None)

    def _peak_by_rank(self, rank):
        return next((p for p in self._peaks if p.get('rank') == rank), None)

    def _get_area(self, peak):
        """Trapezoidal area on mean spectrum.
        Uses min/max to handle NMR convention where start_ppm > end_ppm.
        """
        if self._mean_int is None or self._ppm is None:
            return 0.0
        sp = peak.get('start_ppm', peak['ppm'])
        ep = peak.get('end_ppm',   peak['ppm'])
        lo = min(sp, ep)
        hi = max(sp, ep)
        if lo == hi:
            lo -= 0.01; hi += 0.01
        mask = (self._ppm >= lo) & (self._ppm <= hi)
        if mask.sum() < 2:
            return 0.0
        return float(abs(np.trapezoid(self._mean_int[mask], self._ppm[mask])))

    def _get_area_per_sample(self, peak):
        """Trapezoidal area per sample.
        Uses min/max to handle NMR convention where start_ppm > end_ppm.
        """
        if self._all_int is None or self._ppm is None:
            return np.array([])
        sp = peak.get('start_ppm', peak['ppm'])
        ep = peak.get('end_ppm',   peak['ppm'])
        lo = min(sp, ep)
        hi = max(sp, ep)
        if lo == hi:
            lo -= 0.01; hi += 0.01
        mask = (self._ppm >= lo) & (self._ppm <= hi)
        if mask.sum() < 2:
            return np.zeros(self._all_int.shape[1])
        x = self._ppm[mask]
        return np.array([
            float(abs(np.trapezoid(self._all_int[mask, s], x)))
            for s in range(self._all_int.shape[1])
        ])

    def _get_protons(self, met, mp):
        try:
            n = int(mp.get('protons') or 0)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
        db_ppm = float(mp.get('db_ppm', 0))
        entry  = next(
            (p for p in met.get('db_peaks', [])
             if abs(p.get('ppm_center', p.get('ppm', 0)) - db_ppm) < 0.02),
            None)
        if entry:
            try:
                n = int(entry.get('protons') or 0)
                if n > 0:
                    return n
            except (TypeError, ValueError):
                pass
        return 1

    def _effective_nh(self, met, mp):
        """nH × intensity_fraction for the specific multiplet line this peak represents.

        For a doublet (1:1) each line carries nH/2 of the total signal.
        For a triplet (1:2:1) the centre carries nH/2, each outer carries nH/4.
        Using the correct fraction fixes concentration estimates and overlap
        contributions for non-uniform multiplets (triplet, quartet, dd, etc.).

        Priority 1 — predicted_lines rel_intensity from the scorer (most accurate).
        Priority 2 — Pascal's triangle derived from multiplet type + J coupling.
        Fallback    — raw n_h (singlet or unknown type), same as previous behaviour.
        """
        import re as _re
        n_h = self._get_protons(met, mp)

        # --- Priority 1: predicted_lines from the scorer ----------------------
        pred = mp.get('predicted_lines') or []
        if len(pred) > 1:
            ris   = [float(l.get('rel_intensity', 1.0)) for l in pred]
            total = sum(ris)
            if total > 0:
                peak_ppm = float(mp.get('peak_ppm') or mp.get('ppm') or 0.0)
                closest_idx = min(range(len(pred)),
                                  key=lambda i: abs(
                                      float(pred[i].get('ppm', peak_ppm)) - peak_ppm))
                return n_h * ris[closest_idx] / total

        # --- Priority 2: Pascal's triangle from multiplet type + J spacing ----
        _pascal = {
            's':   [1],
            'd':   [1, 1],
            't':   [1, 2, 1],
            'q':   [1, 3, 3, 1],
            'p':   [1, 4, 6, 4, 1],
            'quin':[1, 4, 6, 4, 1],
            'h':   [1, 5, 10, 10, 5, 1],
            'sex': [1, 5, 10, 10, 5, 1],
            'sep': [1, 6, 15, 20, 15, 6, 1],
            'dd':  [1, 1, 1, 1],
            'ddd': [1, 1, 1, 1, 1, 1, 1, 1],
            'dt':  [1, 1, 2, 2, 1, 1],
            'td':  [1, 2, 1, 1, 2, 1],
            'dq':  [1, 1, 3, 3, 3, 3, 1, 1],
        }
        mult   = (mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
        coeffs = _pascal.get(mult)
        if coeffs and len(coeffs) > 1:
            j_str  = str(mp.get('j_coupling') or '').strip()
            j_nums = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str)
                      if float(x) > 0]
            if j_nums:
                # Use spectrometer freq from app if available, else 600 MHz fallback
                mhz = 600.0
                if self.app is not None:
                    mhz = float(getattr(self.app, 'spectrometer_mhz', None) or 600.0)
                spacing    = j_nums[0] / mhz
                centre     = float(mp.get('ppm_center') or mp.get('db_ppm') or 0.0)
                n_lines    = len(coeffs)
                positions  = [centre + (i - (n_lines - 1) / 2.0) * spacing
                              for i in range(n_lines)]
                peak_ppm   = float(mp.get('peak_ppm') or mp.get('ppm') or centre)
                dists      = [abs(peak_ppm - pos) for pos in positions]
                closest    = dists.index(min(dists))
                return n_h * coeffs[closest] / sum(coeffs)

        # --- Fallback: singlet or unknown type — unchanged behaviour ----------
        return float(n_h)

    # ── Algorithm ─────────────────────────────────────────────────────────────

    def _run(self):
        self._build_overlap()
        self._conc_est.clear()
        self._contributions.clear()
        self._status.clear()
        self._anchor_peaks.clear()
        self._auto_anchored.clear()

        self._phase1()    # user-marked 'good' peaks → conc_est
        self._phase1b()   # auto-anchor: clean (non-overlapping) matched peaks
        self._phase2()    # subtract known contributions from overlapping peaks

        if self._iterate_var.get():
            self._phase3()   # cascade: residual → conc_est if 1 unknown left

        self._phase4()
        self._refresh_ui()

    def _phase1(self):
        """Good peaks → concentration estimate.

        Trusts the user's annotation_quality == 'good' marking completely.
        Peaks without overlap are the strongest anchors; overlapping peaks
        that were explicitly assigned to this metabolite (via Resolve window
        or Ctrl+1) are also accepted — their area is the best available
        estimate for this metabolite's contribution.

        Uses effective nH (nH × line_intensity_fraction) so that non-uniform
        multiplets (triplet, quartet, dd, …) produce the correct concentration
        proxy regardless of which line was used as the anchor.
        """
        for met in self._metabolites:
            mid     = met['id']
            probes  = []
            anchors = []
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') != 'good':
                    continue
                rank = mp.get('peak_rank')
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                area = self._get_area(peak)
                if area <= 0:
                    continue
                eff = self._effective_nh(met, mp)
                if eff > 0:
                    probes.append(area / eff)
                    anchors.append(rank)

            if probes:
                self._conc_est[mid]     = float(np.median(probes))
                self._anchor_peaks[mid] = anchors

    # ── Anchor-quality helpers ────────────────────────────────────────────────

    def _build_line_index(self):
        """Build a sorted position index of every predicted multiplet line.

        Returns (entries, ppms) where:
          entries = [(ppm, met_id), …] sorted by ppm
          ppms    = [ppm, …]           same order, for bisect lookups

        Every metabolite's matched_peaks are scanned.  When predicted_lines are
        available (from the multiplet scorer) each individual line position is
        indexed; otherwise the matched peak PPM is used as a fallback.
        """
        entries = []
        for met in self._metabolites:
            mid = met['id']
            for mp in met['matched_peaks']:
                pred = mp.get('predicted_lines') or []
                if pred:
                    for line in pred:
                        ppm = float(line.get('ppm', 0) or 0)
                        if ppm:
                            entries.append((ppm, mid))
                else:
                    ppm = float(mp.get('peak_ppm') or mp.get('ppm') or 0)
                    if ppm:
                        entries.append((ppm, mid))
        entries.sort(key=lambda x: x[0])
        ppms = [e[0] for e in entries]
        return entries, ppms

    def _is_clean_position(self, peak_ppm, exclude_mid, tolerance, entries, ppms):
        """True if no other metabolite has a predicted line within *tolerance* ppm.

        Uses binary search → O(log N + K) per call.

        A peak is a 'clean anchor' for metabolite A when this returns True:
        no competitor B has a line predicted at that PPM, even if B is a
        triplet whose other lines DO overlap with A's other multiplet lines.
        """
        import bisect
        lo_idx = bisect.bisect_left(ppms,  peak_ppm - tolerance)
        hi_idx = bisect.bisect_right(ppms, peak_ppm + tolerance)
        for _, mid in entries[lo_idx:hi_idx]:
            if mid != exclude_mid:
                return False
        return True

    def _phase1b(self):
        """Auto-anchor: predicted-line competition check (smarter than peak_overlap).

        For metabolites that Phase 1 could not anchor (no user-marked 'good' peaks),
        this pass identifies 'clean' peaks using the multiplet scorer's predicted_lines:

          A detected peak at ppm_X is a clean anchor for metabolite A if NO other
          metabolite has a predicted multiplet line within `_anchor_tol` ppm of ppm_X.

        This correctly handles the case where a competitor is a triplet but none of
        its predicted lines land at ppm_X — even though the triplet's centre might be
        nearby, ppm_X is still free from contamination.

        The tolerance is set via the 'Anchor tol' spinbox (default 0.020 ppm).
        """
        tol             = float(self._anchor_tol.get())
        entries, ppms   = self._build_line_index()

        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est:
                continue   # already anchored by Phase 1 — don't overwrite
            probes  = []
            anchors = []
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                peak_ppm = float(peak['ppm'])
                # Core check: is this position free of competing predicted lines?
                if not self._is_clean_position(peak_ppm, mid, tol, entries, ppms):
                    continue
                area = self._get_area(peak)
                if area <= 0:
                    continue
                eff = self._effective_nh(met, mp)
                if eff > 0:
                    probes.append(area / eff)
                    anchors.append(rank)
            if probes:
                self._conc_est[mid]     = float(np.median(probes))
                self._anchor_peaks[mid] = anchors
                self._auto_anchored.add(mid)

    def _phase2(self):
        """Subtract known contributions from overlapping peaks.

        Uses effective nH (nH × line_intensity_fraction) so that each metabolite's
        expected contribution to an overlapping peak accounts for the actual
        intensity ratio of the specific multiplet line at that position.
        """
        for rank, met_names in self._peak_overlap.items():
            if len(met_names) <= 1:
                continue
            peak = self._peak_by_rank(rank)
            if peak is None:
                continue
            obs_area = self._get_area(peak)
            if obs_area <= 0:
                continue

            contribs  = {}   # met_id → absolute contribution
            for name in met_names:
                met = self._met_by_name(name)
                if met is None:
                    continue
                mid = met['id']
                if mid not in self._conc_est:
                    continue
                mp = next(
                    (m for m in met['matched_peaks']
                     if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                eff = self._effective_nh(met, mp)
                contribs[mid] = self._conc_est[mid] * eff

            total_known = sum(contribs.values())
            residual    = max(0.0, obs_area - total_known)

            result = {mid: v / obs_area * 100 for mid, v in contribs.items()}
            result['_residual_pct'] = residual / obs_area * 100
            result['_observed']     = obs_area
            self._contributions[rank] = result

    def _phase3(self):
        """Iterative: residual → new anchor if only 1 unknown metabolite left.

        Uses effective nH throughout so the cascade correctly propagates the
        J-ratio correction: once metabolite C's concentration is inferred from
        a residual, its contribution to its own overlapping peaks is computed
        with the proper line fraction.
        """
        changed = True
        while changed:
            changed = False
            for rank, info in list(self._contributions.items()):
                if info.get('_residual_pct', 100) < 5:
                    continue   # already well-resolved
                met_names = self._peak_overlap.get(rank, [])
                unknowns  = [
                    self._met_by_name(n) for n in met_names
                    if self._met_by_name(n) is not None
                    and self._met_by_name(n)['id'] not in self._conc_est
                ]
                if len(unknowns) != 1:
                    continue
                met = unknowns[0]
                mid = met['id']
                mp  = next(
                    (m for m in met['matched_peaks']
                     if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                # Residual area for this unknown — use effective_nh for known mets
                peak     = self._peak_by_rank(rank)
                obs_area = self._get_area(peak) if peak else 0
                known    = sum(
                    self._conc_est[self._met_by_name(n)['id']]
                    * self._effective_nh(
                        self._met_by_name(n),
                        next((m for m in self._met_by_name(n)['matched_peaks']
                              if m.get('peak_rank') == rank), {}))
                    for n in self._peak_overlap.get(rank, [])
                    if self._met_by_name(n) is not None
                    and self._met_by_name(n)['id'] in self._conc_est
                )
                residual_area = max(0.0, obs_area - known)
                eff = self._effective_nh(met, mp)
                if eff > 0 and residual_area > 0:
                    new_probe = residual_area / eff
                    # Merge with existing estimate if any
                    prev = self._conc_est.get(mid)
                    self._conc_est[mid] = (
                        float(np.mean([prev, new_probe])) if prev else new_probe)
                    self._anchor_peaks.setdefault(mid, []).append(rank)
                    changed = True

            if changed:
                self._phase2()  # recompute contributions with new conc_ests

    def _phase4(self):
        """Assign status to every metabolite.

        Statuses (in decreasing confidence):
          quantifiable  – has at least 1 unambiguous user-marked 'good' anchor
          auto_anchored – anchored via Phase 1b (clean peak, not user-marked)
          resolved      – no direct anchor, resolved via Phase 3 cascade residual
          unresolvable  – no concentration estimate possible
        """
        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est:
                # Check whether any anchor peak is truly unambiguous (no overlap)
                unambiguous = [
                    r for r in self._anchor_peaks.get(mid, [])
                    if len([n for n in self._peak_overlap.get(r, [])
                            if n != met['name']]) == 0
                ]
                if unambiguous:
                    # Distinguish manually confirmed from auto-detected
                    if mid in self._auto_anchored:
                        self._status[mid] = 'auto_anchored'
                    else:
                        self._status[mid] = 'quantifiable'
                else:
                    self._status[mid] = 'resolved'
            else:
                self._status[mid] = 'unresolvable'

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top control bar ────────────────────────────────────────────────
        ctrl = ttk.Frame(self.win, padding=(6, 4))
        ctrl.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl, text='Overlap Resolving Method',
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Checkbutton(ctrl, text='Phase 3 (iterative)',
                        variable=self._iterate_var).pack(side=tk.LEFT, padx=12)
        ttk.Label(ctrl, text='Anchor tol (ppm):',
                  font=('Arial', 8)).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(ctrl, from_=0.005, to=0.10, increment=0.005, width=6,
                    textvariable=self._anchor_tol,
                    format='%.3f').pack(side=tk.LEFT)
        ttk.Button(ctrl, text='▶ Run', command=self._run).pack(
            side=tk.LEFT, padx=8)
        self._summary_lbl = ttk.Label(ctrl, text='', foreground='#555',
                                      font=('Arial', 8))
        self._summary_lbl.pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text='Export CSV',
                   command=self._export).pack(side=tk.RIGHT, padx=4)
        self._apply_btn = ttk.Button(ctrl, text='✓ Apply to Quantification',
                                     command=self._apply_to_quant,
                                     style='Accent.TButton')
        self._apply_btn.pack(side=tk.RIGHT, padx=4)
        self._apply_lbl = ttk.Label(ctrl, text='', foreground='#2e7d32',
                                    font=('Arial', 8, 'bold'))
        self._apply_lbl.pack(side=tk.RIGHT, padx=2)

        ttk.Separator(self.win, orient='horizontal').pack(fill=tk.X, padx=4)

        # ── Main split ────────────────────────────────────────────────────
        split = tk.PanedWindow(self.win, orient=tk.HORIZONTAL,
                               sashwidth=5, sashrelief=tk.RAISED)
        split.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: metabolite result table
        left = ttk.Frame(split)
        split.add(left, minsize=340, stretch='always')
        self._build_table(left)

        # Right: heatmap
        right = ttk.LabelFrame(split, text='Contribution heatmap', padding=2)
        split.add(right, minsize=520, stretch='always')
        self._build_heatmap_panel(right)

    def _build_table(self, parent):
        ttk.Label(parent,
                  text='Metabolites with overlapping peaks',
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=4, pady=(2, 0))
        ttk.Label(parent,
                  text='✓ Quantifiable: user-marked  '
                       '⊙ Auto-anchored: clean peak  '
                       '⚠ Resolved: cascade  '
                       '✗ Unresolvable',
                  font=('Arial', 7), foreground='#888').pack(
            anchor=tk.W, padx=4, pady=(0, 0))

        # Anchor summary line (updated in _refresh_ui)
        self._anchor_lbl = ttk.Label(
            parent, text='', font=('Arial', 7), foreground='#1a6e1a')
        self._anchor_lbl.pack(anchor=tk.W, padx=4, pady=(0, 2))

        COLS = [
            ('status',   'Status',        80, tk.CENTER),
            ('name',     'Metabolite',   160, tk.W),
            ('anchors',  'Anchors',       58, tk.CENTER),
            ('n_good',   'Good peaks',    68, tk.CENTER),
            ('rel_conc', 'Rel. conc',     72, tk.CENTER),
        ]
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        self._tree = ttk.Treeview(
            parent,
            columns=[c[0] for c in COLS],
            show='headings',
            selectmode='browse',
            yscrollcommand=vsb.set)
        vsb.config(command=self._tree.yview)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        for cid, heading, width, anchor in COLS:
            self._tree.heading(cid, text=heading)
            self._tree.column(cid, width=width, anchor=anchor,
                              minwidth=30, stretch=(cid == 'name'))

        self._tree.tag_configure('quant',     background='#e8ffe8',
                                 font=('Arial', 9, 'bold'))
        self._tree.tag_configure('auto',      background='#e0f7fa')
        self._tree.tag_configure('resolved',  background='#fff8e0')
        self._tree.tag_configure('unresolv',  background='#ffe8e8',
                                 foreground='#880000')

    def _build_heatmap_panel(self, parent):
        self._hm_fig = plt.Figure(figsize=(7, 4), dpi=100)
        self._hm_fig.patch.set_facecolor('#fafafa')
        self._hm_ax  = self._hm_fig.add_subplot(111)
        self._hm_canvas = FigureCanvasTkAgg(self._hm_fig, master=parent)
        self._hm_canvas.draw()
        self._hm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Refresh UI after run ──────────────────────────────────────────────────

    def _refresh_ui(self):
        self._update_anchor_label()
        self._populate_table()
        self._draw_heatmap()
        self._update_summary()

    def _update_summary(self):
        # Count only metabolites involved in overlapping peaks
        ovlp_ids = self._overlapping_met_ids()
        n_q = sum(1 for mid, v in self._status.items()
                  if v == 'quantifiable' and mid in ovlp_ids)
        n_a = sum(1 for mid, v in self._status.items()
                  if v == 'auto_anchored' and mid in ovlp_ids)
        n_r = sum(1 for mid, v in self._status.items()
                  if v == 'resolved' and mid in ovlp_ids)
        n_u = sum(1 for mid, v in self._status.items()
                  if v == 'unresolvable' and mid in ovlp_ids)
        self._summary_lbl.config(
            text=(f"✓ {n_q} quantifiable  ⊙ {n_a} auto-anchored  "
                  f"⚠ {n_r} resolved  ✗ {n_u} unresolvable"))

    def _overlapping_met_ids(self):
        """IDs of metabolites involved in at least one overlapping peak."""
        ids = set()
        for rank, names in self._peak_overlap.items():
            if len(names) > 1:
                for name in names:
                    met = self._met_by_name(name)
                    if met:
                        ids.add(met['id'])
        return ids

    def _update_anchor_label(self):
        """Show which metabolites are used as pure anchors (only unique Good peaks)."""
        lbl = getattr(self, '_anchor_lbl', None)
        if lbl is None:
            return
        ovlp_ids = self._overlapping_met_ids()
        anchors = []
        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est and mid not in ovlp_ids:
                anchors.append(met['name'])
        auto_anch_names = [m['name'] for m in self._metabolites
                           if m['id'] in self._auto_anchored
                           and m['id'] not in ovlp_ids]
        if anchors or auto_anch_names:
            parts = []
            if anchors:
                parts.append('User-marked: ' + ', '.join(anchors))
            if auto_anch_names:
                parts.append('Auto-detected: ' + ', '.join(auto_anch_names))
            txt = ' | '.join(parts)
        else:
            n_good = sum(1 for m in self._metabolites
                         for mp in m['matched_peaks']
                         if mp.get('annotation_quality') == 'good')
            n_auto = len(self._auto_anchored)
            if n_good == 0 and n_auto == 0:
                txt = '⚠ No anchors found — mark peaks as Good or ensure non-overlapping matched peaks exist'
            else:
                txt = (f'{n_good} Good peak(s), {n_auto} auto-anchored — '
                       f'no unique external anchors resolved')
        lbl.config(text=txt)

    def _populate_table(self):
        self._tree.delete(*self._tree.get_children(''))

        # Only show metabolites involved in overlapping peaks
        ovlp_ids = self._overlapping_met_ids()
        order = {'quantifiable': 0, 'auto_anchored': 1,
                 'resolved': 2, 'unresolvable': 3}
        mets  = sorted(
            [m for m in self._metabolites if m['id'] in ovlp_ids],
            key=lambda m: (order.get(self._status.get(m['id'], 'unresolvable'), 3),
                           m['name']))

        for met in mets:
            mid    = met['id']
            status = self._status.get(mid, 'unresolvable')
            ce     = self._conc_est.get(mid)
            n_anch = len(self._anchor_peaks.get(mid, []))
            n_good = sum(1 for mp in met['matched_peaks']
                         if mp.get('annotation_quality') == 'good')
            tag    = {'quantifiable':  'quant',
                      'auto_anchored': 'auto',
                      'resolved':      'resolved',
                      'unresolvable':  'unresolv'}.get(status, 'unresolv')
            sym    = {'quantifiable':  '✓',
                      'auto_anchored': '⊙',
                      'resolved':      '⚠',
                      'unresolvable':  '✗'}.get(status, '✗')
            label  = {'quantifiable':  'Quantifiable',
                      'auto_anchored': 'Auto-anchored',
                      'resolved':      'Resolved',
                      'unresolvable':  'Unresolvable'}.get(status, status.capitalize())
            self._tree.insert('', tk.END,
                              tags=(tag,),
                              values=(
                                  sym + ' ' + label,
                                  met['name'],
                                  n_anch,
                                  n_good,
                                  f"{ce:.4g}" if ce else '—',
                              ))

    def _draw_heatmap(self):
        """Heatmap matrix: metabolites (rows) × overlapping peaks (columns).
        Limited to the 50 most-affected resolved metabolites and their peaks.
        """
        _MAX = 50   # cap on rows shown

        ax = self._hm_ax
        ax.clear()
        ax.set_facecolor('#f8f8f8')

        _resolved_statuses = {'quantifiable', 'auto_anchored', 'resolved'}

        # ── Peaks: only those with at least one resolved contribution ─────────
        all_resolved_ranks = sorted(
            [rank for rank, info in self._contributions.items()
             if any(mid in self._status
                    and self._status[mid] in _resolved_statuses
                    for mid in info if not str(mid).startswith('_'))],
            key=lambda r: (self._peak_by_rank(r) or {}).get('ppm', 0),
            reverse=True)

        if not all_resolved_ranks:
            ax.text(0.5, 0.5,
                    'No peaks resolved yet.\n'
                    'Mark a peak as Good in Annotation Results\n'
                    'or ensure a non-overlapping matched peak exists.',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=9, linespacing=1.7)
            self._hm_canvas.draw_idle()
            return

        # ── Metabolites: resolved only, ranked by number of overlapping peaks,
        #    capped at _MAX ────────────────────────────────────────────────────
        resolved_met_ids = {mid for mid, st in self._status.items()
                            if st in _resolved_statuses}

        def _overlap_count(met):
            """Number of resolved peaks this metabolite is involved in."""
            return sum(
                1 for r in all_resolved_ranks
                if met['name'] in self._peak_overlap.get(r, []))

        order_status = {'quantifiable': 0, 'auto_anchored': 1, 'resolved': 2}
        all_resolved_mets = sorted(
            [m for m in self._metabolites if m['id'] in resolved_met_ids],
            key=lambda m: (-_overlap_count(m),
                           order_status.get(self._status.get(m['id'], 'resolved'), 2),
                           m['name']))

        shown_mets = all_resolved_mets[:_MAX]
        truncated  = len(all_resolved_mets) > _MAX

        if not shown_mets:
            ax.text(0.5, 0.5, 'No resolved metabolites to display.',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=9)
            self._hm_canvas.draw_idle()
            return

        # ── Keep only peaks where at least one *shown* metabolite appears ─────
        shown_names = {m['name'] for m in shown_mets}
        ovlp_ranks  = [r for r in all_resolved_ranks
                       if any(n in shown_names
                              for n in self._peak_overlap.get(r, []))]

        n_rows = len(shown_mets)
        n_cols = len(ovlp_ranks)

        # ── Adaptive figure & font sizes ──────────────────────────────────────
        fs_cell = max(5, min(9,  int(140 / max(n_rows, n_cols))))
        fs_axis = max(5, min(8,  int(120 / max(n_rows, 1))))

        fig_h = max(3.5, 0.32 * n_rows + 1.5)
        fig_w = max(4.0, 0.55 * n_cols + 2.8)
        self._hm_fig.set_size_inches(fig_w, fig_h)

        # ── Colour matrix ─────────────────────────────────────────────────────
        img = np.ones((n_rows, n_cols, 4))

        for ci, rank in enumerate(ovlp_ranks):
            info = self._contributions.get(rank, {})
            for ri, met in enumerate(shown_mets):
                mid  = met['id']
                name = met['name']
                if name not in self._peak_overlap.get(rank, []):
                    img[ri, ci] = [0.95, 0.95, 0.95, 1]   # not involved — grey
                    continue
                is_anchor    = rank in self._anchor_peaks.get(mid, [])
                is_auto_anch = is_anchor and mid in self._auto_anchored
                if is_anchor:
                    img[ri, ci] = mcolors.to_rgba(
                        _COL_AUTO if is_auto_anch else _COL_ANCHOR, alpha=0.85)
                elif mid in info:
                    pct = info[mid]
                    rc, gc, bc, _ = mcolors.to_rgba(_COL_RESOLVED)
                    t = min(pct / 100.0, 1.0)
                    img[ri, ci] = [1-t*(1-rc), 1-t*(1-gc), 1-t*(1-bc), 1]
                else:
                    img[ri, ci] = mcolors.to_rgba(_COL_RESIDUAL, alpha=0.30)

        ax.imshow(img, aspect='auto', interpolation='none',
                  extent=[-0.5, n_cols-0.5, n_rows-0.5, -0.5])

        # ── Cell text ─────────────────────────────────────────────────────────
        for ci, rank in enumerate(ovlp_ranks):
            info = self._contributions.get(rank, {})
            for ri, met in enumerate(shown_mets):
                mid  = met['id']
                name = met['name']
                if name not in self._peak_overlap.get(rank, []):
                    continue
                is_anchor    = rank in self._anchor_peaks.get(mid, [])
                is_auto_anch = is_anchor and mid in self._auto_anchored
                if is_anchor:
                    txt = '⊙' if is_auto_anch else '✓'
                    col = 'white'
                elif mid in info:
                    pct = info[mid]
                    txt = f"{pct:.0f}%"
                    col = 'white' if pct > 45 else '#222'
                else:
                    txt = '…'
                    col = '#666'
                ax.text(ci, ri, txt, ha='center', va='center',
                        fontsize=fs_cell, color=col, fontweight='bold')

        # ── Residual per column ───────────────────────────────────────────────
        for ci, rank in enumerate(ovlp_ranks):
            resid = self._contributions.get(rank, {}).get('_residual_pct', 0)
            if resid > 2:
                ax.text(ci, n_rows - 0.48,
                        f"R:{resid:.0f}%", ha='center', va='bottom',
                        fontsize=max(4, fs_cell - 1),
                        color='#c45000', style='italic',
                        clip_on=False)

        # ── Axes ──────────────────────────────────────────────────────────────
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [f"#{r}\n{(self._peak_by_rank(r) or {}).get('ppm', r):.3f}"
             for r in ovlp_ranks],
            fontsize=fs_axis, rotation=45 if n_cols > 6 else 0,
            ha='right' if n_cols > 6 else 'center')

        ax.set_yticks(range(n_rows))
        max_ch = max(14, int(200 / max(n_rows, 1)))
        ax.set_yticklabels(
            [m['name'][:max_ch] + ('…' if len(m['name']) > max_ch else '')
             for m in shown_mets],
            fontsize=fs_axis)

        title = 'J-Resolve: contributions  (% of observed area)'
        if truncated:
            title += f'  [top {_MAX} of {len(all_resolved_mets)} metabolites]'
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xlabel('Overlapping peaks  (rank / ppm)', fontsize=7)

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        # ── Compact legend ────────────────────────────────────────────────────
        import matplotlib.patches as mpatches
        leg_items = [
            mpatches.Patch(color=_COL_ANCHOR,   label='✓ Anchor (user)'),
            mpatches.Patch(color=_COL_AUTO,      label='⊙ Auto-anchor'),
            mpatches.Patch(color=_COL_RESOLVED,  label='Known %'),
            mpatches.Patch(color=_COL_RESIDUAL,  label='Residual'),
        ]
        ax.legend(handles=leg_items, fontsize=6, loc='lower right',
                  framealpha=0.85, ncol=2, handlelength=1.0,
                  borderpad=0.5, labelspacing=0.3, handletextpad=0.4)

        self._hm_fig.tight_layout(pad=1.0)
        self._hm_canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        import csv
        fn = filedialog.asksaveasfilename(
            title='Export Resolving Results',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
            initialfile='resolving_results.csv',
            parent=self.win)
        if not fn:
            return
        try:
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Metabolite', 'ID', 'Status',
                            'Anchor peaks', 'Rel. conc estimate',
                            'Peak rank', 'Peak PPM',
                            'Contribution %', 'Residual %'])
                for met in self._metabolites:
                    mid    = met['id']
                    status = self._status.get(mid, '?')
                    ce     = self._conc_est.get(mid, '')
                    for rank, info in self._contributions.items():
                        peak = self._peak_by_rank(rank)
                        if met['name'] not in self._peak_overlap.get(rank, []):
                            continue
                        ppm   = peak['ppm'] if peak else ''
                        pct   = info.get(mid, '')
                        resid = info.get('_residual_pct', '')
                        w.writerow([
                            met['name'], mid, status,
                            '; '.join(str(r) for r in self._anchor_peaks.get(mid, [])),
                            f"{ce:.6g}" if ce else '',
                            rank, f"{ppm:.5f}" if ppm else '',
                            f"{pct:.1f}" if pct != '' else '',
                            f"{resid:.1f}" if resid != '' else '',
                        ])
            messagebox.showinfo('Exported', f"Saved to:\n{fn}", parent=self.win)
        except Exception as e:
            messagebox.showerror('Error', str(e), parent=self.win)

    # ── Apply to Quantification ───────────────────────────────────────────────

    def _apply_to_quant(self):
        """Save J-Resolve contributions to annotation_stage so quantification
        can use them when 'J-Resolve correction' is enabled."""
        if not self._contributions:
            messagebox.showinfo('J-Resolve', 'Run the algorithm first (▶ Run).',
                                parent=self.win)
            return

        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation') \
              if self.app else None
        if ann is None:
            messagebox.showwarning('J-Resolve',
                                   'Annotation stage not found.',
                                   parent=self.win)
            return

        # contributions: {rank: {met_id: pct, '_residual_pct': pct, '_observed': float}}
        # Store a clean copy: {rank: {met_id: fraction (0‒1)}}
        stored = {}
        for rank, info in self._contributions.items():
            frac = {}
            for k, v in info.items():
                if k.startswith('_'):
                    continue
                # v is already a % (0‒100); convert to fraction for easy multiplication
                met = self._met_by_name(k)
                mid = met['id'] if met else k
                frac[mid] = float(v) / 100.0
            stored[rank] = frac

        ann._j_resolve_contributions = stored
        ann._j_resolve_status        = dict(self._status)
        ann._j_resolve_conc_est      = dict(self._conc_est)

        n = len(stored)
        self._apply_lbl.config(text=f'Applied ({n} peaks)')
        messagebox.showinfo(
            'J-Resolve applied',
            f'{n} overlapping peak(s) registered.\n\n'
            'Enable "J-Resolve correction" in Quantification\n'
            'to use these fractional areas.',
            parent=self.win)

    def _center(self):
        self.win.update_idletasks()
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f'{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}')
