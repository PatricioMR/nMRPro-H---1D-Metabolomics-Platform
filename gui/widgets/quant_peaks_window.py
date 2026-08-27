# gui/widgets/quant_peaks_window.py
"""
Quant Peaks Window.

Shows all candidate peaks for a given analyte with full quantification
results (computed in _run via _all_peak_quant) per group.
The user can select one and click Apply to override the peak used.

Opened via right-click → "Quant Peaks" in the Quantification tree.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


class QuantPeaksWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    quant_stage : QuantificationStage instance
    result      : one entry from quant_stage._quant_results
    """

    # Fixed columns before per-group mM columns
    _FIXED_COLS = [
        ('current',     '',            28,  'center'),
        ('anchor',      'Overlap%',    72,  'center'),
        ('rank',        'Rank',        50,  'center'),
        ('ppm',         'ppm',         72,  'center'),
        ('snr',         'SNR',         60,  'center'),
        ('width',       'Width(ppm)',  80,  'center'),
        ('r2',          'R²',          55,  'center'),
        ('ann_score',   'Ann Score',   75,  'center'),
        ('overlap',     'Overlap',     60,  'center'),
        ('ann_quality', 'Ann Quality', 90,  'center'),
        ('mean_mm',     'Mean mM',     80,  'center'),
    ]

    def __init__(self, parent, quant_stage, result):
        self.qs     = quant_stage
        self.result = result
        # The window is visualisation-only at the peak level: quantification
        # works per multiplicity. The user selects a whole multiplet (header
        # row) to use for the analyte; applying it re-quantifies every analyte.
        self._selected_hdr   = None   # iid of the currently selected header
        self._hdr_members    = {}     # hdr_iid → [candidate, ...]
        self._original_anchor = self._current_anchor_rank(result)

        self.win = tk.Toplevel(parent)
        self.win.title(f"Quant Signals — {result['metabolite']}")
        self.win.resizable(True, True)

        self._groups       = self._get_groups()
        self._candidates   = self._gather_candidates()
        self._build_ui()

    # ── Close ─────────────────────────────────────────────────────────────────

    def _do_close(self):
        """Close button and WM_DELETE_WINDOW both route here."""
        cb = getattr(self, '_close_cb', None)
        if cb:
            cb()
        else:
            self.win.destroy()

    # ── Current selection helpers ─────────────────────────────────────────────

    @staticmethod
    def _current_anchor_rank(result):
        """Anchor peak_rank of the multiplet currently quantifying this analyte."""
        return result.get('_peak_rank')

    # ── Live update ───────────────────────────────────────────────────────────

    def load_result(self, result):
        """Switch window to a different metabolite result without reopening."""
        self.result          = result
        self._selected_hdr   = None
        self._hdr_members    = {}
        self._original_anchor = self._current_anchor_rank(result)
        self.win.title(f"Quant Signals — {result['metabolite']}")
        self._candidates     = self._gather_candidates()
        self._populate_tree()

    # ── Data collection ───────────────────────────────────────────────────────

    def _get_groups(self):
        """Return ordered group names from samples_info, excluding BLK."""
        si = getattr(self.qs.app, 'samples_info', None) or []
        seen, groups = set(), []
        for s in si:
            lbl = s.get('label', 'Unknown')
            if str(s.get('type', lbl)).upper() == 'BLK':
                continue
            if lbl not in seen:
                seen.add(lbl)
                groups.append(lbl)
        return groups or ['All']

    def _gather_candidates(self):
        """Collect candidates directly from _iid_to_subchild for this metabolite.

        The list is authoritative: it contains exactly the peaks shown in
        Annotation Results for this metabolite, grouped by DB multiplet (dp).
        No additional search is performed.

        Fallback to results hits only when _iid_to_subchild has no entries
        for this metabolite (singlets or metabolites without subchild rows).
        """
        hmdb_id   = str(self.result.get('hmdb_id', ''))
        ann_stage = getattr(self.qs, 'annotation_stage', None)
        ann_args  = getattr(ann_stage, '_last_results_args', None) or {}
        all_hits  = ann_args.get('results', [])

        peaks_by_rank = {p['rank']: p
                         for p in (self.qs.peak_stage.peaks or [])
                         if p.get('rank') is not None}
        apq        = getattr(self.qs, '_all_peak_quant', {}).get(hmdb_id, {})
        current_rk = self.result.get('_peak_rank')

        def _make_candidate(peak, hit, pr, dp=None):
            ppm_r = round(float(peak.get('ppm', 0)), 3)
            quant = apq.get((pr, ppm_r)) or apq.get(pr, {})
            _dp_d   = dp or {}
            dp_ppm  = float(_dp_d.get('ppm_center', _dp_d.get('ppm', _dp_d.get('shift', 0))) or 0)
            dp_mult = str(_dp_d.get('type', _dp_d.get('multiplicity', '?')) or '?').lower().strip()
            return {
                'peak':           peak,
                'hit':            hit,
                'peak_rank':      pr,
                'hit_ppm':        float(hit.get('_hit_ppm', peak.get('ppm', 0))),
                'is_current':     (pr == current_rk),
                'mean_mm':        quant.get('conc_mm', float('nan')),
                'concs_by_group': quant.get('concs_by_group', {}),
                # DB multiplet identity — groups match Annotation Results exactly
                '_dp_key':   (round(dp_ppm, 3), dp_mult) if dp is not None else None,
                '_dp_ppm':   dp_ppm,
                '_dp_mult':  dp_mult,
                '_dp':       dp,
            }

        # ── Primary: _iid_to_subchild — identical to Annotation Results tree ──
        _annot_win = getattr(getattr(self.qs, 'app', None), '_annot_win', None)
        _iid_to_sc = getattr(_annot_win, '_iid_to_subchild', {}) or {}
        sc_ranks   = set()
        candidates = []

        def _met_matches(met):
            for field in ('id', 'hmdb_id', 'bmrb_id'):
                v = met.get(field)
                if v is not None and str(v) == hmdb_id:
                    return True
            return False

        for _sc in _iid_to_sc.values():
            try:
                _met, _dp, _mp_ref, _line_dict, _line_idx, _nearest_pk = _sc
            except (ValueError, TypeError):
                continue
            if _nearest_pk is None:
                continue
            if not _met_matches(_met):
                continue
            pr = _nearest_pk.get('rank')
            if pr is None or pr in sc_ranks:
                continue
            sc_ranks.add(pr)
            c = _make_candidate(_nearest_pk, dict(_mp_ref) if _mp_ref else {}, pr, _dp)
            c['_rel_intensity'] = float(_line_dict.get('rel_intensity', 1.0)
                                        if _line_dict else 1.0)
            candidates.append(c)

        # ── Fallback: ann._metabolites (annotation window closed / loaded from save) ─
        # _metabolites is serialised in the project file — always available after load.
        # Reconstruct subchild-like candidates from matched_peaks + predicted_lines.
        if not sc_ranks:
            _ann_mets = getattr(ann_stage, '_metabolites', None) or []
            for _met in _ann_mets:
                if not _met_matches(_met):
                    continue
                # Build db_peak lookup by rounded db ppm
                _db_peak_by_dp = {
                    round(float(_dp.get('ppm_center', _dp.get('ppm', 0)) or 0), 3): _dp
                    for _dp in _met.get('db_peaks', [])
                }
                # matched_mp_by_dbppm: already built during annotation
                _mp_by_dp = {
                    round(float(_mp.get('db_ppm', 0) or 0), 3): _mp
                    for _mp in _met.get('matched_peaks', [])
                }
                seen_pr = set()
                for _dp_ppm_r, _dp in sorted(_db_peak_by_dp.items(),
                                              key=lambda x: -x[0]):
                    _mp = _mp_by_dp.get(_dp_ppm_r)
                    if _mp is None:
                        continue
                    # Members of the multiplet = the predicted-line peaks (what the
                    # Annotation tree shows), NOT the matched_peak anchor, which for
                    # a multiplet is the CENTRE and can be a stray peak between the
                    # real lines (e.g. 461 between Alanine's doublet 49/51).
                    _line_ppms = [float(l['ppm'])
                                  for l in (_mp.get('predicted_lines') or [])
                                  if 'ppm' in l]
                    _line_ranks = []
                    if _line_ppms:
                        _used_ln = set()
                        for _lp in _line_ppms:
                            _best, _bd = None, 0.015
                            for _pk in (self.qs.peak_stage.peaks or []):
                                _rk = _pk.get('rank')
                                if _rk is None or _rk in _used_ln:
                                    continue
                                _d = abs(float(_pk.get('ppm', 1e9)) - _lp)
                                if _d < _bd:
                                    _bd, _best = _d, _rk
                            if _best is not None:
                                _line_ranks.append(_best)
                                _used_ln.add(_best)
                    if not _line_ranks:
                        _line_ranks = [_mp.get('peak_rank')]
                    for pr in _line_ranks:
                        if pr is None or pr in sc_ranks or pr in seen_pr:
                            continue
                        peak = peaks_by_rank.get(pr)
                        if peak is None:
                            continue
                        seen_pr.add(pr)
                        c = _make_candidate(peak, dict(_mp), pr, _dp)
                        c['_rel_intensity'] = float(_mp.get('rel_intensity', 1.0) or 1.0)
                        candidates.append(c)
                        sc_ranks.add(pr)

        # ── Last-resort fallback: flat results hits ────────────────────────────
        # Only when ann._metabolites also has no data (very old saves).
        if not sc_ranks:
            def _hit_ids(h):
                ids = set()
                for field in ('_mid', 'hmdb_id', 'bmrb_id', 'id'):
                    v = h.get(field)
                    if v is not None:
                        ids.add(str(v))
                if h.get('name'):
                    ids.add(str(h['name']))
                return ids

            by_key = {}
            for h in all_hits:
                if hmdb_id not in _hit_ids(h):
                    continue
                pr = h.get('peak_rank')
                if pr is None:
                    continue
                hit_ppm = float(h.get('peak_ppm', h.get('ppm', 0)) or 0)
                score   = float(h.get('composite_score', h.get('multiplet_score', 0)) or 0)
                if pr not in by_key or score > by_key[pr]['_score']:
                    by_key[pr] = dict(h, _score=score, _hit_ppm=hit_ppm)

            for pr, hit in sorted(by_key.items()):
                peak = peaks_by_rank.get(pr)
                if peak is None:
                    continue
                candidates.append(_make_candidate(peak, hit, pr, dp=None))

        # Inject composite_score from best annotation hit into subchild candidates
        # whose _mp_ref (matched_peaks entry) lacks it.
        _best_score = float(self.result.get('_ann_score', 0) or 0)
        if sc_ranks:
            for h in all_hits:
                s = float(h.get('composite_score', h.get('multiplet_score', 0)) or 0)
                if s > _best_score:
                    _best_score = s
        for c in candidates:
            if not c['hit'].get('composite_score') and not c['hit'].get('multiplet_score'):
                c['hit']['composite_score'] = _best_score

        # All peaks in the multiplicity family are included regardless of quality.
        # Bad-quality marks are reflected visually (grey row) and block Apply,
        # but do not exclude a peak from candidates or from best-peak selection.

        # Sort: by DB multiplet ppm desc (matches Annotation Results order),
        # then by detected ppm desc within group.
        candidates.sort(key=lambda c: (
            -c['_dp_ppm'] if c['_dp_key'] else -float(c['peak'].get('ppm', c['hit_ppm'])),
            -float(c['peak'].get('ppm', c['hit_ppm'])),
        ))
        return candidates

    # ── Quality helpers ───────────────────────────────────────────────────────

    def _effective_quality(self, peak_rank):
        """Return 'good' / 'bad' / 'undefined' / None for a peak rank.

        Manual mark in _quality_marks takes priority over annotation_quality
        stored in the hit dict.  Tries all possible ID fields to handle
        HMDB/BMRB/name format mismatches in _quality_marks keys.
        """
        ann_stage = getattr(self.qs, 'annotation_stage', None)
        qm = getattr(ann_stage, '_quality_marks', {}) or {}
        for _key_id in dict.fromkeys(filter(None, [
            str(self.result.get('_mid', '') or ''),
            str(self.result.get('hmdb_id', '') or ''),
            str(self.result.get('bmrb_id', '') or ''),
            str(self.result.get('id', '') or ''),
            str(self.result.get('name', '') or ''),
        ])):
            manual = qm.get((_key_id, peak_rank))
            if manual is not None:
                return manual
        for c in self._candidates:
            if c['peak_rank'] == peak_rank:
                return c['hit'].get('annotation_quality')
        return None

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        win = self.win

        # Header
        self._hdr_var   = tk.StringVar()
        self._count_var = tk.StringVar()
        self._refresh_header_vars()
        ttk.Label(win, textvariable=self._hdr_var,
                  font=('Arial', 10, 'bold')).pack(padx=10, pady=(8, 2), anchor='w')
        ttk.Label(win, textvariable=self._count_var,
                  foreground='#555555').pack(padx=10, anchor='w')

        # Build column list: fixed + one per group
        group_cols = [(f'g_{g}', f'{g} mM', 80, 'center') for g in self._groups]
        all_cols   = self._FIXED_COLS + group_cols
        col_ids    = [c[0] for c in all_cols]

        # Estimate window width
        total_w = sum(c[2] for c in all_cols) + 160 + 30   # +160 for tree '#0' column
        win_w   = max(800, min(total_w, 1500))
        win.geometry(f'{win_w}x420')

        frm = ttk.Frame(win)
        frm.pack(fill='both', expand=True, padx=10, pady=6)

        self._tree = ttk.Treeview(frm, columns=col_ids, show='tree headings',
                                  selectmode='browse', height=12)
        self._tree.column('#0', width=160, minwidth=120, anchor='w', stretch=False)
        self._tree.heading('#0', text='Signal')
        for cid, hdr_txt, w, anc in all_cols:
            self._tree.heading(cid, text=hdr_txt)
            self._tree.column(cid, width=w, minwidth=w, anchor=anc, stretch=False)

        vsb = ttk.Scrollbar(frm, orient='vertical', command=self._tree.yview)
        hsb = ttk.Scrollbar(frm, orient='horizontal', command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        self._tree.tag_configure('current', background='#e8f5e9',
                                 font=('Arial', 9, 'bold'))
        self._tree.tag_configure('normal', background='')

        self._group_cols = [c[0] for c in group_cols]

        # Bottom bar — must exist before _populate_tree() calls selection_set()
        # which fires <<TreeviewSelect>> → _on_select accesses _apply_btn.
        btn_frm = ttk.Frame(win)
        btn_frm.pack(fill='x', padx=10, pady=(2, 8))

        self._apply_btn = ttk.Button(btn_frm, text='Use multiplet',
                                     command=self._apply_selected, state='disabled')
        self._apply_btn.pack(side='right', padx=(4, 0))
        ttk.Button(btn_frm, text='Close',
                   command=self._do_close).pack(side='right')
        self._info_lbl = ttk.Label(
            btn_frm,
            text='Select a multiplet (header row) to quantify this analyte with it.',
            foreground='#555555')
        self._info_lbl.pack(side='left')

        self._tree.bind('<<TreeviewSelect>>', self._on_select)
        self._tree.bind('<ButtonRelease-1>', self._on_click)
        self._populate_tree()

    @staticmethod
    def _cluster_by_proximity(candidates, gap=0.05):
        """
        Group candidates into multiplet signals by detected ppm proximity.

        Peaks within `gap` ppm of each other belong to the same signal.
        Returns list of [members] sorted by ppm desc.
        """
        if not candidates:
            return []

        def _det_ppm(c):
            return float(c['peak'].get('ppm', c['hit_ppm']))

        sorted_c = sorted(candidates, key=lambda c: -_det_ppm(c))
        groups, current = [], [sorted_c[0]]
        for c in sorted_c[1:]:
            if abs(_det_ppm(c) - _det_ppm(current[-1])) <= gap:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
        groups.append(current)
        return groups

    def _populate_tree(self):
        tree = self._tree
        for row in tree.get_children():
            tree.delete(row)
        self._hdr_members = {}

        comp_count = getattr(self.qs, '_peak_competitor_count', {})

        _MULT_LABEL = {
            's': 'Singlet', 'br s': 'Broad singlet', 'brs': 'Broad singlet',
            'd': 'Doublet', 't': 'Triplet', 'q': 'Quartet',
            'dd': 'Doublet of doublets', 'dt': 'Doublet of triplets',
            'td': 'Triplet of doublets', 'tt': 'Triplet of triplets',
            'ddd': 'Doublet of doublet of doublets',
            'quintet': 'Quintet', 'sextet': 'Sextet', 'm': 'Multiplet',
        }

        def _overlap_pct(members):
            """
            DB-expectation overlap indicator.
            Compares each peak's actual area fraction against the expected
            fraction from annotated relative intensities (BD). A peak whose
            area exceeds its expected share has positive overlap%, indicating
            contamination from an adjacent signal.
            Returns {peak_rank: '+N.N%' | '0.0%' | '-N.N%'} or {} for singletons.
            """
            if len(members) < 2:
                return {}

            areas, rel_ints, ranks = [], [], []
            for c in members:
                pk    = c['peak']
                snr   = float(pk.get('snr', 0) or 0)
                width = float(pk.get('width_ppm', pk.get('width_at_half', 0)) or 0)
                area  = float(pk.get('area', 0) or 0)
                if area <= 0:
                    area = snr * width if width > 0 else snr
                if area <= 0:
                    continue
                ri = float(c.get('_rel_intensity', 0) or 0)
                if ri <= 0:
                    ri = 1.0  # fallback: treat as equal weight
                areas.append(area)
                rel_ints.append(ri)
                ranks.append(c['peak_rank'])

            if len(areas) < 2:
                return {}

            total_area = sum(areas)
            total_ri   = sum(rel_ints)
            if total_area <= 0 or total_ri <= 0:
                return {}

            result = {}
            for rank, area, ri in zip(ranks, areas, rel_ints):
                actual_frac   = area   / total_area
                expected_frac = ri     / total_ri
                pct = (actual_frac - expected_frac) / expected_frac * 100
                result[rank] = '0.0%' if abs(pct) < 0.05 else f'{pct:+.1f}%'
            return result

        def _best_rank_in_group(members, prop_dev):
            """Return peak_rank of the recommended peak in this multiplicity group.
            All peaks are considered regardless of quality — bad-quality marks
            reflect user judgement but do not disqualify a line from the algorithm.
            Criterion: minimum |overlap%|; tiebreak by composite_score desc.
            """
            candidates = []
            for c in members:
                ovlp_s = prop_dev.get(c['peak_rank'], '—')
                try:
                    ovlp_val = abs(float(ovlp_s.replace('%', '').replace('+', '')))
                except (ValueError, AttributeError):
                    ovlp_val = 0.0
                score = float((c['hit'] or {}).get('composite_score',
                               (c['hit'] or {}).get('multiplet_score', 0)) or 0)
                candidates.append((c['peak_rank'], ovlp_val, score))
            if not candidates:
                return None
            candidates.sort(key=lambda x: (x[1], -x[2]))
            return candidates[0][0]

        def _child_values(peak, hit, c, prop_dev, best_rank):
            pr        = c['peak_rank'] if c else '—'
            disp_ppm  = float(peak.get('ppm', c['hit_ppm'])) if peak else c['hit_ppm']
            snr       = peak.get('snr', float('nan')) if peak else float('nan')
            width     = (peak.get('width_ppm', peak.get('width_at_half', float('nan')))
                         if peak else float('nan'))
            r2        = (peak.get('shape_r2', peak.get('gaussian_r2', float('nan')))
                         if peak else float('nan'))
            score     = float((hit or {}).get('composite_score',
                               (hit or {}).get('multiplet_score', float('nan'))) or float('nan'))
            overlap   = comp_count.get(pr, (hit or {}).get('overlap_count', '—')) if c else '—'
            ann_q     = (hit or {}).get('annotation_quality') or '—'
            prop_s    = prop_dev.get(pr, '—') if c else '—'
            mm        = c['mean_mm'] if c else float('nan')
            mm_s      = f'{mm:.4f}' if np.isfinite(mm) else '—'
            cbg       = c['concs_by_group'] if c else {}
            is_curr   = c['is_current'] if c else False
            is_best   = (c is not None and pr == best_rank)
            if is_curr and is_best:
                marker = '◉'
            elif is_curr:
                marker = '●'
            elif is_best:
                marker = '○'
            else:
                marker = ''
            g_vals = []
            for g in self._groups:
                gv = cbg.get(g, [])
                g_vals.append(f'{np.mean(gv):.4f}' if gv else '—')
            return (
                marker,
                prop_s,
                str(pr),
                f'{disp_ppm:.4f}',
                f'{snr:.1f}'   if np.isfinite(snr)   else '—',
                f'{width:.4f}' if np.isfinite(width)  else '—',
                f'{r2:.3f}'    if np.isfinite(r2)     else '—',
                f'{score:.3f}' if np.isfinite(score)  else '—',
                str(overlap),
                ann_q,
                mm_s,
                *g_vals,
            )

        # Group by DB multiplet identity (_dp_key) — mirrors Annotation Results.
        # Fallback candidates (no _dp_key) group by ppm proximity (0.1 ppm gap).
        _grp_order = {}
        for c in self._candidates:
            if c['_dp_key']:
                key = c['_dp_key']
            else:
                _fp   = round(float(c['peak'].get('ppm', c['hit_ppm'])), 1)
                _fmul = (c['hit'].get('multiplicity') or c['hit'].get('type') or '?').lower()
                key   = ('_fb', _fp, _fmul)
            if key not in _grp_order:
                _grp_order[key] = []
            _grp_order[key].append(c)
        signal_groups = list(_grp_order.values())

        for hdr_idx, members in enumerate(signal_groups):
            rep_c    = members[0]
            rep_hit  = rep_c['hit']
            has_curr = any(c['is_current'] for c in members)

            # Use DB ppm + multiplicity for header — matches Annotation Results label
            dp_ppm  = rep_c['_dp_ppm']
            dp_mult = rep_c['_dp_mult']
            mult_lbl = _MULT_LABEL.get(dp_mult, dp_mult.upper() if dp_mult != '?' else '?')
            protons  = rep_hit.get('protons') or (rep_c['_dp'] or {}).get('protons') or 1
            j_str    = (rep_hit.get('j_coupling') or rep_hit.get('j_hz')
                        or (rep_c['_dp'] or {}).get('j_hz') or '')
            j_part   = f'  ·  J={j_str} Hz' if j_str else ''
            if dp_ppm:
                hdr_text = (f'{dp_ppm:.4f} ppm  [{dp_mult.upper()}]'
                            f'  ·  {mult_lbl}  ·  {protons}H{j_part}'
                            f'  ·  {len(members)} peak{"s" if len(members) != 1 else ""}')
            else:
                ppms     = [float(c['peak'].get('ppm', c['hit_ppm'])) for c in members]
                grp_ppm  = sum(ppms) / len(ppms)
                hdr_text = (f'{grp_ppm:.4f} ppm  ·  {mult_lbl}'
                            f'  ·  {protons}H{j_part}'
                            f'  ·  {len(members)} peak{"s" if len(members) != 1 else ""}')

            hdr_iid = f'hdr_{hdr_idx}'
            hdr_tag = 'hdr_current' if has_curr else 'hdr_normal'
            self._hdr_members[hdr_iid] = members
            tree.insert('', 'end', iid=hdr_iid, text=hdr_text,
                        tags=(hdr_tag,), open=True,
                        values=('', '', '', '', '', '', '', '', '', '', '',
                                *([''] * len(self._group_cols))))

            prop_dev  = _overlap_pct(members)
            best_rank = _best_rank_in_group(members, prop_dev)

            # Children sorted by ppm descending (high field → low field)
            for idx, c in enumerate(sorted(members,
                                           key=lambda c: -float(
                                               c['peak'].get('ppm', c['hit_ppm'])))):
                q = self._effective_quality(c['peak_rank'])
                if c['is_current']:
                    tag = 'current'
                elif q == 'bad':
                    tag = 'bad_quality'
                else:
                    tag = 'normal'
                row_iid = f'{hdr_iid}_{idx}'
                tree.insert(hdr_iid, 'end', iid=row_iid, tags=(tag,),
                            values=_child_values(c['peak'], c['hit'], c, prop_dev, best_rank))

        tree.tag_configure('hdr_current',  background='#c8e6c9',
                           font=('Arial', 9, 'bold italic'))
        tree.tag_configure('hdr_normal',   background='#e3f2fd',
                           font=('Arial', 9, 'italic'))
        tree.tag_configure('bad_quality',  foreground='#9e9e9e',
                           background='#fafafa')

        # Pre-select the multiplet (header) currently quantifying this analyte.
        for hdr_iid, members in self._hdr_members.items():
            if any(c['is_current'] for c in members):
                tree.selection_set(hdr_iid)
                tree.focus(hdr_iid)
                tree.see(hdr_iid)
                self._on_select()
                return

    # ── Interaction ───────────────────────────────────────────────────────────

    def _refresh_header_vars(self):
        hdr = (f"{self.result['metabolite']}  ·  "
               f"HMDB: {self.result.get('hmdb_id', '—')}  ·  "
               f"Current peak: rank {self.result.get('_peak_rank', '?')} "
               f"@ {self.result.get('ppm', 0):.4f} ppm")
        self._hdr_var.set(hdr)
        self._count_var.set(f'{len(self._candidates)} candidate peak(s)')

    @staticmethod
    def _multiplet_anchor_rank(members):
        """Anchor peak_rank used to force this multiplet in re-quantification.

        The matched-peak anchor (hit['peak_rank']) is the rank annotation
        recorded for the multiplet and the one present in the search `results`,
        so _quant_peak_overrides resolves to the hit carrying THIS multiplet's
        predicted_lines (not just a single line).
        """
        if not members:
            return None
        anchor = (members[0].get('hit') or {}).get('peak_rank')
        if anchor is not None:
            return anchor
        for c in members:
            if c.get('peak_rank') is not None:
                return c['peak_rank']
        return None

    def _multiplet_conflicts(self, members):
        """Names of OTHER metabolites annotated on any peak of this multiplet.

        Authoritative source is _iid_to_subchild (identical to Annotation
        Results); falls back to annotation_stage._metabolites.matched_peaks
        when the annotation window is closed (e.g. loaded project).
        """
        my_ids = {str(self.result.get('hmdb_id', '') or ''),
                  str(self.result.get('_mid', '') or ''),
                  str(self.result.get('bmrb_id', '') or '')}
        my_ids.discard('')
        ranks = {c['peak_rank'] for c in members if c.get('peak_rank') is not None}
        if not ranks:
            return []

        def _met_id(met):
            for f in ('id', 'hmdb_id', 'bmrb_id'):
                v = met.get(f)
                if v is not None:
                    return str(v)
            return ''

        others = {}
        annot_win  = getattr(getattr(self.qs, 'app', None), '_annot_win', None)
        iid_to_sc  = getattr(annot_win, '_iid_to_subchild', {}) or {}
        if iid_to_sc:
            for sc in iid_to_sc.values():
                try:
                    met, _dp, _mp, _ld, _li, npk = sc
                except (ValueError, TypeError):
                    continue
                if npk is None:
                    continue
                mid = _met_id(met)
                if not mid or mid in my_ids:
                    continue
                if npk.get('rank') in ranks:
                    others[mid] = met.get('name', mid)
        else:
            ann_stage = getattr(self.qs, 'annotation_stage', None)
            for met in (getattr(ann_stage, '_metabolites', []) or []):
                mid = _met_id(met)
                if not mid or mid in my_ids:
                    continue
                for mp in met.get('matched_peaks', []):
                    if mp.get('peak_rank') in ranks:
                        others[mid] = met.get('name', mid)
                        break
        return sorted(set(others.values()))

    def _on_click(self, event):
        iid = self._tree.identify_row(event.y)
        if iid:
            self._tree.selection_set(iid)
            self._on_select()

    def _on_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            self._selected_hdr = None
            self._apply_btn.config(state='disabled')
            return
        iid = sel[0]
        # Child (peak line) rows are view-only — quantification is per multiplet.
        if iid not in self._hdr_members:
            self._selected_hdr = None
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(
                text='Individual signals are view-only — select the multiplet '
                     '(header row) to quantify this analyte with it.')
            return

        self._selected_hdr = iid
        members = self._hdr_members[iid]

        if any(c['is_current'] for c in members):
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(
                text='This multiplet already quantifies the analyte.')
            return

        conflicts = self._multiplet_conflicts(members)
        if conflicts:
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(
                text='Blocked — signal annotated for: ' + ', '.join(conflicts) + '.')
            return

        if self._multiplet_anchor_rank(members) is None:
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(
                text='No detected anchor signal for this multiplet.')
            return

        self._apply_btn.config(state='normal')
        self._info_lbl.config(
            text='Apply re-quantifies all analytes using this multiplet.')

    def _apply_selected(self):
        iid = self._selected_hdr
        if iid is None or iid not in self._hdr_members:
            return
        members = self._hdr_members[iid]

        # ── Block: signal annotated for another metabolite ───────────────────
        conflicts = self._multiplet_conflicts(members)
        if conflicts:
            messagebox.showwarning(
                'Signal shared',
                'This multiplet is also annotated for:\n  '
                + '\n  '.join(conflicts)
                + f'\n\nIt cannot be used to quantify '
                  f'{self.result.get("metabolite", "")}.')
            self._apply_btn.config(state='disabled')
            return

        anchor = self._multiplet_anchor_rank(members)
        if anchor is None:
            self._info_lbl.config(text='No detected anchor signal for this multiplet.')
            return

        if not messagebox.askyesno(
                'Re-quantify all analytes',
                'Changing the multiplet for this analyte re-runs quantification '
                'for EVERY analyte.\n\nContinue?'):
            return

        hmdb_id = str(self.result.get('hmdb_id', ''))
        if getattr(self.qs, '_quant_peak_overrides', None) is None:
            self.qs._quant_peak_overrides = {}
        self.qs._quant_peak_overrides[hmdb_id] = anchor

        self._apply_btn.config(state='disabled')
        self._info_lbl.config(text='Re-quantifying all analytes…')
        # Full re-run (worker thread). Refresh the window when it finishes.
        self.qs._safe_run(on_done=self._after_requant)

    def _after_requant(self):
        """Re-acquire this analyte's (rebuilt) result and refresh the window."""
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        hmdb_id = str(self.result.get('hmdb_id', ''))
        new = next((r for r in (getattr(self.qs, '_quant_results', []) or [])
                    if str(r.get('hmdb_id', '')) == hmdb_id), None)
        if new is not None:
            self.result          = new
            self._original_anchor = self._current_anchor_rank(new)
        self._candidates = self._gather_candidates()
        self._refresh_header_vars()
        self._populate_tree()
        self._info_lbl.config(text='Re-quantified with the selected multiplet.')
