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
        self._selected_rank       = result.get('_peak_rank')
        self._original_peak_rank  = result.get('_peak_rank')
        self._preview_original    = None   # saved fields before preview
        self._has_unsaved_preview = False

        self.win = tk.Toplevel(parent)
        self.win.title(f"Quant Peaks — {result['metabolite']}")
        self.win.resizable(True, True)

        self._groups       = self._get_groups()
        self._candidates   = self._gather_candidates()
        self._build_ui()

    # ── Close ─────────────────────────────────────────────────────────────────

    def _do_close(self):
        """Close button and WM_DELETE_WINDOW both route here."""
        self._cancel_preview()
        cb = getattr(self, '_close_cb', None)
        if cb:
            cb()
        else:
            self.win.destroy()

    # ── Preview / revert ──────────────────────────────────────────────────────

    _PREVIEW_FIELDS = ('conc_mm', '_concs_by_group', 'ppm', '_peak',
                       '_peak_rank', 'snr', 'cv_by_group', '_lorentz')

    def _start_preview(self, peak_rank):
        """Apply peak_rank to self.result for preview; save original if first time."""
        import numpy as _np
        hmdb_id = str(self.result.get('hmdb_id', ''))
        apq = getattr(self.qs, '_all_peak_quant', {}).get(hmdb_id, {})
        entry = (apq.get(peak_rank)
                 or next((v for k, v in apq.items()
                          if isinstance(k, tuple) and k[0] == peak_rank), None))
        pk_by_rank = {p['rank']: p
                      for p in (self.qs.peak_stage.peaks or [])
                      if p.get('rank') is not None}
        new_peak = pk_by_rank.get(peak_rank)

        if entry is None and new_peak is None:
            return  # no data — cannot preview

        # Save original fields only on the first preview (keep until Apply or cancel)
        if not self._has_unsaved_preview:
            self._preview_original = {f: self.result.get(f) for f in self._PREVIEW_FIELDS}
            self._has_unsaved_preview = True

        if entry:
            self.result['conc_mm']         = entry['conc_mm']
            self.result['_concs_by_group'] = entry.get('concs_by_group', {})
            self.result['ppm']             = entry.get('ppm', self.result.get('ppm', 0))
            self.result['_lorentz']        = entry.get('_lorentz')
            cbg = entry.get('concs_by_group', {})
            cv_bg = {}
            for _g, _vals in cbg.items():
                if len(_vals) >= 2:
                    _m = float(_np.mean(_vals))
                    cv_bg[_g] = float(_np.std(_vals) / _m * 100) if _m > 0 else float('nan')
                else:
                    cv_bg[_g] = float('nan')
            self.result['cv_by_group'] = cv_bg
        if new_peak:
            self.result['_peak']      = new_peak
            self.result['_peak_rank'] = peak_rank
            self.result['snr']        = float(new_peak.get('snr', self.result.get('snr', 0)))

        self._refresh_quant_display()

    def _cancel_preview(self, refresh=True):
        """Revert self.result to state before preview.

        refresh=False: restore fields silently (caller will update the display).
        """
        if not self._has_unsaved_preview or self._preview_original is None:
            return
        for f, v in self._preview_original.items():
            self.result[f] = v
        self._has_unsaved_preview = False
        self._preview_original    = None
        if refresh:
            self._refresh_quant_display()

    def _refresh_quant_display(self):
        """Push self.result state to the quant module tree row and detail panel."""
        try:
            if hasattr(self.qs, '_refresh_tree_row'):
                self.qs._refresh_tree_row(self.result)
        except Exception:
            pass
        try:
            if hasattr(self.qs, '_draw_detail'):
                self.qs._draw_detail(self.result)
        except Exception:
            pass

    # ── Live update ───────────────────────────────────────────────────────────

    def load_result(self, result):
        """Switch window to a different metabolite result without reopening."""
        self._cancel_preview(refresh=False)   # revert previous metabolite quietly
        self.result               = result
        self._selected_rank       = result.get('_peak_rank')
        self._original_peak_rank  = result.get('_peak_rank')
        self._preview_original    = None
        self._has_unsaved_preview = False
        self.win.title(f"Quant Peaks — {result['metabolite']}")
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
                    pr = _mp.get('peak_rank')
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

        # Exclude bad-quality peaks — they are rejected for quantification.
        # _quality_marks key = (met['id'], peak_rank). Try all possible ID
        # fields from self.result to handle HMDB/BMRB/name format mismatches.
        ann_stage = getattr(self.qs, 'annotation_stage', None)
        _qm = getattr(ann_stage, '_quality_marks', {}) or {}
        _id_candidates = list(dict.fromkeys(filter(None, [
            str(self.result.get('_mid', '') or ''),
            str(self.result.get('hmdb_id', '') or ''),
            str(self.result.get('bmrb_id', '') or ''),
            str(self.result.get('id', '') or ''),
            str(self.result.get('name', '') or ''),
        ])))
        def _is_bad(c):
            pr = c['peak_rank']
            for _key_id in _id_candidates:
                manual = _qm.get((_key_id, pr))
                if manual is not None:
                    return manual == 'bad'
            return c['hit'].get('annotation_quality') == 'bad'
        candidates = [c for c in candidates if not _is_bad(c)]

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
        self._tree.heading('#0', text='Multiplet')
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

        self._apply_btn = ttk.Button(btn_frm, text='Apply',
                                     command=self._apply_selected, state='disabled')
        self._apply_btn.pack(side='right', padx=(4, 0))
        ttk.Button(btn_frm, text='Close',
                   command=self._do_close).pack(side='right')
        self._info_lbl = ttk.Label(btn_frm, text='Select a peak row then click Apply.',
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
            Width-based overlap indicator for each peak in a multiplet group.

            Ratio = area / height ∝ FWHM.  For peaks of the same multiplet,
            FWHM should be equal (same T2, same field).  A peak wider than its
            siblings has inflated area — indicative of overlap.

            Reference = narrowest peak (ratio_min = 0.0%).
            Others show % excess width relative to reference.
            """
            entries = []
            for c in members:
                pk    = c['peak']
                snr   = float(pk.get('snr', 0) or 0)
                width = float(pk.get('width_ppm', pk.get('width_at_half', 0)) or 0)
                if snr <= 0:
                    continue
                ratio = width if width > 0 else 1.0   # area/height ∝ width
                entries.append((c['peak_rank'], ratio))

            if len(entries) < 2:
                return {}

            ratio_ref = min(r for _, r in entries)
            if ratio_ref <= 0:
                return {}

            return {
                rank: f'{(ratio - ratio_ref) / ratio_ref * 100:.1f}%'
                for rank, ratio in entries
            }

        def _child_values(peak, hit, c, prop_dev):
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
            g_vals = []
            for g in self._groups:
                gv = cbg.get(g, [])
                g_vals.append(f'{np.mean(gv):.4f}' if gv else '—')
            return (
                '●' if (c and c['is_current']) else '',
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
            tree.insert('', 'end', iid=hdr_iid, text=hdr_text,
                        tags=(hdr_tag,), open=True,
                        values=('', '', '', '', '', '', '', '', '', '', '',
                                *([''] * len(self._group_cols))))

            prop_dev = _overlap_pct(members)

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
                            values=_child_values(c['peak'], c['hit'], c, prop_dev))

        tree.tag_configure('hdr_current',  background='#c8e6c9',
                           font=('Arial', 9, 'bold italic'))
        tree.tag_configure('hdr_normal',   background='#e3f2fd',
                           font=('Arial', 9, 'italic'))
        tree.tag_configure('bad_quality',  foreground='#9e9e9e',
                           background='#fafafa')

        # Pre-select current peak
        cr = self.result.get('_peak_rank')
        if cr is not None:
            for hdr_iid in tree.get_children():
                for iid in tree.get_children(hdr_iid):
                    if tree.set(iid, 'rank') == str(cr):
                        tree.selection_set(iid)
                        tree.focus(iid)
                        tree.see(iid)
                        return

    # ── Interaction ───────────────────────────────────────────────────────────

    def _refresh_header_vars(self):
        hdr = (f"{self.result['metabolite']}  ·  "
               f"HMDB: {self.result.get('hmdb_id', '—')}  ·  "
               f"Current peak: rank {self.result.get('_peak_rank', '?')} "
               f"@ {self.result.get('ppm', 0):.4f} ppm")
        self._hdr_var.set(hdr)
        self._count_var.set(f'{len(self._candidates)} candidate peak(s)')

    def _on_click(self, event):
        iid = self._tree.identify_row(event.y)
        if iid and not iid.startswith('hdr_'):
            self._tree.selection_set(iid)
            self._on_select()

    def _on_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            self._apply_btn.config(state='disabled')
            return
        iid = sel[0]
        if not self._tree.parent(iid):
            self._apply_btn.config(state='disabled')
            return
        rank_str = self._tree.set(iid, 'rank')
        try:
            pr = int(rank_str)
        except (ValueError, TypeError):
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(text='No detected peak at this line position.')
            return
        if pr == self._original_peak_rank:
            # User re-selected the original peak — cancel any active preview
            if self._has_unsaved_preview:
                self._cancel_preview()
                self._populate_tree()
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(text='Already the current peak.')
            return
        if self._effective_quality(pr) == 'bad':
            # Cancel preview if user clicked a bad peak after a valid preview
            if self._has_unsaved_preview:
                self._cancel_preview()
                self._populate_tree()
            self._apply_btn.config(state='disabled')
            self._info_lbl.config(
                text=f'Peak rank {pr} is marked Bad — rejected for quantification.')
            return
        self._selected_rank = pr
        self._start_preview(pr)
        self._apply_btn.config(state='normal')
        self._info_lbl.config(
            text=f'Peak rank {pr} preview — click Apply to save, navigate away to cancel.')

    def _apply_selected(self):
        if self._selected_rank is None:
            return
        if self._effective_quality(self._selected_rank) == 'bad':
            self._info_lbl.config(
                text=f'Peak rank {self._selected_rank} is marked Bad — cannot apply.')
            self._apply_btn.config(state='disabled')
            return
        hmdb_id      = str(self.result.get('hmdb_id', ''))
        applied_rank = self._selected_rank

        # Remember override so a future _run() also uses the correct peak.
        if not hasattr(self.qs, '_quant_peak_overrides'):
            self.qs._quant_peak_overrides = {}
        self.qs._quant_peak_overrides[hmdb_id] = applied_rank

        # ── Instant update — no re-run needed ────────────────────────────────
        # _all_peak_quant already has pre-computed concentrations for every
        # candidate peak.  Just swap the displayed values.
        apq = getattr(self.qs, '_all_peak_quant', {}).get(hmdb_id, {})
        entry = (apq.get(applied_rank)
                 or next((v for k, v in apq.items()
                          if isinstance(k, tuple) and k[0] == applied_rank),
                         None))

        pk_by_rank = {p['rank']: p
                      for p in (self.qs.peak_stage.peaks or [])
                      if p.get('rank') is not None}
        new_peak = pk_by_rank.get(applied_rank)

        if entry is None and new_peak is None:
            self._info_lbl.config(text=f'No pre-computed data for rank {applied_rank}.')
            return

        # Update self.result in place (also updates qs._quant_results entry
        # because self.result IS a reference to the same dict).
        if entry:
            self.result['conc_mm']          = entry['conc_mm']
            self.result['_concs_by_group']  = entry.get('concs_by_group', {})
            self.result['ppm']              = entry.get('ppm', self.result.get('ppm', 0))
            self.result['_lorentz']         = entry.get('_lorentz')
            # Recompute CV per group from pre-computed concentrations
            import numpy as _np
            cv_bg = {}
            for _g, _vals in entry.get('concs_by_group', {}).items():
                if len(_vals) >= 2:
                    _m = float(_np.mean(_vals))
                    cv_bg[_g] = float(_np.std(_vals) / _m * 100) if _m > 0 else float('nan')
                else:
                    cv_bg[_g] = float('nan')
            self.result['cv_by_group'] = cv_bg

        if new_peak:
            self.result['_peak']      = new_peak
            self.result['_peak_rank'] = applied_rank
            self.result['snr']        = float(new_peak.get('snr', self.result.get('snr', 0)))

        self._selected_rank       = None
        # Commit preview — clear saved original, update baseline rank
        self._has_unsaved_preview = False
        self._preview_original    = None
        self._original_peak_rank  = applied_rank

        # Update tree row in QuantificationStage
        if hasattr(self.qs, '_refresh_tree_row'):
            try:
                self.qs._refresh_tree_row(self.result)
            except Exception:
                pass

        # Refresh Quant Peaks window in place
        self._candidates = self._gather_candidates()
        self._refresh_header_vars()
        self._populate_tree()
        self._apply_btn.config(state='disabled')
        _mcr_active = (
            bool(getattr(getattr(self.qs, 'mcr_var', None), 'get', lambda: False)())
            or getattr(getattr(self.qs, 'integ_var', None), 'get', lambda: '')() == 'MCR'
        )
        _mcr_note = ' [MCR mode: value is trapz estimate]' if _mcr_active else ''
        self._info_lbl.config(
            text=f'Peak rank {applied_rank} applied.{_mcr_note} Select another to change.'
        )

        # Redraw Peak Info panel
        if hasattr(self.qs, '_draw_detail'):
            try:
                self.qs._draw_detail(self.result)
            except Exception:
                pass
