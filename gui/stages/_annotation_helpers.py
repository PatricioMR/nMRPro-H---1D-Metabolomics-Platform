# gui/stages/_annotation_helpers.py
"""
Pure helper functions for the Annotation Stage.

Extracted from annotation_stage.py to break circular imports between
annotation_stage.py and _annotation_search.py.

Used by both:
  - annotation_stage.py  (module-level helpers + AnnotationStage core)
  - _annotation_search.py (_AnnotationSearchMixin)
"""
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import os
import warnings
from processing.annotation_scoring import (
    rank_candidates, DJ_GOLD_HZ_DEFAULT)


def _align_peaks_to_reference(peaks, ppm_arr, all_intensities,
                               ref_ppm=0.0, window=0.05):
    """
    Return (shift_ppm, alignment_ok).

    shift_ppm:    ppm offset to align reference (DSS/TSP) to ref_ppm.
    alignment_ok: False when the reference peak could not be located —
                  callers should reduce ppm weight and warn the user.
    """
    if ppm_arr is None or all_intensities is None:
        warnings.warn(
            "PPM alignment skipped: spectrum data unavailable. "
            "PPM axis may be misaligned.",
            UserWarning, stacklevel=2,
        )
        return 0.0, False
    try:
        mean_y = np.mean(all_intensities, axis=1) \
                 if all_intensities.ndim == 2 else all_intensities
        mask   = (ppm_arr >= ref_ppm - window) & (ppm_arr <= ref_ppm + window)
        if not np.any(mask):
            warnings.warn(
                f"Calibration reference (DSS/TSP) not found within "
                f"±{window:.3f} ppm of {ref_ppm:.3f} ppm. "
                "Applying zero shift — PPM axis may be misaligned.",
                UserWarning, stacklevel=2,
            )
            return 0.0, False
        idx_in_win = np.argmax(mean_y[mask])
        current    = ppm_arr[mask][idx_in_win]
        return float(ref_ppm - current), True
    except Exception:
        warnings.warn(
            "PPM alignment failed unexpectedly. Applying zero shift.",
            UserWarning, stacklevel=2,
        )
        return 0.0, False


def _assignment_sort_key(score, mp):
    """Ranking key for the per-peak metabolite winner.

    Single source of truth shared by AnnotationResultsWindow._build_peak_assignments
    and the Peak List metabolite column, so both pick the SAME metabolite for a
    given peak rank. A pure composite_score tiebreak (the old Peak List logic)
    diverged from the results window: it ignored these bonuses, so in overlap
    regions the two views could disagree on which metabolite owns a peak.

      • j_anchor_bonus  — a J-confirmed multiplet (ΔJ < 0.5 Hz) beats a higher
                          scoring singlet competing for the same peak.
    """
    dj = mp.get('delta_j_hz')
    j_anchor_bonus = 0.20 if (dj is not None and float(dj) < 0.5) else 0.0
    return float(score) + j_anchor_bonus


def select_metabolite_per_rank(metabolites, manual_quality=None):
    """Canonical peak_rank → metabolite-name map (1 peak → 1 metabolite).

    Mirrors AnnotationResultsWindow._build_peak_assignments candidate selection:
      • discard matched peaks marked annotation_quality == 'bad'
      • a manually-confirmed 'good' match wins over any auto candidate
      • otherwise the highest _assignment_sort_key wins

    Used by the Peak List as the fallback when a detected peak has no
    assigned_metabolite propagated onto it yet.
    """
    cands = {}  # rank -> [(key, name, is_good)]
    for met in metabolites or []:
        name = met.get('name', '')
        for mp in met.get('matched_peaks', []):
            if mp.get('annotation_quality') == 'bad':
                continue
            rank = mp.get('peak_rank')
            if rank is None:
                continue
            score = float(mp.get('composite_score',
                                 mp.get('multiplet_score', 0.0)) or 0.0)
            cands.setdefault(rank, []).append((
                _assignment_sort_key(score, mp),
                name,
                mp.get('annotation_quality') == 'good',
            ))
    # Keep the fallback consistent with the annotation window: close
    # candidates are evidence, not an identity decision.
    ambiguous = ambiguous_peak_ranks(
        metabolites, manual_quality=manual_quality or {})
    out = {}
    for rank, lst in cands.items():
        if rank in ambiguous:
            continue
        good = [c for c in lst if c[2]]
        pool = good if good else lst
        out[rank] = max(pool, key=lambda c: c[0])[1]
    return out


def ambiguous_peak_ranks(metabolites, manual_quality=None,
                         dj_gold_hz=DJ_GOLD_HZ_DEFAULT):
    """Return peak ranks whose top candidates are too close to resolve safely.

    Ambiguity is decided by the ranking rule shared with quantification
    (``processing.annotation_scoring.rank_candidates``): a rank is ambiguous
    when its top two candidates tie on evidence tier, ΔJ band, Δppm band and
    cosine. This used to compare ``composite_score`` values within a fixed
    margin — a blend weighted 45% on Δppm, which drifts by ~18 Hz between
    samples while J does not, so near-ties in that score were near-ties in the
    wrong quantity.

    Manual Good on a specific (metabolite, rank) pair is an explicit identity
    decision and therefore resolves that rank. Automatic quality marks alone do
    not resolve competition between metabolites.
    """
    manual_quality = manual_quality or {}
    by_rank = {}
    for met in metabolites or []:
        mid = str(met.get("id", met.get("name", "")))
        for mp in met.get("matched_peaks", []):
            rank = mp.get("peak_rank")
            if rank is None or mp.get("annotation_quality") == "bad":
                continue
            by_rank.setdefault(rank, {})[mid] = {
                'multiplicity':   mp.get('type') or mp.get('multiplicity'),
                'delta_j_hz':     mp.get('delta_j_hz'),
                'ppm_diff':       mp.get('ppm_diff'),
                'n_found':        mp.get('_n_found_lines'),
                'n_expected':     mp.get('_n_exp_lines'),
                'cosine':         mp.get('_s_cosine'),
                'manual_quality': manual_quality.get((mid, rank)),
                'quality':        mp.get('annotation_quality'),
                'score':          mp.get('composite_score',
                                         mp.get('multiplet_score', 0.0)),
            }

    ambiguous = set()
    for rank, by_mid in by_rank.items():
        if any(manual_quality.get((mid, rank)) == "good" for mid in by_mid):
            continue
        if len(by_mid) < 2:
            continue
        _, is_ambiguous = rank_candidates(list(by_mid.values()),
                                          dj_gold_hz=dj_gold_hz)
        if is_ambiguous:
            ambiguous.add(rank)
    return ambiguous


def _ppm_score(exp_ppm, db_ppm, tolerance):
    """Gaussian score centred at 0 delta. Score=1 at perfect match."""
    sigma = tolerance / 2.0
    return float(np.exp(-0.5 * ((exp_ppm - db_ppm) / sigma) ** 2))


def _multiplet_score(result, peaks, tolerance, mhz,
                     ppm_array=None, intensity_array=None):
    """Wrapper around multiplet_scorer.score_multiplet_match."""
    try:
        from gui.notation.multiplet_scorer import score_multiplet_match
        return score_multiplet_match(result, peaks, tolerance, mhz,
                                     ppm_array=ppm_array,
                                     intensity_array=intensity_array)
    except Exception as e:
        return 0.0, f'scorer error: {e}', []


def _baseline_correct_linear(x, y, edge_frac=0.15):
    """Remove a linear baseline from the edge samples."""
    n = len(y)
    edge_n = max(2, int(n * edge_frac))
    x_left  = float(np.mean(x[:edge_n]))
    y_left  = float(np.mean(y[:edge_n]))
    x_right = float(np.mean(x[-edge_n:]))
    y_right = float(np.mean(y[-edge_n:]))
    if abs(x_right - x_left) < 1e-9:
        return y - y_left
    slope    = (y_right - y_left) / (x_right - x_left)
    baseline = y_left + slope * (x - x_left)
    return y - baseline


def _fit_r2(sim, exp_y):
    """Fit a non-negative scalar scale to align sim → exp_y. Returns (r2, scale)."""
    sim_v = np.asarray(sim).ravel()
    exp_v = np.asarray(exp_y).ravel()
    if np.max(np.abs(sim_v)) < 1e-12 or np.max(np.abs(exp_v)) < 1e-12:
        return 0.0, 1.0
    denom = float(np.dot(sim_v, sim_v))
    if denom < 1e-30:
        return 0.0, 1.0
    scale = float(np.dot(sim_v, exp_v)) / denom
    if scale <= 0:
        return 0.0, scale
    residual  = exp_v - scale * sim_v
    ss_res    = float(np.dot(residual, residual))
    ss_tot    = float(np.var(exp_v)) * len(exp_v)
    if ss_tot < 1e-30:
        return 0.0, scale
    r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return r2, scale


def _observe_j_hz(peaks_in_window, j_db_hz, mhz, n_expected_lines,
                  tol_frac=0.50):
    """Estimate observed J coupling from peak-pair separations."""
    if len(peaks_in_window) < 2 or j_db_hz <= 0 or mhz <= 0:
        return None, None
    ppms    = sorted(p['ppm'] for p in peaks_in_window)
    seps_hz = [abs(ppms[i + 1] - ppms[i]) * mhz
               for i in range(len(ppms) - 1)]
    if not seps_hz:
        return None, None
    tol_hz  = j_db_hz * tol_frac
    cands   = [s for s in seps_hz if abs(s - j_db_hz) <= tol_hz]
    if not cands:
        closest = min(seps_hz, key=lambda s: abs(s - j_db_hz))
        return float(closest), float(abs(closest - j_db_hz))
    j_obs = min(cands, key=lambda s: abs(s - j_db_hz))
    return float(j_obs), float(abs(j_obs - j_db_hz))


def _mult_compat_check(result, aligned_peaks, tolerance, mhz):
    """Return (compat_score 0-1, observed_mult_str) for a DB candidate."""
    try:
        import re as _re
        from gui.notation.multiplet_scorer import (
            classify_observed_multiplicity,
            multiplicity_compat_score,
            _norm as _mult_norm,
        )
        db_ppm   = float(result.get('ppm_center') or result.get('ppm') or 0)
        mult_raw = result.get('multiplicity') or result.get('type') or 's'
        j_str    = result.get('j_coupling', '') or ''
        j_nums   = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str) if float(x) > 0]
        j_hz     = j_nums[0] if j_nums else 0.0

        norm     = _mult_norm(mult_raw)
        n_lines  = {'s':1,'d':2,'t':3,'q':4,'quin':5,'dd':4,'ddd':8,'dt':6,'td':6}.get(norm, 3)
        j_ppm    = (j_hz / mhz) if j_hz > 0 and mhz > 0 else 0.0
        win      = max(j_ppm * (n_lines / 2) * 1.5, tolerance * 1.5, 0.008)

        peaks_near = [p for p in aligned_peaks if abs(p['ppm'] - db_ppm) <= win]
        obs_mult, obs_conf = classify_observed_multiplicity(peaks_near, j_hz, mhz)
        compat = multiplicity_compat_score(mult_raw, obs_mult, obs_conf)
        return compat, obs_mult
    except Exception:
        return 1.0, '?'


def _adaptive_mcr_mask(ppm_full, all_int, peak, cands=None, field_mhz=600.0):
    """Build an adaptive MCR region mask around a peak."""
    p_lo = float(min(peak.get('start_ppm', peak['ppm']), peak.get('end_ppm', peak['ppm'])))
    p_hi = float(max(peak.get('start_ppm', peak['ppm']), peak.get('end_ppm', peak['ppm'])))
    width = max(abs(p_hi - p_lo), 1e-4)
    margin = max(0.01, width * 1.5)
    lo = p_lo - margin
    hi = p_hi + margin

    cands = cands or []
    n_lines_map = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quin': 5, 'dd': 4, 'ddd': 8, 'dt': 6, 'td': 6}
    for c in cands:
        try:
            cp = float(c.get('ppm_center') or c.get('ppm') or peak['ppm'])
            cs = c.get('ppm_start')
            ce = c.get('ppm_end')
            if cs is not None and ce is not None:
                lo = min(lo, float(cs) - 0.006)
                hi = max(hi, float(ce) + 0.006)
            else:
                mult = str(c.get('type') or c.get('multiplicity') or 's').lower().strip()
                n_lines = n_lines_map.get(mult, 2)
                j_raw = str(c.get('j_coupling') or '')
                j_hz = 0.0
                try:
                    j_hz = float(j_raw.split(',')[0].strip()) if j_raw else 0.0
                except Exception:
                    j_hz = 0.0
                split_ppm = (j_hz / max(field_mhz, 1e-9)) * max(1, n_lines - 1)
                side = max(0.006, split_ppm * 0.8)
                lo = min(lo, cp - side)
                hi = max(hi, cp + side)
        except Exception:
            continue

    try:
        y_full = (np.mean(all_int, axis=1) if getattr(all_int, 'ndim', 1) == 2
                  else np.asarray(all_int))
        base_mask = (ppm_full >= lo) & (ppm_full <= hi)
        if np.sum(base_mask) > 4:
            xw = ppm_full[base_mask]
            yw = y_full[base_mask]
            c_idx = int(np.argmin(np.abs(xw - float(peak.get('ppm', (p_lo + p_hi) * 0.5)))))
            y_min = float(np.min(yw))
            y_max = float(np.max(yw))
            thr = y_min + 0.10 * (y_max - y_min)
            l = c_idx
            while l > 0 and yw[l] > thr:
                l -= 1
            r = c_idx
            while r < len(yw) - 1 and yw[r] > thr:
                r += 1
            sig_margin = max(0.01, width * 0.8)
            lo = min(lo, float(xw[l]) - sig_margin)
            hi = max(hi, float(xw[r]) + sig_margin)
    except Exception:
        pass

    ppm_min = float(np.min(ppm_full)); ppm_max = float(np.max(ppm_full))
    lo = max(ppm_min, lo)
    hi = min(ppm_max, hi)
    mask = (ppm_full >= lo) & (ppm_full <= hi)
    return mask


def _ask_search_mode(parent_widget):
    """Ask whether to reset or run incremental. Returns 'reset', 'incremental', or None."""
    result = [None]

    dlg = tk.Toplevel(parent_widget)
    dlg.title('New Search')
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(
        dlg,
        text='A previous annotation already exists.',
        font=('TkDefaultFont', 10, 'bold'),
        pady=6).pack(padx=20, pady=(14, 4))

    tk.Label(
        dlg,
        text='Choose how to proceed:',
        font=('TkDefaultFont', 9)).pack(padx=20)

    bf = tk.Frame(dlg); bf.pack(padx=20, pady=10, fill='x')

    def _choose(v):
        result[0] = v
        dlg.destroy()

    tk.Button(
        bf,
        text='♻  Reset & start over',
        font=('TkDefaultFont', 9),
        width=28,
        relief='groove',
        command=lambda: _choose('reset')).pack(pady=4)
    tk.Label(
        bf,
        text='Discards all marks, Good/Bad annotations\nand previous matches.',
        font=('TkDefaultFont', 8), foreground='#666',
        justify='left').pack(anchor='w', padx=4)

    ttk.Separator(bf, orient='horizontal').pack(fill='x', pady=8)

    tk.Button(
        bf,
        text='🔍  Search only new / unmarked signals',
        font=('TkDefaultFont', 9),
        width=28,
        relief='groove',
        command=lambda: _choose('incremental')).pack(pady=4)
    tk.Label(
        bf,
        text='Keeps existing Good annotations.\nSearches signals without a confirmed Good match.',
        font=('TkDefaultFont', 8), foreground='#666',
        justify='left').pack(anchor='w', padx=4)

    ttk.Separator(bf, orient='horizontal').pack(fill='x', pady=8)

    tk.Button(
        bf,
        text='Cancel',
        font=('TkDefaultFont', 9),
        width=12,
        command=dlg.destroy).pack()

    dlg.update_idletasks()
    px = parent_widget.winfo_rootx() + parent_widget.winfo_width() // 2
    py = parent_widget.winfo_rooty() + parent_widget.winfo_height() // 2
    w, h = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f'+{px - w//2}+{py - h//2}')
    dlg.bind('<Escape>', lambda e: dlg.destroy())
    parent_widget.wait_window(dlg)
    return result[0]


def _pick_database(parent_widget):
    """Show dialog to pick a database: bundled DBs or browse. Returns path or None."""
    from app_paths import databases_dir
    _db_folder = databases_dir()
    bundled = []
    if os.path.isdir(_db_folder):
        for fname in sorted(os.listdir(_db_folder)):
            if fname.lower().endswith(('.db', '.csv')):
                bundled.append(os.path.join(_db_folder, fname))

    if not bundled:
        return filedialog.askopenfilename(
            title='Select Database File',
            filetypes=[('Database files', '*.db *.csv'),
                       ('SQLite', '*.db'), ('CSV', '*.csv'),
                       ('All files', '*.*')],
            initialdir=os.path.expanduser('~'))

    result = [None]

    dlg = tk.Toplevel(parent_widget)
    dlg.title('Select Database')
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(dlg, text='Choose a bundled database or browse for another file:',
             font=('TkDefaultFont', 10, 'bold'), pady=8).pack(padx=16, pady=(12, 4))

    frame = tk.Frame(dlg)
    frame.pack(padx=16, pady=4, fill='both', expand=True)

    scrollbar = tk.Scrollbar(frame, orient='vertical')
    lb = tk.Listbox(frame, yscrollcommand=scrollbar.set, selectmode='single',
                    height=min(len(bundled) + 1, 12), width=52,
                    font=('TkFixedFont', 10))
    scrollbar.config(command=lb.yview)
    scrollbar.pack(side='right', fill='y')
    lb.pack(side='left', fill='both', expand=True)

    for path in bundled:
        name = os.path.splitext(os.path.basename(path))[0].replace('_', ' ').title()
        lb.insert('end', f'  {name}')

    lb.insert('end', '  Browse for another file…')
    lb.select_set(0)

    def _confirm():
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(bundled):
            result[0] = bundled[idx]
            dlg.destroy()
        else:
            path = filedialog.askopenfilename(
                title='Select Database File',
                filetypes=[('Database files', '*.db *.csv'),
                           ('SQLite', '*.db'), ('CSV', '*.csv'),
                           ('All files', '*.*')],
                initialdir=os.path.expanduser('~'))
            if path:
                result[0] = path
                dlg.destroy()

    def _cancel():
        dlg.destroy()

    btn_frame = tk.Frame(dlg)
    btn_frame.pack(padx=16, pady=(8, 14))
    tk.Button(btn_frame, text='Select', width=12, command=_confirm).pack(side='left', padx=6)
    tk.Button(btn_frame, text='Cancel', width=12, command=_cancel).pack(side='left', padx=6)

    lb.bind('<Double-Button-1>', lambda e: _confirm())
    dlg.bind('<Return>', lambda e: _confirm())
    dlg.bind('<Escape>', lambda e: _cancel())

    dlg.update_idletasks()
    pw = parent_widget.winfo_rootx() + parent_widget.winfo_width() // 2
    ph = parent_widget.winfo_rooty() + parent_widget.winfo_height() // 2
    w, h = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f'+{pw - w//2}+{ph - h//2}')

    parent_widget.wait_window(dlg)
    return result[0]


def _multi_peak_coherence(met_id, all_peaks_by_id, observed_peaks, tolerance):
    """Score candidate coherence with one-to-one line matching.

    This is candidate evidence, not ownership: the same observed peak may
    remain a candidate for multiple metabolites. Only lines belonging to one
    candidate are prevented from reusing the same observed peak.
    """
    db_peaks = all_peaks_by_id.get(met_id, [])
    if len(db_peaks) <= 1:
        return 1.0, 1.0, 1.0

    db_max = max((max(float(i), 0.0) for _, i in db_peaks), default=0.0) or 1.0
    db_sticks = [(float(ppm), max(float(intensity), 0.0) / db_max)
                  for ppm, intensity in db_peaks]
    obs = [p for p in (observed_peaks or []) if p.get("ppm") is not None]
    if not obs:
        return 0.0, 0.0, 0.0

    try:
        from gui.notation.multiplet_scorer import _hungarian_line_matches
        matches = _hungarian_line_matches(
            db_sticks, obs, max_distance=float(tolerance))
    except Exception:
        matches = []

    coverage = len(matches) / len(db_sticks)
    if not matches:
        return coverage, coverage, 0.0

    matched_heights = [
        max(float(m.get("peak", {}).get("height", 0.0)), 0.0)
        for m in matches
    ]
    obs_max = max(matched_heights, default=0.0) or 1.0
    errors = []
    for i, m in enumerate(matches):
        expected = float(m.get("pred_rel", 0.0))
        observed = float(matched_heights[i] / obs_max)
        errors.append(min(abs(expected - observed) / max(expected, 1e-6), 1.0))
    ratio_match = max(0.0, 1.0 - float(np.mean(errors))) if errors else 0.0
    coherence = 0.6 * coverage + 0.4 * ratio_match
    return coherence, coverage, ratio_match


def _stocsy_coherence(met_id, all_peaks_by_id, observed_peaks, tolerance,
                      peak_corr):
    """Mean pairwise Pearson r across all matched peak pairs for a metabolite.

    Mirrors ``_multi_peak_coherence`` signature so it can be called at the
    same point in the annotation search loop.

    Returns ``None`` when data is unavailable (singleton metabolite,
    fewer than 3 samples when the matrix was built, or no matched pairs found).
    A ``None`` result means the caller should apply no STOCSY boost.

    Parameters
    ----------
    met_id : str
        Metabolite ID key into ``all_peaks_by_id``.
    all_peaks_by_id : dict
        ``{met_id: [(ppm, intensity), ...]}`` — DB fingerprint peaks.
    observed_peaks : list of dict
        Detected peaks, each with ``'ppm'`` and ``'rank'``.
    tolerance : float
        PPM matching window.
    peak_corr : dict
        ``{(rank_i, rank_j): pearson_r}`` from
        ``processing.stocsy.compute_peak_correlation_matrix``.
        Empty dict → always returns ``None``.

    Returns
    -------
    float or None
    """
    if not peak_corr:
        return None
    db_peaks = all_peaks_by_id.get(met_id, [])
    if len(db_peaks) <= 1:
        return None   # singleton — no pairs possible

    obs_ranks = [p.get('rank') for p in observed_peaks]

    # Greedy matching: each observed peak consumed at most once.
    # Without this, two DB peaks close together could map to the same
    # observed peak, artificially inflating matched_ranks and pair count.
    available = list(range(len(observed_peaks)))
    matched_ranks = []
    for db_ppm, _ in db_peaks:
        if not available:
            break
        dists    = [(abs(observed_peaks[i]['ppm'] - db_ppm), i)
                    for i in available]
        min_dist, best_i = min(dists)
        if min_dist <= tolerance:
            r = obs_ranks[best_i]
            if r is not None:
                matched_ranks.append(r)
            available.remove(best_i)

    if len(matched_ranks) < 2:
        return None

    pairs = [(min(a, b), max(a, b))
             for i, a in enumerate(matched_ranks)
             for b in matched_ranks[i + 1:]]
    rs = [peak_corr[p] for p in pairs if p in peak_corr]
    return float(np.mean(rs)) if rs else None
