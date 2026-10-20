# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Annotating a study against a reference database: the search itself.

This is what the annotation stage runs when the analyst presses Search,
with nothing of the window in it. It takes the study and what detection
measured about it, asks the database which of its signals could stand at
each peak, measures every candidate against the trace -- position,
multiplicity, coupling, pattern, line heights, how the peak behaves
across the samples -- and returns the matches ranked.

It lived inside a Tk mixin, as a nested function of the method that opens
the dialogs, so the only way to run it was to open a window. The bench
that validates the software could not call it and grew a second
annotation of its own; the two then drifted, and two defects lived for
months in the path only the window took -- a fingerprint query that asked
every signal for one proton, and a complex multiplet credited with all of
its protons in a single line. Both are fixed; neither was visible from
the bench. That is what this module is for.

The window keeps what belongs to it: the dialogs, the progress bar, the
thread, and what to do with the answer. Progress arrives through the two
callbacks, which are free to marshal onto the interface's thread.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from gui.stages._annotation_helpers import (
    _baseline_correct_linear, _fit_r2, _multi_peak_coherence,
    _multiplet_score, _ppm_score, _stocsy_coherence, _stocsy_partner_count)
from processing.annotation_consistency import (
    consistent_peaks, reference_per_proton)
from processing.annotation_scoring import (
    PATTERN_LINE_TOL_HZ, PATTERN_MULT, SCORING_DEFAULTS, dj_score,
    expected_line_count, lines_covary_on_trace, measure_multiplet,
    measure_pattern_signal, measured_letter, pattern_governs,
    pattern_offsets_hz, peak_shape_score, search_multiplet_in_window,
    signal_score)
from processing.sample_selection import included_sample_indices
from processing.signal_quality import is_annotatable
from processing.spectral_regions import measure_linewidth_ppm


def is_unquantifiable_multiplicity(mult):
    """True for DB multiplicities that must not be used for annotation/quant.

    Covers generic multiplets ('m') — no J, no predictable pattern — and any
    BROAD signal ('br s', 'br. s.', 'br d', …): broad lines are exchangeable
    protons (OH/NH), whose integral and position drift with pH/temperature/
    solvent suppression, so they are unreliable quantification targets.
    """
    m = str(mult or '').lower().strip()
    return m in ('m', 'multiplet') or m.startswith('br')

def _measured_columns_sql(shift_cols) -> dict:
    """The SQL for a signal's position and coupling: measured where the DB
    carries a measurement, listed otherwise.

    BMRB_Metabolomics_v02 adds, per signal, what its own spectrum says --
    ``ppm_meas``, ``j_meas`` and ``j_sigma`` -- beside what the curators
    listed. The listed coupling is text and may hold several values for an
    unresolved multiplet; the measured one is a number and exists only where
    a first-order pattern was measured, so it overrides the listing where it
    is present and defers to it where it is not. A DB without the columns
    (v01) reads exactly as before.
    """
    cols = set(shift_cols or ())
    ppm = 'COALESCE(s.ppm_meas, s.ppm)' if 'ppm_meas' in cols else 's.ppm'
    # j_use is the build's verdict on which coupling to trust -- the
    # measurement where it corrects the listing by a plausible amount or the
    # listing was never a coupling, the listing where the two disagree
    # beyond that and a reviewer has to look. Its sigma applies only when the
    # measurement is what is being used.
    j = ('CASE WHEN s.j_use IS NOT NULL THEN CAST(s.j_use AS TEXT) '
         'ELSE s.j_coupling END' if 'j_use' in cols else 's.j_coupling')
    j_sigma = ("CASE WHEN s.j_use_source LIKE 'measured%' THEN s.j_sigma "
               "ELSE NULL END" if 'j_use_source' in cols else 'NULL')
    pattern = 's.pattern_lines' if 'pattern_lines' in cols else 'NULL'
    profile = 's.pattern_profile' if 'pattern_profile' in cols else 'NULL'
    profile_w = 's.profile_fwhm_hz' if 'profile_fwhm_hz' in cols else 'NULL'
    profile_nh = 's.profile_nh' if 'profile_nh' in cols else 'NULL'
    # v04 reads the multiplicity off the reference spectra where every
    # agreeing BMRB acquisition contradicts the curators' label or makes a
    # generic 'm' specific (taurine's triplets, listed 'd' and 'm'). The
    # listing stays in its column; this is what the search matches against.
    mult = ('COALESCE(s.mult_meas, s.multiplicity)' if 'mult_meas' in cols
            else 's.multiplicity')
    return {'ppm': ppm, 'j': j, 'j_sigma': j_sigma, 'pattern': pattern,
            'profile': profile, 'profile_fwhm': profile_w,
            'profile_nh': profile_nh, 'mult': mult}

def pattern_of(entry, mhz) -> list:
    """``[(ppm, rel_intensity)]`` of a measured line pattern laid on the
    axis at this field, or ``[]``. Offsets are stored in hertz because they
    are couplings, and couplings do not move with the field."""
    lines = pattern_offsets_hz(entry)
    if not lines or not mhz:
        return []
    centre = float(entry.get('ppm_center') or entry.get('ppm') or 0.0)
    return sorted((centre + off / float(mhz), rel) for off, rel in lines)


class IncompatibleDatabase(ValueError):
    """The file opened is not a reference database this can read."""


@dataclass
class SearchOutcome:
    """What the search found, and what it had to work with."""

    results: list = field(default_factory=list)
    peaks_annotatable: int = 0
    peaks_matched: int = 0
    elapsed_s: float = 0.0


def search_database(*, db_path, db_type, peaks_to_search, annotatable_peaks,
                    tiers, ppm, intensities, samples_info, field_mhz,
                    tolerance_ppm, params=None, ppm_calibration=None,
                    peak_correlation=None, n_samples_correlated=0,
                    min_distance_ppm=0.0, noise_std=0.0, min_snr=0.0,
                    carry_results=(), progress=None, status=None):
    """Match every annotatable peak against the database.

    ``peaks_to_search`` are the peaks the search may anchor on and
    ``annotatable_peaks`` those it counts as its denominator (the analyst
    can exclude a tier). ``carry_results`` are matches from an earlier
    search being added to rather than replaced.

    What detection measured is passed rather than read off its stage:
    ``peak_correlation`` and ``n_samples_correlated`` for the STOCSY
    coherence, ``min_distance_ppm`` for what the detector could resolve,
    and ``noise_std`` with ``min_snr`` for the height under which it
    reports nothing.

    ``progress(done, total, matches, peaks_matched, elapsed)`` and
    ``status(text)`` are called as it goes, from whatever thread the
    caller ran this on.

    Raises ``IncompatibleDatabase`` when the file holds neither of the
    schemas this reads.
    """
    from processing.peak_detection import canonical_peak_order
    peaks_to_search = canonical_peak_order(peaks_to_search)
    annotatable_peaks = canonical_peak_order(annotatable_peaks)
    # names the body still uses under their old spelling
    mhz = field_mhz
    tolerance = tolerance_ppm
    _annotatable = annotatable_peaks
    _tiers = tiers
    t0 = time.monotonic()
    all_results, peaks_matched = [], set()
    total_entries = 1
    conn = None
    # Line counts come from the scoring core's LINE_COUNTS — one table.
    # This local copy had 'quin' where BMRB writes 'quint', so every
    # quintet, sextet and septet fell to the default of 2 lines.
    _n_lines_map = {'s': 1}
    _ppm_arr = ppm
    _raw_int = intensities
    has_spec = (
        _ppm_arr is not None
        and _raw_int is not None
        and len(_ppm_arr) > 10)
    if has_spec:
        _mean_int = (np.mean(_raw_int, axis=1)
                     if getattr(_raw_int, 'ndim', 1) == 2
                     else np.asarray(_raw_int, dtype=float))

    def _observe_peak_mult(pk, ppm_arr, int_arr, field_mhz):
        p_lo = min(float(pk.get('start_ppm', pk['ppm'])),
                   float(pk.get('end_ppm',   pk['ppm'])))
        p_hi = max(float(pk.get('start_ppm', pk['ppm'])),
                   float(pk.get('end_ppm',   pk['ppm'])))
        span = p_hi - p_lo
        if span < 0.002 or ppm_arr is None or int_arr is None:
            return 's', None
        mask = (ppm_arr >= p_lo) & (ppm_arr <= p_hi)
        if np.sum(mask) < 5:
            return 's', None
        x = ppm_arr[mask]
        y = int_arr[mask].copy()
        y -= y.min()
        if y.max() < 1e-12:
            return 's', None
        y /= y.max()
        from scipy.signal import find_peaks as _fp
        min_dist = max(2, int(len(x) * 0.08))
        # Conservative thresholds: a noisy shoulder must not split a true
        # singlet into a 'd' (which would wrongly apply the j_compat=0.3
        # singlet penalty downstream). Require a prominent secondary line.
        pks, _ = _fp(y, height=0.25, prominence=0.18, distance=min_dist)
        n = len(pks)
        if n <= 1:
            return 's', None
        ppms_pk = x[pks]
        seps_hz = [abs(ppms_pk[k+1] - ppms_pk[k]) * field_mhz
                   for k in range(len(ppms_pk) - 1)]
        j_est = float(np.median(seps_hz))
        # J floor: line separations below ~1.5 Hz are not resolvable
        # couplings at metabolomics linewidths — treat as a single line.
        if j_est < 1.5:
            return 's', None
        mult_map = {2: 'd', 3: 't', 4: 'q'}
        return mult_map.get(n, 'm'), j_est

    if has_spec:
        for _pk in peaks_to_search:
            if _pk.get('is_sub_peak'):
                continue
            _om, _oj = _observe_peak_mult(
                _pk, _ppm_arr, _mean_int, mhz)
            _pk['_obs_mult'] = _om
            _pk['_obs_j_hz'] = _oj

    try:
        from gui.widgets.annotation_results_window import (
            _build_stick_positions, lines_from_transitions)
        from gui.utils.mcr_utils import simulate_lorentzian_profile

        _cal = ppm_calibration
        _cal_delta = float(_cal['delta_mean']) if _cal else 0.0
        # Guard against stale/corrupt calibration from old project versions.
        # The in-window calibration logic limits each anchor to Δ < 0.02 ppm,
        # so a valid delta_mean must be well below that.  Values beyond ±0.05
        # ppm indicate a broken or accumulated calibration — discard them.
        if abs(_cal_delta) > 0.05:
            _cal_delta = 0.0

        ppm_vals   = [p['ppm'] for p in peaks_to_search]
        g_lo       = min(ppm_vals) - _cal_delta - max(tolerance * 2, 0.10)
        g_hi       = max(ppm_vals) - _cal_delta + max(tolerance * 2, 0.10)

        db_entries = []
        is_bmrb    = False

        if db_type == 'hmdb':
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur   = conn.cursor()
            tables = {r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'").fetchall()}
            has_shifts   = 'shifts_1h' in tables
            has_multiplets = 'multiplets' in tables

            # Detectar ID primário: bmrb_id (BMRB) vs hmdb_id (HMDB nativo)
            # Prefer bmrb_id only when the column exists AND has non-NULL values;
            # a column present but fully NULL means the DB uses hmdb_id as PK.
            if has_shifts:
                shift_cols = {r[1] for r in cur.execute(
                    "PRAGMA table_info(shifts_1h)").fetchall()}
                if 'bmrb_id' in shift_cols:
                    _bmrb_count = cur.execute(
                        "SELECT COUNT(bmrb_id) FROM shifts_1h").fetchone()[0]
                    id_col = 'bmrb_id' if _bmrb_count > 0 else 'hmdb_id'
                else:
                    id_col = 'hmdb_id'
                met_id_col = id_col
            else:
                id_col = met_id_col = 'hmdb_id'

            if not has_shifts and not has_multiplets:
                raise IncompatibleDatabase(
                    'Database must contain shifts_1h or multiplets.')

            if has_shifts:
                _protons_col = (", s.protons" if 'protons' in shift_cols
                                else ", 1 as protons")
                _meas = _measured_columns_sql(shift_cols)
                cur.execute(
                    f"SELECT s.{id_col} as hmdb_id, met.name,"
                    f" {_meas['ppm']} as ppm_center,"
                    f" {_meas['ppm']} as ppm,"
                    f" s.atom_name, {_meas['mult']} as type,"
                    f" {_meas['j']} as j_coupling,"
                    f" {_meas['j_sigma']} as j_sigma,"
                    f" {_meas['pattern']} as pattern_lines,"
                    f" {_meas['profile']} as pattern_profile,"
                    f" {_meas['profile_fwhm']} as profile_fwhm_hz,"
                    f" {_meas['profile_nh']} as profile_nh,"
                    f" {_meas['mult']} as multiplicity{_protons_col}"
                    " FROM shifts_1h s"
                    " JOIN metabolites met"
                    f"   ON s.{id_col} = met.{met_id_col}"
                    f" WHERE {_meas['ppm']} BETWEEN ? AND ?"
                    f" ORDER BY {_meas['ppm']}",
                    (g_lo, g_hi))
                db_entries = [dict(r) for r in cur.fetchall()]
                # complex multiplets shaped by their spin system at this
                # field (processing.spin_patterns, off by default)
                from processing import spin_patterns as _spin
                _spin.apply(db_entries, mhz)
                db_label = 'BMRB' if id_col == 'bmrb_id' else 'HMDB'
                for e in db_entries:
                    e['match_type'] = 'shift'
                    e['database']   = db_label
            else:
                cur.execute(
                    "SELECT m.*, met.name"
                    " FROM multiplets m"
                    " JOIN metabolites met"
                    "   ON m.hmdb_id = met.hmdb_id"
                    " WHERE m.ppm_center BETWEEN ? AND ?"
                    " ORDER BY m.ppm_center",
                    (g_lo, g_hi))
                db_entries = [dict(r) for r in cur.fetchall()]
                for e in db_entries:
                    e['match_type'] = 'multiplet'
                    e['database']   = 'HMDB'

        elif db_type == 'csv':
            import pandas as pd
            df = pd.read_csv(db_path)
            for _, row in df.iterrows():
                rdict = dict(row)
                ppm_v = float(rdict.get('ppm', 0))
                if g_lo <= ppm_v <= g_hi:
                    rdict['ppm_center']  = ppm_v
                    rdict['database']    = 'CSV'
                    rdict['match_type']  = 'csv_match'
                    rdict.setdefault('type', 's')
                    db_entries.append(rdict)

        if _cal_delta != 0.0:
            for e in db_entries:
                for key in ('ppm_center', 'ppm'):
                    if key in e and e[key] is not None:
                        try:
                            e[key] = float(e[key]) + _cal_delta
                        except (ValueError, TypeError):
                            pass

        total_entries = max(1, len(db_entries))
        if status is not None:
            status(f'Fitting fingerprints for {total_entries} '
                   f'DB entries…')

        # Bulk-fetch all sibling peaks for multi-peak coherence scoring.
        # One query for all candidate metabolites — avoids N+1 overhead.
        all_peaks_by_id = {}
        if conn is not None and db_type == 'hmdb':
            try:
                _cand_ids = list({
                    e.get('hmdb_id') or e.get('bmrb_id')
                    for e in db_entries
                    if (e.get('hmdb_id') or e.get('bmrb_id'))
                })
                if _cand_ids:
                    _id_col_bulk = id_col if has_shifts else 'hmdb_id'
                    _table_bulk  = 'shifts_1h' if has_shifts else 'multiplets'
                    _ppm_col     = 'ppm' if has_shifts else 'ppm_center'
                    _ph          = ','.join('?' * len(_cand_ids))
                    _bulk_rows   = cur.execute(
                        f"SELECT {_id_col_bulk} as mid, {_ppm_col} as ppm,"
                        f" COALESCE(intensity, 1.0) as intensity"
                        f" FROM {_table_bulk}"
                        f" WHERE {_id_col_bulk} IN ({_ph})",
                        _cand_ids,
                    ).fetchall()
                    for _br in _bulk_rows:
                        all_peaks_by_id.setdefault(_br[0], []).append(
                            (_br[1], _br[2]))
            except Exception:
                pass  # coherence will default to 1.0 for all entries

        # Source reliability weights — penalise less-curated databases
        # so well-annotated sources rank above statistically-inferred entries.
        # Scoring parameters: the user's Annotation Parameters when
        # they exist, the scoring core's defaults otherwise. One owner.
        _sc_params = dict(SCORING_DEFAULTS)
        # The Δppm gate follows the analyst's 'Tolerance (ppm)' — the
        # same number that defined the search window. An explicit
        # dppm_gold in the Parameters dialog still wins.
        _sc_params['dppm_gold'] = tolerance
        _sc_params.update(
            params or {})

        _SOURCE_WEIGHTS = {
            'internal': 1.00,
            'COLMAR':   0.90,
            'BMRB':     0.82,
            'HMDB':     0.68,
            'CSV':      0.72,
        }

        # Graph STOCSY — retrieve precomputed peak correlation matrix
        # and sample count from peak_detection_stage.  Both default to
        # safe no-op values when detection hasn't run or n_samples < 3.
        _pk_corr   = peak_correlation or {}
        _n_samp_st = int(n_samples_correlated or 0)

        # What this study can resolve, for the patterns: the wider
        # of its linewidth and the detector's minimum spacing, in Hz.
        # A reference pattern's lines closer than that are one line
        # here, and are expected as one.
        _lw_ppm = (measure_linewidth_ppm(_ppm_arr, _mean_int,
                                         peaks_to_search)
                   if has_spec else None) or 0.0
        _min_dist_ppm = float(min_distance_ppm or 0.0)
        # The detector's floor merges what the linewidth did not:
        # two apices closer than min_distance are one peak in the
        # list whatever the trace shows.
        _lw_hz = _lw_ppm * mhz if mhz else 0.0
        _merge_hz = _min_dist_ppm * mhz if mhz else 0.0
        # the analytical samples, for reading heights across them
        try:
            incl_samples = included_sample_indices(
                samples_info,
                int(getattr(_raw_int, 'shape', (0, 0))[1])
                if getattr(_raw_int, 'ndim', 1) == 2 else 0)
        except Exception:
            incl_samples = None
        # The height under which detection reports nothing: a line
        # of a pattern the study could not have seen is not missing.
        _noise_sr = float(noise_std or 0.0)
        _floor_hz = _noise_sr * float(min_snr or 0.0)

        # How tall each metabolite's signals should be, from the
        # tallest peak any of them could anchor on: the anchor of a
        # signal is then the nearest peak among those its own
        # proportions allow, not the nearest peak of any height.
        _ref_pp = reference_per_proton(
            db_entries, peaks_to_search, tolerance=tolerance,
            met_key=lambda e: e.get('hmdb_id') or e.get('bmrb_id'))

        for i, entry in enumerate(db_entries):
            db_ppm   = float(
                entry.get('ppm_center') or entry.get('ppm') or 0)
            mult_raw = str(
                entry.get('type') or
                entry.get('multiplicity') or 's').lower().strip()
            j_raw    = str(entry.get('j_coupling') or '').strip()
            try:
                j_hz = float(
                    j_raw.split(',')[0].strip()) if j_raw else 0.0
            except (ValueError, AttributeError):
                j_hz = 0.0

            is_singlet  = mult_raw in ('s', 'singlet', '')

            # A measured pattern that reads as a first-order letter
            # is matched as that letter with the spacing measured:
            # taurine's 3.41 is listed 'd' and is a triplet at 6.7
            # Hz, and a triplet it stays -- on the path with lines
            # to find and a coupling to measure.
            _letter = measured_letter(
                pattern_offsets_hz(entry)) if entry.get(
                'pattern_lines') else None
            if _letter is not None and _letter[0] != mult_raw:
                mult_raw, j_hz = _letter[0], float(_letter[1])
                j_raw = f'{j_hz:.2f}'
                is_singlet = False
                entry = dict(entry)
                entry['type'] = mult_raw
                entry['multiplicity'] = mult_raw
                entry['j_coupling'] = j_raw
                entry['listed_multiplicity'] = str(
                    entry.get('type') or '')

            # An unresolved multiplet whose pattern the reference DB
            # measured is a multiplet with lines to find, not a
            # generic 'm'. Relabelled so every rule downstream -- tier,
            # competition, the line count -- treats it as one.
            _pattern = (pattern_of(entry, mhz)
                        if pattern_governs(
                            mult_raw, entry.get('pattern_lines'))
                        else [])
            if _pattern:
                mult_raw = PATTERN_MULT
                is_singlet = False
                entry = dict(entry)
                entry['type'] = PATTERN_MULT
                entry['multiplicity'] = PATTERN_MULT

            # Skip unquantifiable multiplicities: generic 'm' (no J, no
            # predictable pattern) and BROAD/exchangeable signals (br s,
            # br. s., br d…). Broad lines are exchangeable protons whose
            # area/position drift → unreliable for quantification and a
            # source of false matches (previously a broad singlet was
            # mis-built as a 2-line doublet).
            if is_unquantifiable_multiplicity(mult_raw):
                continue

            n_lns       = (len(_pattern) if _pattern else
                           _n_lines_map.get(
                               mult_raw, expected_line_count(mult_raw, 2)))
            j_ppm       = (j_hz / mhz
                           if j_hz > 0 and mhz > 0 else 0.0)
            half_width  = max(
                j_ppm * (n_lns / 2) * 1.4, tolerance, 0.010)


            try:
                protons = int(entry.get('protons') or 1)
            except (ValueError, TypeError):
                protons = 1
            # Prefer EXACT line positions from the DB transitions table
            # (independent of J, which is missing for ~half the lines);
            # fall back to the J-modelled stick pattern when unavailable.
            _met_id_tr = entry.get('hmdb_id') or entry.get('bmrb_id')
            # The pattern the DB measured beats the curated transition
            # list: it is the same spectrum, read to hundredths of a
            # hertz instead of the half hertz the list is rounded to.
            _tr_lines = _pattern or lines_from_transitions(
                conn, _met_id_tr,
                db_ppm - half_width, db_ppm + half_width, mhz)
            if len(_tr_lines) >= 2:
                sticks = _tr_lines
                _sp = [p for p, _ in _tr_lines]
                half_width = max(half_width,
                                 (max(_sp) - min(_sp)) / 2.0 + tolerance)
            else:
                # No database J: do not invent line positions.
                sticks = (_build_stick_positions(
                    db_ppm, mult_raw, protons, j_raw)
                          if j_hz > 0 else [])
            if not sticks:
                sticks = [(db_ppm, 1.0)]

            # Local window: 10% of the DB ppm value, floored at user tolerance.
            # Defines the candidate pool for sibling/J search.
            # The anchor peak must still be within tolerance to be accepted.
            search_window = max(tolerance, abs(db_ppm) * 0.10)
            local_peaks = [p for p in peaks_to_search
                           if abs(p['ppm'] - db_ppm) <= search_window]
            if not local_peaks:
                continue
            # Any detected peak within tolerance can serve as anchor —
            # quality filtering is left to the composite score.
            # Siblings (J-coupling partners) can be any quality; the
            # anchor cannot — a Poor signal is not annotated.
            good_in_tol = [p for p in local_peaks
                           if abs(p['ppm'] - db_ppm) <= tolerance
                           and is_annotatable(p, _tiers)]
            if not good_in_tol:
                # The fallback below covers a signal peak detection
                # missed entirely. It must not cover one that WAS
                # detected and then excluded by its quality grade —
                # rebuilding it from the raw spectrum would annotate
                # exactly the signal the analyst ruled out.
                if any(abs(p['ppm'] - db_ppm) <= tolerance
                       for p in local_peaks):
                    continue
                # Spectrum-based fallback: no detected peak within
                # tolerance, but the signal may still be present in the
                # raw spectrum (peak missed by detection — sub-threshold,
                # merged, or too close to a neighbour).
                # Mirror what the manual right-drag selection does.
                if not has_spec:
                    continue
                _syn_lo = db_ppm - tolerance
                _syn_hi = db_ppm + tolerance
                _syn_mask = (_ppm_arr >= _syn_lo) & (_ppm_arr <= _syn_hi)
                if _syn_mask.sum() < 3:
                    continue
                _syn_y = _mean_int[_syn_mask]
                _syn_x = _ppm_arr[_syn_mask]
                _syn_max_i = int(np.argmax(_syn_y))
                _syn_h = float(_syn_y[_syn_max_i])
                # Require ≥ 3× the local noise floor to avoid false matches
                _syn_noise = float(np.percentile(np.abs(_syn_y), 25)) + 1e-12
                if _syn_h < _syn_noise * 3.0:
                    continue
                # A spectrum-derived peak was never graded, so it
                # carries no status and reads as Average. Claiming
                # 'Good' would let it bypass the grade selection
                # entirely — the one thing this filter exists to stop.
                _syn_peak = {
                    'ppm':           float(_syn_x[_syn_max_i]),
                    'rank':          None,
                    'height':        _syn_h,
                    'width_ppm':     0.010,
                    'width_at_half': 0.010,
                    '_synthetic':    True,
                    '_obs_j_hz':     0,
                    '_obs_mult':     mult_raw,
                }
                if not is_annotatable(_syn_peak, _tiers):
                    continue
                good_in_tol = [_syn_peak]
                # Augment local_peaks with spectrum-derived sibling
                # positions so the sibling loop and _multiplet_score
                # can find them even if peak detection missed them.
                _syn_aug = list(local_peaks)
                if not is_singlet and j_ppm > 0 and n_lns >= 2:
                    for _k in range(1, n_lns):
                        for _sgn in (+1, -1):
                            _sib_c = _syn_peak['ppm'] + _sgn * _k * j_ppm
                            _sm = ((_ppm_arr >= _sib_c - tolerance * 0.7) &
                                   (_ppm_arr <= _sib_c + tolerance * 0.7))
                            if _sm.sum() < 2:
                                continue
                            _sy = _mean_int[_sm]
                            _sx = _ppm_arr[_sm]
                            _si = int(np.argmax(_sy))
                            if float(_sy[_si]) < _syn_noise * 2.0:
                                continue
                            _sp = float(_sx[_si])
                            if any(abs(p['ppm'] - _sp) < 0.005
                                   for p in local_peaks):
                                continue
                            _syn_aug.append({
                                'ppm':           _sp,
                                'rank':          None,
                                'height':        float(_sy[_si]),
                                'width_ppm':     0.010,
                                'width_at_half': 0.010,
                                '_synthetic':    True,
                                '_obs_j_hz':     0,
                                '_obs_mult':     mult_raw,
                            })
                local_peaks = _syn_aug
            nearest   = min(
                consistent_peaks(
                    good_in_tol, entry,
                    _ref_pp.get(entry.get('hmdb_id')
                                or entry.get('bmrb_id'))),
                key=lambda p: abs(p['ppm'] - db_ppm))
            delta_ppm = abs(nearest['ppm'] - db_ppm)
            peak_rank = nearest.get('rank')   # None for spectrum-derived peaks
            peak_ppm  = float(nearest['ppm'])

            # A missing database J is intentionally not inferred from
            # the observed spectrum. The analyst must confirm it.
            _eff_j_ppm = j_ppm if j_hz > 0 else 0.0

            multiplet_centre = peak_ppm
            # A pattern's lines are found by match_pattern, two hertz
            # from where the shape puts them. Walking siblings from
            # the anchor at the list's first coupling -- 1.2 Hz for
            # NAD's 8.197 ppm pattern -- within 1.5 tolerances took
            # any peak 0.075 ppm away for a line, and the multiplet
            # centre, and the plot's window, landed on it.
            #
            # A letter's siblings are looked for the same two hertz
            # from where the coupling puts them (PATTERN_LINE_TOL_HZ).
            # The annotation's tolerance is for where the multiplet
            # sits, not for where its lines sit: at 1.5 tolerances
            # IMP's doublet of 2.05 Hz took a peak six hertz off for
            # its second line, and the centre moved between the two.
            _sib_tol = (PATTERN_LINE_TOL_HZ / mhz) if mhz else tolerance
            if (not is_singlet and _eff_j_ppm > 0 and n_lns >= 2
                    and not _pattern):
                found_line_ppms = [peak_ppm]
                for k in range(1, n_lns):
                    for sign in (+1, -1):
                        expected = peak_ppm + sign * k * _eff_j_ppm
                        best_sib = min(
                            (p for p in local_peaks if p is not nearest),
                            key=lambda p: abs(p['ppm'] - expected),
                            default=None)
                        if best_sib is not None:
                            dist = abs(best_sib['ppm'] - expected)
                            if dist <= _sib_tol:
                                if best_sib['ppm'] not in found_line_ppms:
                                    found_line_ppms.append(float(best_sib['ppm']))
                                break
                if len(found_line_ppms) >= 2:
                    multiplet_centre = float(np.mean(found_line_ppms))

            _shift        = multiplet_centre - db_ppm
            shifted_sticks = [(s + _shift, ri) for s, ri in sticks]

            win_lo = multiplet_centre - half_width
            win_hi = multiplet_centre + half_width

            fitting_r2  = 0.0
            r2_computed = False
            shape_score_val = None
            if has_spec:
                _exp_hwhm = max(
                    float(nearest.get('width_ppm',
                          nearest.get('width_at_half', 0.008))) / 2,
                    0.003)
                spec_mask = (_ppm_arr >= win_lo) & (_ppm_arr <= win_hi)
                if np.sum(spec_mask) >= 4:
                    x_win  = _ppm_arr[spec_mask]
                    y_win  = _mean_int[spec_mask].copy()
                    y_corr = _baseline_correct_linear(x_win, y_win)
                    sim    = simulate_lorentzian_profile(
                        shifted_sticks, x_win, hwhm=_exp_hwhm)
                    fitting_r2, _ = _fit_r2(sim, y_corr)
                    r2_computed = True
                # Shape term of the signal score, measured in this
                # signal's own window — the same function the results
                # window uses, so both see one number.
                shape_score_val = peak_shape_score(
                    _ppm_arr, _mean_int, shifted_sticks,
                    multiplet_centre, entry, hwhm_default=_exp_hwhm)

            # peak_rank is None for spectrum-derived/synthetic peaks;
            # excluding them keeps the "X/Y peaks matched" count honest
            # (a single None would otherwise collapse all of them to 1).
            if peak_rank is not None:
                peaks_matched.add(peak_rank)

            if not is_singlet and shifted_sticks:
                _nearest_stick = min(shifted_sticks,
                                     key=lambda sv: abs(sv[0] - peak_ppm))
                ppm_score_val = _ppm_score(peak_ppm, _nearest_stick[0], tolerance)
            else:
                ppm_score_val = _ppm_score(multiplet_centre, db_ppm, tolerance)
            obs_mult_pk = nearest.get('_obs_mult', 's')

            _scorer_centre = None

            if nearest.get('is_sub_peak'):
                siblings = sorted(
                    [p for p in peaks_to_search
                     if p.get('rank') == peak_rank
                     and p.get('is_sub_peak')],
                    key=lambda p: p['ppm'], reverse=True)
                if siblings:
                    max_h = max(p.get('height', 1) for p in siblings) or 1
                    predicted_lines = [
                        {'ppm': float(p['ppm']),
                         'rel_intensity': float(p.get('height', 1)) / max_h}
                        for p in siblings]
                else:
                    predicted_lines = [
                        {'ppm': float(s), 'rel_intensity': float(ri)}
                        for s, ri in shifted_sticks]
            else:
                # Candidate line positions for the measurement below.
                # J and the line count are NOT taken from here — a mean
                # of whatever separations the scorer returned is not a
                # coupling constant. measure_multiplet owns both.
                _score_r = dict(entry)
                _score_r['peak_ppm'] = multiplet_centre
                if _pattern:
                    # The scorer scores a known letter from its J and
                    # anything else from the first J alone -- which
                    # turned a seven-line pattern into a doublet. It
                    # has a path for a line list; the measured pattern
                    # takes it, at this spectrometer's field.
                    _score_r['db_transitions'] = [
                        {'ppm': pp, 'intensity': ri}
                        for pp, ri in _pattern]
                _ms_main, _md_main, _pl_main = _multiplet_score(
                    _score_r, local_peaks, tolerance, mhz,
                    ppm_array=(_ppm_arr if has_spec else None),
                    intensity_array=(_mean_int if has_spec else None))
                _stick_lines = [{'ppm': float(s), 'rel_intensity': float(ri)}
                                for s, ri in shifted_sticks]
                predicted_lines = _pl_main if _pl_main else _stick_lines
                _scorer_centre = multiplet_centre
                if _pl_main:
                    _srt = sorted(_pl_main, key=lambda l: l['ppm'])
                    _tot_ri = (sum(l.get('rel_intensity', 1.0) for l in _srt)
                               or 1.0)
                    _c_sc = (sum(l['ppm'] * l.get('rel_intensity', 1.0)
                                 for l in _srt) / _tot_ri)
                    # The scorer looks for the lines over the whole local
                    # pool (a tenth of the ppm: a third of a ppm at 3 ppm)
                    # and can centre them on somebody else's peaks. Lines
                    # centred further from the reference than the
                    # tolerance are not this signal's -- the match was
                    # dropped for exactly that at the end, with a detected
                    # peak sitting at the reference (ethanolamine,
                    # 3-ureidopropionate, mesaconate never annotated on
                    # the tube of 100). The signal is judged on its lines
                    # around the anchor instead.
                    if abs(_c_sc - db_ppm) <= tolerance:
                        _scorer_centre = _c_sc
                    else:
                        predicted_lines = _stick_lines

            # ── Pattern measurement ──────────────────────────
            # A measured pattern is matched as a shape, not as a
            # letter with one J. Its lines are looked for within two
            # hertz of where the aligned pattern puts them, and the
            # residual over the found lines stands in for delta-J --
            # scaled to delta-J's gates so every rule downstream reads
            # it unchanged, with the raw residual kept beside it.
            if _pattern:
                _meas_sr = measure_pattern_signal(
                    entry, local_peaks, field_mhz=mhz,
                    dj_gold_hz=_sc_params['dj_gold_hz'],
                    anchor_ppm=peak_ppm, merge_hz=_merge_hz,
                    fwhm_hz=_lw_hz, floor=_floor_hz or None,
                    trace=((_ppm_arr, _mean_int) if has_spec
                           else None),
                    sigma=_noise_sr or None, tolerance_ppm=tolerance,
                    shape_good=_sc_params.get('shape_good'),
                    shape_min=_sc_params.get('shape_min'))
                _pattern_residual = _meas_sr.pattern_residual_hz
                _pattern_agreement = _meas_sr.pattern_intensity_agreement
            else:
                _pattern_residual = None
                _pattern_agreement = None

            # ── Line measurement ─────────────────────────────
            # Snap the DB multiplet's predicted lines onto detected
            # peaks, count how many were actually found, and measure J
            # from their real spacings. This decides whether the signal
            # was identified at all, so it belongs here — before the
            # score and before the candidate can compete — not in the
            # results window, where it used to run.
            #
            # It replaces _observe_j_hz, which searched the peak
            # separations for whichever one best matched the DB J. That
            # is circular: ΔJ came out small by construction, because
            # the separation minimising it was the one selected. It also
            # never counted lines, so a doublet with one leg missing
            # reported a confident J measured from unrelated peaks.
            _n_exp_sr = max(
                expected_line_count(mult_raw,
                                    fallback=len(predicted_lines) or 2),
                n_lns)
            if not _pattern:
                _meas_sr = measure_multiplet(
                    predicted_lines, peaks_to_search,
                    mult=mult_raw, j_coupling=j_raw,
                    db_ppm=db_ppm, centre_hint=multiplet_centre,
                    n_expected=_n_exp_sr, tolerance=tolerance,
                    field_mhz=mhz, linewidth_hz=_lw_hz)
            if (not _pattern and _meas_sr.measured
                    and not _meas_sr.complete and j_hz > 0):
                # Lines missing: before giving up, look in the window for
                # the multiplet's comb — n_exp peaks spaced by J, every
                # spacing agreeing with the DB J AND with each other.
                # Line-matching works outward from stored positions and
                # can miss a multiplet that sits off them; a template
                # match does not depend on where it was expected.
                _hw_sr = (j_hz / mhz) * (_n_exp_sr - 1) / 2 + tolerance
                _comb_sr = search_multiplet_in_window(
                    [p_.get('ppm') for p_ in peaks_to_search],
                    centre_ppm=multiplet_centre, j_hz=j_hz,
                    n_expected=_n_exp_sr, field_mhz=mhz,
                    half_width_ppm=_hw_sr)
                if _comb_sr:
                    _retry_sr = measure_multiplet(
                        [{'ppm': v, 'rel_intensity': 1.0} for v in _comb_sr],
                        peaks_to_search,
                        mult=mult_raw, j_coupling=j_raw,
                        db_ppm=db_ppm,
                        centre_hint=sum(_comb_sr) / len(_comb_sr),
                        n_expected=_n_exp_sr, tolerance=tolerance,
                        field_mhz=mhz, linewidth_hz=_lw_hz)
                    if _retry_sr.complete:
                        _meas_sr = _retry_sr

            n_found_sr = _meas_sr.n_found
            n_exp_sr   = _meas_sr.n_expected
            obs_j_hz   = _meas_sr.observed_j_hz
            delta_j_hz = _meas_sr.delta_j_hz
            if _meas_sr.measured and _meas_sr.predicted_lines:
                predicted_lines = _meas_sr.predicted_lines
            if _meas_sr.centre_ppm is not None:
                _scorer_centre = float(_meas_sr.centre_ppm)

            # ── Signal score ────────────────────────────────────────
            # Same function the Annotation Results window uses, so the
            # number the search produces and the number the user sees
            # are the same quantity. They used to be two different
            # formulas under one key: a Δppm-weighted blend here (45%
            # Δppm, 10% J) and a ΔJ-led one there — the same signal
            # scored 0.317 in one and 0.972 in the other.
            #
            # It is computed ONCE, here, on measured evidence: the
            # lines have been snapped and J measured just above. The
            # results window does not recompute it — a view shows
            # results, it does not produce them.
            _dj_sig = (dj_score(delta_j_hz,
                                _sc_params['dj_gold_hz'],
                                _sc_params['dj_bad_hz'],
                                _sc_params['dj_neutral'])
                       if (not is_singlet and delta_j_hz is not None)
                       else None)
            _shape_sig = (float(shape_score_val)
                          if shape_score_val is not None else None)
            _sig_bd = signal_score(s_ppm=ppm_score_val,
                                   s_dj=_dj_sig,
                                   s_cosine=_shape_sig,
                                   is_singlet=is_singlet,
                                   params=_sc_params)
            composite = _sig_bd.value

            # Multi-peak coherence: penalise candidates whose sibling
            # peaks are absent or have inconsistent relative intensities.
            # MIN_COH=0.05 retains a small floor for genuinely congested
            # spectra while avoiding the 35% inflation of the old value.
            _met_id_coh = entry.get('hmdb_id') or entry.get('bmrb_id', '')
            _coh, _cov, _rm = _multi_peak_coherence(
                _met_id_coh, all_peaks_by_id, peaks_to_search, tolerance)
            _MIN_COH = 0.05
            composite *= _MIN_COH + (1.0 - _MIN_COH) * _coh

            # Graph STOCSY boost: metabolites whose matched peaks
            # co-vary across samples score higher.
            # Weight w uses exponential saturation so n=3 gives a small
            # but non-zero boost while n≥10 approaches the full effect.
            _stocsy_coh = _stocsy_coherence(
                _met_id_coh, all_peaks_by_id, peaks_to_search,
                tolerance, _pk_corr)
            if _stocsy_coh is not None:
                _w_st = float(
                    1.0 - np.exp(-0.25 * max(0, _n_samp_st - 2)))
                composite *= np.exp(
                    0.10 * max(0.0, _stocsy_coh) * _w_st)


            r = dict(entry)
            r['fitting_r2']     = float(fitting_r2)
            r['composite_score']= float(composite)
            r['_score_breakdown'] = _sig_bd.equation()
            r['multiplet_score']= float(fitting_r2)
            r['ppm_score']      = float(ppm_score_val)
            r['db_ppm']         = float(db_ppm)
            _eff_centre = (_scorer_centre if _scorer_centre is not None
                           else multiplet_centre)
            # A pattern's drift is where its reference position fell,
            # not where its found lines centre: the two differ by the
            # pattern's asymmetry and by which lines were found.
            _origin = (_meas_sr.origin_ppm
                       if _pattern and _meas_sr.origin_ppm is not None
                       else _eff_centre)
            r['ppm_diff']       = float(abs(_origin - db_ppm))
            r['peak_ppm']       = _eff_centre
            r['pattern_centre_offset_ppm'] = (
                float(_eff_centre - _origin) if _pattern else None)
            r['peak_rank']      = peak_rank
            r['is_sub_peak']    = bool(nearest.get('is_sub_peak', False))
            r['predicted_lines']= predicted_lines
            # The reference's own lines, at the centre found here:
            # what the letter promises. A quantification plot draws
            # them beside the lines it integrated.
            r['reference_lines'] = [{'ppm': float(sp), 'rel_intensity': float(ri)}
                                    for sp, ri in shifted_sticks]
            # Single source: measure_multiplet above. It reports None
            # when lines are missing, and None must stay None — a ΔJ
            # measured on a different set of peaks is not this signal's.
            r['observed_j_hz']  = (float(obs_j_hz)
                                   if obs_j_hz is not None else None)
            r['delta_j_hz']     = (float(delta_j_hz)
                                   if delta_j_hz is not None else None)
            r['_n_found_lines'] = n_found_sr
            r['_n_exp_lines']   = n_exp_sr
            r['pattern_residual_hz'] = _pattern_residual
            r['pattern_merge_hz'] = _merge_hz if _pattern else None
            r['pattern_floor'] = (_floor_hz or None) if _pattern else None
            r['pattern_sigma'] = (_noise_sr or None) if _pattern else None
            # The field every signal's profile is placed at, not
            # only a pattern's: a shared window fits a letter by
            # the reference's profile too (quant_shared).
            r['pattern_field_mhz'] = mhz
            r['pattern_tolerance_ppm'] = tolerance if _pattern else None
            r['pattern_fit'] = (
                {'snr': _meas_sr.pattern_snr,
                 'explained': _meas_sr.pattern_explained,
                 'r2': _meas_sr.pattern_r2,
                 'amplitude': _meas_sr.pattern_amplitude}
                if _pattern and _meas_sr.pattern_snr is not None
                else None)
            # What the lines were judged at, so a re-measurement after
            # recalibration asks the same question of the same width.
            r['study_linewidth_hz'] = _lw_hz or None
            r['expected_j_hz'] = _meas_sr.expected_j_hz
            # Do the lines rise and fall together across the samples?
            # One molecule's do; two singlets of different molecules at
            # the right spacing do not. Read off the traces, so a
            # pattern matched on the peak list is judged too.
            _line_ppms_cv = [float(l['ppm']) for l in
                             (predicted_lines or ()) if 'ppm' in l]
            r['lines_covary'] = (
                lines_covary_on_trace(
                    _ppm_arr, _raw_int, _line_ppms_cv,
                    included=incl_samples,
                    half_ppm=max(_lw_ppm / 2.0, 0.001),
                    noise=_noise_sr or None)
                if has_spec and len(_line_ppms_cv) >= 2
                and getattr(_raw_int, 'ndim', 1) == 2 else None)
            if _pattern:
                r['intensity_agreement'] = _pattern_agreement
            r['j_coupling']     = j_raw
            r['baseline_corrected'] = has_spec
            r['multiplet_detail']   = (
                _md_main if not nearest.get('is_sub_peak') else '')
            r['observed_mult']      = obs_mult_pk
            r['coherence_score']    = round(_coh, 3)
            r['peak_coverage']      = round(_cov, 3)
            r['ratio_match']        = round(_rm,  3)
            r['stocsy_coherence']   = (round(_stocsy_coh, 3)
                                       if _stocsy_coh is not None
                                       else None)
            # How many of THIS analyte's other resonances co-vary with
            # THIS peak. The coherence above is the metabolite's mean
            # over all its pairs and says nothing about the contested
            # peak in particular; this does, and it is what decides a
            # contest whose direct evidence is identical for both
            # claimants -- which is the majority of them.
            r['stocsy_partners'] = _stocsy_partner_count(
                _met_id_coh, r.get('peak_rank'), all_peaks_by_id,
                peaks_to_search, tolerance, _pk_corr)
            # Database provenance is recorded, not multiplied into the
            # score. Scaling by 0.82 (BMRB) meant a signal needed a raw
            # 0.915 to reach the 0.75 Good threshold, and the Annotation
            # Results window never applied the same factor — so the same
            # signal read 0.317 here and 0.972 there. Provenance is a
            # property of the reference, not evidence about this peak.
            _db_src      = r.get('database', 'HMDB')
            r['source_weight']    = _SOURCE_WEIGHTS.get(_db_src, 0.70)
            # Map detected peak to which predicted line it corresponds to.
            # Stored as line_idx so _effective_nh can use it directly
            # without positional estimation at quantification time.
            _pl_li = r.get('predicted_lines') or []
            if len(_pl_li) > 1:
                _det_ppm_li = float(r.get('peak_ppm') or 0)
                r['line_idx'] = min(
                    range(len(_pl_li)),
                    key=lambda _i: abs(float(_pl_li[_i].get('ppm', 0)) - _det_ppm_li))
            else:
                r['line_idx'] = 0
            if r.get('ppm_diff', float('inf')) > tolerance:
                continue
            all_results.append(r)

            if (i + 1) % 100 == 0 or i == total_entries - 1:
                _el = time.monotonic() - t0
                if progress is not None:
                    progress(i + 1, total_entries, len(all_results),
                             len(peaks_matched), _el)

        seen = {}
        for r in all_results:
            key = (
                r.get('hmdb_id') or r.get('bmrb_id', ''),
                round(float(
                    r.get('ppm_center') or r.get('ppm') or 0),
                    3))
            # Dedup/rank by composite_score (single source of truth),
            # not fitting_r2 — otherwise the prefilter could drop the
            # candidate the window would score best (filter-by-A/sort-by-B).
            if (key not in seen
                    or r.get('composite_score', 0)
                    > seen[key].get('composite_score', 0)):
                seen[key] = r
        all_results = list(seen.values())

        all_results.sort(
            key=lambda r: r.get('composite_score', 0), reverse=True)

        if carry_results:
            all_results = list(carry_results) + all_results

        return SearchOutcome(results=all_results,
                             peaks_annotatable=len(_annotatable),
                             peaks_matched=len(peaks_matched),
                             elapsed_s=time.monotonic() - t0)
    finally:
        if conn:
            conn.close()
