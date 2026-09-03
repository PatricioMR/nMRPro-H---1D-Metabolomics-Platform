# processing/quant_engine.py
"""Elect, integrate, concentrate — the whole of quantification, in that order.

Three decisions, each made once, by a function that can be read on its own:

  elect        which signal quantifies which metabolite, and why the others
               got nothing (processing.quantification)
  integrate    the area of that signal's multiplet in every sample, foot to
               foot (processing.quant_integration)
  concentrate  that area per proton, against the standard, in mM
               (processing.quant_concentration)

Deliberately absent: JG-Deconv. The old engine folded its fractional areas,
its NNLS rejections and its per-line scaling into the same expression that
computed the concentration, so a number could not be read without knowing
which of them had fired. This engine measures what is in the window and says
so; sharing an overlapped window between analytes is a separate question and
gets a separate answer, later.

Also absent: the four modes that were alternatives to multiplet integration,
and the three paths that put a metabolite back after it lost. A metabolite
that lost has no number here, and the reason it has none is recorded.
"""
from __future__ import annotations

import numpy as np

from processing.quant_concentration import (
    apply_pqn, concentrations, signal_nh, standard_per_proton)
from processing.quant_integration import integrate_signal
from processing.quant_snr import noise_level, representative_snr, window_snr
from processing.quantification import REJECTION_TEXT, elect

# The status dots the results table shows. Confidence is about how much of the
# expected evidence this signal actually carried, not about how large the
# number is: a big concentration measured on one unverified line is not a
# confident one.
STATUS_STRONG = '●●●'
STATUS_FAIR = '●●○'
STATUS_WEAK = '●○○'
STATUS_NONE = 'n.q.'


def _cv_percent(values):
    v = [float(x) for x in (values or ()) if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    if m <= 0:
        return float('nan')
    return float(np.std(v, ddof=1) / m * 100.0)


def _confidence(signal) -> tuple[int, str]:
    """How much of the expected evidence this signal carried.

    Three: the multiplet was complete and its coupling agreed. Two: complete,
    coupling off but inside the gate. One: everything else that still produced
    a number — a singlet, which has no coupling to agree with, among them.
    """
    ev = signal.evidence or {}
    n_f, n_e = ev.get('n_found'), ev.get('n_expected')
    complete = (n_f is not None and n_e is not None and n_f >= n_e)
    dj = ev.get('delta_j_hz')
    if complete and dj is not None and abs(float(dj)) <= 0.20:
        return 3, STATUS_STRONG
    if complete and dj is not None:
        return 2, STATUS_FAIR
    return 1, STATUS_WEAK


def quantify(*, candidates, peaks_by_rank, ppm, intensities, samples_info,
             included, groups, dj_gold_hz, ambiguous_ranks=(),
             standard_areas=None, standard_nh=1.0, standard_conc=0.0,
             standard_name='Internal standard', standard_ppm=0.0,
             standard_snr=None,
             external_standard_samples=None, dilution=None, correction=None,
             use_pqn=True, local_baseline=True, proton_correction=True,
             align=False, blank_indices=None) -> list[dict]:
    """Quantify a study and return one row per metabolite.

    Every metabolite that was a candidate gets a row, quantified or not: an
    analyte that vanishes from the table cannot be questioned, and the reason
    it has no number is the part an analyst can act on.

    ``candidates`` are ``processing.quantification.Candidate`` objects whose
    ``payload`` carries whatever the caller needs to build display fields —
    this engine only reads ``met_id``, the evidence, and the line peaks.
    """
    election = elect(candidates, dj_gold_hz=dj_gold_hz,
                    ambiguous_ranks=ambiguous_ranks)

    # ── Integrate every winner, per sample ───────────────────────────────────
    measured, per_proton_areas = {}, {}
    for mid, cand in election.winners.items():
        lines = [peaks_by_rank[r] for r in sorted(cand.line_ranks)
                 if r in peaks_by_rank]
        if not lines and cand.peak_rank in peaks_by_rank:
            lines = [peaks_by_rank[cand.peak_rank]]
        out = integrate_signal(lines, ppm, intensities, included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices)
        if not out or not out['areas']:
            continue
        nh = signal_nh((cand.evidence or {}).get('protons'))
        lo, hi = out['window']
        snr = window_snr(ppm, intensities, lo, hi,
                         noise_level(lines, ppm, intensities),
                         included=included)
        measured[mid] = {'signal': cand, 'nh': nh, 'snr': snr, **out}
        per_proton_areas[mid] = {s: a / nh for s, a in out['areas'].items()}

    # ── Dilution correction across samples ───────────────────────────────────
    # PQN before the standard: it corrects how concentrated the sample is, and
    # the standard then says what the corrected signal is worth. Running it
    # after would divide the answer by a factor derived from the answer.
    pqn_factors = {}
    if use_pqn and len(per_proton_areas) >= 3:
        per_proton_areas, pqn_factors = apply_pqn(per_proton_areas)

    standard = None
    if standard_areas:
        standard = standard_per_proton(
            standard_areas, standard_nh,
            external_samples=external_standard_samples)

    rows = []
    for mid, info in measured.items():
        nh = info['nh']
        areas = {s: v * nh for s, v in per_proton_areas[mid].items()}
        conc = concentrations(areas, nh, standard=standard,
                              standard_conc=standard_conc,
                              external=bool(external_standard_samples),
                              dilution=dilution, correction=correction,
                              proton_correction=proton_correction)
        by_group = {g: [conc[s] for s in idxs
                        if s in conc and np.isfinite(conc[s]) and conc[s] > 0]
                    for g, idxs in (groups or {}).items()}
        good = [v for v in conc.values() if np.isfinite(v) and v > 0]
        level, dots = _confidence(info['signal'])
        rows.append(_row(mid, info, conc, by_group, good, level, dots,
                         pqn_factors))

    for mid, reason in election.rejected.items():
        rows.append(_rejected_row(mid, reason))

    if standard_areas:
        rows.append(_standard_row(standard_areas, standard_nh, standard_conc,
                                  standard_name, standard_ppm, standard_snr,
                                  groups, standard))
    return rows


def _standard_row(areas, nh, conc_nominal, name, ppm_pos, snr, groups,
                  standard) -> dict:
    """The internal standard, measured like everything it standardises.

    Its recovery is the check on the whole run: the standard was added at a
    known concentration, so reading it back against itself must return that
    concentration. When it does not, the areas it divided are not trustworthy
    either, and the analyst needs to see that before reading any other row.
    """
    conc = concentrations(areas, nh, standard=standard,
                          standard_conc=conc_nominal)
    by_group = {g: [conc[s] for s in idxs
                    if s in conc and np.isfinite(conc[s]) and conc[s] > 0]
                for g, idxs in (groups or {}).items()}
    good = [v for v in conc.values() if np.isfinite(v) and v > 0]
    row = _rejected_row(name, 'standard')
    row.update({
        'metabolite': name,
        'hmdb_id': name,
        'ppm': float(ppm_pos or 0.0),
        'n_peaks': 1,
        'raw_area': float(np.mean(list(areas.values()))) if areas else float('nan'),
        'corr_area': float(np.mean(list(areas.values()))) if areas else float('nan'),
        'conc_mm': float(np.mean(good)) if good else float('nan'),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        'snr': float(snr) if snr is not None else 0.0,
        'status': STATUS_STRONG,
        '_confidence': 3,
        '_not_quantified': not good,
        '_status_reasons': [],
        '_nh': float(nh),
        '_areas_per_sample': dict(areas),
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_is_is': True,
    })
    row.pop('_rejection', None)
    return row


def _row(mid, info, conc, by_group, good, level, dots, pqn_factors) -> dict:
    lo, hi = info['window']
    sig = info['signal']
    return {
        'hmdb_id': mid,
        'metabolite': mid,
        'ppm': float((lo + hi) / 2.0),
        'n_peaks': len(sig.line_ranks) or 1,
        'raw_area': float(np.mean(list(info['areas'].values()))),
        'corr_area': float(np.mean(list(info['areas'].values()))),
        'conc_mm': float(np.mean(good)) if good else float('nan'),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        'snr': representative_snr(info.get('snr')),
        'status': dots,
        'vip_score': float('nan'),
        '_confidence': level,
        '_integ_mode': 'Multiplet',
        '_not_quantified': not good,
        '_status_reasons': [] if good else [REJECTION_TEXT.get('zero', '')],
        '_peak': None,
        '_peak_rank': sig.peak_rank,
        '_display_peak_rank': sig.peak_rank,
        '_multiplet_window': [float(lo), float(hi)],
        '_nh': float(info['nh']),
        '_areas_per_sample': info['areas'],
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_pqn_factors': pqn_factors,
        '_snr_per_sample': info.get('snr', {}),
        '_peak_align_shifts': info.get('shifts') or None,
        '_blk_mean_area': info.get('blank_area', float('nan')),
        '_is_singlet': (sig.evidence or {}).get('multiplicity') == 's',
        '_ann_score': float((sig.evidence or {}).get('score') or 0.0),
        '_hits': [sig.payload] if sig.payload is not None else [],
        # JG-Deconv is not part of this engine. The fields stay so the window
        # keeps working, and they say plainly that nothing was applied rather
        # than leaving the reader to guess from a missing key.
        '_used_j_resolve': False,
        '_j_resolve_frac': 1.0,
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }


def _rejected_row(mid, reason) -> dict:
    """A metabolite with no number still gets a row, and says why.

    An analyte that disappears from the table cannot be questioned, and the
    reason is the actionable half: a withdrawn multiplet comes back when
    detection resolves its lines, an ambiguous rank when the analyst marks one
    candidate, a beaten one only if the winner is wrong.
    """
    return {
        'hmdb_id': mid,
        'metabolite': mid,
        'ppm': 0.0,
        'n_peaks': 0,
        'raw_area': float('nan'),
        'corr_area': float('nan'),
        'conc_mm': float('nan'),
        'cv_by_group': {},
        'snr': 0.0,
        'status': STATUS_NONE,
        'vip_score': float('nan'),
        '_confidence': 0,
        '_integ_mode': 'Multiplet',
        '_not_quantified': True,
        '_status_reasons': [REJECTION_TEXT.get(reason, str(reason))],
        '_rejection': reason,
        '_peak': None,
        '_peak_rank': None,
        '_display_peak_rank': None,
        '_multiplet_window': None,
        '_nh': float('nan'),
        '_areas_per_sample': {},
        '_concs_per_sample': {},
        '_concs_by_group': {},
        '_pqn_factors': {},
        '_snr_per_sample': {},
        '_peak_align_shifts': None,
        '_blk_mean_area': float('nan'),
        '_is_singlet': False,
        '_ann_score': 0.0,
        '_hits': [],
        '_used_j_resolve': False,
        '_j_resolve_frac': 1.0,
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }
