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
from processing.quant_detection_limit import (
    BELOW_LOD, BELOW_LOQ, K_LOD, UNRESOLVED, censor, censor_text,
    detection_limits, predicted_window)
from processing.quant_integration import integrate_sample, integrate_signal
from processing.quant_snr import noise_level, representative_snr, window_snr
from processing.quantification import (
    NO_MEASURABLE_WINDOW, REJECTION_TEXT, elect)

# The status dots the results table shows. Confidence is about how much of the
# expected evidence this signal actually carried, not about how large the
# number is: a big concentration measured on one unverified line is not a
# confident one.
STATUS_STRONG = '●●●'
STATUS_FAIR = '●●○'
STATUS_WEAK = '●○○'
STATUS_NONE = 'n.q.'

# Censored rows carry their own marks. They are not failures and not results:
# an upper bound is a measurement, and reading it as either of the other two
# loses what it says.
STATUS_CENSORED = {
    BELOW_LOD: '< LOD',
    BELOW_LOQ: '< LOQ',
    UNRESOLVED: 'overlap?',
}


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
             align=False, blank_indices=None,
             noise_sigma=None, linewidth_ppm=0.008,
             drift_ppm=0.0) -> list[dict]:
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
    # A winner whose lines bound no window is recorded as such rather than
    # dropped. An analyte that simply vanishes between the election and the
    # table cannot be questioned, and this is the one place where a metabolite
    # can win and still have nothing to measure.
    measured, per_proton_areas, unmeasurable = {}, {}, []
    for mid, cand in election.winners.items():
        lines = [peaks_by_rank[r] for r in sorted(cand.line_ranks)
                 if r in peaks_by_rank]
        if not lines and cand.peak_rank in peaks_by_rank:
            lines = [peaks_by_rank[cand.peak_rank]]
        out = integrate_signal(lines, ppm, intensities, included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices)
        if not out or not out['areas']:
            unmeasurable.append(mid)
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
    # A winner with no detected line is not a failure to measure — it is a
    # measurement of an upper bound, and reporting it as one is the whole of
    # processing.quant_detection_limit.
    for mid in unmeasurable:
        rows.append(_censored_row(
            election.winners[mid], ppm, intensities, included=included,
            standard=standard, standard_conc=standard_conc,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
            drift_ppm=drift_ppm, local_baseline=local_baseline))

    if standard_areas:
        rows.append(_standard_row(standard_areas, standard_nh, standard_conc,
                                  standard_name, standard_ppm, standard_snr,
                                  groups, standard,
                                  external_samples=external_standard_samples))
    return rows


def _censored_row(cand, ppm, intensities, *, included, standard,
                  standard_conc, noise_sigma, linewidth_ppm, drift_ppm,
                  local_baseline) -> dict:
    """An analyte the experiment could not see, reported as the bound it is.

    The limits travel through the same chain as any concentration -- divided by
    nH, referenced to the standard -- because a limit in area units cannot be
    compared with the numbers in the table, and a limit nobody can compare is
    not a limit.

    The window is always integrated against a local baseline here, whatever the
    run's integration setting says. Currie's limits describe the NET signal --
    what stands above the background -- so comparing an integral that still
    contains the background against a limit built from noise alone compares two
    different quantities. Measured on a 197-point window: pure noise sitting on
    an offset of +2 sigma, with no peak in it at all, integrates to three times
    the quantification limit and is reported as an unresolved signal. Every one
    of the reference study's 28 censored analytes came back that way.

    The run's own ``local_baseline`` flag is a different decision: it governs
    how a quantified area is measured, where it costs a known ~5% on the ratio
    between two analytes. Here there is no choice to make.
    """
    ev = cand.evidence or {}
    mid = cand.met_id
    win = predicted_window(ev.get('predicted_ppms'),
                           linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm)
    if win is None or noise_sigma is None or not np.isfinite(noise_sigma):
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    lo, hi = win
    ppm = np.asarray(ppm, dtype=float)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_pts = int(((ppm >= lo) & (ppm <= hi)).sum())
    if n_pts < 2:
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    dppm = float(abs(ppm[1] - ppm[0])) if ppm.size > 1 else 0.0

    areas = {}
    for s_i in (included if included is not None else range(arr.shape[1])):
        if 0 <= s_i < arr.shape[1]:
            a = integrate_sample(ppm, arr[:, s_i], lo, hi,
                                 local_baseline=True)
            if a is not None:
                areas[int(s_i)] = float(a)
    area = float(np.median(list(areas.values()))) if areas else float('nan')

    lod_a, loq_a = detection_limits(noise_sigma, dppm, n_pts)
    verdict = censor(area, lod_a, loq_a)

    nh = signal_nh(ev.get('protons'))
    lod_mm, loq_mm = (_as_concentration(v, nh, standard, standard_conc)
                      for v in (lod_a, loq_a))

    row = _rejected_row(mid, verdict)
    row.update({
        'ppm': float((lo + hi) / 2.0),
        'raw_area': area,
        'corr_area': area,
        'snr': (area / (lod_a / K_LOD)
                if np.isfinite(area) and lod_a and np.isfinite(lod_a)
                else float('nan')),
        'status': STATUS_CENSORED[verdict],
        '_status_reasons': [censor_text(verdict, lod_mm, loq_mm)],
        '_censored': verdict,
        '_lod_mm': lod_mm,
        '_loq_mm': loq_mm,
        '_multiplet_window': [float(lo), float(hi)],
        '_nh': float(nh),
        '_areas_per_sample': areas,
        '_ann_score': float(ev.get('score') or 0.0),
        '_hits': [cand.payload] if cand.payload is not None else [],
        '_is_singlet': ev.get('multiplicity') == 's',
    })
    return row


def _as_concentration(area, nh, standard, standard_conc) -> float:
    """One area in the same units the table's concentrations are in."""
    if not np.isfinite(area):
        return float('nan')
    per = float(area) / signal_nh(nh)
    if not standard:
        return per
    refs = [v for v in standard.values() if v and np.isfinite(v) and v > 0]
    if not refs:
        return float('nan')
    return per / float(np.median(refs)) * float(standard_conc)


def _standard_row(areas, nh, conc_nominal, name, ppm_pos, snr, groups,
                  standard, external_samples=None) -> dict:
    """The internal standard, measured like everything it standardises.

    Its recovery is the check on the whole run: the standard was added at a
    known concentration, so reading it back against itself must return that
    concentration. When it does not, the areas it divided are not trustworthy
    either, and the analyst needs to see that before reading any other row.

    Read over the injections that actually contain it. An externally injected
    standard is absent from the analytical samples by design, so integrating
    its window there measures the baseline underneath it; averaging that in
    drags the recovery down in proportion to how many analytical samples the
    study has and reports a preparation failure that did not happen. On a
    21-sample study with 5 external injections it read 57.5%.
    """
    if external_samples:
        keep = {int(s) for s in external_samples}
        areas = {s: a for s, a in (areas or {}).items() if int(s) in keep}
        groups = {g: [i for i in idxs if int(i) in keep]
                  for g, idxs in (groups or {}).items()}
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
