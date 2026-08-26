# processing/annotation_detectability.py
"""What can be expected to show up, and what competes with what.

Two questions the annotation asks of the spectrum itself, before any candidate
is judged:

  * **Detectability** — given how strong a metabolite's best signal is, how
    strong should its other signals be? A resonance predicted to sit under the
    noise floor is absent for a physical reason, not because the assignment is
    wrong.

    This module MEASURES that expectation and stops there. It used to also act
    on it, flagging signals "below detection limit" and hiding them from the
    tree, from the overlap map and from quantification. That decision belongs
    with the singlet-anchoring work — a signal predicted to be invisible is
    exactly the case an anchor from a companion multiplet should settle — and
    until that exists, hiding signals on a proton-ratio estimate removes
    evidence the analyst never gets to see.
  * **Overlap** — which metabolites lay claim to each detected peak, counting
    every line of a multiplet and not only its anchor.

Both were methods on the results window. Neither touches a widget.
"""
from __future__ import annotations

import numpy as np

# ppm tolerance for tying a predicted line to a detected peak when building the
# overlap map. Tighter than the annotation tolerance on purpose: this asks
# "is this the same peak", not "could this be the same signal".
OVERLAP_TOL_PPM = 0.015

# Where the noise is estimated from when the spectrum reaches that far.
NOISE_TAIL_PPM = 9.5


def proton_count(entry, default: float = 1.0) -> float:
    """Proton count of a DB signal, fractional included.

    BMRB splits a group's protons across the signals it is observed in, so
    values like 0.5 and 0.75 are normal and meaningful. Reading them with
    ``int()`` turns 0.5 into 0, and an expected SNR computed from a zero proton
    ratio is zero — which marked the signal as below the detection limit and hid
    it from the analyst. Measured on a real project: 86 of 114 fractional-nH
    signals were hidden that way, with an expected SNR of exactly 0.00.
    """
    try:
        value = float(entry.get('protons', default) or default)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def estimate_noise(ppm_arr, intensity_arr) -> float:
    """Noise level from the far-upfield tail, or a low percentile as fallback."""
    tail_mask = ppm_arr > NOISE_TAIL_PPM
    if tail_mask.sum() < 20:
        noise = float(np.percentile(np.abs(intensity_arr), 5))
    else:
        tail = intensity_arr[tail_mask]
        noise = float(np.median(np.abs(tail - np.median(tail)))) * 1.4826
    return noise if noise > 0 else 1.0


def mark_detectability(metabolites, ppm_arr, intensity_arr) -> float:
    """Annotate every matched peak and DB peak with its expected SNR.

    The anchor is the metabolite's strongest signal weighted by proton count;
    every other signal's expected SNR follows from the proton ratio. Returns the
    noise level used.

    Nothing is hidden or excluded on the strength of this estimate: it is
    reported, and the analyst decides.
    """
    noise = estimate_noise(ppm_arr, intensity_arr)

    for met in metabolites:
        matched = met.get('matched_peaks', [])
        for mp in matched:
            ppk = mp.get('peak_ppm', mp.get('db_ppm', 0))
            idx = int(np.argmin(np.abs(ppm_arr - ppk)))
            mp['obs_intensity'] = float(intensity_arr[idx])
            mp['obs_snr'] = mp['obs_intensity'] / noise

        if not matched:
            continue

        anchor = max(matched,
                     key=lambda p: p.get('obs_snr', 0) * proton_count(p))
        anchor_snr = anchor.get('obs_snr', 0)
        anchor_protons = proton_count(anchor)

        for mp in matched:
            expected = (anchor_snr * (proton_count(mp) / anchor_protons)
                        if anchor_protons > 0 else anchor_snr)
            mp['expected_snr'] = expected

        # DB peaks too, matched or not — the estimate is informative for an
        # unmatched signal as well.
        for dp in met.get('db_peaks', []):
            dp['expected_snr'] = (
                anchor_snr * (proton_count(dp) / anchor_protons)
                if anchor_protons > 0 else anchor_snr)

    return noise


def build_overlap_map(metabolites, peaks, *, tol: float = OVERLAP_TOL_PPM):
    """Return ``{peak_rank: [metabolite names]}`` — who claims each peak.

    Every line of a multiplet is registered, not just its anchor, so a doublet
    belonging to X still counts as a competitor for Y when Y overlaps only one
    of its legs.
    """
    rank_by_ppm = {float(pk.get('ppm', 0)): pk.get('rank')
                   for pk in (peaks or []) if pk.get('rank') is not None}
    ppms = sorted(rank_by_ppm)

    def _rank_near(value):
        best_d, best_r = tol, None
        for p in ppms:
            d = abs(p - value)
            if d < best_d:
                best_d, best_r = d, rank_by_ppm[p]
            elif p > value + tol:
                break
        return best_r

    claims: dict = {}

    def _claim(rank, name):
        if rank is None:
            return
        holders = claims.setdefault(rank, [])
        if name not in holders:
            holders.append(name)

    for met in metabolites:
        name = met['name']
        for mp in met.get('matched_peaks', []):
            rank = mp.get('peak_rank')
            _claim(rank, name)
            lines = mp.get('predicted_lines') or []
            if len(lines) > 1:
                for line in lines:
                    other = _rank_near(float(line.get('ppm', 0)))
                    if other != rank:
                        _claim(other, name)
    return claims
