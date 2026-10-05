# processing/quant_concentration.py
"""From an integrated area to a concentration, per sample.

Three steps, kept apart because they answer different questions and the old
engine ran them as one expression:

  area / nH        how much signal per proton — the quantity that is
                   comparable between two different resonances at all
  ÷ the standard   what that is worth in mM, read against something of known
                   concentration measured the same way
  × dilution       what it was before the sample was prepared

The proton count is the database signal's own, fraction and all. Flooring it at
1.0 under-reports every analyte whose protons are shared across signals in
direct proportion — a quarter of the reference database's quantifiable signals,
and every one of D-glucose's.
"""
from __future__ import annotations

import numpy as np

MODE_IS = 'IS'
MODE_IS_PQN = 'IS+PQN'
MODE_NONE = 'none'


def signal_nh(protons) -> float:
    """The divisor for one signal: its own proton count, fractional included.

    BMRB stores a fractional per-signal nH whenever a group's protons are
    observed across several signals — D-fructose H13-H16 is 4 H over 5 signals,
    0.8 each, and they sum back to the molecule's total. Only a missing or
    non-positive value falls back to 1.0, and that fallback is an admission
    that the database said nothing, not a floor.
    """
    try:
        nh = float(protons)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(nh) or nh <= 0:
        return 1.0
    return nh


def per_proton(areas: dict, nh: float) -> dict:
    """``{sample: area/nH}`` — the only form in which two resonances compare."""
    n = signal_nh(nh)
    return {int(s): float(a) / n for s, a in (areas or {}).items()}


def standard_per_proton(is_areas: dict, is_nh: float,
                        *, external_samples=None) -> dict | None:
    """The reference's area per proton, as each sample should be divided by.

    With the standard in every sample, each sample is divided by its own: the
    dilution it shares with the analyte then cancels in the ratio.

    With the standard injected separately — ``external_samples`` naming those
    injections — every sample is divided by the same mean. Nothing cancels
    then, which is why the caller must apply the full dilution afterwards.

    Returns None when the standard has no usable area.
    """
    per = per_proton(is_areas, is_nh)
    if not per:
        return None
    if external_samples is None:
        return {s: v for s, v in per.items() if v > 0} or None
    vals = [per[s] for s in external_samples if s in per and per[s] > 0]
    if not vals:
        return None
    fixed = float(np.mean(vals))
    return {s: fixed for s in per}


def concentrations(areas: dict, nh: float, *, standard=None,
                   standard_conc: float = 0.0, external=False,
                   dilution=None, correction=None,
                   proton_correction=True) -> dict:
    """Concentration per sample, in the standard's own units.

    ``proton_correction`` only has a say when there is no standard. Against a
    standard the division by nH is not optional — the ratio of two areas is
    only a ratio of concentrations once both are per proton — so the flag is
    ignored there rather than allowed to produce a number that looks like a
    concentration and is not one. Without a standard the output is an area, and
    whether it is per proton is the analyst's choice.

    With a standard::

        conc[s] = (area[s]/nH) / standard[s] × standard_conc × factor[s]

    where ``factor`` is the correction alone when the standard sits in the
    sample — its dilution is the analyte's dilution and cancels — and the full
    dilution when the standard was injected separately, where nothing cancels.
    Getting that backwards scales every concentration in the study by the
    dilution, silently and uniformly, which is exactly the kind of error a
    plausible-looking number hides.

    Without a standard the area per proton is returned scaled by the dilution:
    not a concentration, and the caller must not label it one.

    A sample the standard could not be measured in yields NaN rather than a
    number, and a NaN is not a zero.
    """
    if standard is None and not proton_correction:
        per = {int(s): float(a) for s, a in (areas or {}).items()}
    else:
        per = per_proton(areas, nh)
    dil = dict(dilution or {})
    corr = dict(correction or {})
    out = {}
    for s, v in per.items():
        if standard is None:
            f = float(dil.get(s, 1.0))
            out[s] = v * f
            continue
        ref = standard.get(s)
        if not ref or ref <= 0 or not np.isfinite(ref):
            out[s] = float('nan')
            continue
        f = float(dil.get(s, 1.0)) if external else float(corr.get(s, 1.0))
        out[s] = v / ref * float(standard_conc) * f
    return out


def pqn_factor(areas: dict, reference: dict, *, min_signals: int = 3):
    """The sample's dilution factor, as the median quotient against a reference.

    Probabilistic quotient normalisation: the most common ratio between this
    sample's signals and the reference sample's is what the whole sample is
    diluted by, since a real biological change moves some signals and a
    dilution moves all of them.

    Returns 1.0 — no correction — when fewer than ``min_signals`` signals can be
    compared. A median over two quotients is not a mode, and applying it would
    dress a coincidence as a dilution.
    """
    q = [v / reference[k] for k, v in (areas or {}).items()
         if k in (reference or {}) and reference[k] > 0 and v > 0
         and np.isfinite(v) and np.isfinite(reference[k])]
    if len(q) < int(min_signals):
        return 1.0
    f = float(np.median(q))
    return f if f > 0 and np.isfinite(f) else 1.0


def pqn_reference(area_by_signal_and_sample: dict, *, exclude=()) -> dict:
    """The reference profile PQN measures each sample against.

    The median sample, signal by signal — not a chosen sample, which would make
    every result depend on which one the analyst happened to run first.
    ``exclude`` are samples that are not part of the study's biology.
    """
    skip = {int(s) for s in exclude or ()}
    ref = {}
    for sig, per_sample in (area_by_signal_and_sample or {}).items():
        vals = [v for s, v in per_sample.items()
                if int(s) not in skip
                and v is not None and np.isfinite(v) and v > 0]
        if vals:
            ref[sig] = float(np.median(vals))
    return ref


def apply_pqn(area_by_signal_and_sample: dict, *, min_signals: int = 3,
              exclude=()):
    """Divide every sample by its own dilution factor.

    Returns ``(corrected, factors)``. The factors travel with the result: a
    concentration corrected by an unseen number cannot be checked.

    ``exclude`` are tubes that are not diluted samples -- an external
    standard's tube holds the standard and little else. Against the study's
    median profile its quotient is the emptiness of the tube (0.004 on a
    cell study), and dividing by it multiplied everything read there two
    hundred times: 0.51 mM of fucose in a tube with none. They keep a
    factor of 1 and stay out of the reference.
    """
    skip = {int(s) for s in exclude or ()}
    ref = pqn_reference(area_by_signal_and_sample, exclude=skip)
    samples = {s for per in area_by_signal_and_sample.values() for s in per}
    factors = {}
    for s in samples:
        if int(s) in skip:
            factors[s] = 1.0
            continue
        this = {sig: per[s] for sig, per in area_by_signal_and_sample.items()
                if s in per and per[s] is not None}
        factors[s] = pqn_factor(this, ref, min_signals=min_signals)
    out = {sig: {s: (v / factors.get(s, 1.0)) for s, v in per.items()}
           for sig, per in area_by_signal_and_sample.items()}
    return out, factors


# How far apart the groups' mean PQN factors may stand before the analyst is
# told that PQN is removing a difference between them. The factor reads how
# much of the sample there is; a group whose factors stand 20% from another's
# is a group the normalisation has rescaled.
PQN_GROUP_WARN = 0.20


def pqn_group_divergence(factors: dict, groups: dict, *,
                         threshold: float = PQN_GROUP_WARN) -> dict | None:
    """The mean PQN factor of every group, and whether they stand apart.

    PQN divides each sample by how concentrated it is against the median
    sample, so it answers a question about composition: a difference that
    reaches every metabolite alike -- a group with fewer cells -- is read as
    dilution and taken out (Dieterle et al. 2006, Anal Chem 78:4281). On the
    validation bench a global x0.6 on one group was estimated at 3% and
    removed: the ratio of an unchanged analyte read 1.0 where its amount had
    fallen to 0.6 (ADR-062). A group-wise gap in the factors is the sign that
    this is happening, and it is the analyst's to judge.

    ``factors`` is ``{sample: factor}``, ``groups`` ``{group: [samples]}``
    (blanks are not a group). ``{'by_group': {group: mean}, 'ratio': highest
    over lowest, 'diverges': bool}``; None with fewer than two groups holding
    a factor.
    """
    by_group = {}
    for g, idxs in (groups or {}).items():
        vals = [float(factors[i]) for i in idxs
                if i in factors and factors[i] is not None
                and np.isfinite(factors[i]) and factors[i] > 0]
        if vals:
            by_group[str(g)] = float(np.mean(vals))
    if len(by_group) < 2:
        return None
    ratio = max(by_group.values()) / min(by_group.values())
    return {'by_group': by_group, 'ratio': float(ratio),
            'diverges': bool(ratio - 1.0 > float(threshold))}


def pqn_group_divergence_of_rows(rows, groups, *,
                                 threshold: float = PQN_GROUP_WARN):
    """``pqn_group_divergence`` from the factors the quantified rows carry
    (every row holds the same ones), or None when PQN was not applied."""
    for r in rows or ():
        f = r.get('_pqn_factors')
        if isinstance(f, dict) and f:
            return pqn_group_divergence(
                {int(k): v for k, v in f.items()}, groups, threshold=threshold)
    return None


def pqn_group_note(div) -> str:
    """The sentence the application and the report give for a divergence."""
    by = ', '.join(f'{g} {v:.2f}' for g, v in div['by_group'].items())
    return (f"PQN is taking a difference between groups out of the data: the "
            f"mean factor by group is {by} (a ratio of {div['ratio']:.2f}). "
            "The concentrations are composition relative to the median "
            "sample: a change that reaches every metabolite alike, such as "
            "fewer cells in one group, is read as dilution and is not in "
            "them. Check that this is what the study means (Dieterle 2006).")
