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

from processing.baseline_model import broad_removed_fraction
from processing.quant_cascade import (
    MAX_SIGNAL_DISAGREEMENT, MIN_FREE_TO_QUANTIFY, build_columns,
    cascade_missed, exclusion_weight, fit_curve_record,
    SHIFT_REACH_LW as CASCADE_FINE_REACH_LW,
    slide_step as cascade_slide_step)
from processing.quant_concentration import (
    apply_pqn, concentrations, signal_nh, standard_per_proton)
from processing.quant_deconvolution import (
    MIN_CLAIM_FRACTION, claimants_in_window, fraction_in_window,
    solve_windows, subtract_neighbours)
from processing.quant_joint_fit import (
    PART_SHIFT_REACH_LW, PART_SHIFT_STEPS, joint_measure)
from processing.quant_detection_limit import (
    INTERFERENCE_NOTE,
    BELOW_LOD, BELOW_LOQ, EXCLUDED_REGION, K_LOD, STRONG_ABOVE_LOQ,
    UNRESOLVED, censor, censor_text, detection_limits, drift_offsets,
    integration_noise, local_noise_sigma, predicted_window, strength_band,
    window_is_excluded)
from processing.quant_integration import (
    MIN_TRUSTED_CAPTURE, cluster_baseline,
    integrate_across_samples, integrate_sample, integrate_signal,
    lorentzian_captured_fraction, multiplet_window, pascal_weights,
    pattern_lines, refine_apex_ppm)
from processing.quant_shape import measure_shape
from processing.quant_shared import (
    signals_corroborate,
    REMAINDER_NOTE, RESIDUAL_ACCEPT, RESIDUAL_EXPLAINED,
    claim_listed_signals, fingerprint_holds,
    resolve_free_apices, resolve_listed_intruders, resolve_residuals,
    resolve_shared_windows)
from processing.quant_snr import (
    best_group, group_evidence, noise_level, representative_snr, window_snr)
from processing.quantification import (
    AMOUNT_CONTRADICTED, CASCADE_CROWDED, FIT_MISPLACED, LOST_COMPETITION,
    NO_MEASURABLE_WINDOW,
    OFF_THE_STUDY, PARTIAL_MULTIPLET, REJECTION_TEXT, SHAPE_UNIDENTIFIED,
    SHAPES_OFF, SINGLET_UNSUPPORTED, WINDOW_TAKEN, elect, is_competitor)
from processing.annotation_scoring import (
    DJ_GOLD_HZ_DEFAULT, PATTERN_MULT, PPM_BAND_PPM, dj_band, in_study as _in_study,
    is_singlet_like, normalize_mult, ppm_band, study_drift)

# The status dots the results table shows. Confidence is about how much of the
# expected evidence this signal actually carried, not about how large the
# number is: a big concentration measured on one unverified line is not a
# confident one.
STATUS_STRONG = '●●●'
STATUS_FAIR = '●●○'
STATUS_WEAK = '●○○'
STATUS_NONE = 'n.q.'

# Every censored row reads as weak confidence in the Conf column, and its
# reason says which kind of weakness. The column answers one question -- how
# much is this worth -- and putting a second vocabulary in it ("< LOQ",
# "overlap?") made the reader translate between two scales to compare two rows.
# The bound itself still travels, in the concentration column and the reason.
STATUS_CENSORED = {
    BELOW_LOD: STATUS_WEAK,
    BELOW_LOQ: STATUS_WEAK,
    UNRESOLVED: STATUS_WEAK,
    EXCLUDED_REGION: STATUS_WEAK,
}


def _cv_percent(values):
    v = [float(x) for x in (values or ()) if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    if m <= 0:
        return float('nan')
    return float(np.std(v, ddof=1) / m * 100.0)


_DOTS = {3: STATUS_STRONG, 2: STATUS_FAIR, 1: STATUS_WEAK}


def _structural_evidence(signal, dj_gold_hz=None):
    """How much of the expected evidence this signal's SHAPE carried.

    Three: the multiplet was complete and its coupling agreed. Two:
    complete, coupling off but inside the gate. One: a multiplet whose
    lines were not all found. None for a singlet, which has no coupling
    to agree with and no lines to count.

    None and not one. The confidence is the worst of its axes, so an axis
    that cannot be measured must not be the worst: scoring a singlet one
    here meant no singlet could ever pass one dot however well the rest of
    its molecule agreed, and creatine -- whose other signal stands exactly
    where its amount says it should -- read as poorly as a peak nobody
    could account for. What a singlet is worth is decided by its identity
    (``_identity_evidence``), which is where the evidence for it actually
    is.

    "Agreed" is the study's own band (``dj_band``), the one the election and
    the annotation use, widened where the reference measured its coupling
    with an uncertainty of its own. It used to be a literal fifth of a
    hertz here, so the engine disagreed with the annotation about the same
    signal: inosine, matched at 0.25 Hz against a reference whose own J
    carries 0.19 Hz of uncertainty, was Good in the window and second-rate
    in the table. A reference known to a fifth of a hertz cannot judge a
    match to a fifth of one.
    """
    ev = signal.evidence or {}
    _mult = ev.get('multiplicity') or ev.get('type')
    if _mult and is_singlet_like(_mult):
        return None
    n_f, n_e = ev.get('n_found'), ev.get('n_expected')
    complete = (n_f is not None and n_e is not None and n_f >= n_e)
    dj = ev.get('delta_j_hz')
    if not complete or dj is None:
        return 1
    gold = float(dj_gold_hz if dj_gold_hz else DJ_GOLD_HZ_DEFAULT)
    return 3 if dj_band(dj, gold, ref_sigma_hz=ev.get('j_sigma_hz')) == 0 else 2


def _confirms(mp, band_ppm=None) -> bool:
    """Whether a matched signal confirms the analyte it was matched to.

    Annotation's Good mark says the multiplet is complete and its coupling
    agrees. That is a statement about the SHAPE of the peak, and a shape
    can agree with a reference standing far from it: the peak belongs to
    something else and the reference is merely the nearest listing with
    that shape. So a Good mark confirms only where the peak also stands
    where this study puts things -- inside one band of the study's own
    drift, the same band the election orders candidates by.

    Measured on the reference bench, over every confirmed signal of every
    quantified analyte: the 19 belonging to analytes really in the study
    sat a median 1.3 mppm from the drift, the 6 belonging to the rest a
    median 27.8. Propionic acid -- the one false positive the confidence
    reached three dots on -- held the two worst of the twenty-five: its
    quartet 43 mppm off, on the peak that is glutamine's quartet, and its
    triplet 57 mppm off, in valine's region. Both are inside the 50 mppm
    match tolerance, which is why they matched at all; neither is inside
    the study, which drifted 8 mppm.
    """
    return (((mp.get('annotation_quality') or mp.get('quality')) == 'good')
            and _in_study(mp, band_ppm))


def _matched_signal_peaks(info, band_ppm=None) -> set:
    """The peaks this analyte's listed signals were matched to, inside the
    study. Used to ask whether a signal is the analyte's own."""
    sig = info.get('signal')
    payload = sig.payload if sig is not None and isinstance(sig.payload, dict) else {}
    met = payload.get('met') or {}
    return {round(float(mp['peak_ppm']), 4)
            for mp in met.get('matched_peaks') or ()
            if mp.get('peak_ppm') is not None and _in_study(mp, band_ppm)}


def stands_in_study(met, band_ppm=None) -> bool:
    """Whether any signal this analyte matched stands where the study does.

    The match tolerance (``dppm_gold``, 0.05 ppm) is wider than a study's
    drift on purpose, so that a study which moves can still be matched. It
    was read as though everything inside it were the same signal. Measured
    on the reference study, over every quantified analyte, by the share of
    its listed signals matched INSIDE one band of the common drift: the 16
    really in the mixture ran 0.33 to 1.00, median 0.86, and not one was at
    zero; of the 16 strangers, seven were at zero -- not a single listed
    signal of theirs anywhere near where the study puts things, though all
    of them had matched a peak.

    An analyte that matched nothing at all is not judged here: this asks
    where its matches landed, and no match is not an answer.
    """
    seen = False
    for mp in (met or {}).get('matched_peaks') or ():
        if mp.get('db_ppm') is None:
            continue
        seen = True
        if _in_study(mp, band_ppm):
            return True
    return not seen


def _off_study_detail(met) -> str:
    """How far the analyte's matches landed, so the row can be checked."""
    offs = [abs(float(mp['ppm_off_drift']))
            for mp in (met or {}).get('matched_peaks') or ()
            if mp.get('ppm_off_drift') is not None]
    if not offs:
        return ''
    return (f' The nearest of its {len(offs)} matched signal(s) sits '
            f'{1000 * min(offs):.0f} mppm from the drift this study measured.')


def study_drift_residual(metabolites, band_ppm=None) -> float | None:
    """How far this study's CONFIRMED signals sit from their common drift.

    The drift itself is subtracted already (``ppm_off_drift``); what is left
    is how tightly the study holds together -- shimming, referencing, the
    pH and temperature spread between samples. Measured only on signals
    annotation confirmed: a complete multiplet whose coupling agrees is
    evidence of SHAPE, which a singlet can never have and a position cannot
    fake, so the bar is not calibrated on the thing it is about to judge.
    The median, because two of the twelve analytes contributing confirmed
    signals to the reference study were strangers and a median does not
    care.

    ``None`` where too few signals were confirmed to say anything, and the
    caller then asks nothing of position.

    On the reference study this comes out at 1.5 mppm over 20 signals.
    """
    offs = []
    for met in metabolites or ():
        for mp in (met or {}).get('matched_peaks') or ():
            if _confirms(mp, band_ppm) and mp.get('ppm_off_drift') is not None:
                offs.append(abs(float(mp['ppm_off_drift'])))
    if len(offs) < DRIFT_RESIDUAL_MIN_SIGNALS:
        return None
    return float(np.median(offs))


# The fewest Good matches a study needs before its own position scatter is
# allowed to widen the band the election, the confirmation of a signal and
# the position gates judge by.
BAND_MIN_GOOD_SIGNALS = 20
# What share of the Good matches the band must hold, and the widest it may
# become (the annotation's own match tolerance, 0.05 ppm).
BAND_SHARE = 75.0
BAND_MAX_PPM = 0.05


def study_position_band(metabolites, floor=PPM_BAND_PPM) -> float:
    """The width of "where this study puts things", measured on the study.

    ``PPM_BAND_PPM`` (20 mppm) was drawn on studies whose compounds move
    together with one common drift: on the reference bench the confirmed
    signals of the analytes really in the tube sat a median 1.3 mppm from
    that drift and the strangers' 27.8. In the NMR_Murilo cell extract each
    compound has its own offset from its reference (pH, ionic strength,
    matrix): 55% of the Good matches stood further than 20 mppm from the
    common drift (median 22, upper quartile 37) -- and the ones annotation
    called bad stood about as far (median 27). There the band cut real
    analytes out of the election, out of the confirmation of their signals
    and out of the table.

    The band is the ``BAND_SHARE`` percentile of ``|ppm_off_drift|`` over the
    Good matches, never narrower than ``floor`` and never wider than the
    annotation's match tolerance. ``floor`` where there are fewer than
    ``BAND_MIN_GOOD_SIGNALS`` Good matches: a tight study stays as it was.
    """
    offs = [abs(float(mp['ppm_off_drift']))
            for met in metabolites or ()
            for mp in (met or {}).get('matched_peaks') or ()
            if (mp.get('annotation_quality') or mp.get('quality')) == 'good'
            and mp.get('ppm_off_drift') is not None]
    if len(offs) < BAND_MIN_GOOD_SIGNALS:
        return float(floor)
    return float(min(max(np.percentile(offs, BAND_SHARE), floor), BAND_MAX_PPM))


def study_position_scatter(metabolites, band_ppm=None) -> float | None:
    """How tightly the analytes really in this study sit on its drift: the
    LOWER QUARTILE of ``|ppm_off_drift|`` over the signals annotation
    confirmed. ``None`` with fewer than DRIFT_RESIDUAL_MIN_SIGNALS.

    A quartile and not the median (``study_drift_residual``), because in a
    crowded spectrum the confirmed signals are a mixture: on the reference
    bench filled to 100 analytes nearly half of them belonged to strangers
    -- a complete multiplet with its coupling inside the gate is easy to
    find among a hundred compounds' lines -- and those sat 5 to 20 mppm off
    the drift while the real analytes' clustered under 1 mppm. The median
    came out at 4 to 6 mppm, dragged up by exactly what it was meant to
    judge; the lower quartile, 0.7 to 1.0, holds with up to half the
    signals contaminated.
    """
    offs = []
    for met in metabolites or ():
        for mp in (met or {}).get('matched_peaks') or ():
            if _confirms(mp, band_ppm) and mp.get('ppm_off_drift') is not None:
                offs.append(abs(float(mp['ppm_off_drift'])))
    if len(offs) < DRIFT_RESIDUAL_MIN_SIGNALS:
        return None
    return float(np.percentile(offs, 25))


def _fit_step(linewidth_ppm) -> float:
    """The joint fit's grid step when it slides one signal on its own."""
    reach = PART_SHIFT_REACH_LW * float(linewidth_ppm or 0.0)
    return 2.0 * reach / max(PART_SHIFT_STEPS - 1, 1)


def _fitted_offset(info) -> tuple[float, float] | None:
    """Where the joint fit, with the neighbours fitted beside it, put the
    analyte's signals: ``(|offset| of the one the amount was measured on,
    median |offset| of its others)`` -- ``None`` when the fit did not place
    the measured one. With no other signal the median is 0."""
    jf = info.get('joint_fit') or {}
    exp = (info['signal'].evidence or {}).get('expected_ppm')
    parts = list(jf.get('signal_offsets') or ())
    if exp is None or not parts:
        return None
    k = min(range(len(parts)), key=lambda i: abs(parts[i][0] - float(exp)))
    if abs(parts[k][0] - float(exp)) > 0.01:
        return None
    others = [abs(float(o)) for i, (_c, o) in enumerate(parts) if i != k]
    return (abs(float(parts[k][1])),
            float(np.median(others)) if others else 0.0)


def _position_reason(info) -> list:
    """Said of a row whose measured signal stands off the study."""
    pc = info.get('position_cap')
    if not pc:
        return []
    off, bar = pc
    return [f"The signal this amount was measured on stands "
            f"{1000 * off:.1f} mppm from this study's drift, beyond the "
            f"{1000 * bar:.1f} mppm the analytes confirmed here sit within: "
            f"most likely another compound's peak. One dot, whatever the "
            f"rest of its evidence."]


def _identity_of(info, band_ppm=None, drift_residual=None) -> dict:
    """The identity evidence of a measured analyte, from what the run
    already holds: the signals annotation confirmed (Good, and standing
    where the study puts things -- see ``_confirms``), how much of the
    reference's listing was matched at all, and what the rest of that
    listing does at the amount just measured."""
    sig = info['signal']
    payload = sig.payload if isinstance(sig.payload, dict) else {}
    met = payload.get('met') or {}
    listed = len(met.get('db_peaks') or ()) or 1
    by_db: dict = {}
    for mp in met.get('matched_peaks') or ():
        if mp.get('db_ppm') is None:
            continue
        by_db.setdefault(round(float(mp['db_ppm']), 3), []).append(mp)
    confirmed = sum(1 for group in by_db.values()
                    if any(_confirms(mp, band_ppm) for mp in group))
    # The signal this number was measured on counts for itself when its own
    # multiplet was complete, its coupling agreed and it stands where the
    # study puts things, whether or not the annotation window was ever
    # opened to mark it: the engine must be able to say what it knows
    # without depending on a mark made elsewhere. Hence _in_study and not
    # _confirms -- the shape half of the Good mark is _structural_evidence,
    # measured here; only the drift half is left to check.
    own = sig.evidence or {}
    if _structural_evidence(sig) == 3 and _in_study(own, band_ppm) and not any(
            _confirms(mp, band_ppm)
            and own.get('db_ppm') is not None
            and abs(float(mp.get('db_ppm', 0.0)) - float(own['db_ppm'])) < 0.005
            for group in by_db.values() for mp in group):
        confirmed += 1
    # A singlet has no shape to confirm. What it does have is a position,
    # and an analyte really in the mixture moves WITH the study: the drift
    # is pH, temperature and referencing acting on every resonance at once,
    # while a stranger whose table position happens to fall near a peak
    # sits a random distance from it. Measured over the analytes elected on
    # a singlet: the five really in the reference study sat 0.1 to 1.0 mppm
    # from the common drift, the seven strangers 3.1 to 18.4.
    on_drift = False
    if drift_residual is not None and is_singlet_like(
            own.get('multiplicity') or own.get('type') or ''):
        _off = own.get('ppm_off_drift')
        on_drift = (_off is not None
                    and abs(float(_off)) <= float(drift_residual))
    verdicts = info.get('fingerprint_at_amount') or ()
    present = sum(1 for v in verdicts if v[3] == 'present')
    contradicted = sum(1 for v in verdicts if str(v[3]).startswith('contradicted'))
    # Covered by the matches that landed inside the study. A listed signal
    # whose only match sits further from the drift than the band was not
    # found here: counting it made propionic acid -- both its signals
    # matched, 43 and 57 mppm off, onto peaks belonging to glutamine and
    # to the valine region -- read as a fully covered listing.
    found = sum(1 for group in by_db.values()
                if any(_in_study(mp, band_ppm) for mp in group))
    covered = found / listed >= COVERAGE_CORROBORATES
    return {'confirmed': confirmed, 'contradicted': contradicted,
            'corroborated': bool(covered and present), 'listed': listed,
            'on_drift': on_drift,
            # No analyte's signals are independent peaks: they stand at
            # the distances the reference lists and in the ratio of their
            # proton counts. Counted only on a listing otherwise covered.
            'signals_linked': bool(covered and info.get('signals_linked')),
            'residual': info.get('by_residual')}


def _identity_evidence(*, confirmed=0, contradicted=0, corroborated=False,
                       listed=1, cascade=None, residual=None,
                       signals_linked=False, on_drift=False) -> tuple[int, str]:
    """How much the study itself says this is the analyte it says it is.

    Measured on the reference bench, over every analyte the run quantified,
    against what was actually put in it:

    * a signal **confirmed** -- its multiplet complete and its coupling
      inside the gate, which is what annotation marks Good -- and nothing
      of the rest of its listing missing at the measured amount: 7 of 7
      were real, 0 false. Nothing else reached it: every one of the eight
      false positives had no confirmed signal at all.
    * covered but unconfirmed -- three quarters of its listed signals
      matched and the rest standing where this amount says they must: 3
      of 4 real.
    * anything else: 5 of 12 real, which is what one dot has to mean.

    A signal of its own listing missing at the measured amount puts it at
    one dot whatever else it has: the amount says the signal should be
    there, and it is not.

    An analyte identified on a free apex by its shape alone is one: no
    line of it was confirmed. One taken from the remainder of another
    analyte's window is never more than two, whatever its own signals
    say elsewhere -- the amount is what that window had left over.

    An analyte the cascade brought in is never more than two: its lines
    were never confirmed one by one. Two when its own signals agree on
    the amount to within CASCADE_AGREEMENT_GOOD, one otherwise.

    A singlet standing on the study's own drift is two. It has no shape to
    confirm, but an analyte really in the mixture moves WITH the study --
    the drift is pH, temperature and referencing acting on every resonance
    at once -- while a stranger whose table position happens to fall near a
    peak sits a random distance from it. Over the analytes elected on a
    singlet in the reference study, the five real ones sat 0.1 to 1.0 mppm
    from the common drift and the seven strangers 3.1 to 18.4; the bar,
    which is the study's own residual over its confirmed signals, came out
    at 1.5 and separated them completely. Two and not three: position is
    not identity, and any compound listing a singlet there would look the
    same.

    An analyte's own signals agreeing on one amount reaches three too.
    The integral of a multiplet is nH x (area per proton) whatever its
    multiplicity, so a singlet, a doublet and an m* can all be asked how
    much of the analyte is there, and a molecule with nothing but singlets
    -- five of the eighteen here, nothing in them having a neighbour to
    couple to -- is no longer barred from the top for a fact about its
    chemistry. On peaks nothing else quantified holds and on a listing
    otherwise covered: 8 of 8 promoted on the reference study were real,
    and no stranger reached it. Asking it of the singlets alone reached 2.
    """
    if residual == 'spectrum':
        return 1, ('identified on a free apex by the shape of its signal '
                   'alone: no line of it was confirmed')
    if cascade is not None:
        if cascade.get('wide'):
            # slid further than the study's own drift can account for: a
            # column slid that far agrees with some peak by chance, and its
            # position is no evidence. The number is shown; one dot
            return 1, ('the cascade slid its column well away from where the '
                       'reference lists it: its position is unconfirmed')
        dis = float(cascade.get('disagreement') or float('nan'))
        if np.isfinite(dis) and dis <= CASCADE_AGREEMENT_GOOD:
            return 2, f'the cascade placed it and its own signals agree to {100 * dis:.0f}%'
        # Placed on the study's drift (ADR-042's reading of position: what
        # is in the mixture moves with the study). On the crowded tubes
        # (2026-09-29, seed 0) every stranger the cascade quantified sat two
        # or three steps of its slide off the drift; the reals held at one
        # dot at zero or one -- singlets with nothing to agree with
        # (acetate, glycine, gallic acid) and multiplets agreeing to 21-40%
        # (myo-inositol, 5-methyluridine, spermidine). The one real held off
        # the drift was acetyl phosphate, read 219-407% high. How far its
        # signals may disagree is already the bar on having a number at all.
        if cascade.get('placed'):
            if not np.isfinite(dis):
                return 2, ('the cascade placed its one signal on the study\'s '
                           'drift, within one step of its slide')
            return 2, (f'the cascade placed it on the study\'s drift; its '
                       f'own signals agree to {100 * dis:.0f}%')
        return 1, 'the cascade placed it and no second signal could check the amount'
    if contradicted:
        return 1, (f'{contradicted} signal(s) of its own listing are missing '
                   f'at this amount')
    if confirmed:
        if residual:
            return 2, (f'{confirmed} of its signals confirmed, but its '
                       f'amount came from what the window of {residual} '
                       f'had left over')
        return 3, (f'{confirmed} of its signals confirmed: the multiplet '
                   f'complete and the coupling inside the gate')
    if signals_linked and not residual:
        return 3, ('its own signals agree on one amount, standing where the '
                   'reference puts them and in the ratio of their protons, '
                   'on peaks nothing else quantified holds')
    if on_drift and not residual:
        return 2, ('a singlet, but standing no further from the drift this '
                   'study measured than the signals it confirmed by their '
                   'coupling: it moved with the study')
    if listed <= 1:
        return 1, 'the reference lists one signal for it: nothing corroborates it'
    if corroborated:
        return 2, ('no signal of it was confirmed, but its listing is covered '
                   'and stands at this amount')
    return 1, 'no signal of it was confirmed'


def _confidence(signal, strength=None, shared=False, *, identity=None,
                dj_gold_hz=None) -> tuple[int, str]:
    """What the result is worth: the worst of its identity, its shape and
    its size.

    Three independent things can spoil a number and none makes up for
    another. A textbook multiplet measured at the quantification limit is
    a confident identity and an imprecise quantity; a towering singlet
    with a rival on its peak is the reverse. Averaging them would let
    either hide the other, so the lowest stands.

    ``identity`` is ``_identity_evidence`` -- what the rest of the
    analyte's own listing says about the assignment. Without it the shape
    of the one signal measured is all that speaks, which is how eight
    false positives came to sit at the same dot count as eight real
    analytes on the reference bench.

    ``shared`` says the window is not this analyte's alone -- either the
    area was resolved out of one held with others, or it is a window that
    could not be split and so still holds somebody else's signal. Two
    dots either way, whatever the multiplet's shape: in the first case
    the number rests on another analyte's concentration being right, and
    in the second it is an upper bound wearing a concentration's clothes.
    This is about the amount, not the identity: an analyte whose identity
    is confirmed and whose window is shared keeps its identity level,
    which the row carries separately (``_identity_why``).

    ``strength`` is the band from ``strength_band`` -- 3 when the area
    clears the quantification limit by the agreed factor, 2 otherwise. It
    never reaches 1: an analyte whose peak was identified and whose
    contest it won is not made doubtful by being scarce, and being under
    the limit says the quantity is small, not that the assignment was
    wrong.
    """
    level = _structural_evidence(signal, dj_gold_hz)
    if level is None:                 # a singlet: its shape says nothing
        level = 3
    if identity is not None:
        level = min(level, int(identity))
    if strength is not None:
        level = min(level, int(strength))
    if shared:
        level = min(level, SHARED_MAX_CONFIDENCE)
    return level, _DOTS[level]


# A share resolved by subtraction is worth less than an area measured on a
# window nobody else is in, however clean the multiplet's shape. It rests on
# another analyte's concentration, and on the assumption that the claimants
# listed are all of them. Two dots, never three.
SHARED_MAX_CONFIDENCE = 2

# How many confirmed signals a study needs before its own drift residual
# means anything. Three is the fewest a median is not an accident of.
DRIFT_RESIDUAL_MIN_SIGNALS = 3

# How far past the study's position scatter (study_position_scatter) the
# signal an amount was measured on may stand before the row is held at one
# dot. End to end over three seeds of the reference bench:
#   100 analytes  false positives at >=2 dots 8.0 -> 0.7 per run, at three
#                 dots 3.0 -> 0.0; real analytes at >=2 dots 37.0 -> 30.0,
#                 the ones it took down measured 35% off in median
#   18 analytes   false positives at >=2 dots 1.0 -> 0.3; real analytes
#                 14.0 -> 10.7 -- taurine and glutathione among them, well
#                 measured, whose listed position is the reference's own
#                 error: position alone cannot tell them from a stranger.
# A floor of 2 mppm on the bar spared betaine on the sparse tube and let a
# stranger back to three dots on the crowded one; not kept.
#
# 4.5, not 3.0, since the samples reach quantification aligned
# (2026-09-30): the scatter is measured on the study's signals, and the
# drift between samples had been part of it -- 1.7 mppm unaligned, 1.1
# aligned on the 18-analyte GISSMO tube at 600 MHz. What the bar is for,
# the reference's own error, did not shrink with it. Three seeds,
# aligned: on the 100-analyte tube real at two dots or more 168 -> 170,
# three dots 31 -> 33, strangers at two dots 9 -> 9, at three none; the
# 18-analyte tube unchanged.
POSITION_SCATTER_K = 4.5

# A signal off the drift at its apex is released from the cap when the
# joint fit, with its neighbours fitted beside it, puts it within this many
# of its grid steps (_fit_step) -- an apex pulled by an overlapping line is
# not a stranger's -- and puts the analyte's OTHER signals, in median,
# within FITTED_OTHERS_STEPS: a real analyte fitted with its neighbours
# lands on its listing, a stranger lands one signal on somebody's line and
# the rest at the edge of the reach. The measured signal alone let two
# strangers through to two and three dots at 50 analytes (meso-erythritol,
# histidinol). One and a half: the grid is coarser than the scatter.
# Reference bench, three seeds: recall at two dots 56% -> 67% at 18
# analytes, 47% -> 53% at 50, 35% -> 40% at 100; strangers at two dots and
# at three unchanged (three: none).
FITTED_POSITION_STEPS = 1.5
FITTED_OTHERS_STEPS = 1.0

# Candidates annotation proposed and quantification did not report enter
# the joint fit as columns whose amounts are not reported (ASICS, Tardivel
# et al. 2017, fits the whole library, not only what it reports). On the
# reference bench filled to 100 analytes, three seeds: joint-fit bias
# +20% -> +5%, p90 135% -> 92%; at 50, p90 51% -> 48%; three dots still
# no stranger. The price: 0.6-1.0 more strangers per run at two dots, all
# cascade rows whose residual the better amounts left differently.
JOINT_FIT_CANDIDATES = True

# An analyte whose joint fit is refused (``joint_fit_refusal``) stays in the
# table as a candidate: not quantified, with the integral of its window as an
# upper bound (``_as_candidate``). Measured on the cell extract NMR_Murilo (20
# samples, 500 MHz) the references stand a median 16 mppm -- 4.5 linewidths --
# from the peaks, so most fits are refused. False: the row is dropped.
FIT_MISPLACED_KEEPS_INTEGRAL = True

# The share of its own window an analyte has to keep for the split to
# leave its number standing on its own. Above it, what the others took is
# a rounding of the analyte: an error of half in the neighbour moves the
# analyte by half of what it gave away.
SHARE_KEPT_ALONE = 0.90

# The share of a window's raw area the baseline model may remove before
# the row says so. A tenth is already enough to move a concentration by a
# tenth when it is the internal standard's window.
BROAD_REMOVED_NOTE = 0.10



# How closely an analyte the cascade brought in must have its own signals
# agree before the row is worth two dots rather than one.
#
# What distinguishes its true rows from its false ones on the reference
# study is not how well they agree but whether they could be checked at
# all. The four real ones agree to 3%, 6%, 11% and 17%; of the five
# false, two agree to 2% and 6% and the other three have no second signal
# to check against. A floor of two dots for everything the cascade finds
# -- which is the same as not asking -- takes those three unchecked ones
# up with the rest. At a fifth, inosine and leucine come up and nothing
# false comes with them.
#
# Not the same number as MAX_SIGNAL_DISAGREEMENT, which decides whether
# there is a number at all: one threshold should answer one question.
CASCADE_AGREEMENT_GOOD = 0.20

# The share of an analyte's listed signals that must have been matched
# before the rest of the listing standing at the measured amount counts
# as corroboration rather than as a crowded region obliging.
COVERAGE_CORROBORATES = 0.75

# Below this, the window holds less of an analyte's signal than it leaves
# outside, and dividing the share by it multiplies the error faster than it
# corrects the area. The claimant still counts in the balance -- its
# contribution to the window is real and gets subtracted -- it simply does not
# get a concentration out of a window that is mostly not looking at it.
MIN_OWN_SHARE = 0.5


def _signal_windows(candidates, peaks_by_rank, ppm, intensities, *,
                    included, local_baseline, align, blank_indices,
                    linewidth_ppm, noise_sigma=None, drift_ppm=0.0,
                    excluded_regions=()):
    """One window per candidate signal, and everybody whose lines reach it.

    Built over the candidates and not over the election's winners, because a
    winner has exactly one window here and a claimant of that window could then
    never be anchored anywhere -- every contest would end in two unknowns. The
    metabolite's other matched signals, clean and merely unused by the
    election, are what measure it.

    One window per signal, and not one per connected group of overlapping
    signals. Overlap is transitive and the groups are not: with A on lines
    1-2, B on 2-3 and C on 3-4, the group is one window with three claimants
    and two unknowns, although A and C do not touch. As three windows it is
    three equations, each with the claimants that actually reach it. On the
    reference study the grouped form left 49 of 70 contested windows with more
    unknowns than equations.

    Each claimant is weighed by the fraction of ITS OWN lines the window holds.
    The balance is

        A(W) = sum over claimants of  concentration * protons * fraction

    and the fraction is a term of it, never a test the window has to pass. An
    earlier version refused any window holding less than 75% of its lines --
    the threshold the capture correction uses to decide whether a correction is
    worth applying -- and that threw away 69 of 113 windows, among them the
    uncontested ones that would have anchored the rest. Foot-to-foot windows in
    a crowded region are narrow by construction: the feet are the valleys, and
    the valleys are close when the neighbours are.

    A window containing a detected peak that no candidate claims is refused.
    That peak is a term the balance does not have, and the analyte credited
    with the remainder would be credited with it.

    Where a metabolite's signal was predicted and NOT detected, that is a
    measurement too, and the one the balance was missing. On the reference
    study half the columns were exact multiples of another: an analyte with
    one signal, in one window, shared with an analyte in the same position.
    Nothing in those windows tells them apart. But if the first should also
    show a doublet at 7.5 ppm and there is nothing there, its concentration
    is bounded by the noise in that region and the second's is not -- and
    that is a row in which their columns differ.

    So an undetected signal becomes a window at its predicted position, its
    area whatever is there, and it enters the system as any other equation.
    Two conditions, because a database shift is not where a real sample puts
    the line. The metabolite must have at least one DETECTED signal in the
    study, so the analyte is known to be present and this is a question of
    how much rather than whether. And the window slides over the drift the
    study measured, keeping the position with the MOST area -- so if the
    line moved, the bound follows it, and it can only ever be too loose. A
    bound that is too loose costs nothing; one that is too tight puts an
    analyte at zero that is there.

    A detected peak nobody claims inside the slid window refuses it, as it
    does any window: that peak may be this very signal, drifted further than
    allowed for.
    """
    sigs, by_key, undetected = [], {}, []
    counts = {'candidates': 0, 'no_lines': 0, 'not_competitor': 0,
              'no_protons': 0, 'components': 0, 'contested': 0,
              'no_window': 0, 'foreign_peak': 0, 'thin_share': 0,
              'largest': 0, 'largest_lines': 0,
              'bounds': 0, 'bounds_excluded': 0, 'bounds_foreign': 0,
              'bounds_no_anchor': 0}
    for c in candidates or ():
        counts['candidates'] += 1
        if (c.evidence or {}).get('shape_fit'):
            # a shape holds no peak and shares no window; its region is
            # measured on its own terms (processing.quant_shape)
            counts['shape'] = counts.get('shape', 0) + 1
            continue
        ranks = sorted(int(r) for r in (c.line_ranks or ()))
        if not ranks and c.peak_rank is not None:
            ranks = [int(c.peak_rank)]
        if not ranks:
            counts['no_lines'] += 1
            pred = [float(v) for v in
                    ((c.evidence or {}).get('predicted_ppms') or ())
                    if v is not None]
            nh = signal_nh((c.evidence or {}).get('protons'))
            if pred and nh and np.isfinite(nh) and nh > 0:
                undetected.append({'met_id': str(c.met_id), 'nh': float(nh),
                                   'ppms': pred,
                                   'weights': pascal_weights(len(pred))})
            continue
        if not is_competitor(c.evidence):
            counts['not_competitor'] += 1
            continue
        nh = signal_nh((c.evidence or {}).get('protons'))
        if not nh or not np.isfinite(nh) or nh <= 0:
            counts['no_protons'] += 1
            continue
        key = (str(c.met_id), frozenset(ranks))
        if key in by_key:
            continue
        by_key[key] = c
        own = [peaks_by_rank[r] for r in ranks if r in peaks_by_rank]
        if len(own) != len(ranks):
            counts['no_window'] += 1
            continue
        sigs.append({'met_id': str(c.met_id), 'nh': float(nh),
                     'ranks': tuple(ranks), 'peaks': own,
                     'ppms': [float(pk.get('ppm', 0.0)) for pk in own],
                     'weights': pascal_weights(len(own))})

    hwhm = float(linewidth_ppm) / 2.0
    # Where the lines really are, to better than the grid. Every fraction below
    # is evaluated at the window's edges, and the trapezium the local baseline
    # removes is read off the lineshape there -- half a grid point of error in
    # an apex moves it more than anything else in the model.
    _arr = np.asarray(intensities, dtype=float)
    if _arr.ndim == 1:
        _arr = _arr.reshape(-1, 1)
    _cols = [c for c in (included or range(_arr.shape[1]))
             if 0 <= c < _arr.shape[1]]
    _ref = _arr[:, _cols].mean(axis=1) if _cols else _arr.mean(axis=1)
    _apex = {}
    for sig in sigs:
        sig['ppms'] = [_apex.setdefault(r, refine_apex_ppm(ppm, _ref, pk_ppm))
                       for r, pk_ppm in zip(sig['ranks'], sig['ppms'])]
    claimed_ranks = {r for sig in sigs for r in sig['ranks']}
    windows, reasons = [], {}
    for sig in sigs:
        win = multiplet_window(sig['peaks'])
        if win is None:
            counts['no_window'] += 1
            continue
        lo, hi = win
        # The bounds the integration will actually use. A window whose ends
        # fall between grid points is integrated between the points inside it,
        # and the local baseline's line is drawn through THOSE -- so a model
        # evaluated at the requested ends is describing a window half a point
        # wider at each side. Where the trapezium is large that half point is
        # the difference between agreeing with the integral to 0.06% and being
        # 7% under it.
        _in = np.asarray(ppm, dtype=float)
        _in = _in[(_in >= lo) & (_in <= hi)]
        if _in.size >= 2:
            lo, hi = float(_in.min()), float(_in.max())
        key = '%s@%s' % (sig['met_id'],
                         '_'.join(str(r) for r in sig['ranks']))
        counts['components'] += 1

        unclaimed = [r for r, pk in peaks_by_rank.items()
                     if int(r) not in claimed_ranks
                     and lo <= float(pk.get('ppm', -1e9)) <= hi]
        if unclaimed:
            counts['foreign_peak'] += 1
            reasons[key] = {
                'claimants': [(sig['met_id'], sig['nh'])],
                'reason': ('%d detected peak(s) inside the window are matched '
                           'to no analyte; the balance would be missing a term'
                           % len(unclaimed))}
            continue

        # Only the signals near enough to put anything here. The fraction says
        # how much, and below MIN_CLAIM_FRACTION it says almost nothing.
        contributions, fractions, geometry, own_ranks = [], {}, {}, {}
        for other in sigs:
            if min(other['ppms']) > hi + 40.0 * hwhm:
                continue
            if max(other['ppms']) < lo - 40.0 * hwhm:
                continue
            f = fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                   hwhm, local_baseline=local_baseline)
            if not f or f <= 0:
                continue
            contributions.append((other['met_id'], other['nh'], f))
            mid = other['met_id']
            # Two fractions, and they answer different questions. The one with
            # the baseline in it is the term of the balance -- what integrating
            # this window actually returns for this analyte. The bare
            # geometric one is whether the window is looking at the analyte at
            # all, which is what the floor on reporting asks: a straight line
            # drawn through a signal's own tails is a property of how it was
            # measured, not evidence that the signal is elsewhere.
            g = (fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                    hwhm)
                 if local_baseline else f)
            if f > fractions.get(mid, 0.0):
                fractions[mid] = float(f)
                geometry[mid] = float(g) if g and g > 0 else float(f)
                own_ranks[mid] = other['ranks']
        claimants = claimants_in_window(contributions)
        if not claimants:
            # Not even the signal the window was built on reaches the floor:
            # the window is so much narrower than the line that what the local
            # baseline removes is most of what it holds. Said, not skipped --
            # a window that vanishes without a word is how a step comes to look
            # like it is not there.
            counts['thin_share'] += 1
            reasons[key] = {
                'claimants': [(sig['met_id'], sig['nh'])],
                'reason': ('the window is too narrow for the line: after the '
                           'baseline it holds under %.0f%% of any claimant'
                           % (100.0 * MIN_CLAIM_FRACTION))}
            continue

        n_cl = len(claimants)
        if n_cl > 1:
            counts['contested'] += 1
        if n_cl > counts['largest']:
            counts['largest'] = n_cl
            counts['largest_lines'] = len(sig['ranks'])

        out = integrate_signal(sig['peaks'], ppm, intensities,
                               included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices)
        if not out or not out['areas']:
            counts['no_window'] += 1
            continue
        # One over the noise on this window's area, so a wide noisy window
        # does not outvote a narrow clean one in the joint solution.
        _p = np.asarray(ppm, dtype=float)
        n_pts = int(((_p >= lo) & (_p <= hi)).sum())
        dppm = float(abs(_p[1] - _p[0])) if _p.size > 1 else 0.0
        noise_area = (integration_noise(noise_sigma, dppm, n_pts)
                      if noise_sigma else None)
        windows.append({
            'key': key, 'ranks': sig['ranks'], 'owner': sig['met_id'],
            'weight': (1.0 / float(noise_area)
                       if noise_area and np.isfinite(noise_area)
                       and noise_area > 0 else 1.0),
            'claimants': claimants, 'areas': dict(out['areas']),
            'window': (lo, hi), 'fractions': fractions, 'geometry': geometry,
            'lines': own_ranks,
            'protons': {m: (nh_eff / fractions[m]) for m, nh_eff in claimants
                        if fractions.get(m)},
            'peaks': sig['peaks'],
            'shifts': out.get('shifts'),
            'blank_area': out.get('blank_area', float('nan'))})
    # ── The signals that should be there and are not ─────────────────────────
    present = {sig['met_id'] for sig in sigs}
    _in_all = np.asarray(ppm, dtype=float)
    for u in undetected:
        if u['met_id'] not in present:
            counts['bounds_no_anchor'] += 1
            continue
        win = predicted_window(u['ppms'], linewidth_ppm=linewidth_ppm)
        if win is None:
            continue
        lo0, hi0 = win
        # Slide over the drift and keep the position holding the most: a bound
        # may be loose, never tight.
        best, best_off, best_med = None, 0.0, float('-inf')
        for off in drift_offsets(drift_ppm, linewidth_ppm):
            lo, hi = lo0 + off, hi0 + off
            if window_is_excluded(lo, hi, excluded_regions):
                continue
            areas = integrate_across_samples(
                ppm, intensities, lo, hi, included=included,
                local_baseline=local_baseline)
            if not areas:
                continue
            med = float(np.median(list(areas.values())))
            if med > best_med:
                best, best_off, best_med = areas, off, med
        if best is None:
            counts['bounds_excluded'] += 1
            continue
        lo, hi = lo0 + best_off, hi0 + best_off
        _in = _in_all[(_in_all >= lo) & (_in_all <= hi)]
        if _in.size >= 2:
            lo, hi = float(_in.min()), float(_in.max())
        unclaimed = [r for r, pk in peaks_by_rank.items()
                     if int(r) not in claimed_ranks
                     and lo <= float(pk.get('ppm', -1e9)) <= hi]
        if unclaimed:
            counts['bounds_foreign'] += 1
            continue
        ppms = [float(v) + best_off for v in u['ppms']]
        contributions = [(u['met_id'], u['nh'],
                          fraction_in_window(ppms, u['weights'], lo, hi, hwhm,
                                             local_baseline=local_baseline))]
        for other in sigs:
            if min(other['ppms']) > hi + 40.0 * hwhm:
                continue
            if max(other['ppms']) < lo - 40.0 * hwhm:
                continue
            f = fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                   hwhm, local_baseline=local_baseline)
            if f and f > 0:
                contributions.append((other['met_id'], other['nh'], f))
        claimants = claimants_in_window(contributions)
        if not any(m == u['met_id'] for m, _ in claimants):
            continue
        n_pts = int(_in.size)
        dppm = (float(abs(_in_all[1] - _in_all[0])) if _in_all.size > 1
                else 0.0)
        noise_area = (integration_noise(noise_sigma, dppm, n_pts)
                      if noise_sigma else None)
        counts['bounds'] += 1
        windows.append({
            'key': '%s@predicted_%.3f' % (u['met_id'], float(np.mean(ppms))),
            'ranks': (), 'owner': None, 'bound': True,
            'weight': (1.0 / float(noise_area)
                       if noise_area and np.isfinite(noise_area)
                       and noise_area > 0 else 1.0),
            'claimants': claimants, 'areas': dict(best),
            'window': (lo, hi), 'fractions': {}, 'geometry': {},
            'lines': {}, 'protons': {}, 'peaks': [], 'shifts': None,
            'blank_area': float('nan')})
    return windows, by_key, reasons, counts


def _window_limits(ppm, intensities, lo, hi, *, included, noise_sigma,
                   linewidth_ppm, segments=None):
    """The SNR, the detection limits and the interference of one window.

    One noise for the whole of quantification. The SNR column used to
    reconstruct its own from the peak's recorded ratio, which disagreed with
    the limits beside it by a factor of two and, over a baseline offset,
    produced an SNR of 3149 for an analyte at 0.008 mM.

    That one noise is the BLANK's -- the scatter of a signal-free region --
    for both the ratio and the limits, which is what Currie's framework and
    the qNMR convention define them from. How crowded the window's
    surroundings are is measured too, as a ratio to that blank, and
    returned beside them: it is interference, it costs accuracy rather than
    detectability, and it is the analyst's to see. Putting it inside the
    limit instead marked taurine -- second most concentrated analyte of the
    reference study, group signal-to-noise 639 -- as too weak to quantify.

    Returns ``(snr, lod_area, loq_area, interference)``.
    """
    snr = window_snr(ppm, intensities, lo, hi, noise_sigma, included=included)
    _around = local_noise_sigma(ppm, intensities, lo, hi,
                                linewidth_ppm=linewidth_ppm,
                                global_sigma=noise_sigma)
    interference = (float(_around) / float(noise_sigma)
                    if _around and noise_sigma else float('nan'))
    _p = np.asarray(ppm, dtype=float)
    # The points the integration summed: the segments', where the signal was
    # measured on pieces -- the noise on an area grows with the points in it.
    _in = np.zeros(_p.shape, dtype=bool)
    for a, b in (segments or [(lo, hi)]):
        _in |= (_p >= min(a, b)) & (_p <= max(a, b))
    n_pts = int(_in.sum())
    dppm = float(abs(_p[1] - _p[0])) if _p.size > 1 else 0.0
    lod_a, loq_a = detection_limits(noise_sigma, dppm, n_pts)
    return snr, lod_a, loq_a, interference


def _resolve_shared(candidates, peaks_by_rank, ppm, intensities, *,
                    included, local_baseline, align, blank_indices,
                    linewidth_ppm, noise_sigma, drift_ppm=0.0,
                    excluded_regions=(), diagnostics=None):
    """What each analyte's own share of the shared peaks is.

    Every window is one equation and they are solved together, because they are
    not independent problems: a metabolite appears in as many windows as it has
    signals and each holds a different fraction of it. One at a time, a window
    with two claimants and no independent measurement of either has two
    unknowns and yields nothing -- 215 of the reference study's 242 contested
    windows were of that kind. Together they are 291 equations in about 130
    unknowns.

    Returns ``({met_id: info}, reasons)`` with the same shape the integration
    loop produces, so a share and an uncontested area travel through
    concentration, confidence and the row builder by one path.
    """
    windows, by_key, reasons, counts = _signal_windows(
        candidates, peaks_by_rank, ppm, intensities, included=included,
        local_baseline=local_baseline, align=align,
        blank_indices=blank_indices, linewidth_ppm=linewidth_ppm,
        noise_sigma=noise_sigma, drift_ppm=drift_ppm,
        excluded_regions=excluded_regions)
    per_proton, why, solver_stats = solve_windows(windows)

    # Which window speaks for each metabolite. The one built on its own signal
    # measures it; failing that, the one holding most of it. It decides the
    # region, the limits and the proton count the row reports -- the
    # concentration itself came from all of them at once.
    home: dict = {}
    for w in windows:
        if w.get('bound'):
            continue
        for mid, _nh_eff in w['claimants']:
            better = (mid == w.get('owner'),
                      float((w.get('geometry') or {}).get(mid, 0.0)))
            if better > home.get(mid, ((False, -1.0), None))[0]:
                home[mid] = (better, w)

    out = {}
    for mid, by_sample in per_proton.items():
        entry = home.get(mid)
        if entry is None:
            continue
        w = entry[1]
        cand = by_key.get((mid, frozenset(w['lines'].get(mid) or ())))
        if cand is None:
            cand = next((c for (m, r), c in by_key.items() if m == mid), None)
        if cand is None:
            continue
        g = float((w.get('geometry') or {}).get(mid, 1.0))
        if g < MIN_OWN_SHARE:
            counts['thin_share'] += 1
            why.setdefault(mid, 'no window holds as much as half of it; the '
                                'closest holds %.0f%%' % (100.0 * g))
            continue
        nh = float((w.get('protons') or {}).get(mid, 0.0))
        if nh <= 0:
            continue
        lo, hi = w['window']
        snr, lod_a, loq_a, interference = _window_limits(
            ppm, intensities, lo, hi, included=included,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm)
        areas = {int(s_i): float(v) * nh for s_i, v in by_sample.items()}
        med = (float(np.median(list(areas.values()))) if areas
               else float('nan'))
        own = [peaks_by_rank[r] for r in (w['lines'].get(mid) or ())
               if r in peaks_by_rank]
        out[mid] = {
            'signal': cand, 'nh': nh, 'snr': snr,
            'interference': interference,
            'captured': float((w.get('fractions') or {}).get(mid, 1.0)),
            'detected': {s_i: bool(np.isfinite(v) and v >= K_LOD)
                         for s_i, v in snr.items()},
            'lines': [float(pk.get('ppm', 0.0)) for pk in own],
            'weights': pascal_weights(len(own)),
            'lod_area': lod_a, 'loq_area': loq_a, 'area_median': med,
            'strength': strength_band(med, loq_a),
            'window': (lo, hi), 'areas': areas,
            'width_ppm': abs(hi - lo), 'shifts': w.get('shifts'),
            'blank_area': w.get('blank_area', float('nan')),
            'shared_with': [m for m, _ in w['claimants'] if m != mid],
        }

    # A window's own refusal belongs to the signal it was built on, and to no
    # one else. An analyte reaching 5% into somebody else's window is not
    # answerable for that window failing, and marking it so penalised analytes
    # that had a clean window of their own.
    for entry in reasons.values():
        mid = entry['claimants'][0][0] if entry['claimants'] else None
        if mid is not None and mid not in out:
            why.setdefault(mid, entry['reason'])

    # What an analyte is answerable for. A refusal costs it confidence only
    # when the window that measures it is one it shares -- there the number
    # really is an upper bound holding somebody else's signal. A column the
    # joint system could not pin down, on a window nobody else is in, is a
    # statement about the system and not about that measurement: its own
    # window measured it, and the direct measurement stands.
    #
    # Read off which lines more than one metabolite is matched to, rather than
    # off the windows: a window refused before it was ever built has no
    # claimant list, and the analyte sharing that peak is exactly the one that
    # needs saying so.
    owners: dict = {}
    for (mid, ranks) in by_key:
        for r in ranks:
            owners.setdefault(int(r), set()).add(mid)
    sharing = {mid for (mid, ranks) in by_key
               for r in ranks if len(owners.get(int(r), ())) > 1}
    penalised = {mid: t for mid, t in why.items() if mid in sharing}

    if diagnostics is not None:
        # The widest contests, claimant by claimant, with what annotation said
        # about each. Twelve analytes on three lines is not a quantification
        # problem, and which gate let the twelve through can be read here.
        n_signals: dict = {}
        for (mid, _r) in by_key:
            n_signals[mid] = n_signals.get(mid, 0) + 1
        contests = []
        for w in sorted(windows, key=lambda w: -len(w['claimants'])):
            if w.get('bound') or len(w['claimants']) < 3:
                continue
            rows = []
            for mid, _nh_eff in w['claimants']:
                cand = by_key.get((mid, frozenset(w['lines'].get(mid) or ())))
                if cand is None:
                    cand = next((c for (m, r), c in by_key.items()
                                 if m == mid), None)
                ev = (cand.evidence or {}) if cand is not None else {}
                name = ((cand.payload or {}).get('name') if cand is not None
                        and isinstance(cand.payload, dict) else None)
                rows.append({
                    'met': mid, 'name': name or mid,
                    'mult': ev.get('multiplicity'),
                    'dppm': ev.get('ppm_diff'), 'dj': ev.get('delta_j_hz'),
                    'found': ev.get('n_found'), 'expected': ev.get('n_expected'),
                    'quality': ev.get('quality'),
                    'fraction': float((w.get('geometry') or {}).get(mid, 0.0)),
                    'signals': n_signals.get(mid, 0),
                })
            contests.append({'window': w['window'], 'owner': w.get('owner'),
                             'claimants': rows})
            if len(contests) >= 6:
                break
        tally: dict = {}
        for text in why.values():
            key = text.split(' -- ')[0].split(';')[0]
            tally[key] = tally.get(key, 0) + 1
        counts['resolved'] = len(out)
        counts['unsolved'] = len(why)
        diagnostics['shared'] = {**counts, **(solver_stats or {}),
                                 'reasons': tally,
                                 'metabolites': sorted(out),
                                 'penalised': len(penalised),
                                 'contests': contests}
    return out, penalised



# A singlet is quantified on its own only above this signal-to-noise:
# position is its whole evidence, and a peak of four sigma at a position
# is noise at a position. Ten, set with the mixture bench in view (wrong
# singlets at 1-17 sigma, right ones at 34-202), took creatinine, glycine,
# formate and sarcosine out of an organoid study whose singlets mostly
# stand at 5-10; five is the bar a signal had to clear to be checkable
# at all (quant_shared.FINGERPRINT_SUPPORT_SNR), and under detection's
# own floor.
SINGLET_MIN_SNR = 5.0
# The election is repeated without a winner whose amount its fingerprint
# contradicts, at most this many times: the peak goes to the next.
AMOUNT_ROUNDS = 3
# A measured amount can be twice the truth through overlap alone --
# glutamine's triplet read at 2x with glutamate's line inside it -- so a
# winner is withdrawn only when another of its signals falls short by
# this factor: twice is overlap, three times is absence. (Lysine on the
# same peak fell short eightfold.)
AMOUNT_RATIO = 3.0


def quantify(*, candidates, peaks_by_rank, ppm, intensities, samples_info,
             included, groups, dj_gold_hz, ambiguous_ranks=(),
             standard_areas=None, standard_nh=1.0, standard_conc=0.0,
             standard_name='Internal standard', standard_ppm=0.0,
             standard_snr=None,
             external_standard_samples=None, dilution=None, correction=None,
             use_pqn=True, local_baseline=True, proton_correction=True,
             align=False, blank_indices=None,
             noise_sigma=None, linewidth_ppm=0.008,
             drift_ppm=0.0, excluded_regions=(), deconvolve=True,
             quantify_shapes=True, resolve_shared=False,
             shared_overrides=None, diagnostics=None,
             residual_pool=None, singlet_min_snr=None,
             cascade=False, metabolites=None, field_mhz=None,
             ppm_band_ppm=None, claim_signals=True,
             max_disagreement=None, raw_intensities=None,
             joint_fit=False) -> list[dict]:
    """Quantify a study and return one row per metabolite.

    ``joint_fit`` measures the analytes whose shapes overlap together, with
    a free component at every detected peak no quantified analyte has a
    line under (processing.quant_joint_fit). Analytes that overlap nothing
    keep their integrated amount.

    ``singlet_min_snr`` is the signal-to-noise a singlet on its own must
    reach to be quantified (SINGLET_MIN_SNR when None): the analyst's
    choice per study, since no one number separates a faint real singlet
    from a peak of noise at a position.

    ``ppm_band_ppm`` is the width the election buckets Δppm by, measured
    from the study's own drift (``annotation_scoring.PPM_BAND_PPM`` when
    None). The band is deliberately coarse -- ordering two candidates by a
    Δppm difference smaller than the study's drift is ordering noise -- but
    a fixed width cannot know how much a given study drifts, so a study
    that measured its own can say so here.

    ``raw_intensities`` is the study before the project's baseline model,
    when there is one. With it every window says what share of its area
    the model removed (``broad_removed_fraction``), which is the audit
    trail for subtracting real signal: a window the model took a third of
    is reporting the model as much as the analyte, and a model that takes
    a tenth of the internal standard moves every concentration in the
    study by a tenth.

    ``max_disagreement`` is how far an analyte the cascade brought in may
    have its own signals disagree and still be given a number
    (MAX_SIGNAL_DISAGREEMENT when None). The analyst's choice per study,
    like the signal-to-noise a lone singlet must reach: measured
    leave-one-out on the reference bench, an amount whose signals agree to
    within a fifth came out to a median 8% and at worst 25%, and one that
    did not to a median 30% with the ratio between two groups moving as
    much. Raising it reports analytes that would otherwise be identified
    without a number, each carrying its disagreement.

    ``cascade`` runs the cascade afterwards (quant_cascade, ADR-028) over
    ``metabolites`` -- every annotated analyte with its listed signals --
    on what the quantified analytes leave of the spectrum, and turns what
    it takes in into rows. It only adds: an analyte the election
    quantified is subtracted at its measured amount and never offered.
    Needs ``field_mhz`` to lay the listed patterns.

    Runs the election and the measurement (``_quantify_once``), then asks
    of every winner whether the amount its signal implies is one the rest
    of its fingerprint can carry (``fingerprint_holds``, ADR-012 at the
    measured amount): a winner it contradicts is withdrawn, its row says
    so, and the election is run again without it so the peak goes to the
    next candidate -- lysine at 1.9 mM, on glutamine's triplet, would have
    CH2 multiplets the spectrum does not hold. A singlet is held to more:
    SINGLET_MIN_SNR, and at least one other listed signal seen at the
    amount it implies (an analyte listed with one signal is asked only
    the first). Up to AMOUNT_ROUNDS rounds. See ``_quantify_once`` for
    the rest.
    """
    # One baseline under the whole spectrum for the measurements that fit
    # shapes (joint fit, cascade), read where each cluster of signals ends.
    cluster_base = (cluster_baseline(
        ppm, intensities, [p.get('ppm') for p in (peaks_by_rank or {}).values()],
        linewidth_ppm) if local_baseline and linewidth_ppm else None)
    kw = dict(
        peaks_by_rank=peaks_by_rank, ppm=ppm, intensities=intensities,
        samples_info=samples_info, included=included, groups=groups,
        dj_gold_hz=dj_gold_hz, ambiguous_ranks=ambiguous_ranks,
        standard_areas=standard_areas, standard_nh=standard_nh,
        standard_conc=standard_conc, standard_name=standard_name,
        standard_ppm=standard_ppm, standard_snr=standard_snr,
        external_standard_samples=external_standard_samples,
        dilution=dilution, correction=correction, use_pqn=use_pqn,
        local_baseline=local_baseline, proton_correction=proton_correction,
        align=align, blank_indices=blank_indices, noise_sigma=noise_sigma,
        linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm,
        excluded_regions=excluded_regions, deconvolve=deconvolve,
        quantify_shapes=quantify_shapes, resolve_shared=resolve_shared,
        shared_overrides=shared_overrides, diagnostics=diagnostics,
        residual_pool=residual_pool, singlet_min_snr=singlet_min_snr,
        ppm_band_ppm=ppm_band_ppm, claim_signals=claim_signals,
        raw_intensities=raw_intensities, joint_fit=joint_fit,
        field_mhz=field_mhz, cluster_base=cluster_base)
    cands = list(candidates or ())
    withdrawn: dict = {}
    rows: list = []
    for _round in range(AMOUNT_ROUNDS):
        rows, bad = _quantify_once(candidates=cands, withdrawn=withdrawn,
                                   retry=_round < AMOUNT_ROUNDS - 1, **kw)
        if not bad:
            break
        withdrawn.update(bad)
        cands = [c for c in cands if c.met_id not in withdrawn]
        pool = kw.get('residual_pool')
        if pool:
            kw['residual_pool'] = [c for c in pool if c.met_id not in withdrawn]
    diag = kw.get('diagnostics')
    if diag is not None:
        diag['amount_withdrawn'] = {m: r for m, (r, _d) in withdrawn.items()}
    if cascade and metabolites:
        # The cascade probes the spectrum with an analyte's whole listing,
        # so it never asks whether that listing belongs to this study at
        # all. One whose every matched signal landed further from the
        # common drift than the band allows is not a candidate: it would
        # be fitted on columns built at positions the study does not use.
        _in = [m for m in metabolites if stands_in_study(m, ppm_band_ppm)]
        rows = _cascade_rows(
            rows, _in, ppm=ppm, intensities=intensities,
            included=included, groups=groups, noise_sigma=noise_sigma,
            linewidth_ppm=linewidth_ppm,
            # the study's drift as annotation measured it (the median
            # shift of the matched signals), not the detection limits'
            # search half-width that ``drift_ppm`` carries
            drift_ppm=float(study_drift(metabolites) or 0.0),
            field_mhz=field_mhz, excluded_regions=excluded_regions,
            max_disagreement=max_disagreement,
            standard_areas=standard_areas, standard_nh=standard_nh,
            standard_conc=standard_conc,
            external_standard_samples=external_standard_samples,
            dilution=dilution, correction=correction,
            proton_correction=proton_correction, diagnostics=diag,
            local_baseline=local_baseline, cluster_base=cluster_base)
    return rows


def _cascade_rows(rows, metabolites, *, ppm, intensities, included, groups,
                  noise_sigma, linewidth_ppm, drift_ppm, field_mhz,
                  excluded_regions, standard_areas, standard_nh,
                  standard_conc, external_standard_samples, dilution,
                  correction, proton_correction, max_disagreement=None,
                  diagnostics=None, local_baseline=False,
                  cluster_base=None) -> list:
    """The rows the cascade adds to a quantified study, in place of the
    rejected rows of the analytes it takes in.

    The quantified analytes are handed to the cascade at their area per
    proton before normalisation -- the cascade works on the spectrum as
    measured -- and what it finds is normalised by the same factors, so a
    cascade row and an elected one are comparable numbers.

    ``cluster_base`` (``quant_integration.cluster_baseline``) is the local
    baseline the cascade fits on top of, when the run subtracts one.
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if cluster_base is not None:
        arr = arr - np.asarray(cluster_base, dtype=float).reshape(arr.shape)
    incl = [int(i) for i in (included if included is not None
                             else range(arr.shape[1]))]
    # a row whose joint fit was refused holds its window's integral, which
    # includes its neighbours: the cascade may take the analyte in on the
    # residual instead, and what it subtracts is not that integral
    quantified = [r for r in rows if not r.get('_not_quantified')
                  and not r.get('_is_is') and not r.get('_fit_refused')]
    pqn = {}
    for r in quantified:
        pqn.update(r.get('_pqn_factors') or {})
    fixed = {}
    for r in quantified:
        nh = float(r.get('_nh') or 0.0)
        if not nh or not np.isfinite(nh):
            continue
        fixed[r['hmdb_id']] = {
            int(s): float(a) / nh * float(pqn.get(int(s), 1.0))
            for s, a in (r.get('_areas_per_sample') or {}).items()
            if a is not None and np.isfinite(a) and a > 0}
    found = cascade_missed(
        metabolites, ppm, arr, incl, fixed_per_proton=fixed,
        noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
        drift_ppm=drift_ppm, field_mhz=field_mhz,
        excluded_regions=excluded_regions, diagnostics=diagnostics)
    if not found:
        return rows
    standard = (standard_per_proton(standard_areas, standard_nh,
                                    external_samples=external_standard_samples)
                if standard_areas else None)
    by_id = {r.get('hmdb_id'): r for r in rows}
    cols = {c.mid: c for c in build_columns(
        [m for m in metabolites or () if m.get('id') in found],
        field_mhz=field_mhz, linewidth_ppm=linewidth_ppm,
        drift_ppm=drift_ppm)}
    out = list(rows)
    for mid, res in found.items():
        nh = float(res['nh'])
        areas = {s: a / float(pqn.get(s, 1.0)) * nh
                 for s, a in res['per_proton'].items() if a > 0}
        conc = concentrations(areas, nh, standard=standard,
                              standard_conc=standard_conc,
                              external=bool(external_standard_samples),
                              dilution=dilution, correction=correction,
                              proton_correction=proton_correction)
        by_group = {g: [conc[s] for s in idxs
                        if s in conc and np.isfinite(conc[s]) and conc[s] > 0]
                    for g, idxs in (groups or {}).items()}
        good = [v for v in conc.values() if np.isfinite(v) and v > 0]
        if not good:
            continue
        former = by_id.get(mid)
        _dis = float(res.get('disagreement') or float('nan'))
        _bar = float(MAX_SIGNAL_DISAGREEMENT if max_disagreement is None
                     else max_disagreement)
        if (float(res.get('free_share') or 1.0) < MIN_FREE_TO_QUANTIFY
                or (np.isfinite(_dis) and _dis > _bar)):
            # identified, but the amount would come from the part of it
            # nobody else accounts for; that number is not a measurement
            if former is not None and former.get('_fit_refused'):
                continue      # keeps the integral it has, at one dot
            row = _rejected_row(mid, CASCADE_CROWDED,
                                detail=' ' + _cascade_reason(res))
            row['metabolite'] = res.get('name', mid)
            row['ppm'] = float((res.get('window') or (0.0, 0.0))[0])
            row['_by_cascade'] = True
            row['_cascade'] = {k: res[k] for k in
                               ('apex_snr', 'shift_ppm', 'at_cap', 'signals',
                                'step', 'nh', 'free_share', 'capped_share',
                                'disagreement')}
        else:
            # placed on the study's drift: no further from where the study
            # puts every other signal than one step of the cascade's slide
            res['placed'] = abs(float(res.get('shift_ppm') or 0.0)) <= \
                1.001 * cascade_slide_step(linewidth_ppm)
            res['wide'] = abs(float(res.get('shift_ppm') or 0.0)) > \
                1.001 * CASCADE_FINE_REACH_LW * float(linewidth_ppm)
            row = _cascade_row(mid, res, conc, by_group, good, areas, pqn,
                               former=former)
            if mid in cols and res.get('window'):
                row['_fit_curve'] = fit_curve_record(
                    cols[mid], ppm, res['window'], res['per_proton'], groups,
                    label='Cascade fit', shift=float(res.get('shift_ppm') or 0.0),
                    baseline=cluster_base)
        row['_local_baseline'] = bool(local_baseline)
        if former is not None:
            out[out.index(former)] = row
        else:
            out.append(row)
    return out


def _cascade_row(mid, res, conc, by_group, good, areas, pqn, *, former=None) -> dict:
    """A row for an analyte the cascade took in. One dot: its identity
    rests on its whole listing explaining the residual jointly, not on
    lines confirmed one by one, and the election had rejected it."""
    lo, hi = res.get('window') or (0.0, 0.0)
    sig = sorted(res.get('signals') or (), key=lambda v: -v[3])
    was = [t for t in ((former or {}).get('_status_reasons') or [])]
    level, why = _identity_evidence(cascade=res)
    return {
        'hmdb_id': mid,
        'metabolite': res.get('name', mid),
        'ppm': float((lo + hi) / 2.0),
        'n_peaks': len(sig),
        'raw_area': float(np.mean(list(areas.values()))),
        'corr_area': float(np.mean(list(areas.values()))),
        'conc_mm': float(np.mean(good)),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        'snr': float(res.get('apex_snr') or 0.0),
        'status': _DOTS[level],
        'vip_score': float('nan'),
        '_confidence': level,
        '_identity_why': why,
        '_identity_level': level,
        '_integ_mode': 'Cascade',
        '_not_quantified': False,
        '_by_cascade': True,
        '_cascade': {k: res.get(k) for k in ('apex_snr', 'shift_ppm', 'at_cap',
                                             'signals', 'step', 'nh',
                                             'free_share', 'capped_share',
                                             'disagreement', 'placed', 'wide')},
        '_peak': None,
        '_peak_rank': None,
        '_display_peak_rank': None,
        '_multiplet_window': [float(lo), float(hi)],
        '_line_ppms': [float(v[0]) for v in sig],
        '_nh': float(res['nh']),
        '_areas_per_sample': areas,
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_pqn_factors': dict(pqn),
        '_snr_per_sample': {},
        '_detected_per_sample': {},
        '_peak_align_shifts': None,
        '_blk_mean_area': float('nan'),
        '_is_singlet': len(sig) == 1 and is_singlet_like(sig[0][1]),
        '_multiplicity': sig[0][1] if sig else None,
        '_ann_score': 0.0,
        '_lod_mm': None,
        '_loq_mm': None,
        '_status_reasons': [_cascade_reason(res)] + (
            ['Before the cascade: ' + ' '.join(was)] if was else []),
        '_hits': [],
        '_used_j_resolve': False,
        '_j_resolve_frac': 1.0,
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }


def _cascade_reason(res) -> str:
    sig = sorted(res.get('signals') or (), key=lambda v: -v[3])
    parts = ', '.join(f"{v[1] or '?'} {v[0]:.3f} ({v[3]:.0f} sigma)" for v in sig[:4])
    more = f' and {len(sig) - 4} more' if len(sig) > 4 else ''
    free = float(res.get('free_share') or 1.0)
    capped = float(res.get('capped_share') or 0.0)
    dis = float(res.get('disagreement') or float('nan'))
    agree = (f" Measured again on each of its signals separately, they agree "
             f"to {100 * dis:.0f}% of the amount."
             if np.isfinite(dis) else
             " It has no second signal bright enough to check the amount "
             "against, so the amount has no error bar of its own.")
    held = (f" It was held down in {100 * capped:.0f}% of the samples by the "
            f"emptiest of its regions: the amount reads low."
            if capped else '')
    return (f"Added by the cascade (step {res.get('step')}): won no peak in the "
            f"election, but with every quantified analyte subtracted at its "
            f"measured amount, its whole listing -- {parts}{more} -- is what best "
            f"explains the residual, at one amount for all its signals, and "
            f"stands {float(res.get('apex_snr') or 0):.0f} sigma at its tallest. "
            f"{100 * free:.0f}% of its signal lies where nothing else is "
            f"accounted for; the amount was fitted beside the quantified "
            f"analytes that reach into its regions, each free within a quarter "
            f"of what it measured.{agree} Its lines were not confirmed one by "
            f"one: a weak identification, to be checked.{held}")


def _quantify_once(*, candidates, peaks_by_rank, ppm, intensities,
                   samples_info, included, groups, dj_gold_hz,
                   ambiguous_ranks=(),
                   standard_areas=None, standard_nh=1.0, standard_conc=0.0,
                   standard_name='Internal standard', standard_ppm=0.0,
                   standard_snr=None,
                   external_standard_samples=None, dilution=None,
                   correction=None,
                   use_pqn=True, local_baseline=True, proton_correction=True,
                   align=False, blank_indices=None,
                   noise_sigma=None, linewidth_ppm=0.008,
                   drift_ppm=0.0, excluded_regions=(), deconvolve=True,
                   quantify_shapes=True, resolve_shared=False,
                   shared_overrides=None, diagnostics=None,
                   residual_pool=None, withdrawn=None, singlet_min_snr=None,
                   ppm_band_ppm=None, claim_signals=True,
                   raw_intensities=None, joint_fit=False, field_mhz=None,
                   retry=False, cluster_base=None):
    """One election and measurement. Returns ``(rows, withdrawn)``: the
    rows, and the winners whose measured amount the fingerprint
    contradicts (``{met_id: (reason, detail)}``) -- empty when the table
    stands. ``withdrawn`` (in) are the analytes earlier rounds took out;
    they get their rows here. With ``retry``, a round that contradicts
    anyone returns no rows: the caller elects again without them.

    Every metabolite that was a candidate gets a row, quantified or not: an
    analyte that vanishes from the table cannot be questioned, and the reason
    it has no number is the part an analyst can act on.

    ``candidates`` are ``processing.quantification.Candidate`` objects whose
    ``payload`` carries whatever the caller needs to build display fields —
    this engine only reads ``met_id``, the evidence, and the line peaks.

    ``deconvolve`` off leaves every winner credited with the whole of its
    window, shared or not, and reports nothing for the analytes that lost one.
    That is what the engine did before the balance existed, and it is the
    setting to compare against when a number moves.

    ``quantify_shapes`` off leaves every complex multiplet (``m*``, ADR-013
    -- fitted to the trace or matched on the peak list) out of the election
    and out of the table's numbers, each reported as not quantified with
    the switch named. The rest of the run is then the letter-multiplet and
    singlet engine. It used to leave out only the trace-fitted ones; a
    pattern matched on two apices still competed, and on the mixture bench
    nine such "2/2" patterns won peaks that belonged to the panel's real
    triplets (raffinose took glutamine's) with the switch off.

    ``resolve_shared`` on resolves, by their shapes, the winners' windows
    that hold another accepted analyte's signal (processing.quant_shared):
    the owner is credited with what its shape accounts for, the intruder
    held to its own clean measurement, the rest a remainder on the row.
    Off, a window is its owner's whole, as before. ``shared_overrides`` is
    what the analyst decided per owner in the review window (see
    ``quant_shared.resolve_shared_windows``).

    ``residual_pool`` (with ``resolve_shared``) is the signals the election
    did not accept. A winner's window whose accepted shapes leave a
    remainder is offered them, and one is taken in on three measured
    conditions (processing.quant_shared.resolve_residuals): its analyte is
    then quantified from that window, marked as identified by the
    remainder, at the lowest confidence.

    ``diagnostics``, when a dict is passed, is filled with what the run did
    that no row can show — how many windows were shared, and why the ones that
    resolved nothing resolved nothing. A step that silently does nothing is
    indistinguishable from a step that is not there.
    """
    candidates = list(candidates or ())
    residual_pool = list(residual_pool or ())
    shapes_off: dict = {}
    if not quantify_shapes:
        kept = [c for c in candidates if not _is_complex(c)]
        with_peaks = {c.met_id for c in kept}
        for c in candidates:
            if _is_complex(c) and c.met_id not in with_peaks:
                shapes_off[c.met_id] = SHAPES_OFF
        candidates = kept
        # off means off: not offered to the remainders either
        residual_pool = [c for c in residual_pool if not _is_complex(c)]
        if diagnostics is not None:
            diagnostics['shapes'] = {'off': True, 'left_out': len(shapes_off)}
    else:
        # On, a complex multiplet still never competes for a peak: it is
        # offered to the remainders (resolve_residuals, resolve_free_apices)
        # and identified only where a free apex, its own amount and its
        # whole shape agree. Fitted alone and elected unopposed it was
        # worth 41 false positives on the mixture bench.
        _complex = [c for c in candidates if _is_complex(c)]
        candidates = [c for c in candidates if not _is_complex(c)]
        _pooled = {id(c) for c in residual_pool}
        residual_pool += [c for c in _complex if id(c) not in _pooled]
        # an analyte whose every signal is a shape has a row either way:
        # one that vanishes from the table cannot be questioned
        _with_peaks = {c.met_id for c in candidates}
        for c in _complex:
            if c.met_id not in _with_peaks:
                shapes_off[c.met_id] = SHAPE_UNIDENTIFIED
    election = elect(candidates, dj_gold_hz=dj_gold_hz,
                     ambiguous_ranks=ambiguous_ranks,
                     band_ppm=ppm_band_ppm)
    residual_ids: set = set()
    taken_over: dict = {}

    # ── Integrate every winner, per sample ───────────────────────────────────
    # A winner whose lines bound no window is recorded as such rather than
    # dropped. An analyte that simply vanishes between the election and the
    # table cannot be questioned, and this is the one place where a metabolite
    # can win and still have nothing to measure.
    measured, per_proton_areas, unmeasurable = {}, {}, []
    partial: dict = {}
    for mid, cand in election.winners.items():
        if (cand.evidence or {}).get('shape_fit'):
            # A multiplet judged as a shape holds no peak; its amount is
            # the copy of its shape that fits under each sample's trace
            # (processing.quant_shape). The shape's area is exact, so no
            # capture correction and no neighbour subtraction apply.
            nh = signal_nh((cand.evidence or {}).get('protons'))
            out = measure_shape(cand.evidence, ppm, intensities,
                                included=included, noise_sigma=noise_sigma,
                                blank_indices=blank_indices)
            if not out or not out['areas'] or not nh or nh <= 0:
                unmeasurable.append(mid)
                continue
            measured[mid] = {'signal': cand, 'nh': nh,
                             'noise_sigma': noise_sigma, **out}
            per_proton_areas[mid] = {s: a / nh
                                     for s, a in out['areas'].items()}
            continue
        lines = [peaks_by_rank[r] for r in sorted(cand.line_ranks)
                 if r in peaks_by_rank]
        if not lines and cand.peak_rank in peaks_by_rank:
            lines = [peaks_by_rank[cand.peak_rank]]
        out = integrate_signal(lines, ppm, intensities, included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices,
                               linewidth_ppm=linewidth_ppm,
                               other_peaks=list(peaks_by_rank.values()))
        if not out or not out['areas']:
            unmeasurable.append(mid)
            continue
        nh = signal_nh((cand.evidence or {}).get('protons'))
        lo, hi = out['window']
        # What the window left outside. A Lorentzian's tails never end, and how
        # much a window holds depends on its width in half-widths and on how
        # many lines share it -- so two signals integrated foot to foot lose
        # different fractions, and the ratio between them carries the
        # difference. On a study whose answer was known every analyte came back
        # about 7% high against a standard whose own window held 94% of its
        # line while a doublet's held 83%.
        _lw = (cand.evidence or {}).get('line_weights')
        # The whole pattern, not only the lines detection placed: a triplet
        # with one line found is a third of a pattern, and integrating that
        # line as if it were the signal, then dividing by the triplet's
        # protons, reports half of what is there. The missing lines stand
        # where the annotation predicted them, and the fraction the window
        # holds is computed over all of them -- so the area comes back as
        # the multiplet's. Below MIN_TRUSTED_CAPTURE of the pattern the
        # scale is not believed (one line of a triplet: x2 on a line that
        # may be somebody else's) and the analyte is not quantified.
        _ppms, _weights, _found_mask, _share = pattern_lines(
            (cand.evidence or {}).get('predicted_ppms') or (),
            [float(p.get('ppm', 0.0)) for p in lines],
            weights=(list(_lw) if _lw else None))
        _missing = [p for p, m in zip(_ppms, _found_mask) if not m]
        if _share < MIN_TRUSTED_CAPTURE:
            partial[mid] = (len(lines), len(_ppms), _share, _missing)
            continue
        # A window built from the lines and the neighbours is known: the
        # correction is arithmetic on it. Only a window from detection's
        # feet alone is refused when it holds too little.
        captured = lorentzian_captured_fraction(
            _ppms, _weights, lo, hi, linewidth_ppm / 2.0,
            min_trusted=(0.0 if out.get('geometry') == 'lines'
                         else MIN_TRUSTED_CAPTURE),
            segments=out.get('segments'),
            anchors=out.get('baseline_anchors'))
        if captured:
            out['areas'] = {s_i: a / captured
                            for s_i, a in out['areas'].items()}
        snr, lod_a, loq_a, interference = _window_limits(
            ppm, intensities, lo, hi, included=included,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
            segments=out.get('segments'))
        med_area = (float(np.median(list(out['areas'].values())))
                    if out['areas'] else float('nan'))
        # Which samples actually show this signal, at the same ratio the
        # detection limit is defined by. The CSV reports the others as 'n.d.'
        # rather than as a concentration: a window over noise integrates to a
        # perfectly plausible number, and writing it would put a measurement
        # where the analyte was not seen.
        detected = {s_i: bool(np.isfinite(v) and v >= K_LOD)
                    for s_i, v in snr.items()}
        measured[mid] = {'signal': cand, 'nh': nh, 'snr': snr,
                         'interference': interference,
                         'captured': captured, 'detected': detected,
                         'lines': [float(p.get('ppm', 0.0)) for p in lines],
                         'pattern_lines': list(_ppms),
                         'missing_lines': _missing,
                         'pattern_share': _share,
                         'weights': _weights,
                         'lod_area': lod_a, 'loq_area': loq_a,
                         'area_median': med_area,
                         'strength': strength_band(med_area, loq_a), **out}
        per_proton_areas[mid] = {s: a / nh for s, a in out['areas'].items()}

    # ── The signal on each group's mean ─────────────────────────────────
    # The figure the report draws is one group's mean, and the SNR and the
    # shape agreement beside it must be read on THAT trace, for THIS
    # signal -- not the anchor peak's SNR over the study's mean while the
    # picture shows a group. Measured once here, so the table, the plot and
    # the PDF cannot disagree about which group they are looking at.
    if groups and linewidth_ppm:
        for mid, info in measured.items():
            lo, hi = info['window']
            ev = group_evidence(
                ppm, intensities, groups, lo=lo, hi=hi,
                noise=noise_sigma, lines=info.get('lines') or (),
                weights=info.get('weights') or (), hwhm=linewidth_ppm / 2.0)
            info['group_evidence'] = ev
            info['best_group'] = best_group(ev)

    # ── What each window holds of its neighbours ─────────────────────────────
    # A window is not a container for one signal. Lorentzian tails never reach
    # zero, so a strong neighbour a short way off puts a real amount of itself
    # inside — on a study whose answer was known, alanine read 6.4% high and
    # that 6.4% was lactate, 0.148 ppm away and eight times larger.
    #
    # Only for signals whose own capture correction was accepted. Where it was
    # refused the window is not describing the signal, so its area is the raw
    # one while a leak computed over the same width is the full one — taking
    # the second from the first subtracts a share the area was never credited
    # with. Measured: applying it there pushed every analyte negative and the
    # mean error from 2.8% to 4.1%.
    _trusted = {k: v for k, v in measured.items() if v.get('captured')}
    if len(_trusted) > 1 and linewidth_ppm:
        _corrected, _removed = subtract_neighbours(
            {k: {'window': v['window'], 'lines': v['lines'],
                 'weights': v['weights'], 'areas': v['areas'],
                 'segments': v.get('segments'),
                 'anchors': (v['baseline_anchors'][:2]
                             if v.get('baseline_anchors') else None)}
             for k, v in _trusted.items()},
            linewidth_ppm / 2.0)
        for k, v in _trusted.items():
            v['areas'] = _corrected.get(k, v['areas'])
            v['neighbour_share'] = _removed.get(k, 0.0)
            per_proton_areas[k] = {s_i: a / v['nh']
                                   for s_i, a in v['areas'].items()}

    # ── A window holding another accepted analyte's signal ───────────────
    # Resolved by the shapes of the accepted analytes only, after each has
    # been measured on its own; an intruder is held to what that
    # measurement predicts (processing.quant_shared, ADR-015).
    if resolve_shared:
        _shared_names = {c.met_id: ((c.payload or {}).get('name')
                                    if isinstance(c.payload, dict) else None)
                         or c.met_id for c in candidates}
        _resolved = resolve_shared_windows(
            measured, candidates, ppm, intensities,
            per_proton_areas=per_proton_areas, included=included,
            linewidth_ppm=linewidth_ppm, names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            overrides=shared_overrides, noise_sigma=noise_sigma)
        if diagnostics is not None:
            diagnostics['shape_deconvolution'] = {
                'windows': len(_resolved),
                'remainder_median': {m: float(np.median(list(
                    r['remainder'].values()))) for m, r in _resolved.items()}}
        # ── The remainder as evidence ───────────────────────────────────
        _claimed_ranks = set()
        for _w in election.winners.values():
            _claimed_ranks |= set(_w.line_ranks)
            if _w.peak_rank is not None:
                _claimed_ranks.add(_w.peak_rank)
        _taken = resolve_residuals(
            measured, list(residual_pool or ()), ppm, intensities,
            included=included, linewidth_ppm=linewidth_ppm,
            names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            per_proton_areas=per_proton_areas, noise_sigma=noise_sigma,
            fingerprint_of=lambda c: ((c.payload or {}).get('met') or {}).get(
                'db_peaks') if isinstance(c.payload, dict) else None,
            free_apex_of=lambda c: [
                float(peaks_by_rank[r]['ppm']) for r in _found_ranks(c)
                if r in peaks_by_rank and r not in _claimed_ranks],
            # a singlet's shape says nothing (None), and a window it
            # owns is not one whose remainder can be offered around
            offer_if=lambda rec: (_structural_evidence(rec['signal'])
                                  or 0) >= 2,
            drift_ppm=float((residual_pool or [None])[0].evidence.get(
                'study_drift_ppm') or 0.0) if residual_pool
            and getattr(residual_pool[0], 'evidence', None) else 0.0)
        # ── A free apex nobody holds: the spectrum's own remainder ──────
        _claimed_ranks |= {r for _t in _taken.values()
                           for r in _found_ranks(_t['signal'])}
        _free_peaks = {r: p for r, p in peaks_by_rank.items()
                       if r not in _claimed_ranks}
        _alone = resolve_free_apices(
            measured, residual_pool, _free_peaks, ppm, intensities,
            included=included, linewidth_ppm=linewidth_ppm,
            names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            ranks_of=_found_ranks,
            noise_sigma=noise_sigma,
            fingerprint_of=lambda c: ((c.payload or {}).get('met') or {}).get(
                'db_peaks') if isinstance(c.payload, dict) else None,
            drift_ppm=float((residual_pool or [None])[0].evidence.get(
                'study_drift_ppm') or 0.0) if residual_pool
            and getattr(residual_pool[0], 'evidence', None) else 0.0,
            skip=set(_taken),
            trials_out=(diagnostics.setdefault('free_apex_trials', {})
                        if diagnostics is not None else None))
        _taken.update(_alone)
        for _cm, _t in _taken.items():
            _nh = float(_t['nh'])
            measured[_cm] = {
                'signal': _t['signal'], 'nh': _nh, 'areas': dict(_t['areas']),
                'window': _t['window'], 'segments': [],
                'captured': None, 'lines': [], 'weights': [],
                'shared_with': [_t['owner']],
                'by_residual': _t['owner'],
                'residual_before': _t['remainder_before'],
                'residual_after': _t['remainder_after'],
                'deconv': _t['deconv'],
                'noise_sigma': noise_sigma,
                'area_median': (float(np.median(list(_t['areas'].values())))
                                if _t['areas'] else float('nan')),
                # No limit of its own: the window is the owner's. The
                # strength axis is about the size of the area against that
                # limit and ``strength_band`` gives only 2 or 3; writing 1
                # here said "this identification is weak" on the axis that
                # answers "this quantity is small", and pushed glutamate --
                # three of its signals confirmed -- to one dot. What is
                # weak about a remainder is the window, and the window is
                # already somebody else's: the shared cap says so.
                'lod_area': float('nan'), 'loq_area': float('nan'),
                'snr': {},
            }
            per_proton_areas[_cm] = {s_i: a / _nh
                                     for s_i, a in _t['areas'].items()}
            residual_ids.add(_cm)
        # ── What the quantified analytes list inside other windows ─────
        _listed = resolve_listed_intruders(
            measured, ppm, intensities, included=included,
            linewidth_ppm=linewidth_ppm, names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            per_proton_areas=per_proton_areas, noise_sigma=noise_sigma)
        if diagnostics is not None:
            diagnostics['listed_intruders'] = _listed
        for _m in [m for m, r in measured.items() if r.get('taken_over')]:
            _labels = measured[_m]['taken_over']
            taken_over[_m] = (WINDOW_TAKEN, ' ' + ', '.join(_labels) + '.')
            measured.pop(_m, None)
            per_proton_areas.pop(_m, None)
            residual_ids.discard(_m)
        if diagnostics is not None:
            diagnostics['residual'] = {
                'taken': {m: t['owner'] for m, t in _taken.items()},
                'trials': {m: r.get('residual_trials') for m, r in
                           measured.items() if r.get('residual_trials')}}

    # ── Two analytes on one peak ───────────────────────────────────────
    # The election gives a contested peak to one owner, which is right -- two
    # analytes integrating the same area is the defect that started this work.
    # But the area belongs to both, and until here the winner was credited with
    # all of it while the other was credited with nothing. Where every claimant
    # but one is measured elsewhere, subtracting what those must be
    # contributing leaves the rest, and the rest is arithmetic on measured
    # quantities rather than an estimate.
    if deconvolve:
        shared, shared_reasons = _resolve_shared(
            candidates, peaks_by_rank, ppm, intensities, included=included,
            local_baseline=local_baseline, align=align,
            blank_indices=blank_indices, linewidth_ppm=linewidth_ppm,
            noise_sigma=noise_sigma, drift_ppm=drift_ppm,
            excluded_regions=excluded_regions, diagnostics=diagnostics)
    else:
        shared, shared_reasons = {}, {}
        if diagnostics is not None:
            diagnostics['shared'] = {'off': True}
    recovered = set()
    # A window that resolved nothing leaves its winner credited with the whole
    # of it, including the part that is somebody else's. That has to be on the
    # row: a number silently left too large is worse than one refused.
    for mid, why in shared_reasons.items():
        if mid in measured and mid not in shared:
            measured[mid]['unshared_reason'] = why
    for mid, info in shared.items():
        if mid in measured:
            # Its number came from a window it does not hold alone; what it
            # holds of that window is this.
            # How much of the window the balance left it. A window with
            # claimants the balance gives nothing to is shared in name
            # only, and the number does not rest on their concentrations:
            # on the reference bench every correct analyte sat in such a
            # window, so the cap alone held all of them below the top band
            # and nothing was ever marked confident.
            _before = measured[mid].get('area_median')
            _after = info.get('area_median')
            _kept = (float(_after) / float(_before)
                     if _before and _after is not None and float(_before) > 0
                     else None)
            measured[mid] = {**measured[mid], 'areas': info['areas'],
                             'shared_with': info['shared_with'],
                             'strength': info['strength'],
                             'share_kept': _kept,
                             'area_median': info['area_median']}
        elif election.rejected.get(mid) == LOST_COMPETITION:
            # Only this one. Losing a contested peak is a verdict about which
            # analyte owns an area, and the balance answers exactly that. The
            # other rejections are verdicts about identity -- an ambiguous rank
            # says the search could not tell two candidates apart, an
            # unverified multiplet that the lines were never found -- and no
            # amount of arithmetic on areas settles who a signal belongs to.
            measured[mid] = info
            recovered.add(mid)
        else:
            continue
        per_proton_areas[mid] = {s_i: a / measured[mid]['nh']
                                 for s_i, a in measured[mid]['areas'].items()}

    # ── Dilution correction across samples ───────────────────────────────────
    # PQN before the standard: it corrects how concentrated the sample is, and
    # the standard then says what the corrected signal is worth. Running it
    # after would divide the answer by a factor derived from the answer.
    pqn_factors = {}
    # ── What the baseline model took out of each window ──────────────────
    # The number a window reports is the analyte plus whatever the model
    # left and minus whatever it removed. Where it removed a large share,
    # the row says so: on the reference study a SNIP model with nothing to
    # remove took a tenth of the internal standard and a quarter of
    # glutathione's window, and moved them 9% and 32%.
    if raw_intensities is not None:
        _raw = np.asarray(raw_intensities, dtype=float)
        if _raw.ndim == 1:
            _raw = _raw.reshape(-1, 1)
        _cur = np.asarray(intensities, dtype=float)
        if _cur.ndim == 1:
            _cur = _cur.reshape(-1, 1)
        if _raw.shape == _cur.shape and not np.array_equal(_raw, _cur):
            _p = np.asarray(ppm, dtype=float)
            for mid, info in measured.items():
                lo, hi = info.get('window') or (None, None)
                if lo is None:
                    continue
                m = np.flatnonzero((_p >= min(lo, hi)) & (_p <= max(lo, hi)))
                if m.size < 2:
                    continue
                f = broad_removed_fraction(_raw, _cur, _p, int(m[0]), int(m[-1]))
                info['broad_removed'] = float(np.median(np.atleast_1d(f)))

    # ── Every signal of an analyte, not one ──────────────────────────────
    # A doublet that says ten millimolar says the same molecule's singlet
    # must say ten millimolar too. Each winner is measured again on every
    # signal its reference lists that its own amount predicts should be
    # visible, and the amount becomes the consensus of the ones that
    # agreed (quant_shared.claim_listed_signals). Before the dilution
    # correction and the standard, so both see the consensus.
    _claimed: dict = {}
    if claim_signals and linewidth_ppm:
        _claimed = claim_listed_signals(
            measured, per_proton_areas, ppm, intensities, included=included,
            linewidth_ppm=linewidth_ppm, noise_sigma=noise_sigma,
            excluded_regions=excluded_regions, local_baseline=local_baseline)
        for mid, what in _claimed.items():
            if mid in measured:
                measured[mid]['claimed'] = what
        if diagnostics is not None:
            diagnostics['claimed_signals'] = {
                m: w['used'] for m, w in _claimed.items() if w['used']}

    if use_pqn and len(per_proton_areas) >= 3:
        per_proton_areas, pqn_factors = apply_pqn(
            per_proton_areas, exclude=external_standard_samples or ())

    standard = None
    if standard_areas:
        standard = standard_per_proton(
            standard_areas, standard_nh,
            external_samples=external_standard_samples)

    # ── The amount a winner implies, against the rest of its fingerprint ──
    _mean = None
    _arr_all = np.asarray(intensities, dtype=float)
    if _arr_all.ndim == 1:
        _arr_all = _arr_all.reshape(-1, 1)
    _cols = [i for i in (included if included is not None
                         else range(_arr_all.shape[1]))
             if 0 <= i < _arr_all.shape[1]]
    if _cols:
        _mean = _arr_all[:, _cols].mean(axis=1)
    bad: dict = {}
    # Withdrawn like ``bad``, but WITHOUT re-running the election. A row
    # here is one the run refuses to report; it is not a peak handed to
    # the runner-up. Freeing the peak would be the principled thing, and
    # on the reference study it is the harmful one: the peak propionic
    # acid took at 2.133 carries glutamine's quartet AND glutamate's and
    # glutathione's unresolved multiplets, so the runner-up owns a window
    # it did not fill and reads +38% where the cascade, splitting it,
    # reads -7%. Re-electing here cost p90 13% -> 19%. Until a contested
    # window can be shared rather than handed over (ADR-028's territory),
    # withdrawal stops at not reporting.
    withheld: dict = {}
    if _mean is not None and noise_sigma and linewidth_ppm:
        for mid, info in list(measured.items()):
            if info.get('by_residual'):
                continue                  # checked when it was taken in
            sig = info['signal']
            ev = sig.evidence or {}
            met = (sig.payload or {}).get('met') if isinstance(sig.payload, dict) else None
            db_peaks = (met or {}).get('db_peaks') or ()
            pp = per_proton_areas.get(mid) or {}
            vals = [v for v in pp.values() if v is not None and np.isfinite(v) and v > 0]
            if not vals:
                continue
            held, verdicts = fingerprint_holds(
                db_peaks, float(np.median(vals)), ppm, _mean,
                noise=noise_sigma, linewidth_ppm=linewidth_ppm,
                drift_ppm=float(ev.get('study_drift_ppm') or 0.0),
                skip_ppm=ev.get('db_ppm'), excluded_regions=excluded_regions,
                ratio=AMOUNT_RATIO)
            info['fingerprint_at_amount'] = verdicts
            if not held:
                bad[mid] = (AMOUNT_CONTRADICTED, _fingerprint_detail(verdicts))
                continue
            if not stands_in_study(met, ppm_band_ppm):
                withheld[mid] = (OFF_THE_STUDY, _off_study_detail(met))
                continue
            # a singlet on its own: position is all it has. Its shape
            # axis is None (nothing to confirm), which is exactly the
            # case this gate is for.
            _shape = _structural_evidence(sig)
            if (_shape is None or _shape <= 1) and is_singlet_like(
                    ev.get('multiplicity') or ev.get('type') or 's'):
                if ev.get('manual_quality') == 'good':
                    continue
                # the SNR the table shows: the signal's on the group the
                # figure draws, not the per-sample window estimate
                snr_rep = (info['group_evidence'][info['best_group']]['snr']
                           if info.get('best_group') else
                           representative_snr(info.get('snr')))
                # only what could be checked counts: a signal predicted
                # too faint to look for is not a signal not seen, and an
                # analyte whose every other signal is faint at this
                # amount is asked the signal-to-noise alone
                checkable = [v for v in verdicts
                             if v[3] == 'present' or v[3].startswith('contradicted')]
                seen = any(v[3] == 'present' for v in checkable)
                _min_snr = float(SINGLET_MIN_SNR if singlet_min_snr is None
                                 else singlet_min_snr)
                if (np.isfinite(snr_rep) and snr_rep < _min_snr) or (
                        checkable and not seen):
                    bad[mid] = (SINGLET_UNSUPPORTED,
                                _fingerprint_detail(verdicts, snr=snr_rep))
    if bad and retry:
        # the caller withdraws these and elects again: the rows would be
        # thrown away, and the joint fit below is most of their cost
        return [], bad
    for mid in list(bad) + list(withheld):
        measured.pop(mid, None)
        per_proton_areas.pop(mid, None)
        residual_ids.discard(mid)

    # How tightly this study holds together, from the signals it confirmed
    # by their coupling. A singlet has no shape to be judged on; this is
    # what it is judged on instead.
    _qmets = [(info['signal'].payload or {}).get('met')
              for info in measured.values()
              if isinstance(info['signal'].payload, dict)]
    _drift_residual = study_drift_residual(_qmets, ppm_band_ppm)
    # How tightly the analytes really here sit on the drift -- robust to a
    # crowded spectrum's strangers (see study_position_scatter).
    _scatter = study_position_scatter(_qmets, ppm_band_ppm)

    # Do an analyte's own signals agree with one another on one amount?
    # Asked of what survived, because one of them has to be a peak nobody
    # else quantified holds -- see quant_shared.signals_corroborate for
    # why (adenosine on inosine's aromatics).
    if _mean is not None and linewidth_ppm:
        _peaks_of = {mid: _matched_signal_peaks(info, ppm_band_ppm)
                     for mid, info in measured.items()}
        for mid, info in measured.items():
            taken = set().union(*(v for k, v in _peaks_of.items() if k != mid)) \
                if len(_peaks_of) > 1 else set()
            _ev = info['signal'].evidence or {}
            met = (info['signal'].payload or {}).get('met') \
                if isinstance(info['signal'].payload, dict) else None
            info['signals_linked'] = signals_corroborate(
                met, ppm, _mean, linewidth_ppm=linewidth_ppm,
                drift_ppm=float(_ev.get('study_drift_ppm') or 0.0),
                field_mhz=_ev.get('pattern_field_mhz'),
                taken_peaks=taken, band_ppm=ppm_band_ppm,
                excluded_regions=excluded_regions)

    # Where the analytes' shapes overlap, integration counts the neighbours'
    # area as theirs; measure those together instead, with room for what is
    # not in the library. The integrated amount is kept on the row: the fit
    # replaces it, and the replacement has to be visible and undoable.
    _misplaced: dict = {}
    if joint_fit and measured and linewidth_ppm and _mean is not None:
        _jmets = [(info['signal'].payload or {}).get('met')
                  for info in measured.values()
                  if isinstance(info['signal'].payload, dict)]
        _jmets = [m for m in _jmets if m]
        _jfield = field_mhz or next(
            ((info['signal'].evidence or {}).get('pattern_field_mhz')
             for info in measured.values()
             if (info['signal'].evidence or {}).get('pattern_field_mhz')),
            None)
        if _jfield:
            # Every candidate annotation proposed that stands in the study
            # enters too, as a column whose amount is not reported: the
            # library, not only its winners (ASICS fits the whole library).
            # Without them the area of an analyte really present and not
            # quantified has nowhere to go but into a quantified neighbour.
            _nuis = {}
            if JOINT_FIT_CANDIDATES:
                for _c in candidates or ():
                    _m = ((_c.payload or {}).get('met')
                          if isinstance(_c.payload, dict) else None)
                    if (not _m or not _m.get('id') or _m['id'] in measured
                            or _m['id'] in _nuis):
                        continue
                    if any(_in_study(mp, ppm_band_ppm)
                           for mp in _m.get('matched_peaks') or ()
                           if mp.get('db_ppm') is not None):
                        _nuis[_m['id']] = _m
            _jcols = [c for c in build_columns(
                _jmets + list(_nuis.values()), field_mhz=float(_jfield),
                linewidth_ppm=float(linewidth_ppm),
                drift_ppm=float(study_drift(_jmets) or 0.0))
                if c.mid in measured or c.mid in _nuis]
            _jf = joint_measure(
                _jcols, list((peaks_by_rank or {}).values()), ppm, _arr_all,
                _cols, sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
                weight=exclusion_weight(ppm, excluded_regions),
                baseline=cluster_base)
            _jcol_of = {c.mid: c for c in _jcols}
            for mid, r in _jf.items():
                if mid not in measured:
                    continue          # a candidate: its amount absorbs, unreported
                # What the report draws: the fitted shape, on the baseline
                # the fit stood on -- not a window that was not integrated.
                _rec = (fit_curve_record(
                    _jcol_of[mid], ppm, measured[mid]['window'],
                    r['amplitudes'], groups, label='Joint fit',
                    shift_by_sample=r.get('shift_by_sample'),
                    width=r.get('width_scale', 1.0),
                    baseline=cluster_base,
                    offsets=r.get('signal_offsets') or (),
                    offset_limit=r.get('offset_limit'))
                    if measured[mid].get('window') else None)
                _why = joint_fit_refusal(_rec)
                if _why:
                    # The fit measured a shape that is not where the signal
                    # is, or another signal than the one elected. Its amount
                    # is not this analyte's -- and neither is the integrated
                    # one, whose window holds the neighbours the fit was
                    # there to separate (FIT_MISPLACED).
                    if FIT_MISPLACED_KEEPS_INTEGRAL:
                        measured[mid]['fit_refused'] = _why
                    else:
                        _misplaced[mid] = _why
                    continue
                measured[mid]['fit_curve'] = _rec
                measured[mid]['joint_fit'] = {
                    'group': list(r['group']), 'free': int(r['free']),
                    'residual': float(r['residual']),
                    'signal_offsets': list(r.get('signal_offsets') or ()),
                    'integrated': dict(per_proton_areas.get(mid) or {}),
                    'fitted': dict(r['amplitudes'])}
                # normalised like every other analyte: the fit's amplitudes
                # are the spectrum's, and the PQN above was applied to the
                # integrated amounts they replace
                per_proton_areas[mid] = {
                    s: v / float(pqn_factors.get(s, 1.0))
                    for s, v in r['amplitudes'].items()}
                # the row's areas and its concentrations are one measurement
                _nh = float(measured[mid]['nh'])
                measured[mid]['areas'] = {s: v * _nh
                                          for s, v in r['amplitudes'].items()}

    for mid in _misplaced:
        measured.pop(mid, None)
        per_proton_areas.pop(mid, None)

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
        ident, ident_why = _identity_evidence(
            **_identity_of(info, ppm_band_ppm, _drift_residual))
        # Sharing costs a dot when the others actually took something: a
        # quantity this window could have quantified. Below its own limit
        # of quantification what they took is not a quantity, so the
        # number does not rest on their concentrations.
        _given = info.get('share_given')
        _kept = info.get('share_kept')
        _loq = info.get('loq_area')
        # Two ways of being immaterial, and either is enough. What the
        # others took is below what this window could have quantified, so
        # it is not a quantity; or it is a small enough part of the
        # analyte's own area that its number does not rest on theirs --
        # lactate gave 6% of its window away, and a neighbour wrong by
        # half would move it by 3%.
        _material = (_given is None
                     or ((_loq is None or not np.isfinite(_loq)
                          or float(_given) >= float(_loq))
                         and (_kept is None
                              or float(_kept) < SHARE_KEPT_ALONE)))
        _really_shared = rests_on_shared_window(info, _material)
        level, dots = _confidence(
            info['signal'], info.get('strength'),
            shared=_really_shared, identity=ident, dj_gold_hz=dj_gold_hz)
        # The signal the amount was measured on, off where this study puts
        # things: whatever else agrees, the number is somebody's peak.
        _off = (info['signal'].evidence or {}).get('ppm_off_drift')
        _bar = (POSITION_SCATTER_K * _scatter
                if _scatter is not None else None)
        _fit = _fitted_offset(info)
        _step = _fit_step(linewidth_ppm)
        _pulled = (_fit is not None
                   and _fit[0] <= FITTED_POSITION_STEPS * _step
                   and _fit[1] <= FITTED_OTHERS_STEPS * _step)
        if (_bar is not None and _off is not None and abs(float(_off)) > _bar
                and not _pulled):
            info['position_cap'] = (abs(float(_off)), _bar)
            level = min(level, 1)
            dots = _DOTS[level]
        info['identity_level'] = ident
        info['identity_why'] = ident_why
        info['confidence_axes'] = {
            'identity': ident,
            'shape': _structural_evidence(info['signal'], dj_gold_hz),
            'strength': info.get('strength'),
            'shared': _really_shared,
            'share_kept': info.get('share_kept'),
            'position_off': (info['position_cap'][0]
                             if info.get('position_cap') else None),
        }
        rows.append(_row(mid, info, conc, by_group, good, level, dots,
                         pqn_factors, standard, standard_conc,
                         bool(external_standard_samples), dilution,
                         correction))
        if info.get('fit_refused'):
            _as_candidate(rows[-1], info['fit_refused'], by_group)

    # What the integration subtracted, so a report can draw the very region
    # the number came from: with a local baseline the area is between the
    # trace and the line joining the window's ends, not down to zero.
    for r in rows:
        r['_local_baseline'] = bool(local_baseline)

    names = {c.met_id: ((c.payload or {}).get('name') if isinstance(
        c.payload, dict) else None) or c.met_id for c in (candidates or ())}
    for mid, reason in shapes_off.items():
        if mid in residual_ids or mid in measured:
            continue
        rows.append(_rejected_row(mid, reason))
    for mid, why in _misplaced.items():
        rows.append(_rejected_row(mid, FIT_MISPLACED, detail=f' Here {why}.'))
    for mid, (n_found, n_all, share, missing) in partial.items():
        rows.append(_rejected_row(
            mid, PARTIAL_MULTIPLET,
            detail=(f' Here {n_found} of {n_all} lines were placed, holding '
                    f'{100 * share:.0f}% of the pattern; the missing lines '
                    f'were expected at '
                    f'{", ".join("%.4f" % v for v in missing)} ppm.')))
    for mid, reason in election.rejected.items():
        if mid in recovered or mid in residual_ids:
            continue
        rows.append(_rejected_row(mid, reason, shared_reasons.get(mid),
                                  beaten=election.beaten.get(mid),
                                  names=names))
    # A winner with no detected line is not a failure to measure — it is a
    # measurement of an upper bound, and reporting it as one is the whole of
    # processing.quant_detection_limit.
    for mid in unmeasurable:
        rows.append(_censored_row(
            election.winners[mid], ppm, intensities, included=included,
            standard=standard, standard_conc=standard_conc,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
            drift_ppm=drift_ppm, local_baseline=local_baseline,
            external=bool(external_standard_samples),
            dilution=dilution, correction=correction,
            excluded_regions=excluded_regions))

    for mid, (reason, detail) in (withdrawn or {}).items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    for mid, (reason, detail) in taken_over.items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    for mid, (reason, detail) in bad.items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    for mid, (reason, detail) in withheld.items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    # An analyte that may be quantified but whose every usable signal is a
    # shape sits only in the remainder pool: it competes for no peak, and
    # when no remainder takes it no step above writes its row. On the
    # virtual bench 16 eligible analytes left the table that way, three of
    # them in the tube. One that vanishes cannot be questioned.
    _have = {r.get('hmdb_id') for r in rows}
    for c in residual_pool:
        if ((c.evidence or {}).get('eligible') and c.met_id not in _have
                and c.met_id not in measured):
            rows.append(_rejected_row(c.met_id, SHAPE_UNIDENTIFIED))
            _have.add(c.met_id)

    if standard_areas:
        rows.append(_standard_row(standard_areas, standard_nh, standard_conc,
                                  standard_name, standard_ppm, standard_snr,
                                  groups, standard,
                                  external_samples=external_standard_samples))
    return rows, bad


def _fingerprint_detail(verdicts, snr=None) -> str:
    """What the fingerprint check saw, for the row."""
    bits = []
    if snr is not None and np.isfinite(snr):
        bits.append(f'SNR {snr:.0f}')
    for c, mu, e, v in verdicts or ():
        if v.startswith('contradicted'):
            bits.append(f'{c:.2f} ppm {mu} expected SNR {e:.0f}, {v}')
        elif v == 'present':
            bits.append(f'{c:.2f} ppm {mu} present')
    return (' ' + '; '.join(bits) + '.') if bits else ''


def _censored_row(cand, ppm, intensities, *, included, standard,
                  standard_conc, noise_sigma, linewidth_ppm, drift_ppm,
                  local_baseline, external=False, dilution=None,
                  correction=None, excluded_regions=()) -> dict:
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

    # Where the analyte may have drifted to, tried one position at a time and
    # the best kept. Widening the window to cover the range instead would pay
    # for the whole range in noise, while a signal only ever occupies one
    # window's worth of it — on the reference study that made every limit
    # twice what the spectrum supports.
    idx = list(included if included is not None else range(arr.shape[1]))
    best_area, areas, best_off = float('-inf'), {}, 0.0
    for off in drift_offsets(drift_ppm, linewidth_ppm):
        here = {}
        for s_i in idx:
            if 0 <= s_i < arr.shape[1]:
                # Signed: the limit is built from noise that goes both
                # ways, and the quantified area's clip at the baseline
                # would put every noise window above it.
                a = integrate_sample(ppm, arr[:, s_i], lo + off, hi + off,
                                     local_baseline=True, clip=False)
                if a is not None:
                    here[int(s_i)] = float(a)
        if not here:
            continue
        med = float(np.median(list(here.values())))
        if med > best_area:
            best_area, areas, best_off = med, here, off
    if not areas:
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    lo, hi = lo + best_off, hi + best_off
    area = best_area

    # An excluded region holds no analyte by construction: what is measured
    # there is the suppressed solvent. It gets neither a limit nor a verdict
    # about signal — a limit would report the solvent as the analyte's bound,
    # and an "unresolved" would report it as a buried analyte.
    if window_is_excluded(lo, hi, excluded_regions):
        row = _rejected_row(mid, EXCLUDED_REGION)
        row.update({
            'ppm': float((lo + hi) / 2.0),
            'status': STATUS_CENSORED[EXCLUDED_REGION],
            '_status_reasons': [censor_text(EXCLUDED_REGION, float('nan'),
                                            float('nan'))],
            '_censored': EXCLUDED_REGION,
            '_multiplet_window': [float(lo), float(hi)],
            '_hits': [cand.payload] if cand.payload is not None else [],
        })
        return row

    # The blank's noise, as everywhere: a limit is a property of the method,
    # not of whichever neighbours this analyte happens to have. How crowded
    # the region is is a separate statement -- see local_noise_sigma.
    lod_a, loq_a = detection_limits(noise_sigma, dppm, n_pts)
    _around = local_noise_sigma(ppm, arr, lo, hi,
                                linewidth_ppm=linewidth_ppm,
                                global_sigma=noise_sigma)
    interference = (float(_around) / float(noise_sigma)
                    if _around and noise_sigma else float('nan'))
    verdict = censor(area, lod_a, loq_a)

    nh = signal_nh(ev.get('protons'))
    lod_mm, loq_mm = (_as_concentration(v, nh, standard, standard_conc,
                                        external=external,
                                        dilution=dilution,
                                        correction=correction)
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
        '_noise_sigma': float(noise_sigma) if noise_sigma else float('nan'),
        # How much louder than the blank this window's surroundings are.
        # Not a limit: interference, which costs accuracy, not detectability.
        '_interference': interference,
        '_multiplet_window': [float(lo), float(hi)],
        '_nh': float(nh),
        '_areas_per_sample': areas,
        '_ann_score': float(ev.get('score') or 0.0),
        '_hits': [cand.payload] if cand.payload is not None else [],
        '_is_singlet': ev.get('multiplicity') == 's',
    })
    return row


def _as_concentration(area, nh, standard, standard_conc, *, external=False,
                      dilution=None, correction=None) -> float:
    """One area in the same units the table's concentrations are in.

    Through ``concentrations``, the same function every quantified row uses,
    rather than a second expression that says the same thing. It said almost
    the same thing: it left out the dilution and the correction factor, so a
    study that applied either had its limits on a different scale from the
    numbers they were meant to be compared with -- and a limit that cannot be
    compared is not a limit.

    The study's typical factor is the median across samples: a bound describes
    the run, not one injection of it.
    """
    if not np.isfinite(area):
        return float('nan')
    per_sample = concentrations({0: float(area)}, nh, standard=(
        {0: float(np.median([v for v in standard.values()
                             if v and np.isfinite(v) and v > 0]))}
        if standard else None),
        standard_conc=standard_conc, external=external,
        dilution={0: _median_factor(dilution)},
        correction={0: _median_factor(correction)})
    return float(per_sample.get(0, float('nan')))


def _median_factor(factors) -> float:
    v = [float(x) for x in (factors or {}).values()
         if x is not None and np.isfinite(x) and x > 0]
    return float(np.median(v)) if v else 1.0


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


def _row(mid, info, conc, by_group, good, level, dots, pqn_factors,
         standard=None, standard_conc=0.0, external=False, dilution=None,
         correction=None) -> dict:
    lo, hi = info['window']
    sig = info['signal']
    nh_ = info['nh']
    lod_mm, loq_mm = (_as_concentration(info.get(k), nh_, standard,
                                        standard_conc, external=external,
                                        dilution=dilution,
                                        correction=correction)
                      for k in ('lod_area', 'loq_area'))
    return {
        'hmdb_id': mid,
        'metabolite': mid,
        'ppm': float((lo + hi) / 2.0),
        'n_peaks': len(sig.line_ranks) or 1,
        'raw_area': float(np.mean(list(info['areas'].values()))),
        'corr_area': float(np.mean(list(info['areas'].values()))),
        'conc_mm': float(np.mean(good)) if good else float('nan'),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        # The SNR shown is the signal's on the group the figure draws
        # (best cosine, then highest SNR); the per-sample window SNR the
        # limits are built on stays in _snr_per_sample.
        'snr': (info['group_evidence'][info['best_group']]['snr']
                if info.get('best_group') else
                representative_snr(info.get('snr'))),
        '_snr_group': info.get('best_group'),
        '_snr_by_group': {g: e['snr'] for g, e in
                          (info.get('group_evidence') or {}).items()},
        '_cosine_by_group': {g: e['cosine'] for g, e in
                             (info.get('group_evidence') or {}).items()},
        'status': dots,
        'vip_score': float('nan'),
        '_confidence': level,
        '_integ_mode': 'Shape' if info.get('shape') else 'Multiplet',
        # What the shape was, for a report to draw the fit it quantified
        '_shape_fit': ((sig.evidence or {}).get('shape_fit')
                       if info.get('shape') else None),
        '_shape_explained': info.get('shape_explained'),
        '_deconv': info.get('deconv'),
        '_by_residual': info.get('by_residual'),
        '_residual_trials': info.get('residual_trials'),
        '_noise_sigma': info.get('noise_sigma'),
        '_not_quantified': not good,
        '_peak': None,
        '_peak_rank': sig.peak_rank,
        '_display_peak_rank': sig.peak_rank,
        '_multiplet_window': [float(lo), float(hi)],
        # The pieces of that window the area was summed over, and the
        # foreign peaks between the lines that kept them apart: a triplet
        # measured on its three lines around somebody's doublet says so.
        '_integ_segments': [[float(a), float(b)]
                            for a, b in (info.get('segments') or ())],
        '_foreign_inside': [float(v) for v in (info.get('foreign_inside')
                                               or ())],
        # How much of the window the trace ran under the baseline: what the
        # integration counted as nothing (quant_integration.integrate_sample).
        '_below_baseline_share': info.get('below_baseline_share'),
        # Where the local baseline was read -- beyond the cluster of signals
        # the window sits in -- as [lo_ppm, hi_ppm, half-width], so the plot
        # and the PDF draw the line the area was measured against.
        '_baseline_anchors': ([float(v) for v in info['baseline_anchors']]
                              if info.get('baseline_anchors') else None),
        # How much louder than the blank this window's surroundings read.
        # Not a limit: interference costs accuracy, not detectability.
        '_interference': info.get('interference'),
        # Present when the amount came from the joint fit of an overlapping
        # group: who it was fitted with, how many unknown peaks the fit made
        # room for, what share of the region nothing explains, and the
        # amount integration had given, so the change can be undone.
        '_joint_fit': info.get('joint_fit'),
        # The shape the fit gave this analyte and the baseline under it,
        # per group, when the amount came from a fit: what the report draws.
        '_fit_curve': info.get('fit_curve'),
        # What share of this window's raw area the project's baseline model
        # removed. Nothing to remove and it is zero; a third and the number
        # is the model's as much as the analyte's.
        '_broad_removed': info.get('broad_removed'),
        # Where the signal's own lines are, so a plot can mark who is who --
        # and where the pattern's missing lines were expected, so a signal
        # scaled from part of its multiplet shows which part.
        '_line_ppms': [float(v) for v in (info.get('lines') or ())],
        '_missing_lines': [float(v) for v in (info.get('missing_lines') or ())],
        '_pattern_share': info.get('pattern_share'),
        # Every line the reference lists for this signal, at the centre the
        # annotation found -- the pattern the letter promises, drawn beside
        # the lines actually integrated.
        '_reference_lines': [float(v) for v in
                             ((sig.evidence or {}).get('reference_ppms') or ())],
        '_nh': float(info['nh']),
        '_areas_per_sample': info['areas'],
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_pqn_factors': pqn_factors,
        '_snr_per_sample': info.get('snr', {}),
        '_detected_per_sample': info.get('detected', {}),
        '_peak_align_shifts': info.get('shifts') or None,
        '_blk_mean_area': info.get('blank_area', float('nan')),
        '_is_singlet': (sig.evidence or {}).get('multiplicity') == 's',
        '_multiplicity': (sig.evidence or {}).get('multiplicity'),
        '_ann_score': float((sig.evidence or {}).get('score') or 0.0),
        # Every quantified analyte carries its own limits now. Without them
        # nobody could tell a number ten times the limit from one at 1.1x, and
        # the confidence had no way to say which it was.
        '_lod_mm': lod_mm,
        '_loq_mm': loq_mm,
        '_strength': info.get('strength'),
        '_captured_fraction': info.get('captured'),
        '_neighbour_share': info.get('neighbour_share', 0.0),
        '_shared_with': list(info.get('shared_with') or ()),
        # Why the identity is worth what it is worth, in one sentence the
        # table's tooltip shows beside the dots.
        '_identity_why': info.get('identity_why'),
        '_identity_level': info.get('identity_level'),
        # Each axis of the confidence, so a row can say which one held it
        # back: the shape of the signal measured, the identity the rest of
        # the listing gives it, the size of the area against the
        # quantification limit, and whether the window is its own.
        '_confidence_axes': info.get('confidence_axes'),
        # How much of a shared window the split left this analyte: 1.0 when
        # the other claimants took nothing, so the number rests on nobody.
        '_share_kept': info.get('share_kept'),
        # What the rest of this analyte's listing does at the amount just
        # measured (fingerprint_holds, ADR-012 at the measured amount).
        # The gate above computes it for every winner and used to keep it
        # only when it failed; it is the strongest thing the run knows
        # about an identity, and the confidence and the analyst both read
        # it: 'present' is the analyte's other signals actually standing
        # where this amount says they must.
        '_fingerprint_at_amount': list(info.get('fingerprint_at_amount') or ()),
        '_unshared_reason': info.get('unshared_reason'),
        '_fit_refused': bool(info.get('fit_refused')),
        '_status_reasons': (_residual_reason(info)
                            + (_deconv_reason(info) if info.get('deconv')
                               else _shared_reason(info))
                            + _pattern_reason(info)
                            + _segments_reason(info)
                            + _baseline_reason(info)
                            + _broad_reason(info)
                            + _interference_reason(info)
                            + _joint_fit_reason(info)
                            + _position_reason(info)
                            + _shape_reason(info)
                            + _weak_reason(info, loq_mm)),
        '_hits': [sig.payload] if sig.payload is not None else [],
        # Whether this number came out of a window held with other analytes,
        # and how much of its own signal that window held. The overlap
        # correction the old stage promised, from the engine that now does it.
        '_used_j_resolve': bool(info.get('shared_with')),
        '_j_resolve_frac': (float(info.get('captured') or 1.0)
                            if info.get('shared_with') else 1.0),
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }


def _found_ranks(cand) -> list:
    return sorted(cand.line_ranks or ()) or (
        [cand.peak_rank] if cand.peak_rank is not None else [])


def _found_lines(cand, peaks_by_rank) -> list:
    """The peaks detection found for a signal, in ppm; empty when it found
    none."""
    return [float(peaks_by_rank[r].get('ppm', 0.0))
            for r in _found_ranks(cand) if r in peaks_by_rank]


def _residual_reason(info) -> list:
    """Said of an analyte identified by what another analyte's window left
    unexplained, and of a window that took such a signal in."""
    owner = info.get('by_residual')
    if owner == 'spectrum':
        return [f"Identified on a free apex: a detected peak no accepted "
                f"analyte holds, at which this analyte's reference shape, "
                f"scaled to the apex, fits under the trace over its whole "
                f"width and accounts for "
                f"{100 * (1 - float(info.get('residual_after') or 0)):.0f}% "
                f"of its region, with the rest of its fingerprint able to "
                f"carry the amount. Its lines were not confirmed one by "
                f"one: a weak identification, to be checked."]
    if owner:
        return [f"Identified by the remainder of {owner}'s window: the "
                f"accepted shapes left {100 * float(info.get('residual_before') or 0):.0f}% "
                f"of it unexplained, this analyte's reference shape accounts "
                f"for it (remainder {100 * float(info.get('residual_after') or 0):.0f}% "
                f"with it), and its shape can be told from theirs. Its "
                f"lines were not confirmed one by one: a weak identification, "
                f"to be checked."]
    d = info.get('deconv') or {}
    if d.get('residual'):
        return [f"{', '.join(d['residual'])} was taken into this window on "
                f"the remainder the accepted shapes left; its share is what "
                f"its reference shape accounts for."]
    trials = info.get('residual_trials') or []
    refused = [t['label'] for t in trials if t.get('verdict') != 'taken']
    if refused and not d.get('residual'):
        return [f"The remainder was offered to {', '.join(refused)}; none "
                f"met the three conditions (remainder under "
                f"{100 * RESIDUAL_ACCEPT:.0f}%, at least "
                f"{100 * RESIDUAL_EXPLAINED:.0f}% of it explained, shape "
                f"separable)."]
    return []


def _deconv_reason(info) -> list:
    """Said of a window resolved by shapes: with whom, and what the shapes
    left unexplained."""
    d = info.get('deconv')
    if not d:
        return []
    mid = getattr(info.get('signal'), 'met_id', None)
    others = [c['label'] for c in d.get('components', ())
              if c.get('met_id') != mid]
    rem = d.get('remainder') or {}
    med = float(np.median(list(rem.values()))) if rem else 0.0
    out = [f"Shared window resolved by shape with {', '.join(others)}: this "
           f"area is what this analyte's own shape accounts for."]
    if d.get('listed'):
        out.append(f"{', '.join(d['listed'])}: the reference lists it "
                   f"inside this window; its analyte is quantified elsewhere, "
                   f"and its share here is what that measurement predicts, "
                   f"subtracted (kept only because it fits under the trace).")
    if [f for f in (d.get('fixed') or ()) if f not in (d.get('listed') or ())]:
        out.append(f"{', '.join(f for f in d['fixed'] if f not in (d.get('listed') or ()))}: "
                   f"a shape this signal's own shape makes, so its share "
                   f"was not fitted but fixed at what its own clean signal "
                   f"predicts, and subtracted.")
    placed = [c['label'] for c in d.get('components', ())
              if c.get('placed_at_expected')]
    if placed:
        out.append(f"{', '.join(placed)} was placed where annotation "
                   f"expects it: detection had given it this signal's own "
                   f"line.")
    if d.get('skipped'):
        out.append(f"{', '.join(d['skipped'])} is listed here too but "
                   f"took no share: not placed by detection, a shape this "
                   f"signal's own shape makes, or left out by the analyst.")
    if med >= REMAINDER_NOTE:
        out.append(f'The accepted shapes leave {100 * med:.0f}% of the '
                   f'window unexplained: something else is here.')
    return out


def _pattern_reason(info) -> list:
    """Said of a multiplet measured on part of its lines: which were
    missing, and that the area was scaled by the pattern to the whole."""
    missing = [float(v) for v in (info.get('missing_lines') or ())]
    if not missing:
        return []
    share = float(info.get('pattern_share') or 0.0)
    n_all = len(info.get('pattern_lines') or ()) or (
        len(info.get('lines') or ()) + len(missing))
    return [f'Partial multiplet: {n_all - len(missing)} of {n_all} lines were '
            f'placed by detection, holding {100 * share:.0f}% of the pattern; '
            f'the missing lines were expected at '
            f'{", ".join("%.4f" % v for v in missing)} ppm. The area is the '
            f'placed lines scaled by the pattern to the whole multiplet, '
            f'so it assumes the missing lines are there, under something.']


# Above this share of the window under the baseline, the row says so: a
# window on a signal has a few noise points under its baseline, not a
# quarter of its length.
BASELINE_SILENCED_SHARE = 0.25


def rests_on_shared_window(info, material) -> bool:
    """Whether the row's number rests on how a shared window was divided,
    which costs a dot: the window could not be split, or other analytes
    took a material share of it.

    An amount the joint fit gave does not: the fit replaced that
    measurement, overlap and all. Charging the dot anyway took alanine and
    caffeine off three dots on the tube of 100 the moment the window step
    saw one neighbour more, with the number in the table untouched by it
    (three seeds: three-dot reals 12/4/3 -> 18/12/9, no stranger).
    """
    if info.get('joint_fit'):
        return False
    return (info.get('unshared_reason') is not None
            or (bool(info.get('shared_with')) and bool(material)))


def joint_fit_refusal(rec) -> str | None:
    """Why the joint fit's amount may not replace the integrated one, or
    None when it may.

    Two ways a fit measures something other than the analyte. It left
    signals at the edge of how far it may move them (``at_limit``): the
    reference puts the shape further from the trace's peak than the fit is
    allowed to follow, so the amplitude is that of a shape standing beside
    the signal. Or the signal the fit centres on is not the one the
    election found (``focus_matches_window``). On a cell extract 25 of 46
    fitted rows were held at the edge and 15 fitted another signal; drawn,
    maltose's fitted triplet stood between the trace's peaks.
    """
    if not rec:
        return None
    why = []
    if rec.get('at_limit'):
        why.append(f"{len(rec['at_limit'])} of {rec.get('n_signals')} of "
                   f"its signals stood at the edge of how far the fit may "
                   f"move them (listed at "
                   + ', '.join(f'{c:.3f}' for c in rec['at_limit'])
                   + ' ppm): the shape is not where the peak is')
    if rec.get('focus_matches_window') is False:
        lo, hi = rec.get('focus') or (0.0, 0.0)
        why.append(f'the fit centres on a signal at {lo:.3f}-{hi:.3f} ppm, '
                   f'not on the lines the election found')
    return '; '.join(why) or None


def _as_candidate(row, why, by_group) -> None:
    """Turn a row whose joint fit was refused into a candidate: not
    quantified, not counted as identified, with the integral of its window
    kept as an upper bound (that window holds its neighbours' area).

    On the bench the rows of this kind were one real analyte in five: 18
    real against 72 not in the tube, over three seeds of the 100-compound
    tube. An identification that wrong is a suggestion, not a result."""
    bound = {g: float(np.median(v)) for g, v in (by_group or {}).items() if v}
    row['_upper_bound_mm'] = bound
    row['_fit_refused'] = True
    row['_not_quantified'] = True
    row['_rejection'] = FIT_MISPLACED
    row['_confidence'] = 0
    row['status'] = STATUS_NONE
    row['conc_mm'] = float('nan')
    row['cv_by_group'] = {}
    text = REJECTION_TEXT[FIT_MISPLACED] + f' Here {why}.'
    if bound:
        text += (' Upper bound from integrating its window: ' + ', '.join(
            f'{g} {v:.4g} mM' for g, v in bound.items()) + '.')
    row['_status_reasons'] = [text] + list(row.get('_status_reasons') or ())


def _joint_fit_reason(info) -> list:
    """Said of an amount the joint fit measured."""
    jf = info.get('joint_fit')
    if not jf:
        return []
    others = max(len(jf.get('group') or ()) - 1, 0)
    free = int(jf.get('free') or 0)
    res = jf.get('residual')
    tail = (f" {100 * float(res):.0f}% of the region is explained by nothing."
            if res is not None and np.isfinite(res) else '')
    return [f"Measured by fitting its shape together with {others} other "
            f"analyte(s) and {free} unassigned peak(s) whose signals overlap "
            f"it, instead of integrating a window that holds theirs too. The "
            f"integrated amount is kept on the row.{tail}"]


def _interference_reason(info) -> list:
    """Said of a window whose surroundings are not noise any more."""
    r = info.get('interference')
    if r is None or not np.isfinite(r) or float(r) < INTERFERENCE_NOTE:
        return []
    return [f"The band flanking this window reads {float(r):.0f}x the "
            f"scatter of a signal-free region: what surrounds it is "
            f"neighbouring signal, not noise. The detection and "
            f"quantification limits are the method's and do not change "
            f"with this, but how much of the area is really this "
            f"analyte's does -- check the region."]


def _broad_reason(info) -> list:
    """Said of a window the baseline model took a large share of."""
    f = info.get('broad_removed')
    if f is None or not np.isfinite(f) or float(f) < BROAD_REMOVED_NOTE:
        return []
    return [f"The baseline model removed {100 * float(f):.0f}% of this "
            f"window's area before it was integrated. What is reported is "
            f"what the model left: check that there is a broad component "
            f"here to remove, because on a flat baseline the model takes "
            f"signal instead."]


def _baseline_reason(info) -> list:
    """Said of a signal whose window ran under its local baseline over a
    good part of its length: that part counted nothing, and a window
    standing that much under its own baseline is a window on a flank or a
    dip, not on the signal."""
    share = info.get('below_baseline_share')
    if share is None or float(share) < BASELINE_SILENCED_SHARE:
        return []
    return ['the trace runs under the local baseline on %d%% of the window; '
            'that part counted nothing (only area above the baseline is '
            'integrated)' % round(100 * float(share))]


def _is_complex(cand) -> bool:
    """A complex multiplet: judged as a measured pattern (``m*``), with or
    without a shape fitted to the trace."""
    ev = cand.evidence or {}
    return bool(ev.get('shape_fit')) or normalize_mult(
        ev.get('multiplicity')) == PATTERN_MULT


def _segments_reason(info) -> list:
    """Said of a signal whose window holds a foreign peak between its lines:
    which peaks, and on how many pieces the lines were then integrated."""
    foreign = [float(v) for v in (info.get('foreign_inside') or ())]
    if not foreign or info.get('deconv'):
        return []
    segs = info.get('segments') or ()
    where = ', '.join('%.4f' % v for v in foreign)
    what = ('%d foreign peak%s at %s ppm %s between this signal\'s lines'
            % (len(foreign), 's' if len(foreign) > 1 else '', where,
               'lie' if len(foreign) > 1 else 'lies'))
    if len(segs) > 1:
        return ['%s; integrated on the lines alone, in %d pieces, the '
                'gap left out' % (what, len(segs))]
    return ['%s, too close to its lines to cut around; integrated whole'
            % what]


def _shape_reason(info) -> list:
    """Said of a shape quantified in a region it does not hold alone: the
    copy that fits under the trace is bounded by the least overlapped apex,
    so the number is an upper bound there."""
    if not info.get('shape'):
        return []
    ex = info.get('shape_explained')
    if ex is None:
        return ['Quantified from its fitted shape; no peak claimed.']
    if float(ex) >= 0.9:
        return [f'Quantified from its fitted shape, which explains '
                f'{100 * float(ex):.0f}% of its region.']
    return [f'Upper bound: quantified from its fitted shape in a region it '
            f'shares -- the shape explains {100 * float(ex):.0f}% of it.']


def _weak_reason(info, loq_mm) -> list:
    """Said when the area does not clear the quantification limit by the
    agreed factor. The identity is not in question here — only the precision.
    """
    if info.get('strength', 3) >= 3:
        return []
    return [f'Weak signal: the measured area does not reach '
            f'{STRONG_ABOVE_LOQ:.0f}x the limit of quantification '
            f'({loq_mm:.4g} mM). The analyte was identified and won its peak; '
            f'it is the quantity that is small, and the number less precise.']


def _shared_reason(info) -> list:
    """What the reader needs to know about an area that was shared."""
    with_ = list(info.get('shared_with') or ())
    if not with_:
        why = info.get('unshared_reason')
        if not why:
            return []
        return [f'Shared window, not split: this area is the whole of a '
                f'window other analytes also sit in, so it is an upper bound '
                f'on what belongs to this one. It could not be split because '
                f'{why}.']
    return [f'Shared window: the signal is held with {", ".join(with_)}. '
            f'This area is what remains once their own measurements account '
            f'for what they contribute here, so it depends on those being '
            f'right and on there being no other analyte in the window.']


def _rejected_row(mid, reason, shared_reason=None, *, beaten=None,
                  names=None, detail='') -> dict:
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
        '_status_reasons': ([REJECTION_TEXT.get(reason, str(reason)) + detail]
                            + ([f'Its share of the shared window could not be '
                                f'resolved: {shared_reason}']
                               if shared_reason else [])
                            + [f'Lost peak #{r} to '
                               f'{(names or {}).get(w, w)} on {how}.'
                               for r, w, how in (beaten or ())]),
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
        '_detected_per_sample': {},
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
