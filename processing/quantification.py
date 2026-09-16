# processing/quantification.py
"""Which signal quantifies which metabolite, decided once and auditable.

The old engine settled this inside a 2748-line method, and the same question
was asked in four places that disagreed: a greedy pass, a rescue pass, an APQ
fallback and a fill-in loop. Peaks were handed out by anchor while the lines
underneath were shared, ambiguous contests still elected a winner by
tie-break, and an analyte withdrawn before competing was told it had been
outscored.

Here the election is one function over plain data. It decides nothing about
chemistry that annotation has not already decided: the ranking is
``annotation_scoring.candidate_key``, the same one the Annotation Results
window shows, so what the analyst sees and what gets quantified cannot drift
apart. What this adds is only the allocation — a peak, and the lines under it,
belong to one metabolite.

Nothing is integrated here. The election is separable from the measurement and
must be, because every defect the audit found lived in the seam between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from processing.annotation_scoring import (
    PPM_BAND_PPM, candidate_key, is_competitor)

# Why a metabolite ended without a number. The distinction is the actionable
# part: a withdrawn multiplet returns if detection resolves its lines, an
# ambiguous rank returns the moment the analyst marks one candidate, and a
# metabolite outscored by a rival returns only if the rival is wrong.
LOST_COMPETITION = 'lost_competition'
UNVERIFIED_MULTIPLET = 'unverified_multiplet'
AMBIGUOUS_RANK = 'ambiguous_rank'
NO_ELIGIBLE_SIGNAL = 'no_eligible_signal'
NO_MEASURABLE_WINDOW = 'no_measurable_window'
NOT_DETECTED = 'not_detected'
SHAPES_OFF = 'shapes_off'

REJECTION_TEXT = {
    LOST_COMPETITION:
        'Not quantified: lost the competition — every peak this analyte claims '
        'was won by another with better evidence.',
    UNVERIFIED_MULTIPLET:
        'Not quantified: unverified multiplet — not all the expected lines '
        'were found, so J was never measured. It competed for no peak.',
    AMBIGUOUS_RANK:
        'Not quantified: ambiguous identity — the candidates for this peak tie '
        'on every piece of evidence available. Mark one Good to resolve it.',
    NO_ELIGIBLE_SIGNAL:
        'Not quantified: no eligible signal — every one was marked Bad or has '
        'no detected peak.',
    SHAPES_OFF:
        'Not quantified: complex multiplet (m*) — its lines were never found '
        'one by one, so it holds no peak to integrate, and shape fitting is '
        'off in Quantification. Tick "Quantify complex multiplets by shape '
        'fit" to measure it from its fitted shape.',
    NO_MEASURABLE_WINDOW:
        'Not quantified: no measurable window — the signal won its contest but '
        'no detected line bounds a region to integrate. Nothing is integrated '
        'where the database predicts a peak, only where detection found one.',
    NOT_DETECTED:
        'Not quantified: nothing detected — every signal the database predicts '
        'for this analyte was searched for and none was found. There is no '
        'peak to measure and none to compete with.',
}


@dataclass(frozen=True)
class Candidate:
    """One metabolite's claim on one peak, as annotation left it.

    ``line_ranks`` are the detected peaks matched to this signal's predicted
    lines. They matter as much as ``peak_rank``: two metabolites anchored on
    different peaks were found integrating the same lines, which is the same
    signal counted twice however different the anchors look.
    """
    met_id: str
    peak_rank: int | None
    line_ranks: frozenset = frozenset()
    evidence: dict = field(default_factory=dict)
    payload: object = None


# The criteria of ``candidate_key``, in the words the analyst reads.
_CRITERION_TEXT = {
    'manual': 'a manual Good mark',
    'tier': 'evidence tier (confirmed multiplet > unconfirmed > singlet)',
    'dj': 'delta-J',
    'ppm': 'delta-ppm',
    'quality': 'annotation quality',
    'intensity': 'line intensities',
    'corroboration': 'STOCSY corroboration',
    'cosine': 'fingerprint cosine',
    'score': 'the annotation score',
}


def _decided_on(loser_key, winner_key) -> str:
    """The first criterion on which the winner's key beats the loser's."""
    for name in loser_key._fields:
        a, b = getattr(loser_key, name), getattr(winner_key, name)
        if a != b:
            return _CRITERION_TEXT.get(name, name)
    return 'a tie broken by metabolite id'


@dataclass(frozen=True)
class Election:
    winners: dict          # met_id -> Candidate
    rejected: dict         # met_id -> reason constant
    claimed_peaks: frozenset
    # met_id -> [(peak_rank, winner_met_id, criterion)] for every signal the
    # metabolite stood on and lost. "Lost the competition" says nothing an
    # analyst can act on; "lost 1.33 ppm to N-acetyl-L-alanine on a manual
    # Good mark" does.
    beaten: dict = field(default_factory=dict)


def _sort_key(cand, *, dj_gold_hz, band_ppm=None):
    return candidate_key(cand.evidence, dj_gold_hz=dj_gold_hz,
                         band_ppm=PPM_BAND_PPM if band_ppm is None else band_ppm)


def _detected(c) -> bool:
    """Whether the spectrum shows this signal: a peak, lines, or a shape
    fitted to the trace (ADR-013). A shape claims no peak -- its lines were
    never found one by one -- but it is not an absence either: the fit
    stood over detection's floor where the analyte's shape is."""
    return bool(c.peak_rank is not None or c.line_ranks
                or (c.evidence or {}).get('shape_fit'))


def elect(candidates: Iterable[Candidate], *, dj_gold_hz: float,
          ambiguous_ranks: Sequence = (), band_ppm=None) -> Election:
    """Give every peak to one metabolite, and say why the others got none.

    Order of business, and each step exists because the old engine skipped it:

    1. A multiplet whose expected lines were not all found never verified its
       multiplicity, so it does not compete. ``is_competitor`` decides that,
       the same rule annotation applies.
    2. A rank whose candidates tie on every criterion has no winner. Electing
       one by tie-break reports a decision the data does not support.
    3. The rest are ranked by ``candidate_key`` and taken in order. A candidate
       is skipped when its anchor OR any of its lines is already claimed.
    4. A metabolite keeps competing on its other signals after losing one, so
       losing a peak costs a signal, never the analyte.
    5. And once it has won, its other verified signals stay its own: a peak a
       present analyte holds elsewhere is not free for a rival to take --
       unless the rival's evidence on that peak ranks higher, in which case
       the rival already has it by the time the winner's claim comes round.

    Ties in the key are broken by ``met_id`` so two runs over the same data
    give the same answer; a coin toss here would move concentrations between
    runs with nothing in the data changing.
    """
    amb = set(ambiguous_ranks or ())
    cands = list(candidates or ())
    seen_mids = {c.met_id for c in cands}

    eligible, rejected = [], {}
    for c in cands:
        if c.peak_rank is not None and c.peak_rank in amb:
            rejected.setdefault(c.met_id, AMBIGUOUS_RANK)
            continue
        if not is_competitor(c.evidence):
            rejected.setdefault(c.met_id, UNVERIFIED_MULTIPLET)
            continue
        eligible.append(c)

    # Detected first, always. A signal matched to no peak cannot be judged:
    # never having been matched, its Delta-ppm is zero by default, which is the
    # best value the ranking can see -- absence read as perfection. On the
    # reference study 11 analytes won on a signal with no peak while each held
    # one with a peak, among them L-histidine, L-carnosine, ATP and
    # hypoxanthine, and the peak they held went unmeasured.
    #
    # It is an ordering and not a bar. A metabolite whose every signal is
    # undetected still wins on one of them and is reported as a bound, which is
    # a real measurement of an upper limit; what it must never do is take
    # precedence over a peak the spectrum actually shows.
    eligible.sort(key=lambda c: (0 if _detected(c) else 1,
                                 _sort_key(c, dj_gold_hz=dj_gold_hz,
                                           band_ppm=band_ppm), str(c.met_id)))

    # Which metabolites have a signal the spectrum actually shows. One that
    # does must never fall back to one that it does not: losing a contested
    # peak is losing, and reporting it instead on a signal nobody saw turns a
    # defeat into a measurement.
    has_detected = {c.met_id for c in eligible if _detected(c)}

    winners: dict = {}
    claimed: set = set()
    holder: dict = {}          # rank -> the Candidate that holds it
    beaten: dict = {}

    def _record_loss(c, rank):
        w = holder.get(rank)
        if w is None or w.met_id == c.met_id:
            return
        beaten.setdefault(c.met_id, []).append((
            rank, w.met_id,
            _decided_on(_sort_key(c, dj_gold_hz=dj_gold_hz, band_ppm=band_ppm),
                        _sort_key(w, dj_gold_hz=dj_gold_hz, band_ppm=band_ppm))))

    for c in eligible:
        if not _detected(c) and c.met_id in has_detected:
            continue
        if c.peak_rank is not None and c.peak_rank in claimed:
            if c.met_id not in winners:
                _record_loss(c, c.peak_rank)
            continue
        if c.line_ranks & claimed:
            if c.met_id not in winners:
                _record_loss(c, sorted(c.line_ranks & claimed)[0])
            continue
        if c.met_id in winners:
            # A winner's OTHER verified signals are its own too. It stops
            # competing once it has a window, but the peaks it holds elsewhere
            # do not become free for a rival to take. On the reference study
            # lactate was elected on its quartet at 4.11 ppm and left the
            # election; its methyl doublet at 1.33 -- the strongest signal in
            # the region -- then went uncontested to N-acetyl-L-alanine,
            # reported at 4.5 mM in plasma off the same molecule that stood at
            # 2.3 mM four rows down under its own name.
            #
            # Taken in the same order as everything else, so a rival with
            # better evidence on that peak has already had it: the claim is
            # for the peaks nobody better wanted, not a veto.
            if c.peak_rank is None and not c.line_ranks:
                continue
            for r in ({c.peak_rank} if c.peak_rank is not None else set()) \
                    | set(c.line_ranks):
                claimed.add(r)
                holder[r] = c
            continue
        winners[c.met_id] = c
        for r in ({c.peak_rank} if c.peak_rank is not None else set()) \
                | set(c.line_ranks):
            claimed.add(r)
            holder[r] = c

    for mid in seen_mids:
        if mid in winners:
            rejected.pop(mid, None)
            continue
        if any(c.met_id == mid for c in eligible):
            # It stood and was beaten, whatever else was said about its other
            # signals — telling it otherwise sends the analyst to the wrong one.
            rejected[mid] = LOST_COMPETITION
        else:
            rejected.setdefault(mid, NO_ELIGIBLE_SIGNAL)

    return Election(winners=winners, rejected=rejected,
                    claimed_peaks=frozenset(claimed), beaten=beaten)


def eligible_metabolite_ids(metabolites, *, anchor_met_ids=(),
                            manual_good_ids=()) -> set:
    """Which metabolites may be quantified at all, before any peak is contested.

    Not the same question as the annotation tree's row colour, which aggregates
    every signal to say what the metabolite looks like overall. Here one usable
    signal is enough: a metabolite that loses a contested peak can still
    quantify on another, and judging it on its worst signal would drop analytes
    that are perfectly measurable somewhere else.

    Four ways in, and only four:

    * an anchor whose anchoring peak is itself 'good' — a stale ``auto_anchor``
      flag on an incomplete multiplet must not carry an unconfirmed J into a
      concentration;
    * any signal the annotation search marked 'good';
    * any signal with a DETECTED PEAK that the search did not mark 'bad';
    * a manual Good mark, which is the analyst overriding the search and is
      the only thing that should be able to.

    The third exists because the first two ask a harder question than
    eligibility needs. 'Good' is what the search says when it is confident
    about identity, and a signal it left undecided is not thereby absent — it
    is a peak that was found and not adjudicated. Requiring 'good' meant a
    metabolite whose real signals were all undecided could only get in through
    a signal that was NOT undecided, and until 2026-09-08 that was routinely a
    singlet with no detected peak at all: with no peak there are no rivals and
    an empty window, so "alone and uncontested" promoted it to Good and made it
    an anchor. See ``processing.annotation_quality.promote_singlets``.

    Closing that hole alone took the reference study from 92 eligible
    metabolites to 51, and 26 of the 41 lost had a signal with a detected peak
    that was not marked bad — adenosine, ATP, AMP, L-cystine, dopamine,
    pyridoxine and thiamine among them, all of which were being quantified on
    real windows. The defect was calling a thing identified that was never
    seen, not the existence of those analytes.

    Eligibility is not a verdict on identity. Which of a metabolite's signals
    is measured is settled by the election, and how much the result is worth is
    settled by the evidence that travels with it.
    """
    anchors = {str(m) for m in (anchor_met_ids or ())}
    out = {str(m) for m in (manual_good_ids or ())}
    for met in metabolites or ():
        mid = str(met.get('id', ''))
        if not mid:
            continue
        mps = met.get('matched_peaks', []) or ()
        if mid in anchors or any(mp.get('annotation_quality') == 'good'
                                 and mp.get('auto_anchor') for mp in mps):
            out.add(mid)
            continue
        if any(mp.get('annotation_quality') == 'good' for mp in mps):
            out.add(mid)
            continue
        if any(mp.get('peak_rank') is not None
               and mp.get('annotation_quality') != 'bad' for mp in mps):
            out.add(mid)
    return out
