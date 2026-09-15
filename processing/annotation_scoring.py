# processing/annotation_scoring.py
"""
Pure scoring core for metabolite annotation.

Extracted from ``gui/widgets/annotation_results_window.py`` so the scoring
model can be unit-tested without a Tk window and so the decision logic lives
in ``processing/`` rather than in the view layer.

The model mirrors how a spectrum is read by hand:

  1. Snap the DB multiplet's predicted lines onto detected peaks.
  2. Count how many expected lines were actually found.
  3. Measure J from the confirmed line spacings and compare with the DB value.
  4. Score = agreement in chemical shift (Δppm) and coupling constant (ΔJ).

Singlets carry no J evidence, so they are scored on Δppm alone and capped
below the Good band — a lone singlet is never sufficient identification.

References
----------
Confidence bands follow the Metabolomics Standards Initiative reporting
levels: Sumner et al., "Proposed minimum reporting standards for chemical
analysis", Metabolomics 3(3):211-221, 2007.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from processing.quant_integration import pascal_weights
from processing.signal_quality import SIGNAL_AVERAGE, SIGNAL_GOOD

# ── Multiplicity vocabulary ───────────────────────────────────────────────────
# Broad set: everything that carries no usable J evidence for scoring.
# 'm'/'multiplet' are unresolvable in practice, so they score like singlets.
SINGLET_TYPES_BROAD = frozenset(
    {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'})

# Strict set: literal singlets only. Used where an unresolved 'm' must still be
# treated as a multiplet (candidate exclusion rules).
SINGLET_TYPES_STRICT = frozenset({'s', 'singlet', ''})

# An unresolved multiplet whose line pattern the reference database MEASURED
# on its own spectrum (BMRB_Metabolomics_v02, ``pattern_lines``). It is not
# singlet-like: it has lines to find and positions to agree with. It carries
# no single J, so what stands in for delta-J is the residual of the found
# lines against the pattern, in hertz, after the two are centred on each
# other -- the same quantity, asked of the whole pattern instead of one
# spacing. Seven of D-glucose's nine signals are 'm'; without this it could
# never verify a multiplet, never compete, and its peaks went to singlets.
PATTERN_MULT = 'm*'

# Expected line count per multiplicity label.
# Two multiplicity vocabularies reach this table: BMRB writes quint/sext/sept,
# HMDB writes p/h/hept. Both must resolve, or the expected line count silently
# falls back to 2 and a quintet is judged complete once two of its five lines
# are found — with J then measured from an incomplete set.
# Relative line intensities of each multiplicity — Pascal's triangle for the
# simple ones, and the flat or paired patterns the compound ones show. Two
# copies of this lived in the GUI (gui/notation/multiplet_scorer and
# gui/stages/j_resolve_stage) and disagreed on which keys existed; the
# multiplicity tables belong together, next to LINE_COUNTS, so a name known to
# one is known to both.
#
# Normalised to max = 1 on read, so a pattern can be compared with observed
# apex heights scaled the same way.
LINE_INTENSITIES: dict[str, tuple[float, ...]] = {
    'd': (1, 1), 'doublet': (1, 1),
    't': (1, 2, 1), 'triplet': (1, 2, 1),
    'q': (1, 3, 3, 1), 'quartet': (1, 3, 3, 1),
    'p': (1, 4, 6, 4, 1), 'pentet': (1, 4, 6, 4, 1),
    'quint': (1, 4, 6, 4, 1), 'quintet': (1, 4, 6, 4, 1),
    'quin': (1, 4, 6, 4, 1),
    'h': (1, 5, 10, 10, 5, 1), 'sext': (1, 5, 10, 10, 5, 1),
    'sextet': (1, 5, 10, 10, 5, 1),
    'sept': (1, 6, 15, 20, 15, 6, 1), 'septet': (1, 6, 15, 20, 15, 6, 1),
    'hept': (1, 6, 15, 20, 15, 6, 1), 'heptet': (1, 6, 15, 20, 15, 6, 1),
    # Compound couplings: every line is one combination of the couplings, so
    # they are equal unless two combinations coincide.
    'dd': (1, 1, 1, 1),
    'ddd': (1, 1, 1, 1, 1, 1, 1, 1),
    'dt': (1, 1, 2, 2, 1, 1),
    'td': (1, 2, 1, 1, 2, 1),
    'dq': (1, 1, 3, 3, 3, 3, 1, 1),
    'tt': (1, 2, 1, 2, 4, 2, 1, 2, 1),
    'tq': (1, 3, 3, 1, 2, 6, 6, 2, 1, 3, 3, 1),
}

# Height disagreement below this is a tie. Apex heights are the noisiest
# evidence here — a line sitting on a neighbouring analyte reads tall through
# no fault of the assignment — so the band is looser than the cosine's.
PASCAL_TIE = 0.10


LINE_COUNTS: dict[str, int] = {
    'd': 2, 'doublet': 2, 't': 3, 'triplet': 3,
    'q': 4, 'quartet': 4,
    'p': 5, 'pentet': 5, 'quint': 5, 'quintet': 5, 'quin': 5,
    'h': 6, 'sext': 6, 'sextet': 6,
    'sept': 7, 'septet': 7, 'hept': 7, 'heptet': 7,
    'dd': 4, 'dt': 6, 'td': 6, 'tt': 9, 'ddd': 8,
    'dq': 8, 'tq': 12,
}

# A parent peak and its sub-peak describe the same physical signal. Collapse
# positions closer than a quarter linewidth; genuine multiplet lines
# (J ≈ 6 Hz → ~0.01 ppm at 600 MHz) stay separate.
COINCIDENT_LINE_PPM = 0.003

# Per-spacing tolerance when validating that snapped lines form one multiplet.
J_SPACING_TOL_HZ = 2.0


def normalize_mult(value) -> str:
    """Lower-cased, stripped multiplicity label ('' when absent)."""
    return str(value or '').lower().strip()


def is_singlet_like(mult, *, strict: bool = False) -> bool:
    """True when the multiplicity carries no usable J evidence."""
    types = SINGLET_TYPES_STRICT if strict else SINGLET_TYPES_BROAD
    return normalize_mult(mult) in types


def parse_first_j(j_coupling) -> float | None:
    """First positive J value (Hz) in a DB coupling string, else None."""
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', str(j_coupling or ''))
            if float(x) > 0]
    return nums[0] if nums else None


def expected_line_count(mult, fallback: int = 2) -> int:
    """Number of lines the DB multiplicity predicts."""
    return LINE_COUNTS.get(normalize_mult(mult), fallback)


def ppm_score(ppm_diff: float, sigma_ppm: float) -> float:
    """Gaussian agreement in chemical shift; 1.0 at a perfect match."""
    if sigma_ppm <= 0:
        return 0.0
    return float(np.exp(-0.5 * (float(ppm_diff) / float(sigma_ppm)) ** 2))


def dj_score(delta_j_hz: float | None, gold_hz: float, bad_hz: float,
             neutral: float = 0.0) -> float:
    """Score the coupling-constant agreement.

    ΔJ < gold_hz → 1.0; linear decay to 0.0 at bad_hz; 0.0 beyond.
    Returns ``neutral`` when ΔJ could not be measured.
    """
    if delta_j_hz is None:
        return float(neutral)
    dj = float(delta_j_hz)
    if dj < gold_hz:
        return 1.0
    if dj < bad_hz:
        span = bad_hz - gold_hz
        return 1.0 - (dj - gold_hz) / span if span > 0 else 0.0
    return 0.0


def synthetic_lines(centre_ppm: float, j_hz: float, n_lines: int,
                    field_mhz: float) -> list[dict]:
    """Evenly spaced line positions for a first-order multiplet.

    Used when the stored ``predicted_lines`` are missing or truncated, so the
    line count check still has targets to snap onto.
    """
    if not (centre_ppm and j_hz and field_mhz > 0 and n_lines >= 2):
        return []
    j_ppm = j_hz / field_mhz
    half = (n_lines - 1) / 2.0
    return [{'ppm': centre_ppm + (k - half) * j_ppm, 'rel_intensity': 1.0}
            for k in range(n_lines)]


def snap_lines_to_peaks(pred_lines: Sequence[dict], peaks: Iterable[dict],
                        tolerance: float) -> list[float]:
    """Match each predicted line to its nearest unused detected peak.

    Returns the ppm positions of the confirmed peaks. A peak is consumed by at
    most one line, so a single detected peak cannot satisfy a whole multiplet.
    """
    peaks = list(peaks or [])
    used: set = set()
    found: list[float] = []
    for line in pred_lines or []:
        target = float(line.get('ppm', 0))
        best_pk, best_d = None, float(tolerance)
        for pk in peaks:
            if pk.get('rank') in used:
                continue
            d = abs(float(pk.get('ppm', 0)) - target)
            if d <= best_d:
                best_d, best_pk = d, pk
        if best_pk is not None:
            used.add(best_pk.get('rank'))
            found.append(float(best_pk['ppm']))
    return found


def dedup_coincident(positions: Sequence[float],
                     threshold: float = COINCIDENT_LINE_PPM) -> list[float]:
    """Collapse positions that describe the same physical line."""
    out: list[float] = []
    for p in positions:
        if not any(abs(p - kept) < threshold for kept in out):
            out.append(float(p))
    return out


def spacings_hz(positions: Sequence[float], field_mhz: float) -> list[float]:
    """Consecutive line separations in Hz, ascending ppm order."""
    srt = sorted(positions)
    return [abs(srt[k + 1] - srt[k]) * field_mhz for k in range(len(srt) - 1)]


def drop_worst_j_outlier(positions: Sequence[float], j_db_hz: float,
                         field_mhz: float,
                         tol_hz: float = J_SPACING_TOL_HZ) -> list[float]:
    """Remove one spurious line when a single spacing breaks the pattern.

    Every consecutive spacing must sit within ``tol_hz`` of the DB J. A nearby
    unrelated peak makes exactly one spacing wildly wrong while the others hold;
    dropping the line that minimises the worst spacing deviation recovers the
    genuine multiplet at the cost of one missing line.
    """
    positions = list(positions)
    if len(positions) < 2 or not j_db_hz or field_mhz <= 0:
        return positions
    seps = spacings_hz(positions, field_mhz)
    if not any(abs(s - j_db_hz) > tol_hz for s in seps):
        return positions

    best_drop, best_worst = None, max(abs(s - j_db_hz) for s in seps)
    for i in range(len(positions)):
        cand = [p for j, p in enumerate(positions) if j != i]
        if len(cand) < 2:
            continue
        worst = max(abs(s - j_db_hz) for s in spacings_hz(cand, field_mhz))
        if worst < best_worst:
            best_worst, best_drop = worst, i
    if best_drop is None:
        return positions
    return [p for j, p in enumerate(positions) if j != best_drop]


def sync_predicted_lines(original: Sequence[dict],
                         found_ppms: Sequence[float],
                         tolerance: float) -> list[dict]:
    """Re-anchor predicted lines onto the confirmed peak positions.

    Keeps the fingerprint arrows drawn in the UI identical to the lines counted
    as found, so the displayed evidence and the score share one source.

    A line that found no peak is KEPT, placed where the found ones say it
    should be — the pattern is evenly spaced, so two confirmed positions fix
    the whole comb by their index. Dropping it truncated the list, and a later
    reader seeing fewer lines than the multiplicity calls for had to rebuild
    the pattern from the database J, which then reappeared in anything measured
    against it. Extrapolating from the confirmed peaks keeps the geometry
    measured. Each line records whether it was ``measured``; with fewer than
    two of those there is no spacing to extrapolate from and the original
    positions stand, which is the caller's cue that the pattern is the
    database's, not the spectrum's.
    """
    if not found_ppms:
        return list(original or [])
    used: set = set()
    snapped: list[dict] = []
    unmatched: list[tuple[int, dict]] = []
    fixed: list[tuple[int, float]] = []
    for i, line in enumerate(sorted(original or [],
                                    key=lambda l: float(l.get('ppm', 0)))):
        target = float(line.get('ppm', 0))
        avail = [fp for fp in found_ppms if fp not in used]
        best = min(avail, key=lambda fp: abs(fp - target), default=None)
        if best is not None and abs(best - target) <= tolerance:
            used.add(best)
            fixed.append((i, float(best)))
            snapped.append({'ppm': best, 'measured': True,
                            'rel_intensity': line.get('rel_intensity', 1.0)})
        else:
            unmatched.append((i, line))

    # p(i) = a + b·i through the confirmed lines: b is the spacing the sample
    # actually shows, so a line placed by it carries no database J.
    if unmatched and len(fixed) >= 2:
        xs = np.array([i for i, _ in fixed], dtype=float)
        ys = np.array([p for _, p in fixed], dtype=float)
        b, a = np.polyfit(xs, ys, 1)
        for i, line in unmatched:
            snapped.append({'ppm': float(a + b * i), 'measured': False,
                            'rel_intensity': line.get('rel_intensity', 1.0)})
    else:
        for _i, line in unmatched:
            snapped.append({'ppm': float(line.get('ppm', 0)), 'measured': False,
                            'rel_intensity': line.get('rel_intensity', 1.0)})

    for fp in sorted(found_ppms):
        if fp not in used:
            snapped.append({'ppm': fp, 'rel_intensity': 1.0, 'measured': True})
    if not snapped:
        return list(original or [])
    max_ri = max(l['rel_intensity'] for l in snapped) or 1.0
    for l in snapped:
        l['rel_intensity'] /= max_ri
    return sorted(snapped, key=lambda l: l['ppm'])


@dataclass
class LineMeasurement:
    """What the spectrum actually shows for one DB multiplet."""
    n_expected: int | None = None
    n_found: int | None = None
    found_ppms: list[float] = field(default_factory=list)
    observed_j_hz: float | None = None
    delta_j_hz: float | None = None
    centre_ppm: float | None = None
    predicted_lines: list[dict] = field(default_factory=list)
    measured: bool = False          # False → no lines to snap; caller keeps old data
    # For a measured pattern: the RMS of the found lines against the shape,
    # in hertz, before scaling onto delta-J's gates. delta_j_hz carries the
    # scaled value; this is the number a reader should see.
    pattern_residual_hz: float | None = None
    pattern_intensity_agreement: float | None = None
    # The spacing the observed one was judged against: the reference J as it
    # appears at this study's linewidth, not the reference J itself.
    expected_j_hz: float | None = None
    # For a measured pattern: where the database's reference position falls
    # on this axis, given the lines found -- the found centre minus the mean
    # offset of the lines it was made of. Compared with db_ppm it is the
    # pattern's drift, whichever lines were found and whichever anchored it;
    # centre_ppm is where the signal is seen, and the two differ by the
    # pattern's asymmetry.
    origin_ppm: float | None = None
    # For a pattern fitted as a shape (processing.pattern_shape): the fit's
    # signal-to-noise, the fraction of the trace over its lines it accounts
    # for, its coefficient of determination and its amplitude in trace
    # units. None where the pattern was matched on the peak list instead.
    pattern_snr: float | None = None
    pattern_explained: float | None = None
    pattern_r2: float | None = None
    pattern_amplitude: float | None = None

    @property
    def complete(self) -> bool:
        """True when every expected line was confirmed in the spectrum."""
        return (self.n_found is not None and self.n_expected is not None
                and self.n_found >= self.n_expected)


def apparent_lines(lines, fwhm_hz, *, step_hz=0.02) -> list:
    """Where the apices sit when these lines are Lorentzians of this width.

    Two lines closer than about a linewidth and a half pull each other's apex
    inward, and closer still they are one apex: the maxima of a sum of
    Lorentzians are not at the lines' positions. Glucose's anomeric doublet
    (J = 3.8 Hz) read 2.7 Hz apart in plasma at a 2.7 Hz linewidth and was
    refused as another coupling, when the pulling-in was the linewidth's and
    the coupling was right. A reference position is therefore compared with
    what would be seen of it here, not with itself.

    ``lines`` is ``[(offset_hz, rel_height)]``; the sum of Lorentzians of
    FWHM ``fwhm_hz`` is sampled at ``step_hz`` and its local maxima returned
    in the same form, heights scaled to the tallest. No width, or none of
    the lines, returns the lines unchanged.

    Lorentzian lineshape: Abragam, The Principles of Nuclear Magnetism,
    Oxford 1961, ch. III. The apex pull-in of two overlapping Lorentzians is
    what Posener (Aust. J. Phys. 1959, 12, 184) described for the resolution
    of a doublet.
    """
    pts = [(float(o), float(r)) for o, r in (lines or ())]
    if len(pts) < 2 or not fwhm_hz or float(fwhm_hz) <= 0:
        return pts
    hw = float(fwhm_hz) / 2.0
    lo = min(o for o, _ in pts) - 3 * float(fwhm_hz)
    hi = max(o for o, _ in pts) + 3 * float(fwhm_hz)
    x = np.arange(lo, hi + step_hz, step_hz)
    y = np.zeros_like(x)
    for o, r in pts:
        y += r * hw ** 2 / ((x - o) ** 2 + hw ** 2)
    inner = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    idx = np.nonzero(inner)[0] + 1
    if len(idx) == 0:
        return pts
    top = float(y[idx].max()) or 1.0
    out = []
    for i in idx:
        # parabola through the three samples around the maximum
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        dx = 0.5 * (y0 - y2) / den if den else 0.0
        out.append((float(x[i] + dx * step_hz), float(y1 / top)))
    return out


def apparent_spacing_hz(j_hz, fwhm_hz, n_lines) -> float | None:
    """The spacing a first-order multiplet of coupling ``j_hz`` shows between
    adjacent apices at linewidth ``fwhm_hz`` -- ``j_hz`` itself when the lines
    stand apart, less as they lean on each other, ``None`` once any pair has
    merged and the multiplet no longer shows ``n_lines`` apices."""
    if not j_hz or float(j_hz) <= 0 or not n_lines or int(n_lines) < 2:
        return None
    if not fwhm_hz or float(fwhm_hz) <= 0:
        return float(j_hz)
    n = int(n_lines)
    heights = pascal_weights(n)
    lines = [((k - (n - 1) / 2.0) * float(j_hz), float(h))
             for k, h in enumerate(heights)]
    seen = apparent_lines(lines, fwhm_hz)
    if len(seen) != n:
        return None
    offs = sorted(o for o, _ in seen)
    return float(np.median(np.diff(offs)))


def measure_multiplet(predicted_lines: Sequence[dict], peaks: Iterable[dict],
                      *, mult, j_coupling, db_ppm: float, centre_hint: float,
                      n_expected: int, tolerance: float,
                      field_mhz: float,
                      linewidth_hz: float | None = None) -> LineMeasurement:
    """Snap a DB multiplet onto detected peaks and measure J from the result.

    ``centre_hint`` is the previously observed centre (falls back to db_ppm)
    and is only used to build synthetic line targets when the stored
    ``predicted_lines`` are missing or truncated.

    ``linewidth_hz`` is the study's; the observed spacing is judged against
    the reference coupling as it would appear at that width (see
    ``apparent_lines``), and ``expected_j_hz`` records what that was.
    """
    is_pattern = normalize_mult(mult) == PATTERN_MULT
    # A pattern has no single J. Its coupling list is the curators' and the
    # first value in it is not the spacing of anything in particular.
    j_db = None if is_pattern else parse_first_j(j_coupling)
    pred = list(predicted_lines or [])
    if len(pred) < n_expected and field_mhz > 0 and j_db:
        pred = synthetic_lines(float(centre_hint or db_ppm or 0), j_db,
                               n_expected, field_mhz) or pred

    if not pred or n_expected < 2:
        return LineMeasurement(measured=False)

    j_ref = j_db
    if j_db is not None and linewidth_hz:
        j_ref = apparent_spacing_hz(j_db, linewidth_hz, n_expected) or j_db

    found = dedup_coincident(snap_lines_to_peaks(pred, peaks, tolerance))
    if j_ref is not None and len(found) == n_expected:
        found = drop_worst_j_outlier(found, j_ref, field_mhz)

    m = LineMeasurement(n_expected=n_expected, n_found=len(found),
                        found_ppms=found, measured=True)
    m.predicted_lines = sync_predicted_lines(predicted_lines, found, tolerance)

    if not m.complete:
        # Missing lines: any stored ΔJ describes a different set of peaks and
        # must not be reused. Report it as unmeasured rather than stale.
        return m

    if is_pattern:
        m.delta_j_hz = pattern_residual_hz(
            [float(l.get('ppm')) for l in pred if l.get('ppm') is not None],
            found, field_mhz)
        m.centre_ppm = float(np.mean(found))
        return m

    seps = spacings_hz(found, field_mhz) if field_mhz > 0 else []
    if seps:
        m.observed_j_hz = float(np.median(seps))
        if j_ref is not None:
            m.expected_j_hz = float(j_ref)
            m.delta_j_hz = float(abs(m.observed_j_hz - j_ref))
        m.centre_ppm = float(np.mean(found))
    return m


# A pattern line is looked for this far, in hertz, from where the aligned
# pattern puts it. Two hertz: a linewidth, plus what a system that is not
# quite first order moves between 500 and 600 MHz. The annotation's own
# tolerance -- 0.05 ppm, thirty hertz -- is for where a multiplet sits, not
# for where its lines sit once it has been found; used per line it matched
# any peak in the neighbourhood and the residual measured the neighbourhood.
PATTERN_LINE_TOL_HZ = 2.0

# The residual's gates. Wider than delta-J's, because a residual is over the
# whole shape and the shape was measured at another field: the ring protons
# of a sugar are nearly first order, and "nearly" is a few tenths of a hertz
# per line. A pattern within half a hertz is the reference's; beyond a hertz
# and a half it is another shape.
PATTERN_GOLD_HZ = 0.5
PATTERN_BAD_HZ = 1.5

# A measured line must stand this high against the pattern's tallest to
# count against the listed multiplicity (see pattern_governs). Outer lines
# of a second-order system reach a third; satellites and sidebands do not.
GOVERN_MIN_REL = 0.30


def match_pattern(pattern, peaks, *, anchor_ppm, field_mhz,
                  line_tol_hz=PATTERN_LINE_TOL_HZ, anchor_height=None,
                  floor=None) -> dict:
    """Lay a measured pattern on detected peaks and count what it finds.

    ``pattern`` is ``[(offset_hz, rel_height)]`` from the reference DB;
    ``peaks`` are detected peaks with ``ppm``; ``anchor_ppm`` is the peak the
    search matched the signal on. Each pattern line in turn is placed on the
    anchor, the rest follow at their offsets, and every line is looked for
    within ``line_tol_hz`` of where that puts it. The alignment finding the
    most lines wins, ties going to the smaller residual.

    Returns ``{'n_found', 'n_expected', 'residual_hz', 'centre_ppm',
    'found_ppms', 'found_rel', 'found_idx', 'predicted_ppms'}``;
    ``found_idx`` says which pattern line each found peak is. The residual is over the found lines,
    after centring found and predicted on each other, so a drift of the
    whole multiplet is delta-ppm's business and not the pattern's.

    Given ``anchor_height`` (the anchor peak's) and ``floor`` (the height
    below which detection reports nothing), a line is expected only where
    the pattern's heights put it above the floor: the reference measured
    its lines at a signal-to-noise of thousands, and a line at a twentieth
    of the tallest is real there and invisible in a study at twenty. What
    could not have been seen is not counted missing.
    """
    offs = [float(o) for o, _r in (pattern or ())]
    rels = [float(r) for _o, r in (pattern or ())]
    n = len(offs)
    if n < 2 or not peaks or field_mhz <= 0:
        return {'n_found': 0, 'n_expected': n, 'residual_hz': None,
                'centre_ppm': None, 'found_ppms': [], 'found_rel': [],
                'found_idx': [],
                'predicted_ppms': []}
    ppms = np.array(sorted(float(p['ppm']) for p in peaks
                           if p.get('ppm') is not None))
    tol = float(line_tol_hz) / float(field_mhz)
    best = None
    for k in range(n):
        shift = float(anchor_ppm) - offs[k] / float(field_mhz)
        pred = [shift + o / float(field_mhz) for o in offs]
        if (anchor_height is not None and floor is not None
                and rels[k] > 0 and float(floor) > 0):
            # The floor excuses only the minor lines. A line above
            # GOVERN_MIN_REL of the tallest is always expected: anchored on
            # a small peak, the scale shrank until two lines of fourteen
            # were all that was expected, and the two were found.
            scale = float(anchor_height) / rels[k]
            top = max(rels) or 1.0
            expected = [i for i in range(n)
                        if rels[i] / top >= GOVERN_MIN_REL
                        or scale * rels[i] >= float(floor)]
        else:
            expected = list(range(n))
        found, used = [], set()
        for li, (pp, rel) in enumerate(zip(pred, rels)):
            if li not in expected:
                continue
            # nearest unused peak within tolerance
            order = np.argsort(np.abs(ppms - pp))
            hit = None
            for j in order:
                if abs(ppms[j] - pp) > tol:
                    break
                if j not in used:
                    hit = j
                    break
            if hit is not None:
                used.add(hit)
                found.append((pp, float(ppms[hit]), rel, li))
        n_found = len(found)
        resid = None
        if n_found >= 2:
            pr = np.array([f[0] for f in found])
            fd = np.array([f[1] for f in found])
            resid = float(np.sqrt(np.mean(((fd - fd.mean())
                                            - (pr - pr.mean())) ** 2))
                          * float(field_mhz))
        n_exp = len(expected)
        # whole first, then the most lines, then the closest shape
        key = (n_found >= n_exp and n_found >= 2, n_found,
               -(resid if resid is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, {'n_found': n_found, 'n_expected': n_exp,
                          'residual_hz': resid,
                          'centre_ppm': (float(np.mean([f[1] for f in found]))
                                         if found else None),
                          'found_ppms': [f[1] for f in found],
                          'found_rel': [f[2] for f in found],
                          'found_idx': [f[3] for f in found],
                          'predicted_ppms': pred})
    return best[1]


def pattern_governs(mult, pattern) -> bool:
    """Whether a signal's measured line pattern, not its listed letter, says
    what to look for.

    An unresolved 'm' has no letter to speak of. A letter whose line count
    the measurement contradicts is a curator's simplification: taurine's
    3.41 ppm signal is listed 'd' 6.25 Hz and measured as three lines 6.7
    Hz apart at 1:2:1 -- a triplet, as its two equivalent CH2 neighbours
    make it. Judged as a doublet it found two lines of three and was Bad
    on every study. The measurement was made on the pure compound at a
    signal-to-noise of thousands; where it and the letter disagree on how
    many lines there are, the measurement stands.
    """
    lines = (list(pattern) if isinstance(pattern, list)
             else pattern_offsets_hz({'pattern_lines': pattern}))
    if len(lines) < 2:
        return False
    m = normalize_mult(mult)
    if m in ('m', 'multiplet'):
        return True
    # Only lines of some height argue with the letter. At a signal-to-noise
    # of six hundred thousand acetate's singlet carries a pair of lines at
    # seven percent, 5.5 Hz either side; whatever they are, they are not
    # what makes it a singlet or not.
    # The letter's own lines are held to the same bar -- a septet's outer
    # lines are a twentieth of its centre and a quintet's a sixth -- so a
    # first-order multiplet measured as itself agrees with its letter.
    measured = sum(1 for _o, r in lines if float(r) >= GOVERN_MIN_REL)
    if measured < 2:
        return False
    # A letter this table does not know (a broad or exchangeable label)
    # is left as listed: the pattern of a broad line is not a multiplet.
    if m in ('s', 'singlet'):
        listed = 1
    elif m in LINE_INTENSITIES:
        top = max(LINE_INTENSITIES[m])
        listed = sum(1 for h in LINE_INTENSITIES[m]
                     if h / top >= GOVERN_MIN_REL)
    else:
        listed = LINE_COUNTS.get(m)
    return listed is not None and listed != measured


def pattern_offsets_hz(mp) -> list:
    """``[(offset_hz, rel_height)]`` from a signal's ``pattern_lines`` (the
    JSON the reference DB stores), or ``[]``."""
    raw = mp.get('pattern_lines') if isinstance(mp, dict) else None
    if not raw:
        return []
    try:
        lines = json.loads(raw) if isinstance(raw, str) else list(raw)
        out = [(float(o), float(r)) for o, r in lines]
    except (TypeError, ValueError):
        return []
    return out if len(out) >= 2 else []


def resolvable_pattern(pattern, merge_hz, fwhm_hz=None) -> list:
    """The pattern as THIS study can resolve it.

    A reference pattern is measured on a pure compound at a linewidth of a
    hertz or two; the study's lines are wider, and two pattern lines closer
    together than the study's linewidth are one peak in the study. D-glucose
    at 3.239 ppm lists two lines 1.1 Hz apart: in plasma at 3.2 Hz they are
    one line, and finding two of three was the right count of the wrong
    expectation. Lines closer than ``merge_hz`` collapse into one at their
    height-weighted position, heights added.

    Given ``fwhm_hz`` the lines are first laid down as Lorentzians of that
    width and their apices taken (``apparent_lines``): a line leaning on its
    neighbour is expected where its apex is seen, not where it is.
    """
    pattern = apparent_lines(pattern, fwhm_hz) if fwhm_hz else pattern
    if not pattern or not merge_hz or float(merge_hz) <= 0:
        return list(pattern or ())
    lines = sorted((float(o), float(r)) for o, r in pattern)
    out: list = []
    for o, r in lines:
        if out and o - out[-1][2] < float(merge_hz):
            po, pr, _last = out[-1]
            tot = pr + r
            out[-1] = ((po * pr + o * r) / tot if tot else o, tot, o)
        else:
            out.append((o, r, o))
    top = max((r for _o, r, _l in out), default=1.0) or 1.0
    return [(o, r / top) for o, r, _l in out]


def measure_pattern_signal(mp, peaks, *, field_mhz, dj_gold_hz,
                           anchor_ppm=None, merge_hz=None,
                           fwhm_hz=None, floor=None, trace=None,
                           sigma=None, tolerance_ppm=None) -> LineMeasurement:
    """The one way a measured pattern is matched, wherever it is asked.

    The search asks it when a candidate is built; ``resolve_dj`` asks it
    again when the axis has been recalibrated. Two ways of doing it -- one
    matching the shape, one snapping stored lines with the annotation's
    thirty-hertz tolerance -- had the second overwrite the first: a
    three-line pattern found whole came back as two lines seven hertz off.

    The residual stands in for delta-J, scaled onto delta-J's gates so every
    rule downstream reads it unchanged; the raw residual travels beside it.

    ``merge_hz`` is what this study can resolve -- its linewidth, or the
    detector's minimum spacing, whichever is larger -- and lines of the
    pattern closer than that are expected as one. Remembered on the matched
    peak as ``pattern_merge_hz`` so a later re-measurement asks the same.
    ``floor`` is the height under which detection reports nothing, kept as
    ``pattern_floor``; with the anchor peak's height it says which lines of
    the pattern this study could have seen (see ``match_pattern``).

    Given ``trace`` (``(ppm, intensity)`` of the spectrum the peaks came
    from), ``sigma`` (its noise) and ``tolerance_ppm``, the pattern is
    fitted to the trace as a shape first (``processing.pattern_shape``): the
    largest copy of it that fits under the trace, wherever in the tolerance
    it fits. Its apices at this linewidth are then the lines expected, and
    a line is found where the trace itself has a maximum within two hertz
    of it -- the detector's list, which merges and splits at its own
    thresholds, is not consulted. The fit's signal-to-noise, the fraction
    of the trace it explains and its position ride on the measurement.
    Without a trace the pattern is matched on the peak list as before.
    """
    pat = resolvable_pattern(pattern_offsets_hz(mp),
                             merge_hz if merge_hz is not None
                             else mp.get('pattern_merge_hz'),
                             fwhm_hz if fwhm_hz is not None
                             else mp.get('study_linewidth_hz'))
    if len(pat) < 2 or not field_mhz:
        return LineMeasurement(measured=False)
    anchor = anchor_ppm
    if anchor is None:
        anchor = mp.get('peak_ppm') or mp.get('ppm_center') or mp.get('db_ppm')
    if anchor is None:
        return LineMeasurement(measured=False)
    _floor = floor if floor is not None else mp.get('pattern_floor')
    _fwhm = fwhm_hz if fwhm_hz is not None else mp.get('study_linewidth_hz')
    _sigma = sigma if sigma is not None else mp.get('pattern_sigma')
    _tol = (tolerance_ppm if tolerance_ppm is not None
            else mp.get('pattern_tolerance_ppm'))
    if trace is not None and _fwhm and _sigma and _tol:
        shaped = _measure_pattern_shape(
            mp, pat, trace, field_mhz=float(field_mhz),
            dj_gold_hz=dj_gold_hz, fwhm_hz=float(_fwhm),
            sigma=float(_sigma), tolerance_ppm=float(_tol), floor=_floor)
        if shaped is not None:
            return shaped
    _anchor_h = None
    if _floor:
        _near = min((p for p in peaks if p.get('ppm') is not None
                     and p.get('height') is not None),
                    key=lambda p: abs(float(p['ppm']) - float(anchor)),
                    default=None)
        if (_near is not None and abs(float(_near['ppm']) - float(anchor))
                <= PATTERN_LINE_TOL_HZ / float(field_mhz)):
            _anchor_h = float(_near['height'])
    pm = match_pattern(pat, peaks, anchor_ppm=float(anchor),
                       field_mhz=float(field_mhz),
                       anchor_height=_anchor_h, floor=_floor)
    m = LineMeasurement(n_expected=pm['n_expected'], n_found=pm['n_found'],
                        found_ppms=list(pm['found_ppms']), measured=True)
    # The pattern's own relative heights ride with the lines: quantification
    # weighs a multiplet's tails by them, and a five-line pattern is not a
    # 1:4:6:4:1 quintet.
    m.predicted_lines = [{'ppm': float(v), 'rel_intensity': float(r),
                          'measured': True}
                         for v, r in zip(pm['found_ppms'], pm['found_rel'])]
    m.pattern_residual_hz = pm['residual_hz']
    m.pattern_intensity_agreement = pattern_intensity_agreement(
        pm['found_ppms'], pm['found_rel'], peaks)
    if pm['found_idx'] and pm['centre_ppm'] is not None:
        # Delta-ppm is the pattern's drift, not the anchor's: glucose's
        # 3.234 ppm pattern anchored on its lowest line read +0.002 while the
        # pattern as a whole sat +0.017, with the rest of the analyte.
        mean_off = float(np.mean([pat[i][0] for i in pm['found_idx']]))
        m.origin_ppm = float(pm['centre_ppm']) - mean_off / float(field_mhz)
    if m.complete and pm['residual_hz'] is not None:
        m.delta_j_hz = (float(pm['residual_hz']) * float(dj_gold_hz)
                        / PATTERN_GOLD_HZ)
        m.centre_ppm = pm['centre_ppm']
    return m


def _measure_pattern_shape(mp, pat, trace, *, field_mhz, dj_gold_hz,
                           fwhm_hz, sigma, tolerance_ppm,
                           floor=None) -> LineMeasurement | None:
    """The multiplet judged as a shape and nothing else.

    The reference's profile (``pattern_profile``, with its own width
    ``profile_fwhm_hz``) or, without one, its lines, fitted to the trace by
    ``processing.pattern_shape.fit_shape``. Two numbers decide: the fit's
    signal-to-noise says whether the shape is here, and the fraction of the
    trace it explains says whether it is alone. Under the fit's rule -- the
    largest copy that fits under the trace -- a shape cannot contradict the
    spectrum, so nothing here marks an identity wrong; what it does is rank.
    The explained fraction stands in for delta-J, scaled onto its gates: a
    shape explaining all of its region reads as a perfect coupling, one
    explaining none of it as the worst allowed, never beyond -- overlap is
    sharing, not disagreement.

    None when the trace does not cover the region.
    """
    from processing.pattern_shape import fit_shape, parse_profile
    db_ppm = mp.get('db_ppm') or mp.get('ppm_center') or mp.get('ppm')
    if db_ppm is None:
        return None
    ppm_axis, y_axis = trace
    profile = parse_profile(mp.get('pattern_profile'))
    fit = fit_shape(ppm_axis, y_axis, db_ppm=float(db_ppm),
                    field_mhz=field_mhz, fwhm_hz=fwhm_hz,
                    tolerance_ppm=tolerance_ppm, sigma=sigma,
                    profile=profile,
                    profile_fwhm_hz=mp.get('profile_fwhm_hz'),
                    lines=pattern_offsets_hz(mp) or pat)
    if fit is None or not fit['apex_ppms']:
        return None
    # Here when the fit stands over detection's floor -- the same bar a
    # peak had to clear to be in the list.
    snr_min = (float(floor) / float(sigma)) if floor else 3.0
    present = fit['snr'] >= snr_min
    n = len(fit['apex_ppms'])
    m = LineMeasurement(n_expected=n, n_found=n if present else 0,
                        found_ppms=list(fit['apex_ppms']) if present else [],
                        measured=True)
    m.predicted_lines = [{'ppm': float(a), 'rel_intensity': float(r),
                          'measured': True}
                         for a, r in zip(fit['apex_ppms'], fit['apex_rel'])]
    m.origin_ppm = float(fit['origin_ppm'])
    m.centre_ppm = float(np.mean(fit['apex_ppms']))
    m.pattern_snr = float(fit['snr'])
    m.pattern_explained = float(fit['explained'])
    m.pattern_r2 = float(fit['r2'])
    m.pattern_amplitude = float(fit['amplitude'])
    m.pattern_intensity_agreement = float(fit['explained'])
    if present:
        m.pattern_residual_hz = float(PATTERN_BAD_HZ
                                      * (1.0 - float(fit['explained'])))
        m.delta_j_hz = (m.pattern_residual_hz * float(dj_gold_hz)
                        / PATTERN_GOLD_HZ)
    return m


def pattern_intensity_agreement(found_ppms, found_rel, peaks,
                                *, tol_ppm=0.002) -> float | None:
    """How well the found lines' heights follow the pattern's own.

    The pattern's relative heights were measured on the reference spectrum;
    a first-order multiplet's come from the binomial theorem. Either way the
    question is the same -- do the apices stand in the proportions the claim
    says -- and the answer is compared the same way, each set scaled to its
    own maximum. None when there is nothing to judge, never as disagreement.
    """
    if not found_ppms or not found_rel or len(found_ppms) != len(found_rel):
        return None
    hs = []
    for pp in found_ppms:
        best, bd = None, float(tol_ppm)
        for pk in peaks or ():
            if pk.get('ppm') is None:
                continue
            d = abs(float(pk['ppm']) - float(pp))
            if d <= bd:
                best, bd = pk, d
        h = (best or {}).get('height')
        if h is None or not np.isfinite(float(h)) or float(h) < 0:
            return None
        hs.append(float(h))
    if len(hs) < 2 or max(hs) <= 0:
        return None
    exp = [float(r) for r in found_rel]
    if max(exp) <= 0:
        return None
    err = [abs(o / max(hs) - e / max(exp)) for o, e in zip(hs, exp)]
    return float(max(0.0, 1.0 - sum(err) / len(err)))


def pattern_residual_hz(predicted_ppms, found_ppms, field_mhz) -> float | None:
    """How far the found lines sit from the reference pattern, in hertz.

    Both sets are centred on their own means first, so a shift of the whole
    multiplet -- which is chemistry, and the delta-ppm criterion's business --
    does not count against the pattern. What remains is the root mean square
    of the line-by-line differences: the pattern's version of delta-J, and it
    is compared with the same gates.
    """
    if not predicted_ppms or not found_ppms or field_mhz <= 0:
        return None
    p = np.sort(np.asarray(predicted_ppms, dtype=float))
    f = np.sort(np.asarray(found_ppms, dtype=float))
    if p.size != f.size:
        return None
    p = p - p.mean()
    f = f - f.mean()
    return float(np.sqrt(np.mean((f - p) ** 2)) * float(field_mhz))


# ── Candidate ranking (shared by annotation and quantification) ───────────────
# One decision function, two consumers: the Annotation Results window uses it to
# classify/sort, quantification uses it to elect the single winner of a contested
# peak. Both therefore see the same verdict — previously each computed its own
# composite_score under the same key name and they disagreed.
#
# The decision is LEXICOGRAPHIC, not a weighted sum. A weighted blend lets a
# strong Δppm compensate a bad ΔJ, which inverts the analytically correct answer:
# in a real dataset the CH3 doublet at ~1.0 ppm went to a candidate with
# Δppm = 0.043 over one with Δppm = 0.001 because the blend was 45% Δppm.
#
# Priority of the evidence follows how a spectrum is read by hand, and the order
# is not arbitrary — chemical shift drifts with pH, ionic strength, temperature
# and metal-ion content, while the scalar coupling is a property of the bonding
# path and is far more stable:
#   • Dona et al., "A guide to the identification of metabolites in NMR-based
#     metabonomics/metabolomics experiments", Comput Struct Biotechnol J
#     14:135-153, 2016. doi:10.1016/j.csbj.2016.02.005
#   • Wermter, Bock & Dreher, "1H NMR Chemical Shifts and J-Coupling Constants
#     Dependence on Temperature and pH", NMR Biomed, 2026. doi:10.1002/nbm.70239
#     (shifts strongly T/pH-dependent; only moderate changes in J)
#   • Govindaraju, Young & Maudsley, "Proton NMR chemical shifts and coupling
#     constants for brain metabolites", NMR Biomed 13:129-153, 2000.

# Δppm band width (ppm). Deliberately COARSE. Within one contested peak every
# candidate sees the same observed ppm, but that ppm drifts between samples, so
# when two candidates' DB shifts are closer than the drift the Δppm ordering
# flips with the drift. Measured drift in a 37-sample mouse study: 0.030 ppm
# median peak-position amplitude (18 Hz at 600 MHz) — larger than the 0.017 ppm
# that separates valine from isoleucine. Ordering those by Δppm is ordering
# noise. A per-project value derived from the observed drift is the eventual
# goal; this fixed band is the conservative stand-in.
PPM_BAND_PPM = 0.020

# Defaults for the scoring parameters the Annotation Parameters dialog exposes.
# The dialog's values win whenever it has been opened; these exist so the search
# worker can score before any window is created. Same keys, one owner.
SCORING_DEFAULTS: dict = {
    # Signal grades annotation is allowed to use. Lives here with the rest so a
    # dialog built from these defaults cannot open with every box unticked.
    'annotatable_tiers': [SIGNAL_GOOD, SIGNAL_AVERAGE],
    'good_score_thresh': 0.75,
    'bad_score_thresh':  0.50,
    'dj_gold_hz':        0.20,
    'dj_bad_hz':         0.50,
    'dj_neutral':        0.00,
    'dppm_gold':         0.05,   # mirrors 'Tolerance (ppm)'; one input owns it
    'w_dj':              0.57,
    'w_cosine_ns':       0.14,
    'w_ppm_ns':          0.10,
    'w_cosine_s':        0.32,
    'w_ppm_s':           0.27,
    'w_coverage':        0.05,   # analyte level: penalty for signals not found
    'w_corroboration':   0.09,   # analyte level: lift from the other signals
    'nnls_min_frac':     0.03,
}

# Deprecated keys kept readable so saved projects still load. w_found_ns and
# w_rri_ns were reused as the analyte-level coverage and corroboration weights
# when those terms were introduced, which left their labels describing something
# they no longer did. The rest were weights of a scoring model that no longer
# exists — nothing read them.
LEGACY_PARAM_KEYS: dict[str, str | None] = {
    'w_found_ns':   'w_coverage',
    'w_rri_ns':     'w_corroboration',
    'w_reverse_ns':  None,
    'w_found_s':     None,
    'w_rri_s':       None,
    'w_reverse_s':   None,
    'dppm_gold_hz':  None,       # collapsed into the Tolerance (ppm) input
}


def migrate_params(params: dict) -> dict:
    """Map a saved parameter dict onto the current keys, dropping dead ones."""
    out = {}
    for k, v in (params or {}).items():
        if k in LEGACY_PARAM_KEYS:
            new_key = LEGACY_PARAM_KEYS[k]
            if new_key is not None:
                out.setdefault(new_key, v)
            continue
        out[k] = v
    return out


# One description of the Annotation Parameters, shared by every dialog that
# shows them. Two dialogs used to carry their own copy of this list and had
# already drifted apart.  (group, label, key, help)
PARAMETER_SPEC: list[tuple[str, str, str, str]] = [
    ('Coupling (ΔJ)', 'Gold standard ΔJ (Hz)', 'dj_gold_hz',
     'ΔJ below this is a perfect match — also the tie band between candidates'),
    ('Coupling (ΔJ)', 'Bad ΔJ threshold (Hz)', 'dj_bad_hz',
     'ΔJ at or above this rejects the signal, with no exceptions'),
    ('Coupling (ΔJ)', 's_dj when ΔJ unmeasurable', 'dj_neutral',
     'Score given when J could not be measured; the term is excluded instead'),

    ('Evidence weight — signal', 'ΔJ', 'w_dj',
     'Coupling agreement — the strongest evidence a multiplet carries'),
    ('Evidence weight — signal', 'Δppm', 'w_ppm_ns',
     'Chemical-shift agreement (multiplet)'),
    ('Evidence weight — signal', 'Shape', 'w_cosine_ns',
     "Cosine of the predicted pattern against this signal's own window"),
    ('Evidence weight — signal', 'Δppm (singlet)', 'w_ppm_s',
     'A singlet has no J, so position carries more of the decision'),
    ('Evidence weight — signal', 'Shape (singlet)', 'w_cosine_s',
     'Shape agreement for a singlet'),

    ('Evidence weight — analyte', 'Coverage penalty', 'w_coverage',
     'How much a missing signal lowers the analyte score. Keep it gentle: '
     'signals go missing through overlap and water suppression'),
    ('Evidence weight — analyte', 'Corroboration lift', 'w_corroboration',
     "How much the analyte's other signals raise its score"),

    ('Decision bands', 'Good score threshold', 'good_score_thresh',
     'At or above this a signal is Good (green)'),
    ('Decision bands', 'Bad score threshold', 'bad_score_thresh',
     'Below this a signal is Bad (red); between the two it is Average'),

    ('Other', 'NNLS min fraction', 'nnls_min_frac',
     'Below this fraction of the largest NNLS amplitude, downgrade to Average'),
]

# ΔJ band width (Hz). Mirrors 'dj_gold_hz' in the Annotation Parameters dialog:
# below it the coupling agreement is already perfect, so two candidates there are
# indistinguishable. Used only when the caller has no user-set value.
DJ_GOLD_HZ_DEFAULT = 0.20

# Two candidates whose cosine differs by less than this are not separated by it.
# From the same study: across the contested peaks that ΔJ and Δppm could not
# resolve, cosine separated the top two by more than 0.05 in 59% of cases.
COSINE_TIE = 0.05

# Annotation-score difference below which the score has not separated anything.
# Kept at the margin the previous ambiguity check used, so ranks that were
# resolved by score before stay resolved.
SCORE_TIE = 0.08

# Tier of the evidence a candidate carries (lower = stronger).
TIER_CONFIRMED_MULTIPLET = 0   # every expected line found, J measured
TIER_UNCONFIRMED_MULTIPLET = 1 # multiplet with missing lines or unmeasured J
TIER_SINGLET = 2               # no J evidence by construction

_BIG_BAND = 10 ** 6


def candidate_tier(mult, delta_j_hz, n_found=None, n_expected=None) -> int:
    """Evidence tier of one candidate — the first key of the ranking."""
    if is_singlet_like(mult):
        return TIER_SINGLET
    complete = (n_found is None or n_expected is None
                or (n_found >= n_expected >= 2))
    if delta_j_hz is None or not complete:
        return TIER_UNCONFIRMED_MULTIPLET
    return TIER_CONFIRMED_MULTIPLET


# How many standard deviations of the reference coupling's own uncertainty
# a band spans, when the reference carries one. Three: a delta-J inside three
# sigma of the reference is not a disagreement with the reference, it is the
# reference's own error bar.
REFERENCE_SIGMAS = 3.0


def dj_band(delta_j_hz, gold_hz: float, *, ref_sigma_hz=None) -> int:
    """ΔJ bucketed by ``gold_hz``; unmeasured ΔJ sorts last.

    Banding is what makes a technical tie a tie: ΔJ of 0.046 and 0.096 Hz differ
    by far less than J can be measured (0.28-0.78 Hz between-sample SD in the
    reference study), so both land in band 0 and the decision falls through to
    the next key instead of being settled by noise.

    ``ref_sigma_hz`` is the uncertainty of the reference coupling itself, when
    the database measured it (BMRB_Metabolomics_v02). The band is then never
    narrower than three of those sigma: a reference known to plus or minus
    half a hertz cannot judge a match to a fifth of one, and doing so decided
    contests with the noise of the table -- L-lactic acid, listed at 7.50 Hz
    against a measured 6.94, lost its own doublet to a rival listed at 7.20.
    """
    if delta_j_hz is None or gold_hz <= 0:
        return _BIG_BAND
    width = float(gold_hz)
    if ref_sigma_hz is not None:
        try:
            width = max(width, REFERENCE_SIGMAS * float(ref_sigma_hz))
        except (TypeError, ValueError):
            pass
    return int(abs(float(delta_j_hz)) / width)


def ppm_band(ppm_diff, band_ppm: float = PPM_BAND_PPM) -> int:
    """Δppm bucketed by ``band_ppm`` — see PPM_BAND_PPM for why it is coarse."""
    if ppm_diff is None or band_ppm <= 0:
        return _BIG_BAND
    return int(abs(float(ppm_diff)) / float(band_ppm))


def pascal_agreement(ranks, heights, mult) -> float | None:
    """How well the observed line heights follow the multiplicity's own pattern.

    A triplet is 1:2:1 by the binomial theorem, not by measurement, so three
    apices of equal height are evidence against a triplet claim however well
    their spacing fits. That is evidence no other term in the ranking carries:
    ΔJ and Δppm both come from position.

    ``ranks`` says which expected line each height belongs to, so a partial
    multiplet is judged on the lines it actually has — the outer pair of a
    triplet is 1:1 and must not be scored against 1:2:1.

    Returns 0-1, or None when there is nothing to judge: an unknown
    multiplicity, fewer than two lines, or heights that are missing or all
    zero. None is not disagreement, and the caller must not read it as such.

    Heights are compared after scaling each set to its own maximum, so this
    says nothing about concentration — only about shape.
    """
    pattern = LINE_INTENSITIES.get(normalize_mult(mult))
    if not pattern:
        return None
    rs = list(ranks or ())
    hs = list(heights or ())
    if len(rs) != len(hs) or len(rs) < 2:
        return None
    try:
        obs = [float(h) for h in hs]
    except (TypeError, ValueError):
        return None
    if any(not np.isfinite(o) or o < 0 for o in obs):
        return None
    if not all(0 <= int(r) < len(pattern) for r in rs):
        return None
    exp = [float(pattern[int(r)]) for r in rs]
    o_max, e_max = max(obs), max(exp)
    if o_max <= 0 or e_max <= 0:
        return None
    err = [abs(o / o_max - e / e_max) for o, e in zip(obs, exp)]
    return float(max(0.0, 1.0 - sum(err) / len(err)))


# How strongly a partner resonance must co-vary with the contested peak before
# it counts as corroborating it. Resonances of one molecule move together
# across samples because they are the same molecule at the same concentration;
# 0.80 is well above what unrelated metabolites reach by chance in a study of
# this size, and well below the ~1.0 that two lines of one multiplet show.
STOCSY_MIN_R = 0.80

# Corroboration counts are bucketed rather than compared outright: one partner
# and two partners are different kinds of evidence, four and five are not.
STOCSY_BAND_CAP = 3


def corroborating_partners(peak_rank, partner_ranks, peak_corr,
                           min_r: float = STOCSY_MIN_R) -> int:
    """How many of this analyte's other detected signals co-vary with this one.

    Statistical total correlation spectroscopy: resonances belonging to the
    same molecule rise and fall together across samples, because they are the
    same molecule at the same concentration, while resonances of different
    molecules do not unless the molecules happen to be regulated together.

        Cloarec, O. et al. (2005). Statistical total correlation spectroscopy:
        an exploratory approach for latent biomarker identification from
        metabolic 1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

    This is the evidence that separates two analytes whose contested signal is
    identical. L-lactic acid and N-Acetyl-L-Alanine differ by 0.001 ppm and
    0.30 Hz at the methyl doublet they both claim — nothing about that signal
    can tell them apart. What can is that N-acetyl-alanine has an acetyl
    singlet at 1.999 ppm and lactate has nothing there: if the singlet tracks
    the doublet across the study, the doublet is the acetylated compound's.

    It is a count of partners above ``min_r``, not a mean r, because one strong
    partner is real evidence that a weak second partner should not dilute. A
    mean would let an analyte with four uncorrelated database signals lose to
    one with none.

    Returns 0 when the correlation matrix is unavailable — which must read as
    "no corroboration measured", never as "corroboration refuted".
    """
    if not peak_corr or peak_rank is None:
        return 0
    n = 0
    for other in partner_ranks or ():
        if other is None or other == peak_rank:
            continue
        lo, hi = ((peak_rank, other) if peak_rank < other
                  else (other, peak_rank))
        r = peak_corr.get((lo, hi))
        if r is not None and float(r) >= float(min_r):
            n += 1
    return n


def corroboration_band(n_partners, cap: int = STOCSY_BAND_CAP) -> int:
    """Corroborating partners as a rank band; lower is better.

    Capped because the difference between one partner and two is evidence and
    the difference between four and five is not — past the cap an analyte with
    many database signals would outrank one with few for having a bigger
    molecule rather than better support.

    Zero partners sorts last, and it is the same band whether nothing
    correlated or nothing was measured. That is deliberate: an unmeasured
    correlation is not evidence against the candidate, so it must not be given
    a worse band than a measured refutation would get.
    """
    try:
        n = int(n_partners or 0)
    except (TypeError, ValueError):
        n = 0
    return max(0, int(cap) - min(n, int(cap)))


def pascal_band(agreement, tie: float = PASCAL_TIE) -> int:
    """Height agreement bucketed; unjudgeable sorts last.

    Banded for the same reason ΔJ is: two candidates whose heights agree to
    within the noise of a single apex must not be separated by it.
    """
    if agreement is None or tie <= 0:
        return _BIG_BAND
    return int((1.0 - float(agreement)) / float(tie))


def is_competitor(cand) -> bool:
    """True when this candidate may compete for a peak at all.

    A multiplet whose expected lines were not all found in the spectrum is not
    weak evidence — it is *absent* evidence. J cannot be measured from lines
    that are missing, so the multiplicity claim was never verified and the
    signal was, in effect, not identified. Letting it compete would let an
    unverified claim take a peak away from a verified one.

    The metabolite is not discarded: it remains free to win on any of its other
    signals, and a peak whose every candidate is unverified simply has no
    winner. Singlets carry no J by construction and are unaffected.
    """
    mult = cand.get('multiplicity') or cand.get('type')
    if is_singlet_like(mult):
        return True
    return candidate_tier(mult, cand.get('delta_j_hz'),
                          cand.get('n_found'),
                          cand.get('n_expected')) != TIER_UNCONFIRMED_MULTIPLET


class CandidateKey(NamedTuple):
    """The ranking key, with its criteria named.

    A plain tuple compared the same way, but ``rank_candidates`` read it by
    position — ``k[:5]``, ``k[5]``, ``k[6]`` — so inserting a criterion
    silently made it compare the cosine against the score's tie width. Named
    fields cost nothing at comparison time and make that impossible.
    """
    manual: int
    tier: int
    dj: int
    ppm: int
    quality: int
    intensity: int
    corroboration: int
    cosine: float
    score: float


def candidate_key(cand, *, dj_gold_hz: float, band_ppm: float = PPM_BAND_PPM):
    """Lexicographic ranking key for one candidate (lower = better).

    ``cand`` keys used: ``multiplicity``/``type``, ``delta_j_hz``, ``ppm_diff``,
    ``n_found``, ``n_expected``, ``intensity_agreement``, ``stocsy_partners``,
    ``cosine``, ``manual_quality``, ``quality``, ``score``.

    Order of the evidence, strongest first:

    1. ``manual_quality == 'good'`` — the user looked at the spectrum and
       decided. That judgement is evidence the pipeline does not have, so it
       outranks every computed criterion. Manual 'bad' is handled by the
       caller, which drops the candidate before ranking.
    2. evidence tier — a confirmed multiplet beats an unconfirmed one, which
       beats a singlet.
    3. ΔJ band, then Δppm band — see the module notes on why this order.
    4. annotation quality — a candidate the annotation pipeline did not accept
       as Good must not beat one it did on a similarity tie-break.
    5. line intensities — a triplet is 1:2:1 by the binomial theorem, so three
       apices of equal height argue against the claim however well their
       spacing fits. Position gave ΔJ and Δppm; this is the one thing height
       says. It sits below them because an apex sharing a shoulder with a
       neighbouring analyte reads tall through no fault of the assignment, and
       above the cosine because it judges THIS signal rather than the
       metabolite's fingerprint as a whole. Unjudgeable sorts last, and a
       candidate that cannot be judged is not thereby wrong.
    6. STOCSY corroboration — how many of this analyte's OTHER resonances
       co-vary with the contested peak across the samples. Every criterion
       above judges the contested signal alone, and 61% of this database's
       signals have a rival within 0.005 ppm carrying the same multiplicity, so
       for the majority of contests those criteria are looking at evidence that
       is identical for both claimants by construction. This is the first that
       looks anywhere else, and it is measured from the study rather than read
       from the database. It sits below the direct evidence because a partner
       can co-vary for reasons that are not identity — two metabolites of one
       pathway track each other — and above the cosine and the score because
       both of those are fingerprint blends that this asks a sharper question
       than.
    7. cosine — fingerprint similarity, orthogonal evidence for contests the
       criteria above leave tied.
    8. the annotation score — last resort. It is a Δppm-weighted blend, so it
       must never outrank the criteria above; it only orders what nothing else
       can.
    """
    manual = str(cand.get('manual_quality') or '').lower().strip()
    quality = str(cand.get('quality') or
                  cand.get('annotation_quality') or '').lower().strip()
    mult = cand.get('multiplicity') or cand.get('type')
    return CandidateKey(
        manual=0 if manual == 'good' else 1,
        tier=candidate_tier(mult, cand.get('delta_j_hz'),
                            cand.get('n_found'), cand.get('n_expected')),
        dj=dj_band(cand.get('delta_j_hz'), dj_gold_hz,
                   ref_sigma_hz=cand.get('j_sigma_hz')),
        ppm=ppm_band(cand.get('ppm_diff'), band_ppm),
        quality=0 if quality == 'good' else 1,
        intensity=pascal_band(cand.get('intensity_agreement')),
        corroboration=corroboration_band(cand.get('stocsy_partners')),
        cosine=-float(cand.get('cosine') or 0.0),
        score=-float(cand.get('score') or cand.get('composite_score') or 0.0),
    )


def rank_candidates(candidates, *, dj_gold_hz: float,
                    band_ppm: float = PPM_BAND_PPM):
    """Rank the candidates competing for one peak, best first.

    Candidates that are not competitors (see ``is_competitor``) are dropped
    before ranking, so an unverified multiplet never takes a peak from a
    verified one and never creates an ambiguity of its own.

    Returns ``(ordered, ambiguous)``. ``ambiguous`` is True when the top two are
    indistinguishable on every criterion — same manual mark, tier, ΔJ band,
    Δppm band, quality and line-intensity band, with cosine and score inside
    their tie widths. An ambiguous contest
    has no winner: the caller must not quantify any of them from this peak.
    Reporting a number there would be inventing a decision the data does not
    support (a competitor may still win elsewhere on one of its other signals).
    """
    ordered = sorted(
        [c for c in candidates if is_competitor(c)],
        key=lambda c: candidate_key(c, dj_gold_hz=dj_gold_hz,
                                    band_ppm=band_ppm))
    if len(ordered) < 2:
        return ordered, False
    k0 = candidate_key(ordered[0], dj_gold_hz=dj_gold_hz, band_ppm=band_ppm)
    k1 = candidate_key(ordered[1], dj_gold_hz=dj_gold_hz, band_ppm=band_ppm)
    # Every banded criterion, corroboration included. Leaving one out would
    # declare a contest ambiguous that the ranking had in fact settled, and an
    # ambiguous contest has no winner at all -- so the two analytes the new
    # criterion exists to separate would both end with nothing.
    _banded = ('manual', 'tier', 'dj', 'ppm', 'quality', 'intensity',
               'corroboration')
    ambiguous = (all(getattr(k0, f) == getattr(k1, f) for f in _banded)
                 and abs(k0.cosine - k1.cosine) < COSINE_TIE
                 and abs(k0.score - k1.score) < SCORE_TIE)
    return ordered, ambiguous


# ── Two scores, two questions ────────────────────────────────────────────────
# They were one number, and that conflated two things:
#
#   signal score  — "can I trust THIS signal to quantify with?"
#                   ΔJ, Δppm and the shape agreement measured in this signal's
#                   own window. Nothing about the analyte's other signals: the
#                   evidence carried by alanine's CH3 doublet does not change
#                   because its CH quartet sits under the water suppression.
#
#   analyte score — "is this analyte present at all?"
#                   Built from the signal scores, then softened by how much of
#                   the expected fingerprint was actually found. Missing signals
#                   are normal (overlap, water suppression, below detection), so
#                   the coverage term is deliberately gentle: a single confirmed
#                   multiplet must stay well inside the Good band.
#
# Both return the terms they used, so the UI can show the equation behind the
# number instead of an opaque value. Two different scores lived under the same
# key for a long time precisely because nobody could see how either was built.


@dataclass
class ScoreBreakdown:
    """A score plus the terms that produced it, for display and audit."""
    value: float
    terms: dict = field(default_factory=dict)      # name → contribution
    weights: dict = field(default_factory=dict)    # name → weight applied
    note: str = ''                                 # rule that capped/floored it

    def equation(self) -> str:
        """Human-readable reconstruction, e.g. '0.57·ΔJ(1.00) + 0.10·Δppm(0.88)'."""
        parts = [f'{self.weights[k]:.2f}·{k}({self.terms[k]:.2f})'
                 for k in self.terms if k in self.weights]
        eq = ' + '.join(parts) if parts else f'{self.value:.3f}'
        return f'{eq} = {self.value:.3f}' + (f'  [{self.note}]' if self.note else '')


def signal_score(*, s_ppm: float, s_dj: float | None, s_cosine: float | None,
                 is_singlet: bool, params: dict) -> ScoreBreakdown:
    """Confidence in ONE signal, from its own evidence only.

    ``s_dj`` is None when J could not be measured (component excluded and the
    remaining weights renormalised); ``s_cosine`` is None when the shape could
    not be measured in this signal's window — never substitute a Δppm-derived
    value there, or the term stops being independent of Δppm.

    Weights come from ``params`` (the Annotation Parameters dialog): ``w_dj``,
    ``w_ppm_ns``/``w_ppm_s``, ``w_cosine_ns``/``w_cosine_s``. They are not
    reinvented here.
    """
    if is_singlet:
        # A lone singlet carries no J. It must not reach the Good band on Δppm
        # alone — promotion is left to the corroboration rules, which need
        # evidence a single line cannot provide.
        w = {'Δppm': float(params.get('w_ppm_s', 0.27)),
             'shape': float(params.get('w_cosine_s', 0.32))}
        scores = {'Δppm': float(s_ppm)}
        if s_cosine is not None:
            scores['shape'] = float(s_cosine)
        total_w = sum(w[k] for k in scores) or 1.0
        raw = sum(scores[k] * w[k] for k in scores) / total_w
        cap = float(params.get('bad_score_thresh', 0.50))
        return ScoreBreakdown(value=float(np.clip(raw * cap, 0.0, 1.0)),
                              terms=scores,
                              weights={k: w[k] / total_w for k in scores},
                              note=f'singlet cap ×{cap:g}')

    w = {'ΔJ':    float(params.get('w_dj', 0.57)),
         'Δppm':  float(params.get('w_ppm_ns', 0.10)),
         'shape': float(params.get('w_cosine_ns', 0.14))}
    scores = {'Δppm': float(s_ppm)}
    if s_dj is not None:
        scores['ΔJ'] = float(s_dj)
    if s_cosine is not None:
        scores['shape'] = float(s_cosine)
    total_w = sum(w[k] for k in scores) or 1.0
    raw = sum(scores[k] * w[k] for k in scores) / total_w
    return ScoreBreakdown(value=float(np.clip(raw, 0.0, 1.0)), terms=scores,
                          weights={k: w[k] / total_w for k in scores})


def analyte_score(signal_values, *, coverage: float, params: dict,
                  floor: float | None = None) -> ScoreBreakdown:
    """Confidence that the analyte is present, from its signals' scores.

    ``signal_values`` are the per-signal scores; ``coverage`` is the fraction of
    the analyte's expected DB signals that were matched (0-1). ``floor`` is the
    value the result may not fall below — pass the Good threshold when at least
    one of the analyte's signals satisfied the confirmation rule.

    The best signal is the base: one confirmed multiplet already identifies the
    analyte, and no other resonance can undo that. The remaining signals only
    LIFT the score, never drag it — a weak second signal almost always means
    that resonance is overlapped or suppressed, not that the assignment is
    wrong. (Averaging them in dropped lactate, of all things, out of the Good
    band in a real mouse dataset.)

    The coverage term then applies a gentle penalty weighted by ``w_coverage``
    (0.05 by default), so an analyte seen through a single signal loses a few
    percent, not a band.
    """
    vals = [float(v) for v in (signal_values or []) if v is not None]
    if not vals:
        return ScoreBreakdown(value=0.0, note='no signal')
    best = max(vals)
    others = sorted(vals, reverse=True)[1:]
    mean_others = float(np.mean(others)) if others else 0.0
    # Bounded lift toward 1.0: corroboration helps, and helps less the closer
    # the best signal already is to certainty.
    w_corr = float(params.get('w_corroboration', 0.09))
    lifted = best + (1.0 - best) * w_corr * mean_others

    w_cov = float(params.get('w_coverage', 0.05))
    cov = float(np.clip(coverage, 0.0, 1.0))
    factor = 1.0 - w_cov * (1.0 - cov)
    value = float(np.clip(lifted * factor, 0.0, 1.0))

    note = f'coverage ×{factor:.3f}'
    # A confirmed signal (every expected line found, ΔJ and Δppm within gold)
    # is an identification. The coverage penalty must not undo it — otherwise an
    # analyte confirmed on one resonance is demoted for resonances that are
    # merely hidden. Without this, 13 confirmed analytes in the reference
    # dataset fell from Good to yellow by two thousandths.
    if floor is not None and value < float(floor):
        value = float(floor)
        note += f', floored to {float(floor):.2f} (confirmed signal)'
    return ScoreBreakdown(
        value=value,
        terms={'best': best, 'others': mean_others, 'coverage': cov},
        weights={'best': 1.0, 'others': w_corr, 'coverage': w_cov},
        note=note)


def peak_shape_score(ppm_arr, intensity_arr, shifted_sticks,
                      center_ppm, db_entry, hwhm_default=0.004):
    """
    Compute how well the experimental spectrum shape overlaps the
    simulated DB multiplet (sum of Lorentzianas at shifted_sticks).

    Returns a score 0-1 (1.0 = perfect overlap), or None when the window holds
    too few points to judge. The score is the cosine similarity between the
    experimental slice and the simulated DB shape over a +-6xHWHM window.

    This is the SHAPE term of the signal score, and the only one independent of
    both chemical shift and coupling: it asks whether the predicted pattern of
    line positions and relative intensities is what the spectrum actually shows
    at this signal. Measured in the signal's own window, so it stays local — an
    analyte-wide value cannot separate two candidates competing for one peak.
    """
    if not shifted_sticks:
        return None

    db_width = max(
        abs(db_entry.get('ppm_end', center_ppm) -
            db_entry.get('ppm_start', center_ppm)),
        0.002)
    hwhm = max(db_width / 2, hwhm_default)
    # Window: wide enough to contain the full multiplet + ~4 hwhm of shoulder,
    # but not so wide that neighbouring peaks or broad baseline contaminate the
    # cosine.  hwhm * 8 gives ≈ 4 × linewidth padding on each side.
    win  = hwhm * 8

    lo = center_ppm - win
    hi = center_ppm + win
    mask = (ppm_arr >= lo) & (ppm_arr <= hi)
    if mask.sum() < 3:
        return None

    x   = ppm_arr[mask]
    y_e = intensity_arr[mask].astype(float)

    # Remove linear baseline from experimental slice so that the broad baseline
    # hump doesn't dominate and drag the cosine similarity down.
    # Estimate from the 15% edge samples on each side.
    n = len(y_e)
    edge_n = max(2, int(n * 0.15))
    xl = float(np.mean(x[:edge_n]));   yl = float(np.mean(y_e[:edge_n]))
    xr = float(np.mean(x[-edge_n:]));  yr = float(np.mean(y_e[-edge_n:]))
    if abs(xr - xl) > 1e-9:
        slope = (yr - yl) / (xr - xl)
        y_e   = y_e - (yl + slope * (x - xl))
    y_e = np.clip(y_e, 0, None)   # only positive signal contributes

    # Simulated DB shape at same x points
    y_db = np.zeros_like(x)
    for st_ppm, st_rel in shifted_sticks:
        y_db += st_rel * hwhm**2 / ((x - st_ppm)**2 + hwhm**2)

    # Normalise both to unit vectors
    ne = np.linalg.norm(y_e)
    nd = np.linalg.norm(y_db)
    if ne == 0 or nd == 0:
        return 0.0
    return float(np.dot(y_e / ne, y_db / nd))


# Per-spacing tolerance for the window search. Tied to the ΔJ rejection gate:
# a spacing this search accepts must be one the gate would accept.
SEARCH_SPACING_TOL_HZ = 0.50


def search_multiplet_in_window(peak_ppms, *, centre_ppm: float, j_hz: float,
                               n_expected: int, field_mhz: float,
                               half_width_ppm: float,
                               spacing_tol_hz: float = SEARCH_SPACING_TOL_HZ):
    """Look for a multiplet's comb of lines among the detected peaks.

    A multiplet is an equally spaced comb: ``n_expected`` lines separated by
    ``j_hz``. So the search is a template match — try each peak in the window as
    the leftmost line and ask whether peaks sit at every subsequent position.
    Returns the matched line positions (ascending), or None.

    This replaces an earlier attempt that took the ``n_expected`` TALLEST peaks
    inside the window, ordered them by ppm, and accepted the set when the MEAN
    of their spacings matched the DB J. The mean is blind to the failure it had
    to catch: three peaks spaced 1.2 and 12.8 Hz — two different signals, plainly
    not a triplet — give a mean of exactly 7.0 Hz and were accepted with a
    perfect score. Validating EVERY spacing is what distinguishes one spin
    system from a crowd of neighbours.

    Two geometric conditions, both necessary:
      * every spacing agrees with the DB J within ``spacing_tol_hz``;
      * the spacings agree with EACH OTHER within ``spacing_tol_hz`` — a real
        multiplet is equally spaced, and a set whose spacings drift (11.7, 14.2,
        16.1 Hz for a J of 13.3) is a run of neighbouring signals, not a
        quartet.

    The default tolerance is deliberately tighter than ``J_SPACING_TOL_HZ``: at
    2 Hz the search recovered ten multiplets in a real dataset and most were
    wrong — five different metabolites in one crowded region all matched the
    same peak triple. It is set from ``dj_bad_hz``, the same threshold that
    rejects a ΔJ, so a spacing this search accepts is one the gate would accept.

    No assumption is made about relative intensities; the caller may check the
    pattern separately.
    """
    if n_expected < 2 or j_hz <= 0 or field_mhz <= 0:
        return None
    ppms = sorted(float(p) for p in (peak_ppms or [])
                  if abs(float(p) - centre_ppm) <= half_width_ppm)
    if len(ppms) < n_expected:
        return None

    j_ppm = j_hz / field_mhz
    tol_ppm = spacing_tol_hz / field_mhz
    best = None
    for i, first in enumerate(ppms):
        comb = [first]
        for k in range(1, n_expected):
            target = first + k * j_ppm
            nearest = min((p for p in ppms if p not in comb),
                          key=lambda p: abs(p - target), default=None)
            if nearest is None or abs(nearest - target) > tol_ppm:
                comb = None
                break
            comb.append(nearest)
        if comb is None:
            continue
        seps = [(comb[k + 1] - comb[k]) * field_mhz
                for k in range(len(comb) - 1)]
        # Equally spaced: the spacings must agree with each other, not merely
        # each with the DB J. Without this, a drifting run of neighbouring
        # peaks passes as a multiplet.
        if len(seps) > 1 and (max(seps) - min(seps)) > spacing_tol_hz:
            continue
        # Prefer the comb whose spacings sit closest to the DB J.
        err = sum(abs(sep - j_hz) for sep in seps)
        if best is None or err < best[0]:
            best = (err, comb)
    return sorted(best[1]) if best else None


def coverage_fraction(met, *, dj_bad_hz: float) -> float:
    """Fraction of the analyte's DB signals that were matched in the spectrum.

    A DB signal counts as found when some matched peak sits at its position and
    that peak's J agrees with the database — a multiplet matched at the right
    place with the wrong coupling has not found the signal, it has found
    something else there.

    Unresolved multiplets ('m') are excluded from both sides: the database gives
    them no J and no line pattern, so their absence says nothing.

    Every other unmatched signal lowers the fraction. Signals were once excused
    when their expected SNR fell below the noise floor, which let a metabolite
    look fully covered while most of its fingerprint was missing; that estimate
    is now reported to the analyst instead of quietly rewriting the coverage.
    """
    db_signals = [p for p in (met.get('db_peaks') or [])
                  if normalize_mult(p.get('type')) not in ('m', 'multiplet')]
    if not db_signals:
        return 0.0

    def _j_agrees(mp) -> bool:
        if is_singlet_like(mp.get('type') or mp.get('multiplicity') or 's'):
            return True                      # no J to disagree with
        dj = mp.get('delta_j_hz')
        return dj is None or float(dj) < dj_bad_hz   # None = not measured yet

    matched_at = {round(float(mp.get('db_ppm', 0)), 3)
                  for mp in (met.get('matched_peaks') or []) if _j_agrees(mp)}
    found = sum(1 for p in db_signals
                if round(float(p.get('ppm_center', p.get('ppm', 0))), 3)
                in matched_at)
    return found / len(db_signals)


def expected_lines_for(mp, met, *, mult) -> int:
    """How many lines this multiplet should show.

    The DB fingerprint entry is authoritative: ``mp['predicted_lines']`` may
    have been truncated to the lines that were actually found, and asking a
    truncated list how many lines to expect concludes that every one is present.
    """
    stored = mp.get('predicted_lines') or []
    db_entry = next(
        (p for p in (met.get('db_peaks') or [])
         if round(float(p.get('ppm_center', p.get('ppm', 0))), 3)
         == round(float(mp.get('db_ppm', 0)), 3)), None)
    db_lines = (db_entry.get('predicted_lines') or []) if db_entry else []
    return max(expected_line_count(mult, fallback=len(stored) or 2),
               len(db_lines))


def resolve_dj(mp, met, peaks, *, mult, params, field_mhz: float,
               tolerance: float, remeasure: bool,
               trace=None) -> tuple[float, float]:
    """Settle this signal's ΔJ and return ``(s_dj, dj_confidence)``.

    Writes the measurement back onto *mp* — line counts, observed J, ΔJ, and
    the multiplet centre when it moved.

    ``remeasure`` is False in the normal case: the annotation search already
    snapped the lines and measured J, which is what decided whether the signal
    was identified at all. It is True only when an input has changed since (ppm
    calibration moved the axis, the tolerance changed) or when the search never
    ran for this candidate.

    An incomplete multiplet yields no ΔJ — a coupling measured from a subset
    describes a different set of peaks — and scores on the fraction of lines it
    did find, capped below the singlet boundary.
    """
    if is_singlet_like(mult):
        return 1.0, 1.0                       # no J to validate

    n_expected = expected_lines_for(mp, met, mult=mult)
    stored_lines = mp.get('predicted_lines') or []

    _lw_hz = mp.get('study_linewidth_hz')
    if normalize_mult(mult) == PATTERN_MULT:
        meas = measure_pattern_signal(mp, peaks, field_mhz=field_mhz,
                                      dj_gold_hz=params['dj_gold_hz'],
                                      trace=trace)
        mp['pattern_residual_hz'] = meas.pattern_residual_hz
        if meas.pattern_snr is not None:
            mp['pattern_fit'] = {'snr': meas.pattern_snr,
                                 'explained': meas.pattern_explained,
                                 'r2': meas.pattern_r2,
                                 'amplitude': meas.pattern_amplitude}
            mp['intensity_agreement'] = meas.pattern_intensity_agreement
    elif (not remeasure and mp.get('_n_found_lines') is not None
            and mp.get('_n_exp_lines') is not None):
        meas = LineMeasurement(
            n_expected=int(mp['_n_exp_lines']),
            n_found=int(mp['_n_found_lines']),
            observed_j_hz=mp.get('observed_j_hz'),
            delta_j_hz=mp.get('delta_j_hz'),
            predicted_lines=list(stored_lines),
            measured=True)
    else:
        meas = measure_multiplet(
            stored_lines, peaks, mult=mult,
            j_coupling=mp.get('j_coupling'),
            db_ppm=float(mp.get('db_ppm') or 0),
            centre_hint=float(mp.get('peak_ppm') or 0),
            n_expected=n_expected, tolerance=tolerance, field_mhz=field_mhz,
            linewidth_hz=_lw_hz)

    def _score(dj):
        return dj_score(dj, params['dj_gold_hz'], params['dj_bad_hz'],
                        params['dj_neutral'])

    if not meas.measured:
        # Nothing to snap — fall back to whatever ΔJ was stored.
        stored_dj = mp.get('delta_j_hz')
        return _score(stored_dj), (0.0 if stored_dj is None else 1.0)

    mp['_n_found_lines'] = meas.n_found
    mp['_n_exp_lines'] = meas.n_expected
    if meas.found_ppms:
        mp['predicted_lines'] = meas.predicted_lines

    if not meas.complete:
        mp['delta_j_hz'] = None
        mp['observed_j_hz'] = None
        return float(meas.n_found) / float(meas.n_expected) * 0.5, 1.0

    if meas.observed_j_hz is not None:
        mp['observed_j_hz'] = meas.observed_j_hz
    if meas.delta_j_hz is not None:
        mp['delta_j_hz'] = meas.delta_j_hz
    db_ppm = float(mp.get('db_ppm') or 0)
    if meas.centre_ppm is not None and db_ppm != 0:
        # A pattern's position is judged where its reference position fell,
        # not where its centre of found lines is (see LineMeasurement).
        _at = (meas.origin_ppm if meas.origin_ppm is not None
               else meas.centre_ppm)
        mp['ppm_diff'] = float(abs(_at - db_ppm))
        mp['peak_ppm'] = float(meas.centre_ppm)
        if meas.origin_ppm is not None:
            mp['pattern_centre_offset_ppm'] = float(meas.centre_ppm
                                                    - meas.origin_ppm)
    # Every line found but J still not computable (the DB gave none): the term
    # is excluded rather than scored as a disagreement.
    return _score(meas.delta_j_hz), (0.0 if meas.delta_j_hz is None else 1.0)


def apply_per_sample_consensus(mp, *, params):
    """Replace ΔJ and Δppm with what the samples themselves measured.

    Returns ``(s_dj, dj_confidence)`` when the samples reached a consensus J,
    else None. Δppm is applied whenever a consensus δ exists, independently of
    J — a singlet has a position and no coupling.

    The values these replace were measured on the mean spectrum, which is a
    pointwise maximum across the samples: a trace none of them is. Where a
    signal was read in several samples, the median of those readings is the
    better estimate of the same quantity, and it is the one an analyst can
    check against the per-sample table. ``apply_mcr_dj`` already establishes
    that a later, better measurement replaces the master one.

    Nothing is invented when the samples said nothing: a consensus over zero
    samples leaves the stored value alone.
    """
    con = mp.get('per_sample_consensus') or {}
    if con.get('ppm_diff') is not None and con.get('ppm_centre') is not None:
        mp['ppm_diff'] = float(con['ppm_diff'])
        mp['peak_ppm'] = float(con['ppm_centre'])
        mp['_ppm_source'] = 'per_sample'
    # A measured pattern has no single J. The per-sample fit lays a comb of
    # one spacing across the samples and compares it with the first coupling
    # in the database's list -- 1.15 Hz for D-glucose's doublet of doublets
    # of 8 and 9.5 -- and its consensus overwrote a residual of 0.46 Hz with
    # a disagreement of several. The pattern's own residual stands.
    if normalize_mult(mp.get('type') or mp.get('multiplicity')) == PATTERN_MULT:
        return None
    dj = con.get('delta_j_hz')
    if dj is None or con.get('observed_j_hz') is None:
        return None
    mp['delta_j_hz'] = float(dj)
    mp['observed_j_hz'] = float(con['observed_j_hz'])
    mp['_dj_source'] = 'per_sample'
    return dj_score(dj, params['dj_gold_hz'], params['dj_bad_hz'],
                    params['dj_neutral']), 1.0


def apply_mcr_dj(mp, *, params):
    """Replace ΔJ with the value measured on the deconvolved component.

    Returns ``(s_dj, dj_confidence)`` when MCR resolved this peak, else None.

    The ΔJ measured from lines snapped on the merged spectrum is unreliable by
    construction inside an overlap — the lines belong to more than one analyte.
    MCR measures J on the isolated component instead. Position is untouched:
    deconvolution resolves shape and coupling, not chemical shift.
    """
    mcr_dj = mp.get('mcr_delta_j_hz')
    if mcr_dj is None:
        return None
    mp['delta_j_hz'] = float(mcr_dj)
    mp['_dj_source'] = 'mcr'
    return dj_score(mcr_dj, params['dj_gold_hz'], params['dj_bad_hz'],
                    params['dj_neutral']), 1.0


def is_confirmed(mp, *, dj_conf: float, params) -> bool:
    """True when three independent agreements line up on one signal.

    Every expected line found (a quintet needs five, not three), the coupling of
    those lines matching the database, and the shift within tolerance. That is
    an identification, and a composite number should not be able to rank it
    below a signal with less evidence.
    """
    n_found, n_expected = mp.get('_n_found_lines'), mp.get('_n_exp_lines')
    dj = mp.get('delta_j_hz')
    return bool(
        n_found is not None and n_expected is not None
        and int(n_found) >= int(n_expected) >= 2
        and float(mp.get('ppm_diff', 0) or 0) < params['dppm_gold']
        and dj_conf > 0 and dj is not None
        and float(dj) < params['dj_bad_hz'])


def score_signal(mp, *, s_ppm: float, s_dj: float, dj_conf: float,
                 coverage: float, params) -> ScoreBreakdown:
    """Score one signal and write the result onto *mp*.

    Stores ``composite_score``, the terms behind it, ``_score_breakdown`` and
    ``_confirmed``. Returns the breakdown.

    A signal whose analyte matched nothing at all scores zero: ``coverage`` of
    zero means not one of its database signals was found, so there is no
    fingerprint to have recognised. That gate used to be applied by the caller,
    which left the equation shown to the analyst describing a number the caller
    had since overwritten.
    """
    singlet = is_singlet_like(mp.get('type') or mp.get('multiplicity') or 's')
    breakdown = signal_score(
        s_ppm=s_ppm,
        s_dj=(s_dj if (dj_conf > 0 and not singlet) else None),
        s_cosine=mp.get('_s_cosine_local'),
        is_singlet=singlet, params=params)

    score = breakdown.value
    equation = breakdown.equation()
    note = breakdown.note

    if not singlet and is_confirmed(mp, dj_conf=dj_conf, params=params):
        mp['_confirmed'] = True
        floor = params['good_score_thresh']
        if score < floor:
            score = floor
            note = (f'floored to {floor:.2f} — all lines found, '
                    f'ΔJ and Δppm within gold')
            equation = f'{breakdown.equation()}  → {note}'

    gate = 0.0 if coverage == 0.0 else 1.0
    if gate == 0.0:
        note = 'no database signal of this analyte was matched'
        equation = f'{breakdown.equation()}  → 0.000 ({note})'

    mp['_s_ppm'] = s_ppm
    mp['_s_dj'] = s_dj
    mp['_s_found'] = coverage
    mp['_score_gate'] = gate
    mp['composite_score'] = float(np.clip(score * gate, 0.0, 1.0))
    mp['_score_breakdown'] = equation
    return ScoreBreakdown(value=mp['composite_score'], terms=breakdown.terms,
                          weights=breakdown.weights, note=note)
