"""
Annotation confidence classifier for NMR metabolite identification.

Assigns a confidence class (A–X) to each candidate annotation based on
the available evidence. Classes coexist with the user-assigned
good/bad/undefined quality marks — they represent automatic evidence
strength, not user validation.

Classes (inspired by Metabolomics Standards Initiative levels):
  A — Maximum evidence: multi-peak coverage + independent corroboration
      (STOCSY or MCR) + relative ratio integrity.
  B — Strong evidence: multi-peak coherent match without corroboration.
  C — Reasonable evidence: single-peak high-confidence match.
  D — Tentative: acceptable score but limited evidence.
  X — Rejected: score below minimum threshold.

These thresholds are defaults; callers may pass custom thresholds.
"""

# Default thresholds — can be overridden per-project via _params
_DEFAULT_THRESHOLDS: dict = {
    # Class A
    'a_coverage':   0.80,   # min fraction of DB peaks matched
    'a_stocsy':     0.60,   # min STOCSY Pearson r
    'a_rri':        0.65,   # min relative ratio integrity score
    # Class B
    'b_coverage':   0.65,
    'b_coherence':  0.55,   # min multi-peak coherence score
    # Class C
    'c_score':      0.75,   # min composite_score
    'c_r2':         0.60,   # min fitting_r2  (OR ppm_score ≥ c_ppm_score)
    'c_ppm_score':  0.88,   # min ppm_score   (used when r2 below c_r2)
    # Class D
    'd_score':      0.50,   # min composite_score
}


def classify_annotation(
    candidate: dict,
    thresholds: dict | None = None,
) -> str:
    """
    Return the confidence class ('A'–'X') for a candidate annotation dict.

    The candidate dict is the same structure used throughout the annotation
    pipeline (keys: composite_score, peak_coverage, coherence_score,
    stocsy_coherence, mcr_confirmed, rri, fitting_r2, ppm_score).

    Args:
        candidate:   Annotation candidate dict from search/enrichment pipeline.
        thresholds:  Optional override for classification thresholds.
                     Keys match _DEFAULT_THRESHOLDS; missing keys use defaults.

    Returns:
        One of 'A', 'B', 'C', 'D', 'X'.
    """
    thr = _DEFAULT_THRESHOLDS.copy()
    if thresholds:
        thr.update(thresholds)

    coverage  = float(candidate.get('peak_coverage') or 0.0)
    coherence = float(candidate.get('coherence_score') or 0.0)
    stocsy    = float(candidate.get('stocsy_coherence') or 0.0)
    mcr_ok    = candidate.get('mcr_confirmed') is True
    rri       = float(candidate.get('rri') or 0.0)
    score     = float(candidate.get('composite_score') or 0.0)
    r2        = float(candidate.get('fitting_r2') or 0.0)
    ppm_s     = float(candidate.get('ppm_score') or 0.0)

    # Class A: multi-peak + independent corroboration + intensity ratios
    if (coverage >= thr['a_coverage']
            and (stocsy >= thr['a_stocsy'] or mcr_ok)
            and rri >= thr['a_rri']):
        return 'A'

    # Class B: multi-peak with coherent intensity pattern
    if coverage >= thr['b_coverage'] and coherence >= thr['b_coherence']:
        return 'B'

    # Class C: single-peak but well-fitted
    if score >= thr['c_score'] and (r2 >= thr['c_r2'] or ppm_s >= thr['c_ppm_score']):
        return 'C'

    # Class D: tentative
    if score >= thr['d_score']:
        return 'D'

    return 'X'


# Colour coding for UI badges (background, foreground)
CLASS_COLOURS: dict[str, tuple[str, str]] = {
    'A': ('#1a7a1a', 'white'),   # dark green
    'B': ('#2c7ab5', 'white'),   # blue
    'C': ('#e6a817', 'black'),   # amber
    'D': ('#9e6a00', 'white'),   # brown-orange
    'X': ('#b00020', 'white'),   # red
}

# Publication-level mapping (Metabolomics Standards Initiative)
CLASS_MSI_LEVEL: dict[str, str] = {
    'A': 'Putative ID — Level 2',
    'B': 'Putative ID — Level 2',
    'C': 'Putative ID — Level 3',
    'D': 'Tentative — Level 4',
    'X': 'Rejected',
}
