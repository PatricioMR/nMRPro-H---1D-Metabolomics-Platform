"""
Dynamic evidence weighting for NMR annotation scoring.

Replaces static weighted sums with a renormalizing scheme that excludes
unavailable evidence components rather than defaulting them to neutral values.
"""

# Default weights for the primary annotation search composite score
_BASE_WEIGHTS_SEARCH: dict[str, float] = {
    'ppm':  0.45,
    'r2':   0.30,
    'mult': 0.15,
    'j':    0.10,
}

# Default weights for the results-window enrichment score (non-singlet)
_BASE_WEIGHTS_ENRICHMENT_NS: dict[str, float] = {
    'dj':     0.57,
    'cosine': 0.14,
    'ppm':    0.10,
    'found':  0.05,
    'rri':    0.09,
    'rev':    0.05,
}

# Default weights for the results-window enrichment score (singlet)
_BASE_WEIGHTS_ENRICHMENT_S: dict[str, float] = {
    'cosine': 0.32,
    'ppm':    0.27,
    'found':  0.13,
    'rri':    0.18,
    'rev':    0.10,
}


def dynamic_composite(
    scores: dict,
    confidences: dict,
    base_weights: dict | None = None,
) -> float:
    """
    Compute a weighted composite score that excludes components with
    confidence=0 and renormalises the remaining weights to sum to 1.

    Args:
        scores:       {key: score_value}  — all values in [0, 1]
        confidences:  {key: confidence}   — 0 excludes the component,
                                            1 uses it at full weight,
                                            intermediate values scale it
        base_weights: {key: weight}       — unnormalised base weights;
                                            defaults to _BASE_WEIGHTS_SEARCH

    Returns:
        Composite score in [0, 1], or 0.0 if all components are excluded.

    Examples:
        # J not measurable: j component excluded, rest renormalised
        dynamic_composite(
            scores={'ppm': 0.9, 'r2': 0.8, 'mult': 0.7, 'j': 0.0},
            confidences={'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.0},
        )
        # → (0.9*0.45 + 0.8*0.30 + 0.7*0.15) / (0.45+0.30+0.15)

        # Alignment uncertain: ppm weight reduced
        dynamic_composite(
            scores={'ppm': 0.85, 'r2': 0.75, 'mult': 0.6, 'j': 0.5},
            confidences={'ppm': 0.70, 'r2': 1.0, 'mult': 1.0, 'j': 1.0},
        )
    """
    if base_weights is None:
        base_weights = _BASE_WEIGHTS_SEARCH

    total_w = 0.0
    weighted_sum = 0.0
    for key, score in scores.items():
        w = base_weights.get(key, 0.0)
        c = confidences.get(key, 1.0)
        effective_w = w * c
        total_w += effective_w
        weighted_sum += score * effective_w

    if total_w < 1e-9:
        return 0.0
    return weighted_sum / total_w


def j_confidence(observed_j_hz) -> float:
    """
    Return J evidence confidence: 1.0 if J was measured, 0.0 if absent.
    observed_j_hz: float or None
    """
    return 1.0 if (observed_j_hz is not None) else 0.0


def alignment_ppm_confidence(alignment_ok: bool) -> float:
    """
    Return ppm confidence given whether the DSS/TSP alignment succeeded.
    When alignment fails the PPM axis is uncertain: reduce ppm weight.
    """
    return 1.0 if alignment_ok else 0.70
