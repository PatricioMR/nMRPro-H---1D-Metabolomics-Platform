# processing/pathway_enrichment.py
"""
Two-group pathway enrichment (MetaboAnalyst-style ORA / MSEA).

Pipeline:
  1. per_metabolite_test  — t-test (or Mann-Whitney when a group's profile is
                            non-normal) on each metabolite between two groups,
                            plus log2 fold-change of the group means.
  2. select_hits          — metabolites with p < threshold form the hit set.
  3. compute_ora          — Over-Representation Analysis (Fisher exact, greater)
                            per pathway, Enrichment Ratio = observed / expected
                            hits, Benjamini-Hochberg FDR.

Pure numeric logic — no GUI, no DB writes. Pathway background sizes are passed
in by the caller (from data.pathway_membership.pathway_background_sizes).
"""
import numpy as np


def _shapiro_normal(values, alpha=0.05):
    """True if Shapiro-Wilk cannot reject normality (p >= alpha)."""
    from scipy import stats as spstats
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or np.ptp(v) == 0:
        return True   # too few / constant → assume parametric
    try:
        _, p = spstats.shapiro(v[:5000])
        return p >= alpha
    except Exception:
        return True


def per_metabolite_test(X, group_labels, group_a, group_b, feature_labels):
    """
    Test each metabolite (column of X) between two groups.

    X             : (n_samples, n_features) array of concentrations.
    group_labels  : length-n_samples sequence of group names.
    group_a/b     : the two group names to compare.
    feature_labels: length-n_features metabolite names.

    For each metabolite returns p-value and log2 fold-change of mean(A)/mean(B).
    A Mann-Whitney U test is used when either group's values are non-normal
    (Shapiro), otherwise Welch's t-test.

    Returns
    -------
    dict
        {metabolite_name: {'p': float, 'log2fc': float, 'parametric': bool}}
    """
    from scipy import stats as spstats

    X = np.asarray(X, dtype=float)
    gl = list(group_labels)
    idx_a = [i for i, g in enumerate(gl) if g == group_a]
    idx_b = [i for i, g in enumerate(gl) if g == group_b]

    out = {}
    for j, name in enumerate(feature_labels):
        a = X[idx_a, j]
        b = X[idx_b, j]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            continue

        parametric = _shapiro_normal(a) and _shapiro_normal(b)
        p = 1.0
        try:
            if parametric:
                _, p = spstats.ttest_ind(a, b, equal_var=False)
            else:
                _, p = spstats.mannwhitneyu(a, b, alternative='two-sided')
            p = float(p)
            if not np.isfinite(p):
                p = 1.0
        except Exception:
            p = 1.0

        ma, mb = float(np.mean(a)), float(np.mean(b))
        if ma > 0 and mb > 0:
            log2fc = float(np.log2(ma / mb))
        else:
            log2fc = 0.0

        out[name] = {'p': p, 'log2fc': log2fc, 'parametric': parametric}

    return out


def select_hits(test_results, p_thr=0.05):
    """Metabolite names with p < p_thr (the ORA hit set)."""
    return {name for name, r in test_results.items() if r.get('p', 1.0) < p_thr}


def compute_ora(pathway_table, hit_names, *, background_sizes=None,
                n_background=380):
    """
    Over-Representation Analysis (Fisher exact, alternative='greater') per
    pathway against a fixed background.

    pathway_table   : {pw_name: {'mets': [...], 'smpdb_id': ...}}
    hit_names       : set of significant metabolite names.
    background_sizes: {pw_name: K} pathway sizes in the background; falls back
                      to the number of detected mets when a pathway is missing.
    n_background    : total background metabolite count (N).

    Returns list of dicts sorted by p, each with:
        name, k (hits in pathway), K (pathway size), n (total hits),
        enrichment_ratio (= k / expected), p, fdr, smpdb.
    """
    from scipy.stats import fisher_exact

    hit_names = set(hit_names)
    background_sizes = background_sizes or {}
    N = max(int(n_background), 1)
    n = len(hit_names)

    results = []
    for pw_name, info in pathway_table.items():
        mets = info.get('mets', [])
        K = background_sizes.get(pw_name, len(mets))
        k = len([m for m in mets if m in hit_names])
        if k == 0:
            continue
        table = [[k,     n - k],
                 [K - k, N - n - (K - k)]]
        table[1][1] = max(table[1][1], 0)
        try:
            _, p = fisher_exact(table, alternative='greater')
        except Exception:
            p = 1.0
        expected = n * K / N
        enrichment_ratio = k / max(expected, 1e-9)
        results.append({
            'name':             pw_name,
            'k':                k,
            'K':                K,
            'n':                n,
            'enrichment_ratio': float(enrichment_ratio),
            'p':                float(p),
            'smpdb':            info.get('smpdb_id', ''),
            'mets':             [m for m in mets if m in hit_names],
        })

    # Benjamini-Hochberg FDR
    results.sort(key=lambda r: r['p'])
    m = len(results)
    for rank, r in enumerate(results, 1):
        r['fdr'] = float(min(r['p'] * m / rank, 1.0))

    return results
