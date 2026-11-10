# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# controllers/stocsy_controller.py
"""
What the STOCSY window shows, without Tk.

Reads the detected peaks and the project's spectra; changes nothing in the
project. The numbers are ``processing.stocsy``'s, and the per-sample vector of
a seed is the peak's own ``area_per_sample`` -- the same vector the annotation
scoring correlates -- so the window explains the number the scoring used.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.
"""
import collections
import warnings

import numpy as np

from processing.quant_concentration import pqn_group_divergence
from processing.sample_selection import included_sample_indices
from processing.signal_structure import (
    STRUCTURE_MIN_SNR, first_component, group_means, group_tests,
    mean_pairwise_r, sample_components, sample_distance, signal_modules)
from processing.stocsy import (
    AREA_NOISE_KAPPA, SKIP_REASONS, area_noise, build_basis,
    covariation_parts, dilution_spread, fixed_ratio, peak_vectors,
    pqn_factors, rank_partners, stocsy_trace)

# Above this spread (sd/mean) of the PQN factors the samples differ in how
# concentrated they are by more than the study's own variation. Measured on
# the synthetic bench, 12 samples, 3 seeds: 6-8% with no dilution, 15-24%
# with a 30% dilution, 20-37% when half the molecules differ x9 between
# groups (PQN reads a group difference as dilution, and helps there too).
DILUTION_HINT_CV = 0.20

# Points whose spread across samples is under this many times the spectrum's
# median spread are the noise floor, and r there is the seed's correlation
# with noise.
NOISE_FLOOR_FACTOR = 2.0

SIGNIFICANCE_ALPHA = 0.05

# A module is read as the reference or the water tail when this share of its
# signals stands in that zone: the standard sits at 0 ppm with its 13C
# satellites and spinning sidebands, the residual water leaves a tail on both
# sides of the region cut out of detection.
ARTEFACT_SHARE = 0.7
REFERENCE_HALF_PPM = 0.3
WATER_TAIL_PPM = 0.35


class StocsyAnalysis:
    """The STOCSY of one study: spectrum, peaks, samples, and the options the
    analyst changes (correlation, normalisation, within-group)."""

    def __init__(self, ppm, intensities, peaks, samples_info, *,
                 excluded_regions=(), noise_std=None):
        self.ppm = np.asarray(ppm, dtype=float)
        X = np.asarray(intensities, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n = X.shape[1]
        self.included = included_sample_indices(samples_info, n)
        info = list(samples_info or ())
        self.blanks_left_out = n - len(self.included)
        self.X = X[:, self.included]
        self.groups = [
            str((info[i] if i < len(info) else {}).get(
                'label', (info[i] if i < len(info) else {}).get('group', '')))
            for i in self.included]
        self.sample_names = [
            str((info[i] if i < len(info) else {}).get('name', f'S{i + 1}'))
            for i in self.included]
        self.excluded_regions = [tuple(r) for r in excluded_regions or ()]
        self.peaks = {int(p['rank']): p for p in peaks
                      if p.get('rank') is not None}
        self.ranks, self.M, self.skipped = peak_vectors(
            peaks, included=self.included, min_samples=3)
        self._row = {r: i for i, r in enumerate(self.ranks)}
        for r, flat in zip(self.ranks, np.std(self.M, axis=1) <= 0):
            if flat:
                self.skipped[r] = 'flat'
        self.dilution_cv = (dilution_spread(self.M)
                            if len(self.ranks) >= 3 else 0.0)
        self.method = 'pearson'
        self.normalize = 'none'
        self.within_groups = False
        self._basis = None
        self._trace_cache = {}
        self._modules_cache = {}
        self.area_sigma, self.kappa = self._area_noise(
            samples_info, noise_std)

    # ── options ──────────────────────────────────────────────────────────────

    def set_options(self, *, method=None, normalize=None, within_groups=None):
        if method is not None:
            self.method = method
        if normalize is not None:
            self.normalize = normalize
        if within_groups is not None:
            self.within_groups = bool(within_groups)
        self._basis = None
        self._trace_cache.clear()
        self._modules_cache.clear()

    @property
    def basis(self):
        if self._basis is None:
            M = self.M if len(self.ranks) else np.zeros((1, len(self.included)))
            self._basis = build_basis(
                M, groups=self.groups, normalize=self.normalize,
                within_groups=self.within_groups, method=self.method)
        return self._basis

    @property
    def can_center_by_group(self) -> bool:
        counts = {g: self.groups.count(g) for g in set(self.groups)}
        return sum(1 for c in counts.values() if c >= 2) >= 2

    # ── peaks ────────────────────────────────────────────────────────────────

    def peaks_in_region(self, hi, lo):
        """Detected peaks with δ in [lo, hi], high δ first. Peaks that cannot
        be correlated are listed with their reason (``skipped``), not hidden."""
        lo, hi = min(hi, lo), max(hi, lo)
        out = []
        for rank, p in self.peaks.items():
            ppm = float(p.get('ppm', np.nan))
            if lo <= ppm <= hi:
                out.append({'rank': rank, 'ppm': ppm,
                            'snr': float(p.get('snr') or 0.0),
                            'skipped': (SKIP_REASONS[self.skipped[rank]]
                                        if rank in self.skipped else None)})
        out.sort(key=lambda d: -d['ppm'])
        return out

    def can_seed(self, rank) -> bool:
        """Whether the peak has a per-sample vector that varies."""
        return rank in self._row and rank not in self.skipped

    # ── numbers ──────────────────────────────────────────────────────────────

    def trace(self, seed_rank):
        """STOCSY of the seed peak over the whole spectrum: ``{'r', 'cov',
        'std'}`` per point."""
        key = int(seed_rank)
        return self.trace_of(self.M[self._row[key]], key)

    def trace_of(self, vector, key):
        """STOCSY over the whole spectrum of any per-sample vector (a signal
        made of several peaks, say); ``key`` names it in the cache."""
        if key not in self._trace_cache:
            self._trace_cache[key] = stocsy_trace(
                self.ppm, self.X, np.asarray(vector, dtype=float), self.basis,
                masked_regions=self.excluded_regions,
                min_std_factor=NOISE_FLOOR_FACTOR)
        return self._trace_cache[key]

    def partners(self, seed_rank, region=None):
        """Peaks co-varying with the seed, strongest r first, each with its
        δ and whether it clears the significance bar. ``region`` is (hi, lo)."""
        rows = [r for r in self.ranks
                if region is None or self._in(r, region)]
        seed = self.M[self._row[int(seed_rank)]]
        sub = self.M[[self._row[r] for r in rows]] if rows else self.M[:0]
        crit = self.basis.r_critical
        out = rank_partners(seed, rows, sub, self.basis,
                            seed_rank=int(seed_rank))
        if out:
            at = {r: i for i, r in enumerate(rows)}
            parts = covariation_parts(seed, sub, self.groups,
                                      method=self.method,
                                      factors=self.basis.factors)
            ratio = self.ratio_to(seed_rank)
        for d in out:
            d['ppm'] = float(self.peaks[d['rank']]['ppm'])
            d['significant'] = bool(np.isfinite(d['r'])
                                    and d['p'] < SIGNIFICANCE_ALPHA)
            d['r_critical'] = crit
            k, g = at[d['rank']], self._row[d['rank']]
            d['r_within'] = float(parts['within'][k])
            d['r_between'] = float(parts['between'][k])
            d['ratio'] = float(ratio['ratio'][g])
            d['ratio_sd'] = float(ratio['sd'][g])
            d['ratio_noise_sd'] = float(ratio['noise_sd'][g])
            d['ratio_n'] = int(ratio['n'][g])
        return out

    def matrix(self, ranks):
        """r and p among the chosen peaks: ``(ranks, R, P)``; peaks that
        cannot be correlated are dropped."""
        ranks = [r for r in ranks if r in self._row]
        if len(ranks) < 2:
            return ranks, np.empty((0, 0)), np.empty((0, 0))
        V = self.M[[self._row[r] for r in ranks]]
        R = self.basis.correlate(V, V)
        P = np.vstack([self.basis.stats(row)[0] for row in R])
        return ranks, R, P

    def pair_values(self, rank_x, rank_y):
        """The two area vectors as the correlation sees them (normalised, not
        ranked or centred), with the group and name of each sample."""
        V = self.basis.scale(self.M[[self._row[rank_x], self._row[rank_y]]])
        keep = self.basis.keep
        return (V[0], V[1], [self.groups[i] for i in keep],
                [self.sample_names[i] for i in keep])

    # ── noise of an area ─────────────────────────────────────────────────────

    def _area_noise(self, samples_info, noise_std):
        """sd of each peak's area (aligned with ``self.ranks``) and the kappa
        behind it. The noise of one point comes from ``noise_std`` (what
        detection used) or, without it, from the point-to-point differences of
        every sample. Kappa -- how many times white noise the integration
        adds -- is read from the blank tubes when there are at least three,
        else ``AREA_NOISE_KAPPA``."""
        if not self.ranks:
            return np.array([]), AREA_NOISE_KAPPA
        if noise_std is None or not np.isfinite(noise_std) or noise_std <= 0:
            d = np.abs(np.diff(self.X, axis=0))
            noise_std = float(np.median(np.median(d, axis=0)) / 0.6745
                              / np.sqrt(2))
        self.point_noise = float(noise_std)
        dx = abs(float(self.ppm[1] - self.ppm[0])) if self.ppm.size > 1 else 1.0
        npts = np.maximum(np.array([
            int(self.peaks[r].get('end_idx', 0))
            - int(self.peaks[r].get('start_idx', 0)) + 1
            for r in self.ranks]), 1)
        kappa = AREA_NOISE_KAPPA
        blanks = [i for i, x in enumerate(samples_info or ())
                  if str((x or {}).get('type', '')).upper() == 'BLK']
        if len(blanks) >= 3:
            A = np.array([np.asarray(self.peaks[r]['area_per_sample'], float)
                          for r in self.ranks])
            if A.shape[1] > max(blanks):
                sd = A[:, blanks].std(axis=1, ddof=1)
                white = noise_std * dx * np.sqrt(npts)
                ok = np.isfinite(sd) & (sd > 0) & (white > 0)
                if ok.sum() >= 10:
                    kappa = float(np.clip(np.median(sd[ok] / white[ok]),
                                          1.0, 6.0))
        return area_noise(noise_std, dx, npts, kappa), kappa

    def window_sigma(self, width_ppm) -> float:
        """sd of an area integrated over ``width_ppm``, with the noise and
        kappa the peaks' own areas were given."""
        if not self.ranks or self.ppm.size < 2:
            return np.nan
        dx = abs(float(self.ppm[1] - self.ppm[0]))
        npts = max(int(round(abs(float(width_ppm)) / dx)) + 1, 1)
        return float(area_noise(self.point_noise, dx, np.array([npts]), self.kappa)[0])

    def ratio_to(self, seed_rank):
        """``fixed_ratio`` of every peak against the seed: how constant their
        ratio is across the samples against measurement noise. Dict of arrays
        aligned with ``self.ranks`` (see ``processing.stocsy.fixed_ratio``)."""
        i = self._row[int(seed_rank)]
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            return fixed_ratio(self.M[i], self.M, self.area_sigma[i],
                               self.area_sigma)

    # ── blocks of signals and axes of the samples ────────────────────────────

    def _strong(self, min_snr):
        return [r for r in self.ranks if self.can_seed(r)
                and float(self.peaks[r].get('snr') or 0.0) >= min_snr]

    def ppm_of(self, rank):
        return float(self.peaks[rank]['ppm'])

    def _artefact_kind(self, members):
        """'reference', 'water tail' or None for a block of signals."""
        ppms = np.array([self.ppm_of(r) for r in members])
        if (np.abs(ppms) <= REFERENCE_HALF_PPM).mean() >= ARTEFACT_SHARE:
            return 'reference'
        near = np.zeros(len(ppms), bool)
        for a, b in self.excluded_regions:
            near |= ((ppms >= min(a, b) - WATER_TAIL_PPM)
                     & (ppms <= max(a, b) + WATER_TAIL_PPM))
        if near.mean() >= ARTEFACT_SHARE:
            return 'water tail'
        return None

    def modules(self, r_cut=0.8, min_snr=STRUCTURE_MIN_SNR):
        """Blocks of signals that move together on the current basis, with
        what each does between the groups.

        Each entry: ``id, ranks`` (high δ first), ``n, mean_r, share`` (of the
        block's variance in its first component), ``scores`` (that component,
        one per sample), ``p`` (group tests of the scores: ``group`` and, when
        the labels split in two factors, ``A``, ``B``, ``AxB``), ``means``
        (score by group), ``names`` (annotation counts of its members),
        ``artefact`` ('reference', 'water tail' or None). Biggest first."""
        key = (round(float(r_cut), 4), float(min_snr))
        if key in self._modules_cache:
            return self._modules_cache[key]
        ranks = self._strong(min_snr)
        out = []
        if len(ranks) >= 3:
            V = self.M[[self._row[r] for r in ranks]]
            R = self.basis.correlate(V, V)
            blocks, _ = signal_modules(R, r_cut=r_cut)
            S = self.basis.scale(V)
            groups = [self.groups[i] for i in self.basis.keep]
            for n, b in enumerate(blocks, 1):
                members = sorted((ranks[i] for i in b),
                                 key=lambda r: -self.ppm_of(r))
                sc, share, _ = first_component(S[b])
                names = collections.Counter(
                    self.peaks[r].get('assigned_metabolite') for r in members
                    if self.peaks[r].get('assigned_metabolite'))
                out.append({
                    'id': n, 'ranks': members, 'n': len(members),
                    'mean_r': mean_pairwise_r(R, b), 'share': share,
                    'scores': sc, 'p': group_tests(sc, groups),
                    'means': group_means(sc, groups), 'names': dict(names),
                    'artefact': self._artefact_kind(members)})
        self._modules_cache[key] = out
        return out

    def components(self, k=4, min_snr=STRUCTURE_MIN_SNR):
        """Axes of variation of the samples over the signals standing
        ``min_snr`` over the noise: ``variance``, group tests of the scores,
        group means, ``loadings`` (the eight biggest: rank, δ, loading) and
        the correlation of the axis with the PQN factor."""
        ranks = self._strong(min_snr)
        if len(ranks) < 3 or len(self.included) < 4:
            return []
        V = self.M[[self._row[r] for r in ranks]]
        pc = sample_components(self.basis.scale(V), k)
        keep = self.basis.keep
        groups = [self.groups[i] for i in keep]
        f = np.log(pqn_factors(self.M))[keep]
        out = []
        for j in range(pc['scores'].shape[1]):
            sc = pc['scores'][:, j]
            load = pc['loadings'][:, j]
            top = np.argsort(-np.abs(load))[:8]
            out.append({
                'id': j + 1, 'variance': float(pc['variance'][j]),
                'p': group_tests(sc, groups), 'means': group_means(sc, groups),
                'loadings': [(ranks[i], self.ppm_of(ranks[i]), float(load[i]))
                             for i in top],
                'r_pqn': (float(np.corrcoef(sc, f)[0, 1])
                          if np.ptp(f) > 0 and np.ptp(sc) > 0
                          else float('nan')),
                'scores': sc})
        return out

    def samples(self, k=3, min_snr=STRUCTURE_MIN_SNR):
        """One row per sample: name, group, PQN factor, the scores on the first
        ``k`` axes and the distance from the others (a screening number)."""
        comps = self.components(k, min_snr)
        f = pqn_factors(self.M)
        keep = list(self.basis.keep)
        dist = (sample_distance(np.column_stack([c['scores'] for c in comps]))
                if comps else np.zeros(len(keep)))
        return [{'name': self.sample_names[i], 'group': self.groups[i],
                 'pqn_factor': float(f[i]),
                 'scores': [float(c['scores'][pos]) for c in comps],
                 'distance': float(dist[pos])}
                for pos, i in enumerate(keep)]

    def dilution_by_group(self):
        """``pqn_group_divergence`` of the study's factors, or None."""
        if len(self.ranks) < 3:
            return None
        f = pqn_factors(self.M)
        by = {}
        for pos in range(len(self.included)):
            by.setdefault(self.groups[pos], []).append(pos)
        return pqn_group_divergence({i: float(f[i]) for i in range(len(f))}, by)

    # ── what the analyst is told ─────────────────────────────────────────────

    def notes(self):
        notes = []
        if self.blanks_left_out:
            notes.append(f'{self.blanks_left_out} blank/excluded sample(s) '
                         'left out; correlating '
                         f'{len(self.included)} analytical samples.')
        notes.extend(self.basis.notes)
        lost = {r: w for r, w in self.skipped.items() if w != 'flat'}
        if lost:
            why = sorted({SKIP_REASONS[w] for w in lost.values()})
            notes.append(f'{len(lost)} signal(s) cannot be correlated: '
                         f'{"; ".join(why)}.')
        if self.normalize == 'none' and self.dilution_cv >= DILUTION_HINT_CV:
            notes.append(
                f'The samples differ in concentration by '
                f'{100 * self.dilution_cv:.0f}% (sd of the PQN factors): '
                'unrelated signals co-vary through it. Try PQN.')
        div = self.dilution_by_group()
        if div and div['diverges'] and self.normalize == 'pqn':
            by = ', '.join(f'{g} {v:.2f}' for g, v in div['by_group'].items())
            notes.append(
                f'The PQN factors differ between groups ({by}; ratio '
                f'{div["ratio"]:.2f}): PQN takes part of the group difference '
                'out, so r on PQN is about composition, not amount.')
        if self.normalize == 'pqn' and self.dilution_cv < DILUTION_HINT_CV / 2:
            notes.append(
                f'PQN factors vary by only {100 * self.dilution_cv:.0f}%: '
                'there is little dilution to remove, and PQN also divides '
                'away part of the abundant molecules\' own variation.')
        return notes

    def export_rows(self, seed_rank, region=None):
        """Rows for a CSV of the seed's partners, with the options used."""
        b = self.basis
        rows = []
        for d in self.partners(seed_rank, region):
            rows.append({
                'seed_rank': int(seed_rank),
                'seed_ppm': float(self.peaks[int(seed_rank)]['ppm']),
                'partner_rank': d['rank'], 'partner_ppm': d['ppm'],
                'r': d['r'], 'ci_low': d['ci_low'], 'ci_high': d['ci_high'],
                'p': d['p'], 'n': d['n'], 'dof': d['dof'],
                'significant_p<0.05': d['significant'],
                'r_within_groups': d['r_within'],
                'r_between_groups': d['r_between'],
                'ratio_to_seed': d['ratio'], 'ratio_sd_log': d['ratio_sd'],
                'ratio_noise_sd': d['ratio_noise_sd'],
                'method': self.method, 'normalization': self.normalize,
                'within_groups': b.within_groups})
        return rows

    # ── helpers ──────────────────────────────────────────────────────────────

    def _in(self, rank, region):
        hi, lo = max(region), min(region)
        return lo <= float(self.peaks[rank]['ppm']) <= hi

