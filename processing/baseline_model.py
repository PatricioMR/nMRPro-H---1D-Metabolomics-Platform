"""
The project's single baseline decision.

Baseline correction used to happen in several places, each with its own answer:
the Visualization step overwrote the loaded spectra destructively, signal
detection corrected a working copy, quantification ran its own hard-coded
arPLS, and the statistics and full-spectrum MCR steps read the raw matrix with
no correction at all.  A dataset therefore had no single answer to "was this
baseline-corrected, and how" — and the two steps that skipped it silently lost
statistical power to baseline drift.

This module holds that decision once.  ``BaselineModel`` names the method and
its parameters; ``apply`` turns raw intensities into corrected ones.  Nothing
here mutates its input: the raw matrix stays the record of what was measured,
and the correction is recomputed from it, so the choice is always reversible
and always auditable.

What is being removed is not only instrumental distortion.  In plasma and serum
most of it is the broad macromolecule envelope, which is genuine NMR signal;
subtracting it is what lets a small metabolite resting on that envelope be
measured honestly, but it is a subtraction of real signal and is recorded as
such (see ``broad_removed_fraction``).

References
----------
Baek, S.-J., et al., "Baseline correction using asymmetrically reweighted
    penalized least squares smoothing", Analyst 140 (2015) 250–257.
Ryan, C. G., et al., "SNIP, a statistics-sensitive background treatment for the
    quantitative analysis of PIXE spectra", Nucl. Instr. Meth. B 34 (1988)
    396–402.
"""

import numpy as np

from .baseline_correction import correct_baseline, normalize_method

# Methods offered, and the parameter each one uses to set its width cutoff —
# how broad a feature has to be before it counts as baseline.
METHOD_PARAMS = {
    'snip':     ('niter', 40),
    'arPLS':    ('lam',   1e5),
    'rolling':  (None,    None),
    'constant': (None,    None),
}

METHODS = tuple(METHOD_PARAMS)

# SNIP preserves ~99% of a sharp apex against ~92% for arPLS at lam=1e5, which
# eats the peak skirts and biases integrals low by roughly 20% of the area.
DEFAULT_METHOD = 'snip'


class BaselineModel:
    """Method + parameters for the whole project's baseline correction."""

    __slots__ = ('enabled', 'method', 'params')

    def __init__(self, enabled=True, method=DEFAULT_METHOD, params=None):
        self.enabled = bool(enabled)
        # normalize_method maps the historical 'airPLS' spelling onto 'arPLS',
        # so projects saved before the rename reopen on the same algorithm.
        self.method = normalize_method(method)
        if self.method not in METHOD_PARAMS:
            self.method = DEFAULT_METHOD
        self.params = dict(params) if params else {}
        self._fill_default_param()

    def _fill_default_param(self):
        name, default = METHOD_PARAMS[self.method]
        if name is None:
            self.params = {}
        elif name not in self.params:
            self.params = {name: default}

    # ── Identity ──────────────────────────────────────────────────────────────

    def key(self):
        """Hashable identity — two models with the same key give the same result."""
        return (self.enabled, self.method,
                tuple(sorted((k, float(v)) for k, v in self.params.items())))

    def __eq__(self, other):
        return isinstance(other, BaselineModel) and self.key() == other.key()

    def __repr__(self):
        if not self.enabled:
            return 'BaselineModel(off)'
        p = ', '.join(f'{k}={v:g}' for k, v in self.params.items())
        return f'BaselineModel({self.method}{", " + p if p else ""})'

    def describe(self):
        """One-line description for the UI and for run logs."""
        if not self.enabled:
            return 'no baseline correction'
        p = ', '.join(f'{k}={v:g}' for k, v in self.params.items())
        return f'{self.method}{f" ({p})" if p else ""}'

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self):
        return {'enabled': self.enabled, 'method': self.method,
                'params': dict(self.params)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return cls()
        return cls(enabled=data.get('enabled', True),
                   method=data.get('method', DEFAULT_METHOD),
                   params=data.get('params'))

    # ── Application ───────────────────────────────────────────────────────────

    def param_value(self):
        """The single width-cutoff parameter, or None for methods without one."""
        name, _ = METHOD_PARAMS[self.method]
        return None if name is None else self.params.get(name)

    def apply(self, ppm, intensities):
        """Return baseline-corrected intensities, same shape as the input.

        The input is never modified. Disabled models return the input unchanged
        (as a float array) rather than a zeroed or copied-and-corrupted one.
        """
        arr = np.asarray(intensities, dtype=float)
        if not self.enabled:
            return arr
        ppm = np.asarray(ppm, dtype=float)

        if arr.ndim == 1:
            corrected, _ = correct_baseline(ppm, arr, method=self.method,
                                            **self.params)
            return np.asarray(corrected, dtype=float)

        out = np.empty_like(arr)
        for s in range(arr.shape[1]):
            corrected, _ = correct_baseline(ppm, arr[:, s], method=self.method,
                                            **self.params)
            out[:, s] = corrected
        return out


class BaselineCache:
    """Memoises one corrected matrix for a (raw data, model) pair.

    Correction is recomputed from the raw data rather than stored in place, so
    every reader would otherwise pay for it. The entry is keyed on the raw array
    *identity* and the model's key: replacing the data or changing the model
    both recompute, and nothing else can go stale.
    """

    __slots__ = ('_raw', '_key', '_corrected')

    def __init__(self):
        self.invalidate()

    def invalidate(self):
        self._raw = None
        self._key = None
        self._corrected = None

    def corrected(self, ppm, raw, model, already_corrected=False):
        """Corrected intensities for ``raw`` under ``model``.

        Returns ``raw`` itself — not a copy — when no work is needed, so an
        off model or an already-corrected dataset costs nothing.
        """
        if raw is None:
            return None
        if model is None or not model.enabled or already_corrected:
            return raw

        key = model.key()
        if self._raw is raw and self._key == key:
            return self._corrected

        corrected = model.apply(ppm, raw)
        self._raw, self._key, self._corrected = raw, key, corrected
        return corrected


def broad_removed_fraction(raw, corrected, x, lo_idx, hi_idx):
    """Share of a window's raw area that the baseline model removed.

    0 = nothing removed; →1 = the window was almost entirely broad component,
    so the reported area depends more on the baseline model than on the analyte.
    Returned per sample for 2-D input, as a scalar for 1-D.

    This is the audit trail for subtracting real NMR signal: without it the
    macromolecule envelope simply vanishes from the record.
    """
    raw = np.asarray(raw, dtype=float)
    corrected = np.asarray(corrected, dtype=float)
    x_win = np.asarray(x, dtype=float)[lo_idx:hi_idx + 1]
    if x_win.size < 2:
        return 0.0 if raw.ndim == 1 else np.zeros(raw.shape[1])

    def _one(r, c):
        a_signal = abs(float(np.trapezoid(c, x_win)))
        a_base = abs(float(np.trapezoid(r - c, x_win)))
        denom = a_base + a_signal
        return a_base / denom if denom > 1e-30 else 0.0

    if raw.ndim == 1:
        return _one(raw[lo_idx:hi_idx + 1], corrected[lo_idx:hi_idx + 1])
    return np.array([_one(raw[lo_idx:hi_idx + 1, s],
                          corrected[lo_idx:hi_idx + 1, s])
                     for s in range(raw.shape[1])])
