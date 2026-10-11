"""What a new analysis starts with, before the analyst touches anything.

One place, read by the interface when it builds its controls and by
``validation.fresh_run`` when it replays an analysis the way a new user
would. Kept apart from the widgets so the two cannot drift: a default that
lived only in a ``tk.BooleanVar(value=...)`` was invisible to everything
that tried to reproduce the application, and the validation bench ended up
measuring settings no user would have had.

Every value here is a decision with a measurement behind it -- the ADR it
came from is named beside it. Changing one changes what every new
analysis reports, so it goes through the same A/B as any other change.
"""
from __future__ import annotations

from processing.spectral_regions import DEFAULT_EXCLUDED_REGIONS

DETECTION = {
    # The project-level baseline model (SNIP), applied before DETECTION.
    # Whether quantification integrates on it is QUANTIFICATION's
    # 'baseline_model', and the two answers differ. Over three seeds of the
    # reference study, with quantification on the unmodelled data: the
    # model at detection took false positives from 11.3 to 5.7, p90 from
    # 47% to 20% and F1 from 76% to 86%. At quantification it takes signal
    # -- a tenth of the internal standard, a quarter of glutathione's
    # window, on a baseline that is flat.
    'baseline_model': True,
    'baseline_method': 'snip',
    'baseline_lam': 1e5,
    'baseline_niter': 40,
    # Where the noise is measured, high -> low ppm. Where the spectrum has no
    # such region the detector measures it on the quietest segment instead.
    'noise_region': (12.0, 10.0),
    'min_snr': 6.0,
    'min_distance_ppm': 0.004,
    'per_sample': True,
    'min_group_presence': 3,
    'detect_shoulders': False,
    'shoulder_k': 3.0,
    'excluded_regions': tuple(DEFAULT_EXCLUDED_REGIONS),
}

ANNOTATION = {
    'tolerance_ppm': 0.05,
    # Used only when the data does not carry its own field: a Bruker
    # dataset's procs does, and read_spectrometer_mhz reads it.
    'fallback_field_mhz': 600.0,
}

QUANTIFICATION = {
    # Integrate on the baseline-model-corrected spectrum. Off: the model
    # finds peaks better than it measures them -- see DETECTION. Measured on
    # a flat synthetic baseline; a real spectrum with a broad envelope is the
    # case to check, and _broad_removed on every row says what the model
    # takes whenever this is on.
    'baseline_model': False,
    'integration': 'Multiplet',
    'normalisation': 'IS',
    'proton_correction': True,
    'local_baseline': False,
    'align': False,
    'subtract_blanks': False,
    # Both on, as every result of the reference study was measured: with
    # them a new user's run went from 13 of 18, p90 90% to 13 of 18, p90
    # 23%, and false positives from 7 to 6.
    'quantify_shapes': True,
    'resolve_shared': True,
    # The windows' balance (quant_engine._resolve_shared, ADR-010/011):
    # every contested window one equation, solved together. It resolves
    # few (Mice ISS: 2 of 109 contested windows) but says which winners
    # stand on a shared window nobody resolves, and their confidence drops
    # for it. Over three seeds of the GISSMO tube of 100, with m* judged by
    # its spin system (ADR-074): found 237 -> 241, strangers 75 -> 73, two
    # dots or more 196/20 -> 192/13, p90 73% -> 79%; tube of 18 strangers
    # 29 -> 23 (ADR-075). Was tied to app_paths.DECONVOLUTION_ENABLED,
    # which also hides the unfinished deconvolution stages.
    'split_shared': True,
    # Overlapping analytes fitted together, with free components for the
    # peaks no quantified analyte explains (processing.quant_joint_fit).
    # Over three seeds of the reference bench, against integration:
    #   18 analytes  p90 20% -> 18%, bias -2% -> 0%
    #   50 analytes  p90 67% -> 51%, bias +6% -> +1%, false positives 19 -> 15
    #   100 analytes p90 218% -> 135%, bias +28% -> +20%
    # identification and three-dot precision unchanged or better. It costs
    # time: about 16 s, 83 s and 254 s to quantify those tubes against 5,
    # 17 and 39 s.
    'joint_fit': True,
    'singlet_min_snr': 5.0,
    'cascade': True,
    # percent, as the interface shows it; the measurement is beside
    # quant_cascade.MAX_SIGNAL_DISAGREEMENT
    'max_disagreement_pct': 45.0,
}


# How crowded a spectrum must be before the cascade is worth its false rows.
# The cascade adds analytes the election missed, at two dots at most. On the
# GISSMO bench (three seeds each, 2026-10-01), analytes it added at two dots
# against strangers it added at two dots:
#
#   peaks detected   500 MHz    600 MHz
#   ~90  (18 cpds)   9 : 11     19 : 6
#   ~200 (50 cpds)   56 : 13    54 : 13
#   ~300 (100 cpds)  100 : 11   102 : 8
#
# At the sparse tube the rows of any level it added were 23 real and 17
# false at 600 MHz, 16 real and 25 false at 500 MHz; at 100 compounds, 122
# real and 20 false. Between the ~97 peaks of the sparse tube and the ~190
# of the next there is no measurement; the bar sits in the gap.
CASCADE_MIN_PEAKS = 150


def cascade_advice(n_peaks, enabled=True) -> str:
    """A recommendation about the cascade for a spectrum with ``n_peaks``
    detected peaks, or ``''`` when there is nothing to say.

    A hint, not a setting: the analyst's choice stands.
    """
    if not n_peaks or n_peaks >= CASCADE_MIN_PEAKS:
        return ''
    if enabled:
        return (f'Only {n_peaks} peaks detected: on spectra this sparse the '
                f'cascade adds about as many false rows as real ones '
                f'(measured on mixtures of 18 compounds). The election alone '
                f'is recommended here.')
    return ''
