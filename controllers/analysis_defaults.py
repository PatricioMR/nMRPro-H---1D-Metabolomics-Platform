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
    # percent, as the interface shows it
    'max_disagreement_pct': 30.0,
}
