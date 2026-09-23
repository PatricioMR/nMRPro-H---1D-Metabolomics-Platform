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
    # The project-level baseline model (SNIP), applied before detection and
    # carried into quantification as app.corrected_intensities(). Off: on
    # the reference study -- whose baseline is flat -- SNIP took a tenth of
    # the internal standard and a quarter of glutathione's window, and a new
    # user's run found 11 of 18 with 22 false positives against 13 and 7
    # without it. Measured on a synthetic baseline only: a real spectrum
    # with a broad macromolecule envelope is the case to check before this
    # is trusted everywhere, and _broad_removed on every row says what the
    # model took wherever it is turned on.
    'baseline_model': False,
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
    'singlet_min_snr': 5.0,
    'cascade': True,
    # percent, as the interface shows it
    'max_disagreement_pct': 30.0,
}
