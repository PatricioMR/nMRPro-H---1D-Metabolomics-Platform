"""A signal whose peak sits at the reference is annotated, even when a
stronger multiplet of the same coupling sits a tenth of a ppm away.

The search scored the lines over a pool a tenth of the ppm wide, centred
them on the stronger neighbour, and then dropped the whole match for being
off the reference -- with the right peak detected at the reference. On the
virtual bench's tube of 100, ethanolamine, 3-ureidopropionate and mesaconate
were never annotated that way, and about 300 matches were lost in all.

Real reference database (the repository's BMRB v04), synthetic spectrum.
"""
import io
import contextlib
import os

import numpy as np
import pytest

from controllers.annotation_controller import search_database
from processing.annotation_scoring import SCORING_DEFAULTS
from processing.peak_detection import as_app_peak, detect_peaks
from processing.signal_quality import is_annotatable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(REPO, 'databases', 'BMRB_Metabolomics_Mammalian_v04.db')
MHZ = 500.0
PPM = np.linspace(4.2, 2.8, 14001)
FWHM = 1.2 / MHZ
J = 5.25 / MHZ


def _triplet(centre, height):
    g = FWHM / 2.0
    y = np.zeros_like(PPM)
    for off, w in ((-J, 0.5), (0.0, 1.0), (J, 0.5)):
        y += height * w * g ** 2 / ((PPM - centre - off) ** 2 + g ** 2)
    return y


@pytest.mark.skipif(not os.path.exists(DB), reason='reference database absent')
def test_a_signal_at_its_reference_is_not_lost_to_a_stronger_neighbour():
    rng = np.random.default_rng(0)
    # ethanolamine's two triplets where the reference puts them, and a
    # stronger triplet of the same coupling 0.13 ppm above the first
    base = _triplet(3.129, 100.0) + _triplet(3.810, 100.0) + _triplet(3.260, 400.0)
    X = np.column_stack([base * f + rng.normal(0, 0.5, PPM.size)
                         for f in (0.8, 1.0, 1.2, 0.9, 1.1, 1.0)])
    samples = [{'index': i, 'name': f'S{i}', 'label': 'G'} for i in range(6)]
    with contextlib.redirect_stdout(io.StringIO()):
        det = detect_peaks(PPM, X, noise_std=0.5, min_snr=6.0,
                           min_distance=0.002, baseline_correct=False,
                           samples_info=samples, return_details=True,
                           detect_shoulders=False)
    peaks = [as_app_peak(p) for p in det['peaks']]
    assert any(abs(p['ppm'] - 3.129) < 0.002 for p in peaks)
    tiers = {'Good', 'Average'}
    sc = dict(SCORING_DEFAULTS)
    sc['dppm_gold'] = 0.05
    with contextlib.redirect_stdout(io.StringIO()):
        found = search_database(
            db_path=DB, db_type='hmdb', peaks_to_search=peaks,
            annotatable_peaks=[p for p in peaks if is_annotatable(p, tiers)],
            tiers=tiers, ppm=PPM, intensities=X, samples_info=samples,
            field_mhz=MHZ, tolerance_ppm=0.05, params=sc, noise_std=0.5,
            min_snr=6.0, min_distance_ppm=0.002, peak_correlation={},
            n_samples_correlated=0)
    hits = [r for r in found.results if r.get('name') == 'Ethanolamine'
            and abs(float(r['db_ppm']) - 3.129) < 0.01]
    assert hits, 'ethanolamine at 3.129 was dropped'
    assert abs(float(hits[0]['peak_ppm']) - 3.129) < 0.01
