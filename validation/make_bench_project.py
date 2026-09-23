"""Write the synthetic study out as a project the application can open.

The bench builds its spectrum in memory and runs the engine on it from a
script. That leaves everything the analyst does by hand -- the alignment,
the baseline, the exclusions, the marks -- outside the measurement, and
the bench's own numbers were taken with detection's alignment switched
off while the application runs it by default.

This writes the same twelve samples into a .nmrproj with nothing done to
them: the spectra, the sample list and the groups, and no detection, no
annotation, no quantification. Open it, work it as any study, and every
number that comes out is the software's on a spectrum whose true
concentrations are known exactly.

    python -m validation.make_bench_project [path.nmrproj]

Written alongside it is a text file naming what is in the spectrum and at
what concentration, so the answer is at hand without reading this code.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation import mixture_bench as mb

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out',
                           'MixtureBench_v01.nmrproj')

GROUP_COLOURS = {'A': '#2E86AB', 'B': '#A23B72', 'BLK': '#c0c0c0'}


def build(seed=0):
    """``(state, truth)`` -- the project as the application saves one, and
    what the spectrum actually holds."""
    X, sigma, truth, foreign = mb.build_study(mb.DB_DEFAULT, seed=seed)
    samples = []
    for s in mb.SAMPLES:
        i, label = int(s['index']), s['label']
        name = f'BENCH_{label}_{i % 6 + 1:02d}'
        samples.append({'index': i, 'name': name, 'label': label,
                        'type': 'Sample', 'correction_factor': 1.0,
                        'display': f'{name} ({label})', 'excluded': False})
    state = {
        'version': 6,
        'ppm': np.asarray(mb.PPM, dtype=float),
        'ppm_calibration': None,
        'all_intensities': np.asarray(X, dtype=float),
        'baseline_model': None,
        'baseline_corrected': False,
        'alignment_corrected': False,
        'samples_info': samples,
        'group_colors': dict(GROUP_COLOURS),
        'source_file': None,
        'group_dilution_factors': {},
        'active_stage': 'visualization',
        'axes_settings': {},
        'plot_settings': {},
        'show_legend': True,
        'peak_list_filters': {},
        'eligible_overrides': {},
        # The field the spectrum was built at. Without it the project opens
        # at the interface's 600 MHz fallback and every predicted J line
        # lands a fifth of its splitting away from where it is.
        'ann': {'spectrometer_mhz': f'{mb.MHZ:g}'},
    }
    return state, truth, sigma, foreign


def answer_sheet(truth, sigma, foreign) -> str:
    lines = [
        'The mixture bench, as a project',
        '=' * 60,
        '',
        f'{len(mb.SAMPLES)} samples: six labelled A and six labelled B.',
        f'Field {mb.MHZ:.0f} MHz, linewidth {mb.LINEWIDTH_HZ} Hz, '
        f'{mb.PPM.size} points from {mb.PPM.min():.1f} to {mb.PPM.max():.1f} ppm.',
        f'Noise sigma {sigma:,.0f} (lactate reads {mb.NOISE_LACTATE_SNR:.0f} '
        f'at its tallest).',
        '',
        f'Internal standard: DSS at {mb.STANDARD_PPM} ppm, '
        f'{mb.STANDARD_NH} protons, {mb.STANDARD_MM} mM.',
        f'Water suppressed over {mb.WATER}.',
        '',
        'Every analyte is laid down from the reference database with its own',
        'measured profile where the reference has one, scaled to its listed',
        'protons, shifted by the study drift of '
        f'{mb.DRIFT_PPM} ppm, and given a per-sample',
        'concentration of its group value times a log-normal of 12%.',
        '',
        'What is in it (mM, group A and group B):',
        '-' * 60,
    ]
    for mid, t in sorted(truth.items(), key=lambda kv: -kv[1]['A']):
        lines.append(f"  {t['name'][:34]:34s} A {t['A']:6.3f}   B {t['B']:6.3f}")
    if foreign:
        lines += ['', 'Also present, and not in the reference panel:']
        for name, conc in foreign.items():
            lines.append(f'  {name[:34]:34s}   {conc:6.3f}')
    lines += [
        '', '-' * 60,
        'Anything the software reports that is not on this list is a false',
        'positive; anything on it that the software does not report is a',
        'false negative. The numbers beside each name are the truth.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    state, truth, sigma, foreign = build()
    with open(out, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    sheet = os.path.splitext(out)[0] + '_truth.txt'
    with open(sheet, 'w', encoding='utf-8') as f:
        f.write(answer_sheet(truth, sigma, foreign))
    X = state['all_intensities']
    print(f'wrote {out}')
    print(f'  {X.shape[1]} samples, {X.shape[0]} points, '
          f'{len(truth)} analytes')
    print(f'  nothing has been done to it: no detection, no annotation, '
          f'no quantification')
    print(f'wrote {sheet}')


if __name__ == '__main__':
    main()
