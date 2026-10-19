"""Write the 18/50/100-analyte crowded studies out as projects the
application can open, already run: detection, annotation and
quantification done, with the application's own current defaults
(controllers.analysis_defaults -- joint fit included).

Unlike make_bench_project.py, which writes a blank project for the
analyst to work by hand, this writes the run itself: the peaks
detection found, the metabolites annotation matched, and the table
quantification produced, in the exact schema gui.project_manager saves
-- assembled here from validation.fresh_run and validation.crowded_bench
rather than from a live GUI session, but through the identical engine
calls, so what opens is what those calls produced, not a re-enactment.

    python -m validation.export_bench_projects [out_dir]

Writes, per tube size, ``MixtureBench_<n>.nmrproj`` and a
``MixtureBench_<n>_truth.txt`` answer sheet.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import project_file
from processing.baseline_model import BaselineModel
from processing.spectral_regions import format_regions
from validation import mixture_bench as mb
from validation.crowded_bench import panel
from validation.fresh_run import defaults, prepare, quantify_prepared
from validation.make_bench_project import GROUP_COLOURS, answer_sheet

SIZES = (18, 50, 100)
SEED = 0


def build(n, seed=SEED, overrides=None):
    """``(state, truth, sigma, foreign)`` for the ``n``-analyte tube, run
    with the application's current defaults -- everything ``fresh_run``
    does, assembled into the schema ``gui.project_manager`` saves."""
    pan = panel(mb.DB_DEFAULT, n, seed=seed)
    X, sigma, truth, foreign = mb.build_study(mb.DB_DEFAULT, seed=seed,
                                              panel=pan)
    prep = prepare(mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                   field_mhz=mb.MHZ, overrides=overrides, log=lambda *a: None)
    rows = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                             standard_nh=mb.STANDARD_NH,
                             standard_conc=mb.STANDARD_MM,
                             standard_name='DSS', overrides=overrides)
    return project_state(prep, rows, mb.PPM, mb.MHZ, overrides), truth, sigma, foreign, pan


def project_state(prep, rows, ppm, field_mhz, overrides=None) -> dict:
    """The schema ``gui.project_manager`` saves, for a bench study already
    detected, annotated (``prep``, validation.fresh_run.prepare) and
    quantified (``rows``) with the application's defaults."""
    d = defaults(overrides)
    samples = []
    for s in mb.SAMPLES:
        i, label = int(s['index']), s['label']
        name = f'BENCH_{label}_{i % 6 + 1:02d}'
        samples.append({'index': i, 'name': name, 'label': label,
                        'type': 'Sample', 'correction_factor': 1.0,
                        'display': f'{name} ({label})', 'excluded': False})

    model = BaselineModel(
        enabled=d['baseline_model'], method=d['baseline_method'],
        params={'niter': d['baseline_niter']} if d['baseline_method'] == 'snip'
        else {'lam': d['baseline_lam']})
    excl_str = format_regions(prep['excluded']) or None

    state = {
        'version': 6,
        'ppm': np.asarray(ppm, dtype=float),
        'ppm_calibration': None,
        'all_intensities': prep['raw'],
        'baseline_model': model.to_dict(),
        'baseline_corrected': False,
        'alignment_corrected': False,
        'samples_info': samples,
        'group_colors': dict(GROUP_COLOURS),
        'source_file': None,
        'group_dilution_factors': {},
        'active_stage': 'quantification',
        'axes_settings': {},
        'plot_settings': {},
        'show_legend': True,
        'peak_list_filters': {},
        'eligible_overrides': {},
        'pd': {
            'noise_std': prep['noise_std'],
            'noise_source': 'ppm',
            'peaks': prep['peaks'],
            '_overlaps': [],
            'step1_completed': True,
            'noise_start_var': str(d['noise_region'][0]),
            'noise_end_var': str(d['noise_region'][1]),
            'min_snr_var': d['min_snr'],
            'min_width_var': d['min_distance_ppm'],
            'excluded_regions': excl_str,
            'excluded_regions_applied': excl_str,
            'per_sample_detection': bool(d['per_sample']),
            'detect_shoulders': bool(d['detect_shoulders']),
            'shoulder_k_var': d['shoulder_k'],
            'min_group_presence_var': d['min_group_presence'],
        },
        'ann': {
            'db_path': mb.DB_DEFAULT,
            'last_results_args': None,
            'annotations': {},
            'tolerance': d['tolerance_ppm'],
            'spectrometer_mhz': f'{field_mhz:g}',
            'align': False,
            'exclude_water': True,
            'water_lo': 4.7,
            'water_hi': 4.9,
            'internal_library_mode': 'Metabolomics (default)',
            'lipid_solvent': None,
            'annotation_params': dict(prep['params']) if prep['params'] else None,
            'quality_marks': {},
            'child_quality_marks': {},
            'anchor_lines': {},
            'subline_quality_marks': {},
            'metabolites': prep['mets'],
            'jr_contributions': None,
            'jr_contributions_per_sample': None,
            'jr_status': None,
            'jr_conc_est': None,
        },
        'qnt': {
            'quant_results': rows,
            'quant_peak_overrides': {},
            'deconv_overrides': {},
            'integ': d['integration'],
            'norm': d['normalisation'],
            'proton_corr': d['proton_correction'],
            'mcr': False,
            'blk_sub': d['subtract_blanks'],
            'local_baseline': d['local_baseline'],
            # 'baseline_model' above is DETECTION's; quantification's own
            # answer to the same question is exposed by defaults() as
            # 'quant_on_model' precisely so the two are never confused.
            'baseline_model_quant': bool(d['quant_on_model']),
            'shape_fit': d['quantify_shapes'],
            'shared_deconv': d['resolve_shared'],
            'singlet_min_snr': d['singlet_min_snr'],
            'cascade': d['cascade'],
            'joint_fit': d['joint_fit'],
            'max_disagreement': d['max_disagreement_pct'],
            'j_resolve': False,
            'ext_is': False,
            'ext_is_indices': [],
            'ref_ppm': str(mb.STANDARD_PPM),
            'ref_nh': str(mb.STANDARD_NH),
            'ref_conc': str(mb.STANDARD_MM),
            'ref_std': 'DSS',
            'ref_rank': '',
            'last_norm_mode': None,
            'last_ref_areas': None,
            'last_ref_conc': None,
            'last_sample_dil': None,
            'last_groups': None,
            'last_blk_indices': None,
            'last_non_blk': None,
        },
    }
    return state


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out')
    for n in SIZES:
        print(f'{n} analytes: preparing (detection, annotation, '
              f'quantification)...', flush=True)
        state, truth, sigma, foreign, pan = build(n)
        folder = os.path.join(out_dir, f'{n} sample')
        os.makedirs(folder, exist_ok=True)
        proj = os.path.join(folder, f'MixtureBench_{n}.nmrproj')
        project_file.save(proj, state, keep_backup=False)
        sheet = os.path.join(folder, f'MixtureBench_{n}_truth.txt')
        with open(sheet, 'w', encoding='utf-8') as f:
            f.write(answer_sheet(truth, sigma, foreign))
        n_q = sum(1 for r in state['qnt']['quant_results']
                  if not r.get('_not_quantified') and not r.get('_is_is'))
        print(f'  wrote {proj}')
        print(f'    {len(pan)} analytes in the tube, {len(state["pd"]["peaks"])} '
              f'peaks, {n_q} rows quantified, joint_fit='
              f'{state["qnt"]["joint_fit"]}')
        print(f'  wrote {sheet}')


if __name__ == '__main__':
    main()
