# validation/gissmo_bench.py
"""The crowded bench built from GISSMO spin systems, at 500 and 600 MHz.

The reference bench (``mixture_bench``, ``crowded_bench``) draws each
analyte from the very profiles the annotation searches for -- the inverse
crime: the answer is shaped like the question. Here each pure spectrum is
the GISSMO spin system of the compound solved at the study's field
(``gissmo_sim``), exact to second order, and only then summed:

    A mixture spectrum is the linear sum of its components' pure spectra,
    each scaled by concentration. Muncey HJ, Jones R, De Iorio M, Ebbels
    TMD. MetAssimulo: Simulation of Realistic NMR Metabolic Profiles. BMC
    Bioinformatics 2010;11:496. The same linear model underlies ASICS
    (Tardivel P et al., Metabolomics 2017;13:109).

Everything else is the reference bench's design: 12 samples in two groups,
log-normal 12% between samples, common drift, noise anchored to lactate's
doublet at SNR 500, DSS 0.5 mM, methanol as a compound the database does not
hold, the 18 reference analytes at their amounts and the rest log-uniform
0.05-1 mM. The tube holds the 47 compounds of the reference with a BMRB
spectrum acquired at 600 MHz and a GISSMO system (``--sizes c600``), or the
crowded bench's tubes of 18, 50 and 100 with every compound simulated (a
compound GISSMO has no system for -- mostly sugars, NAD(P)(H), polyamines --
passed over in the draw). The field reaches the annotation the way a
project's spectrometer field does.

``--snr-b0`` divides the noise by (field/500.13)^1.5: sensitivity grows as
B0^(3/2) (Hoult DI, Richards RE. The signal-to-noise ratio of the nuclear
magnetic resonance experiment. J Magn Reson 1976;24:71-85).

Each tube is made once and kept (keyed on field, seed, noise and the data it
is made of -- never on code); detection and annotation are kept too, keyed on
their own code, so a quantification change reruns only quantification.

    python -m validation.gissmo_bench                    # 18/50/100 at 500 and 600
    python -m validation.gissmo_bench --sizes c600       # the 47 of Banco-a-600-MHz
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import pickle
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation import mixture_bench as mb
from validation.crowded_bench import CACHE_DIR, _CACHED_SOURCES, pool_draw
from validation.fresh_run import (
    defaults, prepare, quantify_prepared, score, tables)
from validation.gissmo_sim import (
    SYSTEMS_JSON, load_systems, render, stick_spectrum)

FIELDS = {500: 500.13, 600: 600.13}
REF_FIELD = 500.13
PPM = mb.PPM


def _data_tag():
    h = hashlib.sha1()
    with open(SYSTEMS_JSON, 'rb') as f:
        h.update(f.read())
    st = os.stat(mb.DB_DEFAULT)
    h.update(f'{st.st_size}'.encode())
    return h.hexdigest()[:10]


def _tube_order():
    import json
    with open(SYSTEMS_JSON, encoding='utf-8') as f:
        return json.load(f)['tube_order']


def c600_tube(seed):
    """``[(name, conc, factor)]``: the 47 compounds with a BMRB spectrum
    acquired at 600 MHz and a GISSMO system (Banco-a-600-MHz), drawn over
    the 58 in ``tube_order`` so the tube is the one that note measured."""
    systems = load_systems()
    rng_c = np.random.default_rng(20_000 + seed)
    panel = {n: (c, f) for n, c, f in mb.PANEL}
    out = []
    for name in _tube_order():
        draw = float(np.exp(rng_c.uniform(np.log(0.05), np.log(1.0))))
        if name in systems:
            c, f = panel.get(name, (round(draw, 3), 1.0))
            out.append((name, c, f))
    return out


def crowded_tube(n, seed):
    """``[(name, conc, factor)]``: the crowded bench's tube of ``n``
    (``crowded_bench.panel``) with every compound simulated: the 18 of the
    reference study, then the pool in the order and at the amounts that
    bench draws, a compound GISSMO has no system for passed over."""
    systems = load_systems()
    tube = [(nm, c, f) for nm, c, f in mb.PANEL if nm in systems][:n]
    for name, c in pool_draw(mb.DB_DEFAULT, seed=seed):
        if len(tube) >= n:
            break
        if name in systems:
            tube.append((name, c, 1.0))
    if len(tube) < n:
        raise ValueError(f'GISSMO simulates only {len(tube)} of the pool')
    return tube


def build(field, seed, composition, *, snr_gain=1.0, ref_shift=0.0):
    """``(X, truth)``: one tube of ``composition`` at ``field``.

    ``ref_shift`` (ppm) gives every compound its own offset from where the
    reference lists it, the same in every sample: what a real extract's pH,
    ionic strength and matrix do (BMRB's references were measured at their
    own). A normal draw of that standard deviation per compound, from a
    generator of its own so the rest of the tube is the one made without it.
    In the NMR_Murilo cell extract the peaks the annotation matched stood a
    median 16 mppm from the listing."""
    systems = load_systems()
    con = sqlite3.connect(mb.DB_DEFAULT)
    lw = mb.LINEWIDTH_HZ
    mb.MHZ = field
    rng = np.random.default_rng(seed)
    shift_rng = np.random.default_rng(seed + 7_000_003)
    comps, truth = [], {}
    for name, conc, factor in composition:
        mid, _sigs = mb._db_signals(con, name)
        sp, si = stick_spectrum(systems[name], field)
        own = float(shift_rng.normal(0.0, ref_shift)) if ref_shift else 0.0
        comps.append((sp + own, si * mb.AREA_UNIT, mid))
        truth[mid] = {'name': name, 'A': conc, 'B': conc * factor}
    _m, lac = mb._db_signals(con, 'L-lactic acid')
    sigma = (float(mb.compound_trace(PPM, lac, 4.0, shift=0.0).max())
             / (mb.NOISE_LACTATE_SNR * snr_gain))
    cols = []
    for info in mb.SAMPLES:
        grp = info['label']
        shift = mb.DRIFT_PPM + rng.normal(0.0, 0.0015)
        y = rng.normal(0.0, sigma, PPM.size)
        y += mb._lorentz_area(PPM, mb.STANDARD_PPM + rng.normal(0, 0.0003),
                              mb.STANDARD_MM * mb.STANDARD_NH * mb.AREA_UNIT, lw)
        for sp, si, mid in comps:
            c = truth[mid][grp] * float(np.exp(rng.normal(0.0, 0.12)))
            y += c * render(PPM, sp + shift, si, lw, field)
        for _name, conc, fsigs in mb.FOREIGN:
            c = conc * float(np.exp(rng.normal(0.0, 0.12)))
            for p, _mu, nh in fsigs:
                y += mb._lorentz_area(PPM, p + shift, c * nh * mb.AREA_UNIT, lw)
        cols.append(y)
    return np.column_stack(cols), truth


def _composition(size, seed):
    return c600_tube(seed) if size == 'c600' else crowded_tube(int(size), seed)


def _shift_tag(ref_shift) -> str:
    return f'_rs{1000 * ref_shift:.0f}' if ref_shift else ''


def tube(field, seed, size='c600', *, snr_b0=False, ref_shift=0.0):
    """The tube, made once and kept."""
    gain = (field / REF_FIELD) ** 1.5 if snr_b0 else 1.0
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'gissmo_{size}_{field:.0f}'
                                   f'{"snr" if snr_b0 else ""}'
                                   f'{_shift_tag(ref_shift)}'
                                   f'_s{seed}_{_data_tag()}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    X, truth = build(field, seed, _composition(size, seed), snr_gain=gain,
                     ref_shift=ref_shift)
    with open(path, 'wb') as f:
        pickle.dump((X, truth), f, protocol=pickle.HIGHEST_PROTOCOL)
    return X, truth


def prepared(field, seed, size='c600', *, snr_b0=False, ref_shift=0.0):
    """Detection and annotation of the tube, kept until their code changes."""
    X, truth = tube(field, seed, size, snr_b0=snr_b0, ref_shift=ref_shift)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1(f'{size}|{field}|{seed}|{snr_b0}|{_data_tag()}'
                     f'{_shift_tag(ref_shift)}'.encode())
    d = defaults()
    for k in sorted(d):
        h.update(f'{k}={d[k]!r};'.encode())
    for rel in _CACHED_SOURCES:
        with open(os.path.join(root, rel), 'rb') as f:
            h.update(f.read())
    path = os.path.join(CACHE_DIR, f'gissmo_prep_{size}_{field:.0f}_s{seed}_'
                                   f'{h.hexdigest()[:16]}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f), truth
    mb.MHZ = field
    with contextlib.redirect_stdout(io.StringIO()):
        prep = prepare(PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                       field_mhz=field, log=lambda *a: None)
    with open(path, 'wb') as f:
        pickle.dump(prep, f, protocol=pickle.HIGHEST_PROTOCOL)
    return prep, truth


# What a quantified table is made of, beyond detection and annotation: a
# change to any of these quantifies the tube again.
_QUANT_SOURCES = ('processing', 'gui/stages', 'controllers', 'validation/fresh_run.py')


def _quant_hash() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for rel in _QUANT_SOURCES:
        path = os.path.join(root, rel)
        files = ([path] if os.path.isfile(path) else
                 sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(path)
                        for f in fs if f.endswith('.py')))
        for fp in files:
            with open(fp, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def run(field, seed, size='c600', *, snr_b0=False, overrides=None,
        ref_shift=0.0):
    prep, truth = prepared(field, seed, size, snr_b0=snr_b0,
                           ref_shift=ref_shift)
    path = None
    if not overrides:
        path = os.path.join(CACHE_DIR, f'gissmo_rows_{size}_{field:.0f}'
                                       f'{"snr" if snr_b0 else ""}'
                                       f'{_shift_tag(ref_shift)}_s{seed}_'
                                       f'{_data_tag()}_{_quant_hash()}.pkl')
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            rows = pickle.load(f)
    else:
        mb.MHZ = field
        with contextlib.redirect_stdout(io.StringIO()):
            rows = quantify_prepared(prep, mb.SAMPLES,
                                     standard_ppm=mb.STANDARD_PPM,
                                     standard_nh=mb.STANDARD_NH,
                                     standard_conc=mb.STANDARD_MM,
                                     standard_name='DSS', overrides=overrides)
        if path:
            with open(path, 'wb') as f:
                pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    s = score(rows, truth)
    core = {m: t for m, t in truth.items()
            if t['name'] in {n for n, _c, _f in mb.PANEL}}
    s['core'] = score([r for r in rows if r['hmdb_id'] in core
                       or r['hmdb_id'] not in truth], core)
    return s, rows, truth, len(prep['peaks'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--fields', type=int, nargs='+', default=[500, 600],
                    choices=sorted(FIELDS))
    ap.add_argument('--sizes', nargs='+', default=['18', '50', '100'],
                    help="tube sizes of the crowded bench, or 'c600' for the "
                         "47 compounds with a BMRB spectrum at 600 MHz")
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--snr-b0', action='store_true')
    ap.add_argument('--ref-shift-mppm', type=float, default=0.0,
                    help='each compound offset from its listing, normal, in mppm')
    a = ap.parse_args()
    res, core = [], []
    for size in a.sizes:
        for fk in a.fields:
            for sd in a.seeds:
                s, _rows, truth, n_peaks = run(
                    FIELDS[fk], sd, size, snr_b0=a.snr_b0,
                    ref_shift=a.ref_shift_mppm / 1000.0)
                lab = f'GISSMO {size} {fk}{" +SNR" if a.snr_b0 else ""} s{sd}'
                print(f'{lab}: {len(truth)} analitos, {n_peaks} picos',
                      flush=True)
                res.append((lab, s))
                core.append((lab, s['core']))
    print('\n=== TODOS OS ANALITOS DO TUBO ===\n')
    tables(res)
    print('\n=== SO OS DO PAINEL DE REFERENCIA ===\n')
    tables(core)


if __name__ == '__main__':
    main()
