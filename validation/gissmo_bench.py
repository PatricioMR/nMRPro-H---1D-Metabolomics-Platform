# validation/gissmo_bench.py
"""The crowded bench built from GISSMO spin systems, at 500 and 600 MHz.

The reference bench (``mixture_bench``, ``crowded_bench``) draws each
analyte from the very profiles the annotation searches for -- the inverse
crime: the answer is shaped like the question. Here each pure spectrum is
the GISSMO spin system of the compound solved at the study's field
(``processing.spin_system``), exact to second order, and only then summed:

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
from processing.spin_system import (
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


# Each chemical environment of a compound moves on its own, as in an extract:
# every group of spins with one shift gets its own normal offset of this
# standard deviation (ppm) before the spin system is solved. Every proton has
# its own titration curve (Tredwell et al., Metabolomics 2016;12:152) and the
# matrix moves each differently (Takis et al., Nat Commun 2017;8:1662); a
# compound shifted as a block is the optimistic case. 0: off. Set by
# --signal-shift-mppm; part of every cache key.
SIGNAL_SHIFT = 0.0

# Each chemical environment of a compound moves from SAMPLE TO SAMPLE, by its
# own sensitivity to what differs between the samples (pH, ionic strength,
# Ca2+): offset = SAMPLE_SHIFT * k * z, k ~ N(0, 1) per environment (fixed in
# the tube) and z per sample, centred +0.7 in group A and -0.7 in B with a
# spread of 0.7 inside each group. The Mice ISS skin extract is the case
# measured: glycine 3.567 in one group and 3.561 in the other, acetate 1.926
# and 1.917, aromatic imidazoles up to 20 mppm apart, while lactate moves
# 2 mppm. The rigid per-sample drift below (sigma 1.5 mppm, every compound
# alike) never tested this. 0: off. --sample-shift-mppm; part of every
# cache key.
SAMPLE_SHIFT = 0.0

# How much each compound's amount varies from sample to sample (sd of its
# log), and a dilution every compound of a sample shares (sd of its log). The
# bench drew 12% and no dilution; the studies vary far more -- within a group
# the median analyte's CV is 67% in the Mice ISS skin extract, 46% in the
# liver, 14% in the NMR_Murilo cells (42% study-wide), part of it the
# tissue mass every metabolite of a sample shares. The variation across
# samples is the evidence of identity position cannot fake (one molecule's
# signals move together); a bench at 12% hides it. Part of every cache key.
AMOUNT_SD = 0.12
DILUTION_SD = 0.0

# What detection is handed: the tube as built (None), or aligned first by
# 'valley' (processing.peak_alignment.valley_segment_align) or 'icoshift'.
# --align-first; part of the detection cache key.
ALIGN_FIRST = None

# Settings a variant changed that detection or annotation read
# (bench_runs --set on one of their modules); part of the cache key, so the
# variant is annotated under its own settings and not handed the default's.
PREP_SETTINGS = ''


def _spread_signals(sys_, sigma, rng):
    """A copy of a GISSMO system whose distinct shifts each moved by a normal
    draw of ``sigma`` (spins sharing a shift move together)."""
    out = dict(sys_)
    ppm = np.asarray(sys_['ppm'], dtype=float).copy()
    for v in np.unique(np.round(ppm, 5)):
        ppm[np.round(ppm, 5) == v] += float(rng.normal(0.0, sigma))
    out['ppm'] = ppm
    return out


def build(field, seed, composition, *, snr_gain=1.0, ref_shift=0.0,
          drift=None):
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
    signal_rng = np.random.default_rng(seed + 9_000_011)
    sample_rng = np.random.default_rng(seed + 13_000_027)
    z = np.array([(0.7 if info['label'] == 'A' else -0.7)
                  + float(sample_rng.normal(0.0, 0.7)) for info in mb.SAMPLES])
    comps, truth = [], {}
    for name, conc, factor in composition:
        mid, _sigs = mb._db_signals(con, name)
        sys_ = (_spread_signals(systems[name], SIGNAL_SHIFT, signal_rng)
                if SIGNAL_SHIFT else systems[name])
        own = float(shift_rng.normal(0.0, ref_shift)) if ref_shift else 0.0
        if SAMPLE_SHIFT:
            # one spin system per sample, each environment moved by its own
            # sensitivity times the sample's state
            env = np.round(np.asarray(sys_['ppm'], dtype=float), 5)
            k = {v: float(sample_rng.normal(0.0, 1.0)) for v in np.unique(env)}
            per = []
            for zs in z:
                moved = dict(sys_)
                moved['ppm'] = (np.asarray(sys_['ppm'], dtype=float)
                                + np.array([SAMPLE_SHIFT * k[v] * zs for v in env]))
                sp, si = stick_spectrum(moved, field)
                per.append((sp + own, si * mb.AREA_UNIT))
            comps.append((per, None, mid))
        else:
            sp, si = stick_spectrum(sys_, field)
            comps.append((sp + own, si * mb.AREA_UNIT, mid))
        truth[mid] = {'name': name, 'A': conc, 'B': conc * factor}
    _m, lac = mb._db_signals(con, 'L-lactic acid')
    sigma = (float(mb.compound_trace(PPM, lac, 4.0, shift=0.0).max())
             / (mb.NOISE_LACTATE_SNR * snr_gain))
    cols = []
    dil_rng = np.random.default_rng(seed + 17_000_041)
    for j, info in enumerate(mb.SAMPLES):
        grp = info['label']
        dil = float(np.exp(dil_rng.normal(0.0, DILUTION_SD))) if DILUTION_SD else 1.0
        shift = (mb.DRIFT_PPM if drift is None else drift) + rng.normal(0.0, 0.0015)
        y = rng.normal(0.0, sigma, PPM.size)
        y += mb._lorentz_area(PPM, mb.STANDARD_PPM + rng.normal(0, 0.0003),
                              mb.STANDARD_MM * mb.STANDARD_NH * mb.AREA_UNIT, lw)
        for sp, si, mid in comps:
            if si is None:
                sp, si = sp[j]
            c = dil * truth[mid][grp] * float(np.exp(rng.normal(0.0, AMOUNT_SD)))
            y += c * render(PPM, sp + shift, si, lw, field)
        for _name, conc, fsigs in mb.FOREIGN:
            c = dil * conc * float(np.exp(rng.normal(0.0, AMOUNT_SD)))
            for p, _mu, nh in fsigs:
                y += mb._lorentz_area(PPM, p + shift, c * nh * mb.AREA_UNIT, lw)
        cols.append(y)
    return np.column_stack(cols), truth


def random_tube(n, seed):
    """``[(name, conc, factor)]``: ``n`` compounds drawn at random from every
    one GISSMO simulates among the reference's (the 18 of the reference
    study and the crowded bench's pool alike), each at a log-uniform amount
    in ``CONC_RANGE_MM`` and the same in both groups. A draw of its own for
    every (n, seed): tubes of different sizes share no composition, so the
    replicates of one size measure what changing the analytes does."""
    from validation.crowded_bench import CONC_RANGE_MM
    systems = load_systems()
    names = [nm for nm, _c, _f in mb.PANEL if nm in systems]
    names += [nm for nm, _c in pool_draw(mb.DB_DEFAULT, seed=0)
              if nm in systems and nm not in names]
    names.sort()
    if n > len(names):
        raise ValueError(f'GISSMO simulates only {len(names)} compounds')
    rng = np.random.default_rng(50_000 + 1_000 * int(seed) + int(n))
    pick = rng.choice(len(names), size=n, replace=False)
    lo, hi = np.log(CONC_RANGE_MM[0]), np.log(CONC_RANGE_MM[1])
    concs = np.exp(rng.uniform(lo, hi, size=n))
    return [(names[i], float(round(c, 3)), 1.0) for i, c in zip(pick, concs)]


# Group B against group A, by the compound's place in the tube: nearly absent
# in B (x0.05), up three times in B, induced in B (A holds 5% of what B does),
# unchanged. B keeps the amounts of the tube, so no compound is taken to a
# concentration no extract has. This is the tube for asking what happens to an
# analyte that falls under the detection limit in one group and stands in the
# other.
CONTRAST_FACTORS = (0.05, 3.0, 20.0, 1.0)


def contrast_tube(composition):
    """``composition`` with the contrast of ``CONTRAST_FACTORS`` cycled over
    its compounds: ``[(name, amount in A, B/A)]``."""
    out = []
    for i, (nm, c, _f) in enumerate(composition):
        f = CONTRAST_FACTORS[i % len(CONTRAST_FACTORS)]
        out.append((nm, round(c / f, 4) if f > 3.0 else c, f))
    return out


def _composition(size, seed):
    """'c600', a crowded tube ('18', '50', ...: the reference study's 18
    first), a random one ('r18', 'r150', ...: ``random_tube``) or a crowded
    one with contrast between the groups ('k100': ``contrast_tube``)."""
    if size == 'c600':
        return c600_tube(seed)
    if str(size).startswith('k'):
        return contrast_tube(crowded_tube(int(str(size)[1:]), seed))
    if str(size).startswith('r'):
        return random_tube(int(str(size)[1:]), seed)
    return crowded_tube(int(size), seed)


def _shift_tag(ref_shift, drift=None) -> str:
    return ((f'_rs{1000 * ref_shift:.0f}' if ref_shift else '')
            + (f'_dr{1000 * drift:.0f}' if drift is not None else '')
            + (f'_ss{1000 * SIGNAL_SHIFT:.0f}' if SIGNAL_SHIFT else '')
            + (f'_sm{1000 * SAMPLE_SHIFT:.0f}' if SAMPLE_SHIFT else '')
            + (f'_cv{100 * AMOUNT_SD:.0f}' if AMOUNT_SD != 0.12 else '')
            + (f'_dl{100 * DILUTION_SD:.0f}' if DILUTION_SD else ''))


def tube(field, seed, size='c600', *, snr_b0=False, ref_shift=0.0,
         drift=None):
    """The tube, made once and kept."""
    gain = (field / REF_FIELD) ** 1.5 if snr_b0 else 1.0
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'gissmo_{size}_{field:.0f}'
                                   f'{"snr" if snr_b0 else ""}'
                                   f'{_shift_tag(ref_shift, drift)}'
                                   f'_s{seed}_{_data_tag()}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    X, truth = build(field, seed, _composition(size, seed), snr_gain=gain,
                     ref_shift=ref_shift, drift=drift)
    with open(path, 'wb') as f:
        pickle.dump((X, truth), f, protocol=pickle.HIGHEST_PROTOCOL)
    return X, truth


def _spin_on() -> bool:
    from processing import spin_patterns
    return bool(spin_patterns.SPIN_PATTERNS)


def prepared(field, seed, size='c600', *, snr_b0=False, ref_shift=0.0,
             drift=None):
    """Detection and annotation of the tube, kept until their code changes."""
    X, truth = tube(field, seed, size, snr_b0=snr_b0, ref_shift=ref_shift,
                    drift=drift)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1(f'{size}|{field}|{seed}|{snr_b0}|{_data_tag()}'
                     f'{_shift_tag(ref_shift, drift)}'
                     f'{"|align=" + ALIGN_FIRST if ALIGN_FIRST else ""}'
                     f'{"|spin" if _spin_on() else ""}'
                     f'{"|set=" + PREP_SETTINGS if PREP_SETTINGS else ""}'.encode())
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
                       field_mhz=field, aligned_first=ALIGN_FIRST or False,
                       log=lambda *a: None)
    # beside it, then moved: two variants may prepare the same tube at once
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(prep, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
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
        ref_shift=0.0, drift=None):
    prep, truth = prepared(field, seed, size, snr_b0=snr_b0,
                           ref_shift=ref_shift, drift=drift)
    path = None
    if not overrides:
        path = os.path.join(CACHE_DIR, f'gissmo_rows_{size}_{field:.0f}'
                                       f'{"snr" if snr_b0 else ""}'
                                       f'{_shift_tag(ref_shift, drift)}_s{seed}_'
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
    ap.add_argument('--signal-shift-mppm', type=float, default=0.0,
                    help='each chemical environment offset on its own, normal, in mppm')
    ap.add_argument('--sample-shift-mppm', type=float, default=0.0,
                    help='each environment moved per sample by its own sensitivity')
    ap.add_argument('--align-first', choices=('valley', 'icoshift'), default=None)
    ap.add_argument('--drift-mppm', type=float, default=None,
                    help='the study-wide shift every compound shares (default 8)')
    a = ap.parse_args()
    global SIGNAL_SHIFT, SAMPLE_SHIFT, ALIGN_FIRST
    SIGNAL_SHIFT = a.signal_shift_mppm / 1000.0
    SAMPLE_SHIFT = a.sample_shift_mppm / 1000.0
    ALIGN_FIRST = a.align_first
    res, core = [], []
    for size in a.sizes:
        for fk in a.fields:
            for sd in a.seeds:
                s, _rows, truth, n_peaks = run(
                    FIELDS[fk], sd, size, snr_b0=a.snr_b0,
                    ref_shift=a.ref_shift_mppm / 1000.0,
                    drift=None if a.drift_mppm is None else a.drift_mppm / 1000.0)
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
