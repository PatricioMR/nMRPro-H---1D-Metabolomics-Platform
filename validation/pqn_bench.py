"""PQN validation (python -m validation.pqn_bench 18 S0,S1,S2 [seeds]). Tube of N analytes, 600 MHz, 3 seeds, 12 samples (6 A, 6 B).

Every analyte and the foreign methanol of each sample is multiplied by that
sample's dilution (log-normal, CV 30%); the internal standard is not (it is
added equally to every tube). Scenarios:
  S0  no dilution, no global difference
  S1  dilution CV 30%
  S2  dilution CV 30% and group B x0.6 on every analyte (a real global change)
Methods on the same spectra:
  A  IS only (the application's default)
  B  IS + PQN of the application (areas of the elected analytes)
  C  textbook PQN on the whole spectrum (Dieterle 2006), applied to A
"""
import sys, os, io, contextlib, collections, sqlite3
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats
from validation import gissmo_bench as gb
from validation import mixture_bench as mb
from validation.fresh_run import prepare, quantify_prepared, score
from validation.gissmo_sim import load_systems, render, stick_spectrum

size = sys.argv[1]
scen = sys.argv[2].split(',')
SEEDS = [int(a) for a in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0', '1', '2'])]
CV = 0.30
GLOBAL_B = 0.6
SIG_LN = float(np.sqrt(np.log(1 + CV ** 2)))
FIELD = gb.FIELDS[600]
PPM = np.asarray(gb.PPM, float)
lab = [s['label'] for s in mb.SAMPLES]
n = len(lab)
iA = [i for i, l in enumerate(lab) if l == 'A']
iB = [i for i, l in enumerate(lab) if l == 'B']


def build(seed, composition, dilution, gfac):
    """gb.build with every sample's analytes scaled by dilution[s] and group B by gfac."""
    systems = load_systems()
    con = sqlite3.connect(mb.DB_DEFAULT)
    lw = mb.LINEWIDTH_HZ
    mb.MHZ = FIELD
    rng = np.random.default_rng(seed)
    comps, truth = [], {}
    for name, conc, factor in composition:
        mid, _s = mb._db_signals(con, name)
        sp, si = stick_spectrum(systems[name], FIELD)
        comps.append((sp, si * mb.AREA_UNIT, mid))
        truth[mid] = {'name': name, 'A': conc, 'B': conc * factor}
    _m, lac = mb._db_signals(con, 'L-lactic acid')
    sigma = float(mb.compound_trace(PPM, lac, 4.0, shift=0.0).max()) / mb.NOISE_LACTATE_SNR
    cols = []
    for k, info in enumerate(mb.SAMPLES):
        grp = info['label']
        d = dilution[k] * (gfac if grp == 'B' else 1.0)
        shift = mb.DRIFT_PPM + rng.normal(0.0, 0.0015)
        y = rng.normal(0.0, sigma, PPM.size)
        y += mb._lorentz_area(PPM, mb.STANDARD_PPM + rng.normal(0, 0.0003),
                              mb.STANDARD_MM * mb.STANDARD_NH * mb.AREA_UNIT, lw)
        for sp, si, mid in comps:
            c = truth[mid][grp] * float(np.exp(rng.normal(0.0, 0.12))) * d
            y += c * render(PPM, sp + shift, si, lw, FIELD)
        for _name, conc, fsigs in mb.FOREIGN:
            c = conc * float(np.exp(rng.normal(0.0, 0.12))) * d
            for p, _mu, nh in fsigs:
                y += mb._lorentz_area(PPM, p + shift, c * nh * mb.AREA_UNIT, lw)
        cols.append(y)
    return np.column_stack(cols), truth, sigma


def textbook_pqn(X, sigma):
    """Dieterle 2006: integral normalisation, median reference, median quotient.
    The standard (0 ppm) and the water window are left out. Returns the relative
    dilution factor of every sample (median 1)."""
    m = (PPM > 0.5) & (PPM < 9.5) & ~((PPM > 4.6) & (PPM < 5.0))
    I = np.array([np.sum(np.clip(X[m, k], 0, None)) for k in range(X.shape[1])])
    Xn = X[m] / I
    ref = np.median(Xn, axis=1)
    use = ref * np.median(I) > 10.0 * sigma          # bins with signal in the reference
    q = np.array([np.median(Xn[use, k] / ref[use]) for k in range(X.shape[1])])
    f = I * q
    return f / np.median(f)


def welch(v):
    a, b = np.asarray(v)[iA], np.asarray(v)[iB]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or not (np.ptp(a) or np.ptp(b)):
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def cv_by_group(v):
    out = []
    for idx in (iA, iB):
        x = np.asarray(v)[idx]
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) >= 2:
            out.append(np.std(x, ddof=1) / np.mean(x))
    return float(np.mean(out)) if out else np.nan


def reported(row):
    from gui.stages._quant_adapter import reported_values
    return np.asarray(reported_values(row, n)[0], float)


results = {}
for sc in scen:
    R = collections.defaultdict(list)
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        dil = np.ones(n) if sc == 'S0' else np.exp(rng.normal(-SIG_LN ** 2 / 2, SIG_LN, n))
        gfac = GLOBAL_B if sc == 'S2' else 1.0
        comp = gb._composition(size, sd)
        X, truth, sigma = build(sd, comp, dil, gfac)
        target = dil * np.array([gfac if l == 'B' else 1.0 for l in lab])
        target = target / np.median(target)             # what a normalisation can see
        mb.MHZ = FIELD
        with contextlib.redirect_stdout(io.StringIO()):
            prep = prepare(PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT, field_mhz=FIELD, log=lambda *a: None)
            rowsA = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
                                      standard_conc=mb.STANDARD_MM, standard_name='DSS')
            rowsB = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
                                      standard_conc=mb.STANDARD_MM, standard_name='DSS',
                                      overrides={'normalisation': 'IS+PQN'})
        fC = textbook_pqn(X, sigma)
        # factors
        facB = None
        for r in rowsB:
            pf = r.get('_pqn_factors')
            if pf:
                facB = np.array([float(pf.get(s, 1.0)) for s in range(n)])
                break
        facB_n = facB / np.median(facB) if facB is not None else None
        for nm, f in (('B app', facB_n), ('C livro', fC)):
            if f is not None:
                R[('fator', nm, 'corr')].append(float(np.corrcoef(np.log(f), np.log(target))[0, 1]) if np.ptp(np.log(target)) > 1e-9 else np.nan)
                R[('fator', nm, 'erro')].append(float(np.median(np.abs(f / target - 1))))
        # engine (A vs B)
        for nm, rows in (('A', rowsA), ('B', rowsB)):
            s = score(rows, truth)
            R[('motor', nm, 'achados')].append(s['found'] if 'found' in s else np.nan)
            R[('motor', nm, 'fp')].append(s['fp'] if 'fp' in s else np.nan)
        # per-analyte metrics
        byA = {r['hmdb_id']: r for r in rowsA if not r.get('_not_quantified') and not r.get('_is_is')}
        byB = {r['hmdb_id']: r for r in rowsB if not r.get('_not_quantified') and not r.get('_is_is')}
        for mid, rA in byA.items():
            vals = {'A': reported(rA)}
            if mid in byB:
                vals['B'] = reported(byB[mid])
            cA = vals['A']
            if np.isfinite(fC).all():
                vals['C'] = cA / fC
            real = mid in truth
            t = truth.get(mid)
            changed = bool(real and abs(t['B'] / t['A'] - 1) > 1e-9)
            cls = 'estranho' if not real else ('real COM mudanca' if changed else 'real sem mudanca')
            for mth, v in vals.items():
                R[('cv', mth, cls)].append(cv_by_group(v))
                p = welch(v)
                R[('sig', mth, cls)].append(bool(np.isfinite(p) and p < 0.05))
                if changed:
                    a, b = v[iA], v[iB]
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) and len(b) and np.mean(a) > 0:
                        tr = t['B'] / t['A']
                        meas = np.mean(b) / np.mean(a)
                        R[('razao', mth, 'rel')].append(abs(meas / tr - 1))
                        R[('razao', mth, 'abs')].append(abs(meas / (tr * gfac) - 1))
        print(f'{sc} seed {sd} done', flush=True)
    results[sc] = R

print('\n#### PQN fase 1 -- tubo', size, '(600 MHz, 3 sementes, diluicao CV', CV, ')')
for sc, R in results.items():
    print(f'\n=== {sc}')
    for nm in ('B app', 'C livro'):
        if ('fator', nm, 'corr') in R:
            print(f'  fator estimado x verdadeiro, {nm}: correlacao(log) {np.nanmean(R[("fator", nm, "corr")]):.2f}, erro mediano {100*np.nanmean(R[("fator", nm, "erro")]):.0f}%')
    for mth in ('A', 'B'):
        print(f'  motor {mth}: achados {np.nanmean(R[("motor", mth, "achados")]):.1f}, falsos {np.nanmean(R[("motor", mth, "fp")]):.1f} (media por semente)')
    for cls in ('real sem mudanca', 'real COM mudanca', 'estranho'):
        row = []
        for mth in ('A', 'B', 'C'):
            cvs = [x for x in R.get(('cv', mth, cls), []) if np.isfinite(x)]
            sg = R.get(('sig', mth, cls), [])
            row.append(f'{mth}: CV {100*np.median(cvs):.0f}% sig {sum(sg)}/{len(sg)}' if cvs else f'{mth}: -')
        print(f'  {cls:17s} ' + ' | '.join(row))
    for kind, txt in (('rel', 'erro da razao B/A contra a verdade RELATIVA (so o analito)'), ('abs', 'erro da razao B/A contra a verdade ABSOLUTA (analito x efeito global)')):
        parts = []
        for mth in ('A', 'B', 'C'):
            v = R.get(('razao', mth, kind), [])
            if v:
                parts.append(f'{mth}: {100*np.median(v):.0f}%')
        if parts:
            print(f'  {txt}: ' + ' | '.join(parts))
