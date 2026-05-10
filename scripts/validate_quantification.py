"""
Valida quantificação IS contra MRP_ValidationData_v01.csv.

Correções face à versão anterior:
  - TSP: integração sem local_baseline (pico isolado a 0 ppm)
  - Janelas de integração: calculadas a partir do multiplet (J, mult)
    para capturar a área real do multiplet completo
  - Fallback: 5× a largura da linha individual quando J desconhecido
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bruker_loader import _find_experiments, _load_spectrum, _sample_name_from_path

SPECTRA_ROOT = r"G:\My Drive\Mestrado\BEPE\NMR 042426"
DB_PATH      = os.path.join("databases", "HMDB_ValidationPipe.db")
EXPNO, PROCNO = 2, 1
SPECTROMETER_MHZ = 600.7

TSP_PPM  = 0.0
TSP_NH   = 9
TSP_CONC = 1.0   # mM — ajustar ao valor real

# Picos de referência do CSV + nH + mult + J da ValidationPipe DB
# hw = half-width da janela de integração calculada abaixo
PEAKS = {
    'Taurine':       {'ppm': 3.2648, 'line_fwhm': 0.0125, 'nH': 2,  'mult': 't',  'j_hz': 6.57},
    'Glutamic Acid': {'ppm': 3.7499, 'line_fwhm': 0.0095, 'nH': 1,  'mult': 'dd', 'j_hz': 7.186},
    'Glucose':       {'ppm': 3.5170, 'line_fwhm': 0.0166, 'nH': 1,  'mult': 'dd', 'j_hz': 9.823},
    'Mannitol':      {'ppm': 3.7864, 'line_fwhm': 0.0109, 'nH': 1,  'mult': 'dd', 'j_hz': 11.87},
    'Cysteine':      {'ppm': 3.0484, 'line_fwhm': 0.0106, 'nH': 2,  'mult': 'm',  'j_hz': None},
    'Caffeine':      {'ppm': 3.3961, 'line_fwhm': 0.0064, 'nH': 3,  'mult': 's',  'j_hz': None},
    'Itaconic acid': {'ppm': 3.1430, 'line_fwhm': 0.0054, 'nH': 2,  'mult': 's',  'j_hz': None},
    'Arabinose':     {'ppm': 3.9306, 'line_fwhm': 0.0054, 'nH': 1,  'mult': 'm',  'j_hz': None},
}

EXPECTED = {
    'Taurine':       [3.0429, 2.5250, 3.3060, 10.0990, 10.4079, 10.0702],
    'Glutamic Acid': [0.5881, 0.6502, 1.1038,  5.2174,  4.3079,  4.6108],
    'Glucose':       [0.6932, 0.4652, 0.8052,  1.7504,  1.5816,  1.7351],
    'Mannitol':      [0.3100, 0.1147, 0.1967,  1.5325,  1.2804,  1.4250],
    'Cysteine':      [0.0002, 0.0004, 0.0015,  0.4860,  0.4437,  0.5849],
    'Caffeine':      [0.1030, 0.1004, 0.1618,  0.1790,  0.1723,  0.1446],
    'Itaconic acid': [0.0000, 0.0000, 0.0000,  0.0772,  0.0683,  0.0943],
    'Arabinose':     [0.0001, 0.0013, 0.0000,  0.0875,  0.1124,  0.0799],
}


def multiplet_hw(mult, j_hz, line_fwhm, mhz=600.7):
    """
    Calcula half-width da janela de integração para capturar o multiplet completo.
    Estratégia: (n_lines/2 × J_ppm) + 2 × line_fwhm para caudas
    """
    line_fwhm_ppm = line_fwhm

    if mult in ('s', 'singlet', 'm', 'multiplet', '') or j_hz is None:
        # Singlet/multiplet: 3× FWHM da linha
        return line_fwhm_ppm * 3.0

    j_ppm = j_hz / mhz

    mult_lines = {
        's':  1, 'd':  2, 't':  3, 'q':  4, 'p':  5, 'h':  6,
        'dd': 4, 'dt': 6, 'td': 6, 'ddd': 8, 'tt': 9,
    }
    n = mult_lines.get(mult.lower(), 3)
    half_span = ((n - 1) / 2.0) * j_ppm
    return half_span + 2.5 * line_fwhm_ppm


def trapz(y, x):
    try:
        return float(np.trapezoid(y, x=x))
    except AttributeError:
        return float(np.trapz(y, x=x))


def integrate_raw(ppm, spec, center, hw, local_bl=True):
    """Integra [center-hw, center+hw] com baseline linear opcional."""
    lo, hi = center - hw, center + hw
    mask   = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 2:
        return 0.0
    x = ppm[mask]
    y = spec[mask].copy()
    if local_bl and len(y) >= 2:
        bl = np.linspace(float(y[0]), float(y[-1]), len(y))
        y  = y - bl
    return abs(trapz(np.clip(y, 0, None), x))


def tsp_integrate(ppm, spec, tsp_ppm=0.0, search=0.15):
    """Integra TSP detectando o FWHM real do pico."""
    mask = (ppm >= tsp_ppm - search) & (ppm <= tsp_ppm + search)
    if mask.sum() < 4:
        return 0.0, 0.03
    seg  = spec[mask]
    apex = float(seg.max())
    if apex <= 0:
        return 0.0, 0.03
    half  = apex / 2.0
    above = np.where(seg >= half)[0]
    ppms  = ppm[mask]
    if len(above) >= 2:
        fwhm = abs(float(ppms[above[-1]] - ppms[above[0]]))
        hw   = max(fwhm * 3.0, 0.020)   # 3× FWHM para caudas Lorentzianas
    else:
        hw = 0.030
    # Integrar sem local baseline — TSP isolado a 0 ppm
    area = integrate_raw(ppm, spec, tsp_ppm, hw, local_bl=False)
    return area, hw


def main():
    print("=" * 75)
    print("VALIDAÇÃO DE QUANTIFICAÇÃO — IS (TSP) vs valores teóricos CSV")
    print("(janelas baseadas em multiplet real, sem local_bl para TSP)")
    print("=" * 75)

    # Pré-calcular janelas de integração
    print("\nJanelas de integração por metabolito:")
    for met, info in PEAKS.items():
        hw = multiplet_hw(info['mult'], info['j_hz'], info['line_fwhm'], SPECTROMETER_MHZ)
        info['hw'] = hw
        total_w = hw * 2
        print(f"  {met:<20} mult={info['mult']:<4} J={str(info['j_hz']):<6}Hz  "
              f"hw={hw:.4f} ppm  total={total_w:.4f} ppm")

    # Carregar espectros raw
    exps     = _find_experiments(SPECTRA_ROOT, expno=EXPNO, procno=PROCNO)
    spectra  = []
    for proc_dir in sorted(exps):
        ppm_r, spec_r = _load_spectrum(proc_dir)
        name = _sample_name_from_path(proc_dir, SPECTRA_ROOT)
        if ppm_r[0] > ppm_r[-1]:
            ppm_r, spec_r = ppm_r[::-1], spec_r[::-1]
        spectra.append({'name': name, 'ppm': ppm_r, 'spec': spec_r})

    sample_names = [s['name'] for s in spectra]
    print(f"\nAmostras: {sample_names}\n")

    # TSP
    tsp_areas = []
    print(f"{'Amostra':<15} {'TSP apex':>12} {'TSP hw':>8} {'TSP area':>14} {'area/nH':>12}")
    print("-" * 65)
    for s in spectra:
        area, hw = tsp_integrate(s['ppm'], s['spec'], TSP_PPM)
        tsp_areas.append(area)
        mask = (s['ppm'] >= TSP_PPM - hw) & (s['ppm'] <= TSP_PPM + hw)
        apex = float(s['spec'][mask].max()) if mask.sum() > 0 else 0.0
        print(f"  {s['name']:<13} {apex:>12.3e} {hw:>8.4f} {area:>14.4e} {area/TSP_NH:>12.4e}")

    tsp_ref = np.array(tsp_areas) / TSP_NH

    mmex_i = [i for i, n in enumerate(sample_names) if 'MMEX' in n.upper()]
    mmnx_i = [i for i, n in enumerate(sample_names) if 'MMNX' in n.upper()]
    if mmex_i and mmnx_i:
        r_meas = np.mean([tsp_ref[i] for i in mmnx_i]) / np.mean([tsp_ref[i] for i in mmex_i])
        print(f"\n  MMNX/MMEX TSP ratio = {r_meas:.2f}x  (CSV esperado ≈ 13.4x)")

    # Quantificação
    print(f"\n{'─'*75}")
    print(f"  {'Metabolito':<16} {'Amostra':<12} {'Esperado':>9} {'Calculado':>10} {'Erro%':>9}")
    print(f"{'─'*75}")

    errors_all  = []
    per_met_err = {}

    for met_name, info in PEAKS.items():
        ref_ppm = info['ppm']
        hw      = info['hw']
        nH      = info['nH']
        expected_list = EXPECTED[met_name]
        met_errors = []

        for s_idx, s in enumerate(spectra):
            exp  = expected_list[s_idx] if s_idx < len(expected_list) else float('nan')
            area = integrate_raw(s['ppm'], s['spec'], ref_ppm, hw, local_bl=True)
            ref  = tsp_ref[s_idx]

            calc = (area / nH) / ref * TSP_CONC if ref > 0 and area > 0 else float('nan')

            if exp < 0.001:
                err_str = '      n/a'
            elif np.isnan(calc):
                err_str = '      NaN'
            else:
                err_pct = (calc - exp) / exp * 100
                err_str = f'{err_pct:+9.1f}%'
                met_errors.append(err_pct)
                errors_all.append(err_pct)

            print(f"  {met_name:<16} {s['name']:<12} {exp:>9.4f} "
                  f"{calc if not np.isnan(calc) else 0:>10.4f} {err_str}")

        if met_errors:
            mae = np.mean(np.abs(met_errors))
            bias = np.mean(met_errors)
            per_met_err[met_name] = mae
            print(f"  {'':16} {'':12} {'':9}  MAE={mae:6.1f}%  bias={bias:+.1f}%\n")
        else:
            print()

    print(f"{'='*75}")
    print("SUMÁRIO")
    print(f"{'─'*75}")
    for met, mae in sorted(per_met_err.items(), key=lambda x: x[1]):
        flag = '  ← PROBLEMA' if mae > 25 else ('  ← OK' if mae < 15 else '')
        print(f"  {met:<20} MAE={mae:6.1f}%{flag}")

    if errors_all:
        omae  = np.mean(np.abs(errors_all))
        obias = np.mean(errors_all)
        print(f"\n  OVERALL MAE : {omae:.1f}%")
        print(f"  OVERALL BIAS: {obias:+.1f}%")
        print(f"\n  Interpretação:")
        if abs(obias) < 20 and omae < 30:
            print(f"  ✓ Quantificação dentro de tolerância analítica (< 30% MAE)")
        elif abs(obias) > 30:
            sign = "alto" if obias > 0 else "baixo"
            print(f"  ✗ Bias sistemático {sign}: TSP_CONC incorreto OU nH errado")
        else:
            print(f"  ~ Erros aleatórios altos: overlap, baseline, ou janelas incorretas")

    print(f"\n  TSP_CONC = {TSP_CONC} mM | nH_TSP = {TSP_NH} | espectr. = {SPECTROMETER_MHZ} MHz")


if __name__ == "__main__":
    main()
