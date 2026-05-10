"""Diagnóstico do pico TSP em cada amostra."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bruker_loader import load_bruker_dataset, _find_experiments, _load_spectrum, _read_jcamp

SPECTRA_ROOT = r"G:\My Drive\Mestrado\BEPE\NMR 042426"

# Carregar raw sem interpolação (para ver o espectro real por amostra)
exps = _find_experiments(SPECTRA_ROOT, expno=2, procno=1)
print(f"{'Sample':<15} {'NC_proc':>8} {'OFFSET':>8} {'SF':>8} {'SW_p':>8} "
      f"{'max_int':>12} {'ppm@max':>8} {'TSP_area[-0.1,0.1]':>20} {'TSP_apex':>12}")
print("-" * 115)

for proc_dir in exps:
    name = proc_dir.split(os.sep)[-4] if os.sep in proc_dir else proc_dir
    procs = _read_jcamp(os.path.join(proc_dir, 'procs'))
    nc = int(procs.get('NC_proc', 0))
    offset = float(procs.get('OFFSET', 12.0))
    sf = float(procs.get('SF', 600.0))
    sw_p = float(procs.get('SW_p', 10000.0))

    ppm, spec = _load_spectrum(proc_dir)

    # Encontrar apex na janela TSP
    mask_tsp = (ppm >= -0.15) & (ppm <= 0.15)
    if mask_tsp.sum() < 2:
        print(f"{name:<15} NC={nc:+3d}  TSP region not in spectrum!")
        continue

    tsp_spec  = spec[mask_tsp]
    tsp_ppms  = ppm[mask_tsp]
    apex_val  = float(tsp_spec.max())
    apex_ppm  = float(tsp_ppms[int(np.argmax(tsp_spec))])
    tsp_area  = abs(float(np.trapezoid(np.clip(tsp_spec, 0, None), tsp_ppms)))

    max_idx   = int(np.argmax(np.abs(spec)))
    max_val   = float(spec[max_idx])
    max_ppm   = float(ppm[max_idx])

    print(f"{name:<15} {nc:>8d} {offset:>8.2f} {sf:>8.1f} {sw_p:>8.0f} "
          f"{max_val:>12.2e} {max_ppm:>8.4f} {tsp_area:>20.4e} {apex_val:>12.4e}")
