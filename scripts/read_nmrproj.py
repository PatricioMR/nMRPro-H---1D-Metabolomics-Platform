"""Lê .nmrproj e mostra quantificação e metabolitos com concentrações."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

proj_path = sys.argv[1] if len(sys.argv) > 1 else \
    r"G:\My Drive\Mestrado\BEPE\NMR 042426\MRP_ExtractionTest_042426_v01.nmrproj"

with open(proj_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 75)
print(f"Projecto: {os.path.basename(proj_path)}")
print(f"Chaves: {list(data.keys())}")
print()

# Amostras
samples = data.get('samples_info', [])
names   = [s.get('name', f'S{i}') for i, s in enumerate(samples)]
print(f"Amostras ({len(names)}): {names}\n")

# Quantificação — 'qnt'
qnt = data.get('qnt', {})
print(f"qnt keys: {list(qnt.keys()) if isinstance(qnt, dict) else type(qnt)}\n")

if isinstance(qnt, dict):
    results = qnt.get('results', qnt.get('raw_areas', qnt.get('concentrations', {})))
    # Tentar encontrar concentrações
    for k, v in qnt.items():
        if isinstance(v, dict) and v:
            print(f"  qnt['{k}'] ({len(v)} entries): sample keys={list(list(v.values())[0].keys())[:8] if isinstance(list(v.values())[0], dict) else type(list(v.values())[0])}")
        elif isinstance(v, list) and v:
            print(f"  qnt['{k}'] list len={len(v)}: first={str(v[0])[:100]}")
        else:
            print(f"  qnt['{k}']: {str(v)[:100]}")

# Annotation — 'ann'
ann = data.get('ann', {})
print(f"\nann keys: {list(ann.keys()) if isinstance(ann, dict) else type(ann)}")

# Peaks — 'pd'
pd_ = data.get('pd', {})
print(f"\npd keys: {list(pd_.keys()) if isinstance(pd_, dict) else type(pd_)}")
if isinstance(pd_, dict):
    peaks = pd_.get('peaks', [])
    print(f"  peaks: {len(peaks)} detected")
    if peaks:
        print(f"  peak[0] keys: {list(peaks[0].keys())[:12]}")

# Tentar extrair resultados de quantificação directamente
print("\n" + "=" * 75)
print("RESULTADOS DE QUANTIFICAÇÃO (se existirem)")
print("=" * 75)

def find_concentrations(obj, path=''):
    """Procura recursivamente dicts com chave 'concentrations' ou 'conc'."""
    if isinstance(obj, dict):
        if any(k in obj for k in ('concentrations', 'conc', 's_concs', 'sample_concs')):
            print(f"\nEncontrado em {path}: keys={list(obj.keys())[:10]}")
            for k in ('concentrations', 'conc', 's_concs', 'sample_concs'):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, (list, np.ndarray)):
                        print(f"  {k}: {v}")
                    elif isinstance(v, dict):
                        print(f"  {k}: {v}")
        for k, v in obj.items():
            find_concentrations(v, f"{path}/{k}")

find_concentrations(qnt, 'qnt')
find_concentrations(ann, 'ann')
