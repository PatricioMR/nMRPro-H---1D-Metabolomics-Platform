# data/hmdb_api.py
import time
import random

# Dados simulados de metabólitos (nome, ppm central, intervalo típico, número de prótons)
METABOLITE_DB = [
    {'name': 'Lactate', 'ppm': 1.33, 'range': 0.05, 'protons': 3},
    {'name': 'Alanine', 'ppm': 1.48, 'range': 0.05, 'protons': 3},
    {'name': 'NAA', 'ppm': 2.02, 'range': 0.05, 'protons': 3},
    {'name': 'Glutamate', 'ppm': 2.35, 'range': 0.10, 'protons': 2},
    {'name': 'Glutamine', 'ppm': 2.45, 'range': 0.10, 'protons': 2},
    {'name': 'Creatine', 'ppm': 3.03, 'range': 0.05, 'protons': 3},
    {'name': 'Choline', 'ppm': 3.20, 'range': 0.05, 'protons': 9},
    {'name': 'Glucose', 'ppm': 3.48, 'range': 0.15, 'protons': 7},
    {'name': 'Myo-inositol', 'ppm': 3.55, 'range': 0.10, 'protons': 6},
]

def search_by_interval(ppm_start, ppm_end, tolerance=0.05):
    """
    Simula busca no HMDB por metabólitos cujo deslocamento químico
    esteja dentro do intervalo [ppm_start, ppm_end] com tolerância extra.
    Retorna lista de dicionários com nome, ppm, intervalo e protons.
    """
    time.sleep(0.5)  # simula latência
    results = []
    center = (ppm_start + ppm_end) / 2
    half_width = (ppm_end - ppm_start) / 2 + tolerance
    for met in METABOLITE_DB:
        if abs(met['ppm'] - center) <= half_width:
            results.append({
                'name': met['name'],
                'ppm': met['ppm'],
                'range': met['range'],
                'protons': met['protons'],
                'matched_interval': (ppm_start, ppm_end)
            })
    return results