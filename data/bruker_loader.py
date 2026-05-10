# data/bruker_loader.py
"""
Loader para dados brutos Bruker NMR.

Estrutura esperada (qualquer nível de profundidade):

    RootFolder/
        SampleA/
            1/                  ← expno (número da experiência)
                acqus           ← parâmetros de aquisição
                pdata/
                    1/          ← procno
                        1r      ← espectro real (int32 binário)
                        procs   ← parâmetros de processamento
        SampleB/
            1/
                ...

O root pode ser o próprio expno (1r directamente em pdata/1/1r) ou
uma pasta de estudo com múltiplas amostras — o loader detecta ambos.
"""

import os
import struct
import re
import numpy as np


# ── Leitura de parâmetros Bruker (formato ##$KEY= VALUE) ──────────────────────

def _read_jcamp(filepath):
    """Lê um ficheiro JCAMP-DX Bruker (acqus / procs) e devolve dict."""
    params = {}
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                # ##$KEY= value   ou   ##KEY= value
                m = re.match(r'^##\$?(\w+)=\s*(.+)$', line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    # Tentar converter para número
                    try:
                        params[key] = int(val)
                    except ValueError:
                        try:
                            params[key] = float(val)
                        except ValueError:
                            params[key] = val
    except Exception:
        pass
    return params


# ── Encontrar todos os experimentos válidos numa pasta ────────────────────────

def _find_experiments(root_folder, expno=1, procno=1):
    """
    Detecta automaticamente o nível da pasta seleccionada e devolve
    a lista de proc_dirs com '1r' + 'procs'.

    Casos suportados (do mais específico ao mais geral):

    A) Utilizador seleccionou o proc_dir directamente:
           root_folder/1r  →  [root_folder]

    B) Utilizador seleccionou o expno:
           root_folder/pdata/<procno>/1r  →  [root_folder/pdata/<procno>]

    C) Utilizador seleccionou uma única amostra:
           root_folder/<expno>/pdata/<procno>/1r  →  [proc_dir]

    D) Utilizador seleccionou o dataset (pasta pai com múltiplas amostras):
           root_folder/<sample>/<expno>/pdata/<procno>/1r  →  [proc_dir, ...]
    """
    expno_str  = str(expno)
    procno_str = str(procno)

    # A) A própria pasta já contém '1r'
    if (os.path.isfile(os.path.join(root_folder, '1r')) and
            os.path.isfile(os.path.join(root_folder, 'procs'))):
        return [root_folder]

    # B) Seleccionou o expno (contém pdata/)
    proc_dir_b = os.path.join(root_folder, 'pdata', procno_str)
    if (os.path.isfile(os.path.join(proc_dir_b, '1r')) and
            os.path.isfile(os.path.join(proc_dir_b, 'procs'))):
        return [proc_dir_b]

    # C) Seleccionou uma amostra (contém <expno>/pdata/<procno>/1r)
    proc_dir_c = os.path.join(root_folder, expno_str, 'pdata', procno_str)
    if (os.path.isfile(os.path.join(proc_dir_c, '1r')) and
            os.path.isfile(os.path.join(proc_dir_c, 'procs'))):
        return [proc_dir_c]

    # D) Dataset com múltiplas amostras — cada subpasta é uma amostra
    experiments = []
    try:
        entries = sorted(os.listdir(root_folder))
    except OSError:
        return []

    for entry in entries:
        sample_dir = os.path.join(root_folder, entry)
        if not os.path.isdir(sample_dir):
            continue
        proc_dir = os.path.join(sample_dir, expno_str, 'pdata', procno_str)
        if (os.path.isfile(os.path.join(proc_dir, '1r')) and
                os.path.isfile(os.path.join(proc_dir, 'procs'))):
            experiments.append(proc_dir)

    return experiments


def _sample_name_from_path(pdata_proc_path, root_folder):
    """
    Extrai nome da amostra a partir do caminho.

    Casos:
      root/SampleA/1/pdata/1  →  'SampleA'   (dataset multi-amostra)
      root/pdata/1            →  basename(root)  (seleccionou expno)
      root                    →  basename(root)   (seleccionou proc_dir)
    """
    rel = os.path.relpath(pdata_proc_path, root_folder)
    if rel == '.':
        # proc_dir IS root_folder (caso A)
        return os.path.basename(root_folder)

    parts = rel.replace('\\', '/').split('/')
    # Caso C: parts = ['1', 'pdata', '1']  → nome é o basename do root
    # Caso D: parts = ['SampleA', '1', 'pdata', '1']  → nome é parts[0]
    if parts[0] in ('pdata',) or parts[0].isdigit():
        return os.path.basename(root_folder)
    return parts[0]


# ── Leitura de um espectro 1r ─────────────────────────────────────────────────

def _load_spectrum(proc_dir):
    """
    Lê o espectro real (1r) e devolve (ppm_array, intensity_array).

    Parâmetros relevantes de procs:
        SI      — número de pontos do espectro processado
        SW_p    — largura espectral em Hz
        OFFSET  — ppm mais à esquerda (ppm máximo)
        SF      — frequência do spectrometer (MHz)
        BYTORDP — byte order: 0 = little-endian, 1 = big-endian
        NC_proc — factor de escala (intensidade × 2^NC_proc)
    """
    procs_path = os.path.join(proc_dir, 'procs')
    spec_path  = os.path.join(proc_dir, '1r')

    procs = _read_jcamp(procs_path)

    si      = int(procs.get('SI',      32768))
    sw_p    = float(procs.get('SW_p',  10000.0))   # Hz
    offset  = float(procs.get('OFFSET', 12.0))     # ppm (extremo esquerdo)
    sf      = float(procs.get('SF',    600.0))      # MHz
    bytord  = int(procs.get('BYTORDP', 0))          # 0=LE, 1=BE
    nc_proc = int(procs.get('NC_proc', 0))

    # Eixo ppm: de OFFSET (esquerda) até OFFSET - SW_p/SF (direita), SI pontos
    sw_ppm  = sw_p / sf
    ppm     = np.linspace(offset, offset - sw_ppm, si)

    # Ler binário int32
    endian  = '<' if bytord == 0 else '>'
    filesize = os.path.getsize(spec_path)
    n_points = filesize // 4  # cada int32 = 4 bytes

    with open(spec_path, 'rb') as f:
        raw = f.read(n_points * 4)

    fmt = f'{endian}{n_points}i'
    data = np.array(struct.unpack(fmt, raw), dtype=np.float64)

    # Factor de escala NC_proc
    data = data * (2.0 ** nc_proc)

    # Ajustar comprimento ao SI declarado
    if len(data) != si:
        if len(data) > si:
            data = data[:si]
        else:
            data = np.pad(data, (0, si - len(data)))

    return ppm, data


# ── Loader principal ──────────────────────────────────────────────────────────

def load_bruker_dataset(root_folder, ppm_min=0.0, ppm_max=10.0,
                        n_points=32768, expno=1, procno=1):
    """
    Carrega todos os espectros Bruker encontrados em root_folder.

    Todos os espectros são interpolados para um eixo ppm comum
    (de ppm_max a ppm_min, com n_points pontos) para ficarem compatíveis
    com o formato interno da aplicação.

    Devolve:
        ppm             : np.ndarray shape (n_points,)
        all_intensities : np.ndarray shape (n_points, n_samples)
        sample_names    : list[str]
        sample_labels   : list[str]   (grupo inferido do nome ou 'Sample')
    """
    experiments = _find_experiments(root_folder, expno=expno, procno=procno)

    if not experiments:
        raise ValueError(
            f"Nenhum espectro Bruker encontrado em:\n{root_folder}\n\n"
            "Certifica-te que a pasta contém subdirectorias com "
            "pdata/1/1r e pdata/1/procs."
        )

    # Eixo ppm comum (ppm_max → ppm_min, como em NMR convencional)
    common_ppm = np.linspace(ppm_max, ppm_min, n_points)

    sample_names   = []
    sample_labels  = []
    intensity_cols = []

    for proc_dir in experiments:
        try:
            ppm_raw, spec_raw = _load_spectrum(proc_dir)

            # Interpolação para eixo comum
            # np.interp exige x crescente — inverter se necessário
            if ppm_raw[0] > ppm_raw[-1]:
                ppm_raw  = ppm_raw[::-1]
                spec_raw = spec_raw[::-1]

            common_ppm_asc = common_ppm[::-1]   # crescente para interp
            spec_interp    = np.interp(common_ppm_asc, ppm_raw, spec_raw,
                                       left=0.0, right=0.0)
            spec_interp    = spec_interp[::-1]   # voltar à ordem NMR

            name  = _sample_name_from_path(proc_dir, root_folder)
            label = _infer_label(name)

            sample_names.append(name)
            sample_labels.append(label)
            intensity_cols.append(spec_interp)

            print(f"  ✓  {name}  ({proc_dir})")

        except Exception as e:
            print(f"  ✗  Ignorado {proc_dir}: {e}")
            continue

    if not intensity_cols:
        raise ValueError("Nenhum espectro foi carregado com sucesso.")

    # all_intensities: shape (n_points, n_samples)  ← convenção da app
    all_intensities = np.column_stack(intensity_cols)

    print(f"\nBruker dataset: {len(sample_names)} amostras, "
          f"{n_points} pontos, ppm {ppm_max:.1f}→{ppm_min:.1f}")

    return common_ppm, all_intensities, sample_names, sample_labels


# ── Inferência de grupo a partir do nome ─────────────────────────────────────

def _infer_label(name):
    """
    Tenta inferir o grupo a partir do nome da pasta.
    Exemplos:  'CTRL_01' → 'CTRL',  'BLK'  → 'BLK',  'G1_03' → 'G1'
    """
    upper = name.upper()
    for kw in ('CTRL', 'CONTROL', 'BLK', 'BLANK', 'QC', 'IS'):
        if kw in upper:
            return kw[:4]   # 'CTRL', 'BLK ', 'QC  ', 'IS  ' → normaliza
    # Tentar extrair prefixo antes de _ ou dígitos
    m = re.match(r'^([A-Za-z]+)', name)
    if m:
        prefix = m.group(1)
        if len(prefix) <= 6:   # prefixo curto = provavelmente grupo
            return prefix
    return 'Sample'
