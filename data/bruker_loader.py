# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
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


# ── Detecção e carregamento de experimentos 2D ───────────────────────────────

_PULPROG_TYPES = {
    'cosy': 'COSY', 'dqfcosy': 'COSY', 'cosydf': 'COSY', 'cosygp': 'COSY',
    'jres': 'JRES', 'jresgp': 'JRES',
    'tocsy': 'TOCSY', 'mlev': 'TOCSY', 'dipsi': 'TOCSY',
    'hsqc': 'HSQC', 'hsqcph': 'HSQC', 'hsqcetgp': 'HSQC',
    'hmbc': 'HMBC', 'hmbcgp': 'HMBC',
    'noesy': 'NOESY', 'roesy': 'ROESY',
}


def _classify_pulprog(pulprog):
    pp = pulprog.lower().strip('<> ')
    for kw, etype in _PULPROG_TYPES.items():
        if kw in pp:
            return etype
    return '2D'


def _find_expno_dir(root_folder, expno=1):
    """Devolve o caminho para a pasta expno (onde está acqus), ou None."""
    expno_str = str(expno)

    if os.path.isfile(os.path.join(root_folder, 'acqus')):
        return root_folder

    cand = os.path.join(root_folder, expno_str)
    if os.path.isfile(os.path.join(cand, 'acqus')):
        return cand

    try:
        for entry in sorted(os.listdir(root_folder)):
            d = os.path.join(root_folder, entry)
            if not os.path.isdir(d):
                continue
            cand = os.path.join(d, expno_str)
            if os.path.isfile(os.path.join(cand, 'acqus')):
                return cand
    except OSError:
        pass

    return None


def detect_experiment_type(root_folder, expno=1, procno=1):
    """
    Detecta se a pasta Bruker contém um experimento 1D ou 2D.

    Devolve dict:
        ndim     : 1 ou 2
        pulprog  : str  (valor bruto de PULPROG)
        exp_type : '1D' | 'COSY' | 'JRES' | 'TOCSY' | 'HSQC' | 'HMBC' | '2D'
        expno_dir: str | None
        proc_dir : str | None
    """
    expno_dir = _find_expno_dir(root_folder, expno)
    if expno_dir is None:
        return {'ndim': 1, 'pulprog': 'unknown', 'exp_type': '1D',
                'expno_dir': None, 'proc_dir': None}

    acqus    = _read_jcamp(os.path.join(expno_dir, 'acqus'))
    pulprog  = str(acqus.get('PULPROG', '')).strip('<> ')
    proc_dir = os.path.join(expno_dir, 'pdata', str(procno))

    has_acqu2s = os.path.isfile(os.path.join(expno_dir, 'acqu2s'))
    has_2rr    = os.path.isfile(os.path.join(proc_dir,  '2rr'))

    if has_acqu2s and has_2rr:
        return {'ndim': 2, 'pulprog': pulprog,
                'exp_type': _classify_pulprog(pulprog),
                'expno_dir': expno_dir, 'proc_dir': proc_dir}

    return {'ndim': 1, 'pulprog': pulprog, 'exp_type': '1D',
            'expno_dir': expno_dir, 'proc_dir': proc_dir}


def _detile_2d(flat, si_f1, si_f2, xdim_f1, xdim_f2):
    """
    Reconstrói a matriz 2D a partir do formato de submatrizes Bruker.

    O ficheiro 2rr não é guardado linha-a-linha — é dividido em blocos
    (submatrizes) de XDIM_F1 linhas × XDIM_F2 colunas, ordenados em row-major.
    Sem este de-tiling o espectro aparece embaralhado em bandas.

    Se XDIM == SI em ambas as dimensões (sem tiling), reduz-se a um reshape.
    """
    expected = si_f1 * si_f2
    if len(flat) < expected:
        flat = np.concatenate([flat, np.zeros(expected - len(flat))])
    else:
        flat = flat[:expected]

    # Sem submatrizes → reshape directo
    if xdim_f1 >= si_f1 and xdim_f2 >= si_f2:
        return flat.reshape(si_f1, si_f2)

    n_tiles_f2 = si_f2 // xdim_f2
    n_tiles_f1 = si_f1 // xdim_f1
    tile_size  = xdim_f1 * xdim_f2

    out = np.empty((si_f1, si_f2), dtype=flat.dtype)
    idx = 0
    for tr in range(n_tiles_f1):          # blocos ao longo de F1 (linhas)
        r0 = tr * xdim_f1
        for tc in range(n_tiles_f2):      # blocos ao longo de F2 (colunas)
            c0 = tc * xdim_f2
            tile = flat[idx:idx + tile_size].reshape(xdim_f1, xdim_f2)
            out[r0:r0 + xdim_f1, c0:c0 + xdim_f2] = tile
            idx += tile_size
    return out


def load_bruker_2d(expno_dir, procno=1):
    """
    Carrega o espectro 2D processado Bruker (ficheiro 2rr).

    Devolve:
        f2_ppm  : np.ndarray (SI_F2,)  — dimensão directa  (eixo horizontal)
        f1_ppm  : np.ndarray (SI_F1,)  — dimensão indirecta (eixo vertical)
        data_2d : np.ndarray (SI_F1, SI_F2)
        meta    : dict com pulprog, exp_type, sf_f2, sf_f1, sw_ppm_f2, sw_ppm_f1
    """
    proc_dir = os.path.join(expno_dir, 'pdata', str(procno))
    procs    = _read_jcamp(os.path.join(proc_dir, 'procs'))
    proc2s   = _read_jcamp(os.path.join(proc_dir, 'proc2s'))

    # F2 — dimensão directa (horizontal)
    si_f2     = int(procs.get('SI',      1024))
    xdim_f2   = int(procs.get('XDIM',    si_f2))   # dim. da submatriz em F2
    sw_p_f2   = float(procs.get('SW_p',  5000.0))
    offset_f2 = float(procs.get('OFFSET', 12.0))
    sf_f2     = float(procs.get('SF',    600.0))
    bytord    = int(procs.get('BYTORDP', 0))
    nc_proc   = int(procs.get('NC_proc', 0))

    # F1 — dimensão indirecta (vertical)
    si_f1     = int(proc2s.get('SI',      1024))
    xdim_f1   = int(proc2s.get('XDIM',    si_f1))  # dim. da submatriz em F1
    sw_p_f1   = float(proc2s.get('SW_p',  5000.0))
    offset_f1 = float(proc2s.get('OFFSET', 12.0))
    sf_f1     = float(proc2s.get('SF',    600.0))

    sw_ppm_f2 = sw_p_f2 / sf_f2
    sw_ppm_f1 = sw_p_f1 / sf_f1

    f2_ppm = np.linspace(offset_f2, offset_f2 - sw_ppm_f2, si_f2)
    f1_ppm = np.linspace(offset_f1, offset_f1 - sw_ppm_f1, si_f1)

    endian   = '<' if bytord == 0 else '>'
    rr_path  = os.path.join(proc_dir, '2rr')
    filesize = os.path.getsize(rr_path)
    n_pts    = filesize // 4

    with open(rr_path, 'rb') as fh:
        raw = fh.read(n_pts * 4)

    data = np.frombuffer(raw, dtype=f'{endian}i4').astype(np.float64)
    data *= 2.0 ** nc_proc

    # Dados 2rr Bruker estão em formato de submatrizes (tiles) de
    # XDIM_F1 × XDIM_F2 — é preciso de-tile para reconstruir a matriz real.
    data = _detile_2d(data, si_f1, si_f2, xdim_f1, xdim_f2)

    acqus   = _read_jcamp(os.path.join(expno_dir, 'acqus'))
    acqu2s  = _read_jcamp(os.path.join(expno_dir, 'acqu2s'))
    pulprog = str(acqus.get('PULPROG', '')).strip('<> ')

    # Núcleos: F2 (directa) = NUC1 do acqus; F1 (indirecta) = NUC1 do acqu2s
    nuc_f2 = str(acqus.get('NUC1', '1H')).strip('<> ') or '1H'
    nuc_f1 = str(acqu2s.get('NUC1',
                  acqus.get('NUC2', '1H'))).strip('<> ') or '1H'

    meta = {
        'pulprog'  : pulprog,
        'exp_type' : _classify_pulprog(pulprog),
        'nuc_f2'   : nuc_f2,
        'nuc_f1'   : nuc_f1,
        'sf_f2'    : sf_f2,
        'sf_f1'    : sf_f1,
        'sw_ppm_f2': sw_ppm_f2,
        'sw_ppm_f1': sw_ppm_f1,
    }
    return f2_ppm, f1_ppm, data, meta


# ── Loader principal ──────────────────────────────────────────────────────────

def load_bruker_dataset(root_folder, ppm_min=0.0, ppm_max=10.0,
                        n_points=32768, expno=1, procno=1,
                        reference_ppm=None, reference_report=None):
    """
    Carrega todos os espectros Bruker encontrados em root_folder.

    Todos os espectros são interpolados para um eixo ppm comum
    (de ppm_max a ppm_min, com n_points pontos) para ficarem compatíveis
    com o formato interno da aplicação.

    ``reference_ppm`` (ex.: 0.0 para TSP/DSS) referencia cada amostra antes
    da interpolação: o ápice do padrão, procurado no eixo nativo
    (``processing.peak_alignment.reference_offset``), é levado a esse valor.
    O TopSpin entrega cada amostra com a sua própria referência (no estudo
    Mice ISS, -0.088 a +0.078 ppm), além do alcance do alinhamento rígido da
    deteção. ``reference_report`` (dict) recebe, por amostra,
    ``{'offset_ppm', 'apex_ppm', 'snr'}``; ``offset_ppm`` None quando o
    padrão não foi encontrado e a amostra ficou como estava.

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

            name  = _sample_name_from_path(proc_dir, root_folder)
            if reference_ppm is not None:
                from processing.peak_alignment import reference_offset
                off, apex, snr = reference_offset(ppm_raw, spec_raw,
                                                  target_ppm=reference_ppm)
                if off is not None:
                    ppm_raw = ppm_raw - off
                if reference_report is not None:
                    reference_report[name] = {'offset_ppm': off,
                                              'apex_ppm': apex, 'snr': snr}

            # Interpolação para eixo comum
            # np.interp exige x crescente — inverter se necessário
            if ppm_raw[0] > ppm_raw[-1]:
                ppm_raw  = ppm_raw[::-1]
                spec_raw = spec_raw[::-1]

            common_ppm_asc = common_ppm[::-1]   # crescente para interp
            spec_interp    = np.interp(common_ppm_asc, ppm_raw, spec_raw,
                                       left=0.0, right=0.0)
            spec_interp    = spec_interp[::-1]   # voltar à ordem NMR

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


def reference_summary(report, target_ppm=0.0):
    """``(status_text, missing_names)`` for the referencing a
    ``load_bruker_dataset`` call recorded in ``reference_report``."""
    moved = {n: r['offset_ppm'] for n, r in report.items()
             if r.get('offset_ppm') is not None}
    missing = [n for n, r in report.items() if r.get('offset_ppm') is None]
    if not moved:
        return (f'No reference standard found near {target_ppm:.2f} ppm: '
                f'spectra left as TopSpin referenced them', missing)
    lo, hi = min(moved.values()) * 1000, max(moved.values()) * 1000
    text = (f'Reference standard set to {target_ppm:.2f} ppm in {len(moved)} '
            f'spectra (TopSpin had it {lo:+.1f} to {hi:+.1f} mppm away)')
    if missing:
        text += f'; {len(missing)} without a standard left as they were'
    return text, missing


def read_spectrometer_mhz(root_folder, expno=1, procno=1):
    """The spectrometer frequency (MHz) the dataset was acquired at.

    Read from ``SF`` in each spectrum's ``procs`` -- the same value
    ``_load_spectrum`` uses to build the ppm axis, which until now was
    discarded after that. Every J-coupling the annotation predicts is
    converted from Hz to ppm with this number, so a wrong one moves every
    predicted line: annotating the 500 MHz reference study at the
    interface's fallback of 600 MHz found 10 of 18 analytes instead of 16,
    with 16 false positives instead of 10 and a p90 error of 222% instead
    of 14%.

    Returns the frequency when every spectrum agrees to 0.1 MHz, ``None``
    when none could be read. Spectra acquired at different fields cannot
    share one ppm-to-Hz conversion, and that is an error the analyst has to
    see, not a value to average.
    """
    fields = []
    for proc_dir in _find_experiments(root_folder, expno=expno, procno=procno):
        try:
            sf = _read_jcamp(os.path.join(proc_dir, 'procs')).get('SF')
            if sf is not None and float(sf) > 0:
                fields.append(float(sf))
        except (OSError, ValueError, TypeError):
            continue
    if not fields:
        return None
    lo, hi = min(fields), max(fields)
    if hi - lo > 0.1:
        raise ValueError(
            f"Os espectros foram adquiridos a campos diferentes "
            f"({lo:.2f} a {hi:.2f} MHz): não partilham uma conversão "
            f"ppm↔Hz e não podem ser anotados juntos.")
    return float(np.median(fields))


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
