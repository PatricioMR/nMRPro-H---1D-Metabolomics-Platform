"""
pdf_report.py — Relatório técnico PDF para quantificação NMR.

Estrutura:
  Pág. 1  : Capa — logo, título (NMRPro-H vX), parâmetros, stacked spectrum por grupo
  Pág. 2+ : 5 analitos por página — recorte centrado no pico + tabela de resultados
             Lorentz: curva ajustada sobreposta + R² na tabela.
             Ordenados por concentração decrescente.
"""
from __future__ import annotations

import io
import math
import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit

import os
from pathlib import Path
from reportlab.lib               import colors
from reportlab.lib.pagesizes     import A4
from reportlab.lib.units         import cm, mm
from reportlab.lib.styles        import ParagraphStyle
from reportlab.platypus          import (SimpleDocTemplate, Paragraph, Spacer,
                                         Image, Table, TableStyle, HRFlowable,
                                         PageBreak, KeepTogether)
from reportlab.pdfbase           import pdfmetrics
from reportlab.pdfbase.ttfonts   import TTFont

# ── Registar DejaVu Sans (suporte a ● ○ e outros Unicode) ───────────────────
_DEJAVU_PATH = os.path.join(os.path.dirname(matplotlib.__file__),
                             'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')
_DEJAVU_BOLD = os.path.join(os.path.dirname(matplotlib.__file__),
                             'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')
_HAS_DEJAVU = False
try:
    if os.path.exists(_DEJAVU_PATH):
        pdfmetrics.registerFont(TTFont('DejaVuSans', _DEJAVU_PATH))
    if os.path.exists(_DEJAVU_BOLD):
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', _DEJAVU_BOLD))
    _HAS_DEJAVU = True
except Exception:
    pass

_UNICODE_FONT      = 'DejaVuSans'      if _HAS_DEJAVU else 'Helvetica'
_UNICODE_FONT_BOLD = 'DejaVuSans-Bold' if _HAS_DEJAVU else 'Helvetica-Bold'

# ── Constantes ───────────────────────────────────────────────────────────────
from app_paths import APP_VERSION
PAGE_W, PAGE_H = A4
MARGIN = 1.5 * cm

_BLUE_DARK  = colors.HexColor('#1a3a5c')
_BLUE_MID   = colors.HexColor('#2e6da4')
_BLUE_LIGHT = colors.HexColor('#d0e4f7')
_GREEN      = colors.HexColor('#2e7d32')
_YELLOW     = colors.HexColor('#f9a825')
_RED        = colors.HexColor('#c62828')
_GRAY_LIGHT = colors.HexColor('#f5f5f5')
_GRAY_MID   = colors.HexColor('#bdbdbd')
_WHITE      = colors.white

# Cores default por grupo (fallback se app.group_colors não disponível)
_DEFAULT_GROUP_COLORS = {
    'CTRL':    '#2E86AB',
    'G1':      '#A23B72',
    'G2':      '#F18F01',
    'G3':      '#17BEBB',
    'BLK':     '#C73E1D',
    'Unknown': '#593C8F',
}


# ── Helper: matplotlib fig → ReportLab Image ─────────────────────────────────
def _fig_to_rl_image(fig, width_cm, height_cm, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


# ── Estilos ───────────────────────────────────────────────────────────────────
def _make_styles():
    return {
        'title':    ParagraphStyle('t',   fontSize=22, textColor=_BLUE_DARK,
                                   fontName='Helvetica-Bold', spaceAfter=0, leading=26),
        'sub':      ParagraphStyle('s',   fontSize=12, textColor=_BLUE_MID,
                                   fontName='Helvetica', spaceAfter=0, leading=16),
        'date':     ParagraphStyle('dt',  fontSize=8,  textColor=colors.HexColor('#777'),
                                   fontName='Helvetica', spaceAfter=0, leading=12),
        'small':    ParagraphStyle('sm',  fontSize=8,  textColor=colors.HexColor('#555'),
                                   fontName='Helvetica'),
        'h2':       ParagraphStyle('h2',  fontSize=10, textColor=_BLUE_DARK,
                                   fontName='Helvetica-Bold', spaceBefore=4, spaceAfter=2),
        'cell':     ParagraphStyle('c',   fontSize=8,  fontName='Helvetica', leading=10),
        'cell_b':   ParagraphStyle('cb',  fontSize=8,  fontName='Helvetica-Bold', leading=10),
        # Estilo com fonte Unicode para caracteres ● ○
        'cell_uni': ParagraphStyle('cu',  fontSize=8,  fontName=_UNICODE_FONT, leading=10),
        'cell_uni_b': ParagraphStyle('cub', fontSize=8, fontName=_UNICODE_FONT_BOLD, leading=10),
    }


# ── Spectral overview: média por grupo ───────────────────────────────────────
def _stacked_spectrum_fig(ppm, all_int, samples_info, group_colors,
                          width_in=14, height_in=5,
                          ppm_range: tuple | None = None):
    """Uma linha por grupo (média). Sem stack."""
    import matplotlib.patches as mpatches

    si = samples_info or []
    n  = all_int.shape[1] if all_int.ndim > 1 else 1
    ppm_arr = np.asarray(ppm, dtype=float)

    def _grp_color(lbl):
        return group_colors.get(lbl) or _DEFAULT_GROUP_COLORS.get(lbl, '#888888')

    groups_ordered = list(dict.fromkeys(
        s.get('label', 'Unknown') for s in si
        if str(s.get('type', s.get('label', ''))).upper() != 'BLK'))

    grp_idx: dict[str, list[int]] = {}
    for i, s in enumerate(si):
        if i >= n:
            break
        lbl   = s.get('label', 'Unknown')
        stype = str(s.get('type', lbl)).upper()
        if stype == 'BLK':
            continue
        grp_idx.setdefault(lbl, []).append(i)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Máscara ppm para a janela visível — Y scale só sobre o que se vê
    lo_ppm, hi_ppm = (ppm_range if ppm_range is not None
                      else (float(ppm_arr.min()), float(ppm_arr.max())))
    mask = (ppm_arr >= lo_ppm) & (ppm_arr <= hi_ppm)

    means = []
    if groups_ordered:
        for grp in groups_ordered:
            idxs = grp_idx.get(grp, [])
            if not idxs:
                continue
            mat  = all_int[:, idxs] if all_int.ndim > 1 else all_int[:, None]
            mean = mat.mean(axis=1)
            means.append(mean[mask])
            ax.plot(ppm_arr, mean, color=_grp_color(grp), lw=1.0, alpha=0.9,
                    label=grp)
    else:
        mean = np.mean(all_int, axis=1) if all_int.ndim > 1 else all_int
        means.append(mean[mask])
        ax.plot(ppm_arr, mean, color='#2e6da4', lw=1.0)

    # Y: pico máximo visível no topo
    if means:
        y_top = float(np.concatenate(means).max()) * 2.0
        ax.set_ylim(0, y_top if y_top > 0 else 1.0)

    ax.set_xlim(hi_ppm, lo_ppm)  # invertido (δ decresce →)
    ax.set_xlabel('Chemical shift (ppm)', fontsize=9)
    ax.set_ylabel('Intensity', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    if groups_ordered:
        handles = [mpatches.Patch(color=_grp_color(g), label=g)
                   for g in groups_ordered if grp_idx.get(g)]
        ax.legend(handles=handles, fontsize=8, loc='upper right',
                  framealpha=0.8, edgecolor='#bdbdbd')

    fig.tight_layout(pad=0.4)
    return fig


# ── Recorte espectral centrado no pico ───────────────────────────────────────
def _peak_crop_fig(ppm, all_int, result, samples_info, group_colors,
                   width_in=5.0, height_in=3.0):
    """
    Recorte centrado em r['ppm']. Todas as amostras em baixa opacidade + média
    por grupo a cheio. Se Lorentz: sobrepõe curva ajustada a vermelho tracejado.
    """
    peak    = result.get('_peak') or {}
    ppm_ctr = result.get('ppm', 0.0)
    lo_idx  = peak.get('start_idx', 0)
    hi_idx  = peak.get('end_idx',   len(ppm) - 1)

    # Multiplet mode: use the actual integration window and re-centre the crop on it.
    _mw = result.get('_multiplet_window')
    if _mw and result.get('_integ_mode', '') == 'Multiplet':
        ppm_lo, ppm_hi = min(_mw), max(_mw)
        ppm_ctr = (ppm_lo + ppm_hi) / 2.0   # centre on multiplet, not anchor peak
    else:
        ppm_lo    = min(float(ppm[lo_idx]), float(ppm[hi_idx]))
        ppm_hi    = max(float(ppm[lo_idx]), float(ppm[hi_idx]))
    half_span = max((ppm_hi - ppm_lo) / 2.0, 0.015)
    pad       = max(half_span * 2.0, 0.05)
    win_lo, win_hi = ppm_ctr - pad, ppm_ctr + pad

    ppm_f = np.asarray(ppm, dtype=float)
    step  = abs(float(ppm_f[1] - ppm_f[0])) if len(ppm_f) > 1 else 0.001
    ppm_max = float(ppm_f.max())
    i_lo  = max(0,            int((ppm_max - win_hi) / step))
    i_hi  = min(len(ppm_f)-1, int((ppm_max - win_lo) / step))
    if i_lo >= i_hi:
        i_lo, i_hi = max(0, lo_idx - 15), min(len(ppm_f)-1, hi_idx + 15)

    x = ppm_f[i_lo:i_hi + 1]

    si = samples_info or []
    n  = all_int.shape[1] if all_int.ndim > 1 else 1

    _jr_frac     = result.get('_j_resolve_frac')
    _show_deconv = (result.get('_used_j_resolve')
                    and _jr_frac is not None
                    and 0.0 < float(_jr_frac) < 1.0)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    y_peak_max = -np.inf
    all_ys     = []

    # Outline is ALWAYS the total observed peak — the same crop the user sees in
    # the J-Resolve visualizer. For an NNLS-resolved analyte the fill below paints
    # only THIS analyte's resolved fraction of that shared peak (blue outline =
    # full peak, green fill = the analyte), mirroring the visualizer's stacked
    # contribution fill. A non-deconv analyte fills its own whole peak in blue.
    _INTEG_COL    = '#2e6da4'   # blue — peak outline / standard fill
    _ANALYTE_FILL = '#43a047'   # green — this analyte's resolved portion
    _y_mean_full = (all_int[i_lo:i_hi+1] if all_int.ndim == 1
                    else np.mean(all_int[i_lo:i_hi+1, :], axis=1))
    ax.plot(x, _y_mean_full, color=_INTEG_COL, lw=1.5, zorder=3)
    y_peak_max = float(np.max(_y_mean_full))
    all_ys.append(_y_mean_full)

    # Integration window indices
    _mw = result.get('_multiplet_window')
    if _mw and result.get('_integ_mode', '') == 'Multiplet':
        _f_lo_ppm, _f_hi_ppm = min(_mw), max(_mw)
        _step_f   = abs(float(ppm_f[1] - ppm_f[0])) if len(ppm_f) > 1 else 0.001
        _ppm_max_f = float(ppm_f.max())
        lo_f = max(0, int((_ppm_max_f - _f_hi_ppm) / _step_f))
        hi_f = min(len(ppm_f) - 1, int((_ppm_max_f - _f_lo_ppm) / _step_f))
    else:
        lo_f = max(lo_idx, i_lo)
        hi_f = min(hi_idx, i_hi)

    # Single-colour fill of the integrated area (NNLS → this analyte's fraction).
    if lo_f < hi_f:
        xf = ppm_f[lo_f:hi_f + 1]
        yf = (np.mean(all_int[lo_f:hi_f + 1, :], axis=1)
              if all_int.ndim > 1 else all_int[lo_f:hi_f + 1])
        _line_fracs = result.get('_jr_line_fracs') if _show_deconv else None
        if _show_deconv and _line_fracs:
            # Per-line fractions: paint each point by the fraction of its nearest
            # predicted line, so legs that NNLS resolved differently show their own
            # share — the same per-leg reality folded into the quantified area.
            _lps = np.asarray([p for p, _ in _line_fracs], dtype=float)
            _lfs = np.asarray([f for _, f in _line_fracs], dtype=float)
            _near = np.argmin(np.abs(xf[:, None] - _lps[None, :]), axis=1)
            yf = yf * _lfs[_near]
        elif _show_deconv:
            yf = yf * float(_jr_frac)
        _fill_col = _ANALYTE_FILL if _show_deconv else _INTEG_COL
        ax.fill_between(xf, 0, np.maximum(yf, 0),
                        alpha=0.35, color=_fill_col, linewidth=0, zorder=2)

    # Média global para Lorentz overlay
    y_mean = (np.mean(all_int[i_lo:i_hi+1, :], axis=1)
              if all_int.ndim > 1 else all_int[i_lo:i_hi+1])

    # ── Overlay Lorentziana ───────────────────────────────────────────────
    lorentz_r2 = float('nan')
    lorentz    = result.get('_lorentz')
    if lorentz and result.get('_integ_mode', '').lower() == 'lorentz':
        try:
            cen      = float(lorentz.get('cen',      ppm_ctr))
            wid      = float(lorentz.get('wid',      0.01))
            bl_y0    = float(lorentz.get('bl_y0',    0.0))
            bl_slope = float(lorentz.get('bl_slope', 0.0))
            bl_x0    = float(lorentz.get('bl_x0',    float(ppm_f[lo_idx])))
            lorentz_r2 = float(lorentz.get('r2', float('nan')))

            # Skip overlay when fit is invalid (avoids diagonal artefact)
            if not (np.isfinite(lorentz_r2) and lorentz_r2 >= 0.70):
                raise ValueError('low R²')

            # Amplitude from baseline-corrected mean in integration window
            lo_r = max(lo_idx, i_lo); hi_r = min(hi_idx, i_hi)
            x_r  = ppm_f[lo_r:hi_r+1]
            y_r  = (np.mean(all_int[lo_r:hi_r+1, :], axis=1)
                    if all_int.ndim > 1 else all_int[lo_r:hi_r+1])
            bl_r = bl_y0 + bl_slope * (x_r - bl_x0)
            cen_i = int(np.argmin(np.abs(x_r - cen)))
            A     = max(float((y_r - bl_r)[cen_i]), 1e-30)

            # Plot Lorentz curve only over the integration window (x_r), not
            # the full view — prevents extrapolated baseline from looking like
            # a diagonal ramp when the window contains multiple peaks.
            x_plot        = np.linspace(float(x_r.min()), float(x_r.max()), 300)
            baseline_plot = bl_y0 + bl_slope * (x_plot - bl_x0)
            shape_plot    = wid**2 / ((x_plot - cen)**2 + wid**2)
            ax.plot(x_plot, baseline_plot + A * shape_plot,
                    color='#e53935', lw=1.4, ls='--', zorder=5,
                    label=f'Lorentz  R²={lorentz_r2:.4f}')
            ax.legend(fontsize=6, loc='upper left',
                      framealpha=0.7, edgecolor='#cccccc')
        except Exception:
            pass

    if _show_deconv:
        ax.text(0.02, 0.97,
                f'JG-Deconv: {float(_jr_frac) * 100:.1f}% of peak',
                transform=ax.transAxes, fontsize=6, va='top',
                color='#1565C0',
                bbox=dict(fc='white', alpha=0.75, lw=0))
        if not (result.get('_lorentz') and
                result.get('_integ_mode', '').lower() == 'lorentz'):
            ax.legend(fontsize=6, loc='upper left',
                      framealpha=0.7, edgecolor='#cccccc')

    # Centrar Y: topo do pico a ~85%
    y_base = min((float(np.min(y)) for y in all_ys), default=0.0)
    y_base = min(y_base, 0.0)
    if y_peak_max > y_base:
        span = y_peak_max - y_base
        ax.set_ylim(y_base - span * 0.04, y_base + span / 0.82)

    # Clip ao intervalo da janela — impede linhas de sair do plot
    ax.set_xlim(float(x.max()), float(x.min()))   # ppm invertido
    ax.set_xlabel('δ (ppm)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout(pad=0.3)
    return fig, lorentz_r2


# ── Tabela de DB + anotação ───────────────────────────────────────────────────
def _annotation_table(annotation_stage, avail_w):
    ann  = annotation_stage
    lra  = getattr(ann, '_last_results_args', None) or {}
    db_path = (lra.get('db_path') or
               getattr(ann, '_db_path', None) or '—')
    db_name = Path(db_path).name if db_path != '—' else '—'

    try:
        tol = f"{float(ann.tolerance_var.get()):.3f} ppm"
    except Exception:
        tol = '—'
    try:
        mhz = f"{float(ann.spectrometer_var.get()):.0f} MHz"
    except Exception:
        mhz = '—'

    # Contagem de resultados
    results = lra.get('results', [])
    n_annotated = len({r.get('metabolite', '') for r in results}) if results else '—'
    elapsed = lra.get('elapsed', None)
    elapsed_s = f'{elapsed:.1f} s' if elapsed else '—'

    rows = [
        ['Annotation Parameter', 'Value'],
        ['Database file',    db_name],
        ['Database path',    str(db_path)],
        ['Shift tolerance',  tol],
        ['Spectrometer',     mhz],
        ['Annotated compounds', str(n_annotated)],
        ['Search time',      elapsed_s],
    ]
    col_w = [avail_w * 0.38, avail_w * 0.55]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), _BLUE_MID),
        ('TEXTCOLOR',   (0, 0), (-1, 0), _WHITE),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 8),
        ('FONTNAME',    (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [_WHITE, _GRAY_LIGHT]),
        ('GRID',        (0, 0), (-1, -1), 0.3, _GRAY_MID),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING',  (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('WORDWRAP',    (1, 2), (1, 2), 1),   # path pode ser longo
    ]))
    return tbl


# ── Tabela de parâmetros (pág.1) ─────────────────────────────────────────────
def _params_table(qs, app, annotation_stage=None):
    norm_mode  = qs.norm_var.get() if hasattr(qs, 'norm_var') else '—'
    integ_mode = qs.integ_var.get() if hasattr(qs, 'integ_var') else '—'
    proton_cor = bool(getattr(qs, 'proton_corr_var', None) and
                      qs.proton_corr_var.get())
    ref_peak   = getattr(qs, 'ref_peak', None)
    is_ppm     = f"{ref_peak.get('ppm', 0):.4f}" if ref_peak else '—'
    is_conc    = qs.ref_conc_var.get() if hasattr(qs, 'ref_conc_var') else '—'
    is_nh      = qs.ref_nh_var.get()   if hasattr(qs, 'ref_nh_var')   else '—'

    si        = getattr(app, 'samples_info', []) or []
    n_samples = len([s for s in si if str(s.get('type', '')).upper() != 'BLK'])
    n_blk     = len([s for s in si if str(s.get('type', '')).upper() == 'BLK'])
    groups    = list(dict.fromkeys(
        s.get('label', '') for s in si
        if str(s.get('type', '')).upper() != 'BLK'))

    avail_w = PAGE_W - 2 * MARGIN
    rows = [
        ['Parameter', 'Value'],
        ['Normalisation',     norm_mode],
        ['Integration mode',  integ_mode],
        ['Proton correction', 'Yes' if proton_cor else 'No'],
        ['IS chemical shift', f'{is_ppm} ppm'],
        ['IS concentration',  f'{is_conc} mM'],
        ['IS nH',             str(is_nh)],
        ['Samples (non-BLK)', str(n_samples)],
        ['Blank samples',     str(n_blk)],
        ['Groups',            ', '.join(groups) or '—'],
        ['Report date',       datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
    ]

    if annotation_stage is not None:
        ann = annotation_stage
        lra = getattr(ann, '_last_results_args', None) or {}
        db_path = lra.get('db_path') or getattr(ann, '_db_path', None) or '—'
        db_name = Path(db_path).name if db_path != '—' else '—'
        try:
            tol = f"{float(ann.tolerance_var.get()):.3f} ppm"
        except Exception:
            tol = '—'
        try:
            mhz = f"{float(ann.spectrometer_var.get()):.0f} MHz"
        except Exception:
            mhz = '—'
        rows += [
            ['Annotation DB',    db_name],
            ['Shift tolerance',  tol],
            ['Spectrometer',     mhz],
        ]
    col_w = [avail_w * 0.38, avail_w * 0.55]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), _BLUE_DARK),
        ('TEXTCOLOR',   (0, 0), (-1, 0), _WHITE),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 8),
        ('FONTNAME',    (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [_WHITE, _GRAY_LIGHT]),
        ('GRID',        (0, 0), (-1, -1), 0.3, _GRAY_MID),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING',  (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return tbl


# ── Qualidade ─────────────────────────────────────────────────────────────────
def _quality_info(r, theoretical):
    conc   = r.get('conc_mm', float('nan'))
    conf   = r.get('_confidence', 0) or 0

    dev_pct = None
    if theoretical:
        key  = str(r.get('hmdb_id', r['metabolite']))
        theo = theoretical.get(key)
        if theo is None:
            nl = r['metabolite'].lower()
            for k, v in theoretical.items():
                if k.lower() in nl or nl in k.lower():
                    theo = v; break
        if theo and theo > 0 and np.isfinite(conc) and conc > 0:
            dev_pct = (conc - theo) / theo * 100.0

    if dev_pct is not None:
        if   abs(dev_pct) < 10:  return f'PASS  (Dev {dev_pct:+.1f}%)', _GREEN
        elif abs(dev_pct) < 15:  return f'WARN  (Dev {dev_pct:+.1f}%)', _YELLOW
        else:                    return f'FAIL  (Dev {dev_pct:+.1f}%)', _RED

    # _confidence é int 1/2/3
    ci = int(conf)
    if   ci >= 3: return 'CONF ●●●  100%', _GREEN
    elif ci >= 2: return 'CONF ●●○   67%', _YELLOW
    else:         return 'CONF ●○○   33%', _RED


# ── Bloco de 1 analito ───────────────────────────────────────────────────────
def _analyte_block(r, ppm, all_int, samples_info, group_colors,
                   theoretical, styles, avail_w,
                   crop_w_cm, crop_h_cm, unit='mM'):
    crop_fig, lorentz_r2 = _peak_crop_fig(
        ppm, all_int, r, samples_info, group_colors,
        width_in=crop_w_cm * 0.55, height_in=crop_h_cm * 0.55)
    crop_img = _fig_to_rl_image(crop_fig, crop_w_cm, crop_h_cm, dpi=130)

    conc   = r.get('conc_mm', float('nan'))
    conc_s = f'{conc:.4f} {unit}' if np.isfinite(conc) else 'N/A'
    snr    = r.get('snr', float('nan'))
    snr_s  = f'{snr:.1f}' if snr and np.isfinite(snr) else '—'

    # CV por grupo
    cv_grp = r.get('cv_by_group', {}) or {}
    cv_lines = []
    for grp, cv in cv_grp.items():
        if grp == 'BLK': continue
        cv_val = cv.get('cv') if isinstance(cv, dict) else cv
        if cv_val is not None and np.isfinite(cv_val):
            cv_lines.append(f'{grp}: {cv_val:.1f}%')

    # Conc por grupo (com IC95 bootstrap não-paramétrico quando disponível)
    cbg = r.get('_concs_by_group', {}) or {}
    cig = r.get('_conc_ci_by_group', {}) or {}
    conc_lines = []
    for grp, vals in cbg.items():
        if grp == 'BLK': continue
        fv = [v for v in vals if np.isfinite(v) and v > 0]
        if not fv: continue
        mn = np.mean(fv)
        sd = np.std(fv, ddof=1) if len(fv) > 1 else 0.0
        _lo, _hi = cig.get(grp, (float('nan'), float('nan')))
        ci_s = (f'  [CI95 {_lo:.4f}–{_hi:.4f}]'
                if np.isfinite(_lo) and np.isfinite(_hi) else '')
        conc_lines.append(
            f'{grp}: {mn:.4f} ± {sd:.4f} {unit} (n={len(fv)}){ci_s}')

    # Desvio teórico
    dev_line = ''
    if theoretical:
        key  = str(r.get('hmdb_id', r['metabolite']))
        theo = theoretical.get(key)
        if theo is None:
            nl = r['metabolite'].lower()
            for k, v in theoretical.items():
                if k.lower() in nl or nl in k.lower():
                    theo = v; break
        if theo and theo > 0 and np.isfinite(conc) and conc > 0:
            dev = (conc - theo) / theo * 100.0
            dev_line = f'Theoretical: {theo:.4f} mM  |  Dev: {dev:+.1f}%'

    # Status confidence: já tem ● e ○
    status = r.get('status', '—')

    qual_label, qual_color_rl = _quality_info(r, theoretical)
    qhex = ('#2e7d32' if qual_color_rl == _GREEN else
            '#f9a825' if qual_color_rl == _YELLOW else '#c62828')

    def P(txt, bold=False, uni=False):
        if uni:
            st = styles['cell_uni_b'] if bold else styles['cell_uni']
        else:
            st = styles['cell_b'] if bold else styles['cell']
        return Paragraph(str(txt).replace('\n', '<br/>'), st)

    integ_label = f"{r.get('_integ_mode','—')}  |  nH={r.get('_nh',1):.0f}"
    if np.isfinite(lorentz_r2):
        integ_label += f"  |  R²={lorentz_r2:.4f}"

    rows = [
        [P(r['metabolite'], bold=True),
         P(f"ppm {r.get('ppm', 0):.4f}  |  HMDB {r.get('hmdb_id', '—')}", bold=True)],
        [P('Integration'),      P(integ_label)],
        [P('Annotation score'), P(f"{r.get('_ann_score', 0.0):.3f}")],
        [P('SNR'),              P(snr_s)],
        [P('Method'),
         P(('MCR ' if r.get('_used_mcr') else '') +
           ('JG-Deconv ' if r.get('_used_j_resolve') else '') or 'Standard')],
    ]
    for cl in conc_lines:
        rows.append([P('Concentration'), P(cl)])
    for cv in cv_lines:
        rows.append([P('CV%'), P(cv)])
    if dev_line:
        rows.append([P('vs Theoretical'), P(dev_line)])

    # Linha de qualidade final (span) — também Unicode
    rows.append([
        Paragraph(f'<font color="{qhex}"><b>{qual_label}</b></font>',
                  ParagraphStyle('ql', fontSize=8,
                                 fontName=_UNICODE_FONT_BOLD, leading=11)),
        Paragraph('', styles['cell'])
    ])

    info_w  = avail_w - crop_w_cm * cm - 3 * mm
    col1_w  = info_w * 0.36
    col2_w  = info_w * 0.64
    info_tbl = Table(rows, colWidths=[col1_w, col2_w])
    info_tbl.setStyle(TableStyle([
        ('SPAN',        (0, 0),  (1, 0)),
        ('SPAN',        (0, -1), (1, -1)),
        ('BACKGROUND',  (0, 0),  (1, 0),  _BLUE_LIGHT),
        ('FONTSIZE',    (0, 0),  (-1, -1), 8),
        ('GRID',        (0, 0),  (-1, -1), 0.2, _GRAY_MID),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [_WHITE, _GRAY_LIGHT]),
        ('VALIGN',      (0, 0),  (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0),  (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING',  (0, 0),  (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND',  (0, -1), (1, -1),
         colors.HexColor('#e8f5e9') if qual_color_rl == _GREEN else
         colors.HexColor('#fffde7') if qual_color_rl == _YELLOW else
         colors.HexColor('#ffebee')),
    ]))

    outer = Table(
        [[crop_img, info_tbl]],
        colWidths=[crop_w_cm * cm, info_w],
    )
    outer.setStyle(TableStyle([
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING',   (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BOX',          (0, 0), (-1, -1), 0.5, _GRAY_MID),
    ]))
    return outer


# ── Ponto de entrada ──────────────────────────────────────────────────────────
def generate_pdf(output_path: str, quant_stage, app,
                 analytes_per_page: int = 5,
                 annotation_stage=None):
    ppm     = getattr(app, 'ppm', None)
    all_int = getattr(app, 'all_intensities', None)
    si      = getattr(app, 'samples_info', None) or []
    gc      = getattr(app, 'group_colors', {}) or {}
    results = [r for r in (quant_stage._quant_results or [])
               if not r.get('_is_is') and not r.get('_not_quantified')]

    # Lipidomics is semi-quantitative (nH=1): label values as relative a.u.,
    # never absolute mM, even when IS-normalised.
    _lp_args = getattr(annotation_stage, '_last_results_args', None) or {}
    _is_lipid = str(_lp_args.get('internal_library_mode', '')) == 'Lipidomics'
    unit = 'a.u. (rel.)' if _is_lipid else 'mM'

    if ppm is None or all_int is None:
        raise ValueError('Dados espectrais nao disponiveis.')
    if not results:
        raise ValueError('Sem resultados de quantificacao.')

    results_sorted = sorted(
        results,
        key=lambda r: r.get('conc_mm', 0.0) if np.isfinite(r.get('conc_mm', 0.0)) else 0.0,
        reverse=True)

    theoretical = dict(getattr(quant_stage, '_validation_theoretical', {}) or {})
    styles  = _make_styles()
    avail_w = PAGE_W - 2 * MARGIN

    # Logo
    from app_paths import asset_path
    logo_path = Path(asset_path('utils', 'Logo_NMRPro-H.png'))
    logo_img = None
    if logo_path.exists():
        try:
            logo_img = Image(str(logo_path), width=3.2 * cm, height=3.2 * cm)
        except Exception:
            pass

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='NMR Quantification Report', author=APP_VERSION)

    story = []

    # ══════════════ PÁG. 1: CAPA ═════════════════════════════════════════════
    now_str  = datetime.datetime.now().strftime('%d %B %Y  —  %H:%M')
    logo_w   = 3.6 * cm
    text_w   = avail_w - logo_w - 4 * mm

    title_tbl_data = [[
        Paragraph(APP_VERSION, styles['title']),
    ], [
        Paragraph('Quantification Technical Report', styles['sub']),
    ], [
        Spacer(1, 3),
    ], [
        Paragraph(now_str, styles['date']),
    ]]

    title_inner = Table(title_tbl_data, colWidths=[text_w])
    title_inner.setStyle(TableStyle([
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    if logo_img:
        # Linha azul vertical entre logo e título
        from reportlab.platypus.flowables import Flowable
        hdr_data = [[logo_img, title_inner]]
        hdr = Table(hdr_data, colWidths=[logo_w, text_w])
        hdr.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING',   (0, 0), (0, -1), 0),    # logo: sem padding esq
            ('LEFTPADDING',   (1, 0), (1, -1), 14),   # título: 14pt afastado da linha
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('LINEAFTER',     (0, 0), (0, -1), 1.5, _BLUE_MID),
        ]))
        story.append(hdr)
    else:
        story.append(Paragraph(APP_VERSION, styles['title']))
        story.append(Paragraph('Quantification Technical Report', styles['sub']))
        story.append(Paragraph(now_str, styles['date']))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width='100%', thickness=1.5,
                             color=_BLUE_DARK, spaceAfter=6))
    story.append(Paragraph('Analysis Parameters', styles['h2']))
    story.append(_params_table(quant_stage, app, annotation_stage=annotation_stage))
    story.append(Spacer(1, 0.3 * cm))

    # Vista fixa -0.5–4 ppm
    _ppm_range = (-0.5, 4.0)

    # Stacked spectrum
    story.append(Paragraph('Spectral Overview — All Samples by Group', styles['h2']))
    wf_fig = _stacked_spectrum_fig(ppm, all_int, si, gc, width_in=13, height_in=4.5,
                                   ppm_range=_ppm_range)
    story.append(_fig_to_rl_image(wf_fig, width_cm=avail_w / cm,
                                   height_cm=6.5, dpi=130))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f'Quantified metabolites: <b>{len(results_sorted)}</b>  |  '
        f'Ordered by concentration (decreasing)',
        styles['small']))
    story.append(PageBreak())

    # ══════════════ PÁG. 2+: ANALITOS ════════════════════════════════════════
    # Fixed block size so exactly 5 fit per A4 page.
    # KeepTogether prevents each block from splitting across pages.
    # No explicit PageBreak — ReportLab paginates naturally.
    per_page  = max(1, min(analytes_per_page, 6))
    row_h_cm  = (PAGE_H - 2 * MARGIN - 1.4 * cm) / per_page / cm
    crop_h_cm = row_h_cm * 0.86
    crop_w_cm = (avail_w / cm) * 0.44

    story.append(Paragraph('Quantification Results', styles['h2']))
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=_BLUE_MID, spaceAfter=3))

    for rank_i, r in enumerate(results_sorted):
        block = _analyte_block(
            r, ppm, all_int, si, gc, theoretical,
            styles, avail_w,
            crop_w_cm=crop_w_cm, crop_h_cm=crop_h_cm, unit=unit)
        spacer = Spacer(1, 0.15 * cm)
        story.append(KeepTogether([block, spacer]))

    doc.build(story)
