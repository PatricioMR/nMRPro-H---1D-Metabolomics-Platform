# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

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

from plotting import contributions
from plotting.fit_curve import (FIT_ABOVE_TRACE, VIEW_ZOOM_BELOW, draw_fit_residual,
                                fit_group, fit_over_trace, fit_view_top, fit_warnings)

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
def _grp_color(group_colors, lbl):
    return (group_colors or {}).get(lbl) or _DEFAULT_GROUP_COLORS.get(lbl, '#888888')


def _group_means(all_int, samples_info):
    """``[(group, mean trace)]`` in the analyst's order, blanks left out --
    the same grouping the quantification reports by
    (``_quant_adapter.sample_groups``). Nothing is computed here that the
    table does not already show per group; this is the trace behind it."""
    from gui.stages._quant_adapter import sample_groups
    arr = np.asarray(all_int, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = []
    for grp, idxs in sample_groups(samples_info, arr.shape[1]).items():
        if idxs:
            out.append((grp, np.mean(arr[:, idxs], axis=1)))
    return out


def _best_group(group_means, mask, prefer=None):
    """``(group, trace)`` the numbers were read on: the group quantification
    chose (``prefer``, the row's ``_snr_group``); without one, the group
    whose mean stands highest inside ``mask`` -- the annotation window's
    rule (``_best_group_mean``). ``(None, None)`` without groups."""
    for grp, y in group_means:
        if prefer is not None and str(grp) == str(prefer):
            return grp, y
    best, best_val = (None, None), -np.inf
    for grp, y in group_means:
        v = float(np.max(y[mask])) if mask.any() else -np.inf
        if v > best_val:
            best, best_val = (grp, y), v
    return best


def _group_amplitude(amps, samples_info, group) -> float:
    """The amplitude a window resolved by shape found for ``group``: the mean
    over its samples where it is positive -- the statistic the table reports
    per group -- so the component drawn on the group's mean trace is the
    component of the group's number. Without a group, the median over every
    sample the fit holds."""
    from gui.stages._quant_adapter import sample_groups
    members = (sample_groups(samples_info, max(int(k) for k in amps) + 1)
               .get(group) if group is not None and amps else None)
    vals = [float(amps[s]) for s in (members or ()) if s in amps
            and np.isfinite(amps[s]) and amps[s] > 0]
    return float(np.mean(vals)) if vals else float(np.median(list(amps.values())))


def _best_group_trace(group_means, mask, fallback, prefer=None):
    """The trace of ``_best_group``, or ``fallback``."""
    _g, y = _best_group(group_means, mask, prefer)
    return fallback if y is None else y


def _stacked_spectrum_fig(ppm, all_int, samples_info, group_colors,
                          width_in=14, height_in=5,
                          ppm_range: tuple | None = None):
    """Uma linha por grupo (média). Sem stack."""
    import matplotlib.patches as mpatches

    ppm_arr = np.asarray(ppm, dtype=float)
    grp_means = _group_means(all_int, samples_info)
    groups_ordered = [g for g, _ in grp_means]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Máscara ppm para a janela visível — Y scale só sobre o que se vê
    lo_ppm, hi_ppm = (ppm_range if ppm_range is not None
                      else (float(ppm_arr.min()), float(ppm_arr.max())))
    mask = (ppm_arr >= lo_ppm) & (ppm_arr <= hi_ppm)

    means = []
    if groups_ordered:
        for grp, mean in grp_means:
            means.append(mean[mask])
            ax.plot(ppm_arr, mean, color=_grp_color(group_colors, grp),
                    lw=1.0, alpha=0.9, label=grp)
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
        handles = [mpatches.Patch(color=_grp_color(group_colors, g), label=g)
                   for g in groups_ordered]
        ax.legend(handles=handles, fontsize=8, loc='upper right',
                  framealpha=0.8, edgecolor='#bdbdbd')

    fig.tight_layout(pad=0.4)
    return fig


def resolution_panels(quant_stage, app, annotation_stage=None, field_mhz=None):
    """``[(group, out)]``: the RP-Method resolution plot of every group, as
    the window computes it with its defaults (the confirmed analytes' other
    signals subtracted) -- ``processing.resolution_plot.resolution`` on
    ``rp_method_window.project_inputs`` -- so the report shows what the
    window shows. Raises when the project has no run to read."""
    from gui.widgets.rp_method_window import project_inputs
    from processing import resolution_plot as rp
    ppm, X, groups, rows, excluded, sigma, min_snr = project_inputs(quant_stage)
    mets = list(getattr(annotation_stage, '_metabolites', None) or [])
    try:
        tol = float(annotation_stage.tolerance_var.get())
    except Exception:
        from controllers.analysis_defaults import ANNOTATION
        tol = float(ANNOTATION['tolerance_ppm'])
    si = getattr(app, 'samples_info', None) or []
    out = []
    for g, idx in groups.items():
        trace = np.mean(X[:, idx], axis=1)
        res = rp.resolution(rows, ppm, trace, g,
                            lambda a, g=g: _group_amplitude(a, si, g),
                            metabolites=mets, sigma=sigma, field_mhz=field_mhz,
                            tolerance_ppm=tol, excluded=excluded, min_snr=min_snr)
        res.update(ppm=ppm, trace=trace, excluded=excluded, rows=rows,
                   is_height=is_height(rows, ppm, trace))
        out.append((g, res))
    return out


def is_height(rows, ppm, trace):
    """The internal standard's height on ``trace``: the highest point of
    the region its number came from. None without an internal standard
    row."""
    ppm = np.asarray(ppm, dtype=float)
    for r in rows:
        if not r.get('_is_is'):
            continue
        sp = contributions.number_region(r, ppm)
        if not sp and r.get('ppm') is not None:
            sp = (float(r['ppm']) - 0.05, float(r['ppm']) + 0.05)
        inside = (ppm >= min(sp) - 1e-9) & (ppm <= max(sp) + 1e-9) if sp else None
        if inside is not None and inside.any():
            h = float(np.max(np.asarray(trace)[inside]))
            if h > 0:
                return h
    return None


def resolved_text(group, res) -> str:
    """The line under a group's resolution plot: the share resolved and the
    residual's regions."""
    share = res.get('share')
    miss = sum(1 for g in res['regions'] if g['kind'] == 'missing')
    over = len(res['regions']) - miss
    sec = res.get('secondary') or []
    from xml.sax.saxutils import escape
    return (f"<b>{escape(str(group))}</b>: spectrum resolved <b>"
            + (f"{100 * share:.0f}%" if share is not None else 'not measured')
            + f"</b> of its signal area above the fits' baselines  ·  {miss} missing "
            f"signal(s), {over} over-claim(s)"
            + (f"  ·  {len(sec)} other signal(s) of {len({c['id'] for c in sec})} "
               f"confirmed analyte(s) subtracted as their numbers predict"
               if sec else ''))


# The report's view of the resolution plot: where the metabolites are, at
# half the internal standard's height, so the signals between the tallest
# show (the RP-Method window keeps the whole spectrum).
RP_REPORT_PPM = (0.0, 8.0)
RP_REPORT_IS_SHARE = 0.5


def _resolution_fig(group, res, width_in=13, height_in=4.2):
    """A group's resolution plot, drawn as the RP-Method window draws it
    (``plotting.resolution_plot``), over ``RP_REPORT_PPM`` and up to
    ``RP_REPORT_IS_SHARE`` of the internal standard's height (the window's
    scale without an internal standard); the region markers left out."""
    from plotting import resolution_plot as rp_draw
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    palette = contributions.stable_palette(res['rows'], res['ppm'])
    rp_draw.draw_resolution(ax, res['ppm'], res['trace'], res, palette)
    lo, hi, m = rp_draw.view_limits(res['ppm'], res['trace'], res['resolved'],
                                    res['excluded'], RP_REPORT_PPM)
    scale = 'y: whole height'
    if res.get('is_height'):
        m = RP_REPORT_IS_SHARE * res['is_height']
        scale = f"y: {100 * RP_REPORT_IS_SHARE:.0f}% of the internal standard's height"
    rp_draw.apply_limits(ax, lo, hi, m)
    share = res.get('share')
    ax.set_title(f"{group}: resolved "
                 + (f"{100 * share:.0f}%" if share is not None else '—'),
                 fontsize=10, loc='left')
    ax.set_title(scale, fontsize=7, loc='right', color='#777777')
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(8)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=5,
              frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(pad=0.4)
    return fig


def _over_trace_text(ratio) -> str:
    return (f'the fit holds {ratio:.2f}x the area the trace has under this '
            f'signal')


def _crop_legends(ax, grp, group_colors, drawn, shown):
    """The groups, upper right, when there is more than one; upper left the
    analytes of the crop (this one in bold) and the curves named after what
    they are (fit, shape fit, components, baseline, Lorentz)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = {str(g) for g, _y in grp}
    if len(grp) > 1:
        ax.add_artist(ax.legend(
            handles=[Line2D([], [], color=_grp_color(group_colors, g),
                            lw=1.6 if shown is not None and str(g) == str(shown) else 0.8,
                            label=g) for g, _y in grp],
            fontsize=6, loc='upper right', framealpha=0.7, edgecolor='#cccccc'))
    order = list(reversed(drawn))
    handles = [Patch(facecolor=c, alpha=contributions.FILL_ALPHA, edgecolor=c,
                     lw=1.6 if sel else 0.6,
                     label=(n if len(n) <= 26 else n[:25] + '\u2026'))
               for n, c, sel in order]
    seen = set()
    for ln in ax.get_lines():
        lb = ln.get_label()
        if lb and not lb.startswith('_') and lb not in seen and lb not in names:
            seen.add(lb)
            handles.append(ln)
    if handles:
        leg = ax.legend(handles=handles, fontsize=5.5, loc='upper left',
                        framealpha=0.7, edgecolor='#cccccc')
        for txt, (_n, _c, sel) in zip(leg.get_texts(), order):
            if sel:
                txt.set_fontweight('bold')


# ── Recorte espectral centrado no pico ───────────────────────────────────────
def _peak_crop_fig(ppm, all_int, result, samples_info, group_colors,
                   width_in=5.0, height_in=3.0, rows=None):
    """
    Recorte centrado em r['ppm']: a média de cada grupo, e cada analito de
    ``rows`` que cai no recorte desenhado pelo que deu o seu número
    (``plotting.contributions``, ADR-103), este por cima e mais carregado.
    Se Lorentz: sobrepõe curva ajustada a vermelho tracejado.
    """
    peak    = result.get('_peak') or {}
    ppm_ctr = result.get('ppm', 0.0)
    lo_idx  = peak.get('start_idx', 0)
    hi_idx  = peak.get('end_idx',   len(ppm) - 1)

    # Multiplet mode: use the actual integration window and re-centre the crop on it.
    _mw = result.get('_multiplet_window')
    _windowed = result.get('_integ_mode', '') in ('Multiplet', 'Shape', 'Cascade')
    if _mw and _windowed:
        ppm_lo, ppm_hi = min(_mw), max(_mw)
        ppm_ctr = (ppm_lo + ppm_hi) / 2.0   # centre on multiplet, not anchor peak
    else:
        ppm_lo    = min(float(ppm[lo_idx]), float(ppm[hi_idx]))
        ppm_hi    = max(float(ppm[lo_idx]), float(ppm[hi_idx]))
    # An amount from a fit is shown on the whole fitted signal, every line
    # of it, and on the lines the election found -- both, so a disagreement
    # between them is in the picture rather than cropped out of it.
    _focus = (result.get('_fit_curve') or {}).get('focus')
    if _focus:
        ppm_lo = min(ppm_lo, float(min(_focus)))
        ppm_hi = max(ppm_hi, float(max(_focus)))
        ppm_ctr = (ppm_lo + ppm_hi) / 2.0
    # and a window shared by shapes, whole: the components it was cut into
    _dcw = (result.get('_deconv') or {}).get('window')
    if _dcw:
        ppm_lo = min(ppm_lo, float(min(_dcw)))
        ppm_hi = max(ppm_hi, float(max(_dcw)))
        ppm_ctr = (ppm_lo + ppm_hi) / 2.0
    half_span = max((ppm_hi - ppm_lo) / 2.0, 0.015)
    pad       = max(half_span * 2.0, 0.05)
    win_lo, win_hi = ppm_ctr - pad, ppm_ctr + pad

    ppm_f = np.asarray(ppm, dtype=float)
    # By membership, not by counting steps down from the axis maximum: that
    # assumed δ stored decreasing, and on an axis stored increasing (the
    # bench projects) cut the mirror image of the region -- another analyte.
    _in_crop = np.nonzero((ppm_f >= win_lo) & (ppm_f <= win_hi))[0]
    i_lo, i_hi = ((int(_in_crop.min()), int(_in_crop.max())) if _in_crop.size
                  else (0, -1))
    if i_lo >= i_hi:
        i_lo, i_hi = max(0, lo_idx - 15), min(len(ppm_f)-1, hi_idx + 15)

    x = ppm_f[i_lo:i_hi + 1]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    y_peak_max = -np.inf
    all_ys     = []

    _INTEG_COL = '#2e6da4'      # the trace when the study has no groups
    # One mean per group, as the quantification stage draws them; blanks
    # and the global mean are not traces anybody measured. The fill and
    # every overlay sit on the group whose mean stands highest in the
    # integration window -- the trace the annotation window shows.
    _grp = _group_means(all_int, samples_info)
    _in_win = np.zeros(ppm_f.shape, dtype=bool)
    if _mw and _windowed:
        _in_win = (ppm_f >= ppm_lo) & (ppm_f <= ppm_hi)
    else:
        _in_win[min(lo_idx, hi_idx):max(lo_idx, hi_idx) + 1] = True
    _global = (np.asarray(all_int, dtype=float) if all_int.ndim == 1
               else np.mean(all_int, axis=1))
    _shown = result.get('_snr_group')
    if result.get('_fit_curve'):
        # the group where the fit found the most, when the SNR's group holds
        # next to none of it: the card reads this back
        _shown = fit_group(result['_fit_curve'], _shown)
        result['_drawn_group'] = _shown
    _best_full = _best_group_trace(_grp, _in_win, _global, prefer=_shown)
    for _g, _y in _grp:
        _is_shown = _shown is not None and str(_g) == str(_shown)
        ax.plot(x, _y[i_lo:i_hi+1], color=_grp_color(group_colors, _g),
                lw=1.6 if _is_shown else 0.8, alpha=0.95 if _is_shown else 0.6,
                zorder=3, label=_g)
    _y_mean_full = _best_full[i_lo:i_hi+1]
    if not _grp:
        ax.plot(x, _y_mean_full, color=_INTEG_COL, lw=1.5, zorder=3)
    y_peak_max = float(np.max(_y_mean_full))
    all_ys.append(_y_mean_full)

    # Integration window indices: the same ppm membership the integration
    # used (processing.quant_integration._window_mask), not a step-count
    # estimate that can land a point wide on either side.
    _mw = result.get('_multiplet_window')
    if _mw and _windowed:
        _f_lo_ppm, _f_hi_ppm = min(_mw), max(_mw)
        _in_w = np.nonzero((ppm_f >= _f_lo_ppm) & (ppm_f <= _f_hi_ppm))[0]
        if _in_w.size >= 2:
            lo_f, hi_f = int(_in_w.min()), int(_in_w.max())
        else:
            lo_f, hi_f = 0, -1
    else:
        lo_f = max(lo_idx, i_lo)
        hi_f = min(hi_idx, i_hi)

    # This analyte as what gave its number (ADR-103), alone in colour -- of a
    # window it shares, the other components grey and dotted -- on the mean
    # of the group the number was read on. A fit is the fitted
    # signal on the baseline it stood on, a shape the copy of the reference
    # that fits under the trace (ADR-013), a window resolved by shape every
    # component as the fit placed it (ADR-015), an integral the window above
    # the baseline subtracted, on the pieces summed.
    _fg, _ = _best_group(_grp, _in_win, prefer=_shown)
    _rows = list(rows) if rows is not None else []
    if not any(r_ is result for r_ in _rows):
        _rows.append(result)
    _drawn = contributions.draw_analytes(
        ax, _rows, result, ppm_f, _best_full, _fg, float(x.min()), float(x.max()),
        contributions.stable_palette(_rows, ppm_f),
        lambda amps: _group_amplitude(amps, samples_info, _fg), only_selected=True)
    # a fit made on a residual: the residual it is aligned to
    if contributions.method(result) == 'fit':
        draw_fit_residual(ax, result['_fit_curve'], _fg)

    _fit_group = None
    if (contributions.method(result) == 'fit'
            and contributions.fit_curves(result['_fit_curve'], _fg) is not None):
        _fit_group = _fg
        _warn = fit_warnings(result['_fit_curve'])
        _over = fit_over_trace(result['_fit_curve'], _fg, ppm_f, _best_full)
        # the card says the same as the figure: read back by it
        result['_fit_over_trace'] = _over
        if _over is not None and _over > FIT_ABOVE_TRACE:
            _warn.append(_over_trace_text(_over))
        if _warn:
            ax.text(0.98, 0.97, '\n'.join(_warn),
                    transform=ax.transAxes, fontsize=6, ha='right',
                    va='top', color='#b71c1c',
                    bbox=dict(fc='white', alpha=0.8, lw=0))

    if contributions.method(result) == 'integral' and lo_f < hi_f:
        # Who is who: a singlet's line in the software's peak marker (red,
        # dotted) -- a multiplet is its filled shape -- and a foreign peak
        # inside the window in light grey.
        for _lp in contributions.own_line_marks(result):
            ax.axvline(_lp, color='#e63946', lw=0.8, ls=':', alpha=0.6,
                       zorder=4)
        for _fp in result.get('_foreign_inside') or ():
            ax.axvline(_fp, color='#b0b0b0', lw=0.8, ls=':', alpha=0.9,
                       zorder=4)
        # The whole pattern the reference lists for this signal, dashed
        # grey; a reference line with no red line on it was not integrated
        # -- fused into a neighbour at this linewidth, or never placed.
        for _rl in (result.get('_reference_lines')
                    or result.get('_missing_lines') or ()):
            ax.axvline(_rl, color='#9e9e9e', lw=0.8, ls='--', alpha=0.9,
                       zorder=3)

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
            y_r  = _best_full[lo_r:hi_r+1]
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
        except Exception:
            pass

    _crop_legends(ax, _grp, group_colors, _drawn, _shown)

    # Centrar Y: topo do pico a ~85%
    y_base = min((float(np.min(y)) for y in all_ys), default=0.0)
    y_base = min(y_base, 0.0)
    if y_peak_max > y_base:
        span = y_peak_max - y_base
        ax.set_ylim(y_base - span * 0.04, y_base + span / 0.82)
    # A fit made on what the other analytes leave, standing under a trace far
    # taller (a neighbour's peak): the axis is cut on the fit, the traces run
    # off it, and the figure says so.
    _view = (fit_view_top(result['_fit_curve'], _fit_group)
             if _fit_group is not None else None)
    if _view is not None and _view < VIEW_ZOOM_BELOW * ax.get_ylim()[1]:
        ax.set_ylim(min(0.0, y_base) - 0.04 * _view, _view)
        ax.text(0.02, 0.02, 'y axis cut on the fit: the traces run off it',
                transform=ax.transAxes, fontsize=6, va='bottom', color='#555555',
                bbox=dict(fc='white', alpha=0.8, lw=0))

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
    # An edit of the spectrum is part of how the numbers were made.
    _ra = list(getattr(app, 'region_alignments', None) or [])
    if _ra:
        rows.append(['Regional alignment', '; '.join(
            f"{a['region_ppm'][0]:.3f}-{a['region_ppm'][1]:.3f} ppm "
            f"({a.get('mode')}, signal #{a.get('signal_rank')})"
            for a in _ra[:6]) + (f' and {len(_ra) - 6} more' if len(_ra) > 6 else '')])
    # How this run was made: the options the saved results were produced
    # with, so the report can be repeated.
    _run = getattr(qs, '_last_run_settings', None)
    if not _run:
        from gui.project_vars import collect_vars
        _run = collect_vars(qs)
    _yn = lambda k: 'Yes' if _run.get(k) else 'No'
    _baseline = {(False, False): 'None', (True, False): 'Local per window',
                 (False, True): 'Baseline model',
                 (True, True): 'Local + model'}[
        (bool(_run.get('local_baseline_var')),
         bool(_run.get('bl_model_quant_var')))]
    rows[3:3] = [
        ['Baseline',            _baseline],
        ['Cascade',             _yn('cascade_var')],
        ['Joint fit',           _yn('joint_fit_var')],
        ['Resolve windows by shapes', _yn('shape_windows_var')],
        ['Singlet min SNR',     str(_run.get('singlet_snr_var', '—'))],
        ['Max signal disagreement',
         f"{_run.get('max_disagreement_var', '—')} %"],
        ['Blank subtraction',   _yn('blk_sub_var')],
        ['External IS',         _yn('ext_is_var')],
    ]
    # With PQN the groups' mean factors are said, and a gap between them is
    # flagged: the normalisation may be taking a real difference out.
    pqn_note_row = None
    if 'PQN' in str(norm_mode):
        from gui.stages._quant_adapter import sample_groups
        from processing.quant_concentration import (
            pqn_group_divergence_of_rows, pqn_group_note)
        _div = pqn_group_divergence_of_rows(
            getattr(qs, '_quant_results', None),
            sample_groups(si, len(si)))
        if _div:
            rows.insert(2, ['PQN factor by group', ',  '.join(
                f'{g}: {v:.2f}' for g, v in _div['by_group'].items())])
            if _div['diverges']:
                pqn_note_row = 3
                rows.insert(3, ['PQN notice', pqn_group_note(_div)])

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
    if pqn_note_row is not None:
        rows[pqn_note_row][1] = Paragraph(
            rows[pqn_note_row][1],
            ParagraphStyle('pqnnote', fontSize=7.5, leading=9.5,
                           textColor=colors.HexColor('#b71c1c')))
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
def _quality_info(r, theoretical, precision=None):
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

    # The level with the identity precision it was MEASURED to have (the
    # lowest over the validation mixtures, see the validation page) -- not
    # a fraction of three, which the report printed before and which no
    # measurement stood behind.
    ci = min(max(int(conf), 1), 3)
    dots = {3: '●●●', 2: '●●○', 1: '●○○'}[ci]
    col = {3: _GREEN, 2: _YELLOW, 1: _RED}[ci]
    p = (precision or {}).get(ci)
    return (f'CONF {dots}  ≥{100 * p:.0f}%*' if p is not None
            else f'CONF {dots}'), col


# ── Multiplicity, spelled out ────────────────────────────────────────────────
_MULT_NAMES = {
    's': 'Singlet', 'singlet': 'Singlet',
    'd': 'Doublet', 'doublet': 'Doublet',
    't': 'Triplet', 'triplet': 'Triplet',
    'q': 'Quartet', 'quartet': 'Quartet',
    'p': 'Quintet', 'quint': 'Quintet', 'quintet': 'Quintet', 'pentet': 'Quintet',
    'h': 'Sextet', 'sext': 'Sextet', 'sextet': 'Sextet',
    'sept': 'Septet', 'septet': 'Septet', 'hept': 'Septet', 'heptet': 'Septet',
    'dd': 'Doublet of doublets', 'dt': 'Doublet of triplets',
    'td': 'Triplet of doublets', 'tt': 'Triplet of triplets',
    'ddd': 'Doublet of doublets of doublets', 'dq': 'Doublet of quartets',
    'm': 'Multiplet', 'multiplet': 'Multiplet',
    'm*': 'Complex multiplet',
    'br s': 'Broad singlet', 'brs': 'Broad singlet',
}


def _multiplicity_name(mult) -> str:
    key = str(mult or '').lower().strip()
    if not key:
        return '—'
    return _MULT_NAMES.get(key, key)


# ── Methods and their references ─────────────────────────────────────────────
# Every method a row can have been quantified by, with the reference the
# reader needs to judge it. Only the ones a report actually used are listed
# at its end.
METHOD_REFERENCES = {
    'Standard': ('Peak integration',
                 'Trapezoidal integration of the elected signal over its '
                 'window, foot to foot over its detected lines; Lorentzian '
                 'capture correction (Abragam, The Principles of Nuclear '
                 'Magnetism, Oxford 1961, ch. III).'),
    'Shape': ('Shape fit (targeted profiling)',
              'The reference compound\'s measured multiplet profile, broadened '
              'to the study\'s linewidth, fitted to each sample as the largest '
              'copy that fits under the trace; area = amplitude \u00d7 profile '
              'area. Weljie A.M., Newton J., Mercier P., Carlson E., Slupsky '
              'C.M., Targeted Profiling: Quantitative Analysis of 1H NMR '
              'Metabolomics Data, Anal. Chem. 2006, 78, 4430-4442. One-sided '
              'amplitude: Lawson C.L., Hanson R.J., Solving Least Squares '
              'Problems, Prentice-Hall 1974 (NNLS).'),
    'MCR': ('MCR-ALS',
            'Multivariate curve resolution by alternating least squares '
            '(Tauler R., Chemom. Intell. Lab. Syst. 1995, 30, 133-146).'),
    'JG-Deconv': ('Shared-window balance',
                  'Non-negative least squares over the windows two analytes '
                  'share (Lawson & Hanson 1974).'),
    'JointFit': ('Joint fit of overlapping analytes',
                 'Every analyte whose signals overlap, and every candidate '
                 'the annotation proposed there, fitted together by '
                 'non-negative least squares on the reference shapes, with '
                 'free Lorentzian components for lines no reference claims; '
                 'each signal placed locally with its neighbours fitted '
                 'beside it. Hao J., Astle W., De Iorio M., Ebbels T.M.D., '
                 'BATMAN, Bioinformatics 2012, 28, 2088-2090; Tardivel P.J.C. '
                 'et al., ASICS, Metabolomics 2017, 13, 109; Lawson & Hanson '
                 '1974.'),
    'Cascade': ('Residual cascade',
                'An analyte the election left out, identified on what the '
                'quantified analytes leave by its whole reference listing at '
                'one amount, and measured beside the neighbours that reach '
                'into its regions. A logistic model of twelve features of the '
                'fit (signal-to-noise of its tallest signal and of all of them '
                'together, agreement between its signals, how much of its '
                'listing the residual left it, how far its column slid) gives '
                'the identity score printed on the row; it was fitted on '
                'simulated mixtures. Orthogonal matching pursuit: Pati Y.C., '
                'Rezaiifar R., Krishnaprasad P.S., Proc. 27th Asilomar Conf. '
                'Signals, Systems and Computers 1993; targeted profiling: '
                'Weljie et al., Anal. Chem. 2006, 78, 4430-4442.'),
    'Resolved': ('Shared window resolved by shape',
                 'The window is modelled per sample as a non-negative sum of '
                 'the shapes of the analytes the annotation accepted in it, '
                 "each at the study's linewidth on its detected lines; the "
                 "area is what this analyte's own shape accounts for, an "
                 'intruder held to at most twice what its own clean signal '
                 'predicts. Bounded least squares: Lawson & Hanson 1974; '
                 'targeted profiling: Weljie et al., Anal. Chem. 2006, 78, '
                 '4430-4442.'),
}


def _methods_used(r) -> list:
    """The methods the row's NUMBER came from. An amount a fit replaced
    (joint fit, cascade) did not come from integration, and the report does
    not say it did."""
    if r.get('_by_cascade'):
        return ['Cascade']
    if r.get('_joint_fit'):
        return ['JointFit']
    out = []
    if r.get('_integ_mode', '') == 'Shape':
        out.append('Shape')
    else:
        out.append('Standard')
    if r.get('_used_mcr'):
        out.append('MCR')
    if r.get('_used_j_resolve'):
        out.append('JG-Deconv')
    if r.get('_deconv'):
        out.append('Resolved')
    return out


# What every report used, whatever each row was measured by.
PIPELINE_REFERENCES = (
    ('Baseline model (detection)',
     'SNIP: Ryan C.G. et al., Nucl. Instr. Meth. B 1988, 34, 396-402.'),
    ('Peak boundaries',
     'Feet located on a Savitzky-Golay smoothed copy of the trace (half a '
     'linewidth): Savitzky A., Golay M.J.E., Anal. Chem. 1964, 36, '
     '1627-1639. Heights and areas read on the trace itself.'),
    ('Reference database',
     'Biological Magnetic Resonance Data Bank: Ulrich E.L. et al., Nucleic '
     'Acids Res. 2008, 36, D402-D408; Hoch J.C. et al., Nucleic Acids Res. '
     '2023, 51, D368-D376. Positions, couplings and line shapes measured on '
     'the deposited spectra; from v04 checked against every deposited '
     'spectrum of the compound, with the curators’ listing kept beside '
     'each measured value.'),
    ('Line position precision',
     'Posener D.W., J. Magn. Reson. 1974, 14, 121-128.'),
)


def _method_label(r) -> str:
    keys = _methods_used(r)
    label = ' + '.join(METHOD_REFERENCES[k][0] for k in keys)
    if 'Shape' in keys:
        ex = r.get('_shape_explained')
        label += ' \u2014 Weljie et al. 2006'
        if ex is not None:
            label += (f'; explains {100 * float(ex):.0f}% of its region'
                      + ('' if float(ex) >= 0.9 else ' (upper bound)'))
    if 'Resolved' in keys:
        d = r['_deconv']
        others = [c['label'] for c in d.get('components', ())
                  if c.get('met_id') != r.get('hmdb_id')]
        rem = list((d.get('remainder') or {}).values())
        label += f"; with {', '.join(others)}"
        if rem:
            label += f'; remainder {100 * float(np.median(rem)):.0f}%'
    if r.get('_local_baseline'):
        label += '; on the local baseline'
    return label


def _references_section(results, styles) -> list:
    """A 'Methods and references' block for the methods this report used."""
    used = []
    for r in results:
        for k in _methods_used(r):
            if k not in used:
                used.append(k)
    if not used:
        return []
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.units import cm as _cm
    out = [Spacer(1, 0.3 * _cm),
           Paragraph('Methods and references', styles['h2'])]
    for title, text in PIPELINE_REFERENCES:
        out.append(Paragraph(f'<b>{title}.</b> {text}', styles['small']))
    for k in used:
        title, text = METHOD_REFERENCES[k]
        out.append(Paragraph(f'<b>{title}.</b> {text}', styles['small']))
    return out


# ── Bloco de 1 analito ───────────────────────────────────────────────────────
def _quant_quality_line(r, ppm, all_int, samples_info) -> str:
    """The quantification quality of each group's number
    (``processing.quant_quality``), coloured, with the first reason for a
    group that is not Good -- the card's, on the report."""
    from processing import quant_quality as qq
    colour = {qq.GOOD: '#1a7a1a', qq.WARN: '#856404', qq.POOR: '#9a0000'}
    ppm_f = np.asarray(ppm, dtype=float)
    parts = []
    for g, trace in _group_means(all_int, samples_info):
        q = qq.assess(r, g, ppm=ppm_f, trace=trace)
        if q['level'] is None:
            continue
        why = f' ({q["reasons"][0]})' if q['level'] != qq.GOOD and q['reasons'] else ''
        parts.append(f'<font color="{colour[q["level"]]}"><b>{g}: {q["level"]}</b></font>{why}')
    return '  |  '.join(parts)


def quality_order(results, ppm, all_int, samples_info) -> list:
    """The analytes as the report lists them: by quantification quality,
    the worst of their groups' tiers (``processing.quant_quality``), Good
    first; within a tier by concentration, highest first."""
    from processing import quant_quality as qq
    ppm_f = np.asarray(ppm, dtype=float)
    means = _group_means(all_int, samples_info)

    def conc(r):
        c = r.get('conc_mm', 0.0)
        return float(c) if c is not None and np.isfinite(c) else 0.0

    def tier(r):
        return qq.worst(qq.assess(r, g, ppm=ppm_f, trace=y)['level'] for g, y in means)

    return sorted(results, key=lambda r: (qq.rank(tier(r)), -conc(r)))


def _analyte_block(r, ppm, all_int, samples_info, group_colors,
                   theoretical, styles, avail_w,
                   crop_w_cm, crop_h_cm, unit='mM', precision=None, rows=None):
    crop_fig, lorentz_r2 = _peak_crop_fig(
        ppm, all_int, r, samples_info, group_colors,
        width_in=crop_w_cm * 0.55, height_in=crop_h_cm * 0.55, rows=rows)
    crop_img = _fig_to_rl_image(crop_fig, crop_w_cm, crop_h_cm, dpi=130)

    conc   = r.get('conc_mm', float('nan'))
    conc_s = f'{conc:.4f} {unit}' if np.isfinite(conc) else 'N/A'
    snr    = r.get('snr', float('nan'))
    snr_s  = f'{snr:.1f}' if snr and np.isfinite(snr) else '—'
    # Which group the figure draws and the SNR was read on, and the shape
    # agreement there -- so the picture and the numbers are one trace.
    _sg = r.get('_snr_group')
    if _sg is not None:
        _cos = (r.get('_cosine_by_group') or {}).get(_sg)
        snr_s += f'  (group {_sg}' + (f', cosine {_cos:.2f}' if _cos is not None else '') + ')'
    _dg = r.get('_drawn_group')
    if _dg is not None and _sg is None:
        snr_s += f'; figure drawn on group {_dg}, where the fit found the most'
    elif _dg is not None and str(_dg) != str(_sg):
        snr_s += (f'; figure drawn on group {_dg}, where the fit found the '
                  f'most (the fit holds next to none in {_sg})')

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

    qual_label, qual_color_rl = _quality_info(r, theoretical, precision)
    qhex = ('#2e7d32' if qual_color_rl == _GREEN else
            '#f9a825' if qual_color_rl == _YELLOW else '#c62828')

    def P(txt, bold=False, uni=False):
        if uni:
            st = styles['cell_uni_b'] if bold else styles['cell_uni']
        else:
            st = styles['cell_b'] if bold else styles['cell']
        return Paragraph(str(txt).replace('\n', '<br/>'), st)

    # What the number was made of: the method, the multiplicity the signal
    # was matched as, and the protons it was divided by.
    _nh = float(r.get('_nh', 1) or 1)
    integ_label = (f"{r.get('_integ_mode', '—')}  |  "
                   f"{_multiplicity_name(r.get('_multiplicity'))}  |  "
                   f"nH={_nh:.0f}" if float(_nh).is_integer()
                   else f"{r.get('_integ_mode', '—')}  |  "
                   f"{_multiplicity_name(r.get('_multiplicity'))}  |  "
                   f"nH={_nh:.2f}")
    _fitc = r.get('_fit_curve')
    if _fitc and (r.get('_joint_fit') or r.get('_by_cascade')):
        # The number is the fit's: the cell says what was fitted, with
        # whom, and in how many samples it was found -- not the window
        # the fit replaced.
        _jf = r.get('_joint_fit') or {}
        integ_label = ('Joint fit' if r.get('_joint_fit') else 'Cascade fit')             + f"  |  {_multiplicity_name(r.get('_multiplicity'))}  |  "             + (f"nH={_nh:.0f}" if float(_nh).is_integer() else f"nH={_nh:.2f}")
        _focus_nh = _fitc.get('focus_nh')
        if (not r.get('_joint_fit') and _focus_nh
                and abs(float(_focus_nh) - float(_nh)) > 1e-6):
            # the multiplicity and the curve are the focus signal's; the
            # row's nH is the whole listing's: said, so the card and the
            # figure describe the same thing
            _fl = f"{float(_focus_nh):.0f}" if float(_focus_nh).is_integer() \
                else f"{float(_focus_nh):.1f}"
            integ_label = (integ_label.rsplit('  |  nH=', 1)[0]
                           + f"  |  focus signal {_fl} H of {_nh:.0f} H listed")
        _casc_score = (r.get('_cascade') or {}).get('score')
        if _casc_score is not None:
            integ_label += f"  |  identity score {100 * float(_casc_score):.0f}%"
        if _jf.get('group'):
            integ_label += (f"  |  with {len(_jf['group']) - 1} analyte(s) and "
                            f"{int(_jf.get('free') or 0)} unassigned peak(s)")
        if _jf.get('residual') is not None and np.isfinite(float(_jf['residual'])):
            integ_label += f"  |  {100 * float(_jf['residual']):.0f}% of the region unexplained"
        _n = [f'{g}: {a} of {b}' for g, (a, b) in
              (_fitc.get('n_by_group') or {}).items() if b and a < b]
        if _n:
            integ_label += '  |  found in ' + ', '.join(_n) + ' samples'
        for _w in fit_warnings(_fitc):
            integ_label += f"  |  {_w}"
        _over = r.get('_fit_over_trace')
        if _over is not None and _over > FIT_ABOVE_TRACE:
            integ_label += f"  |  {_over_trace_text(_over)}"
    # How many of the pattern's lines the number stands on: a triplet
    # measured on two of its lines says so beside its name.
    _n_own = len(r.get('_line_ppms') or ())
    _n_miss = len(r.get('_missing_lines') or ())
    if _n_miss:
        integ_label += f"  |  {_n_own} of {_n_own + _n_miss} lines"
    # What separates the shaded area from the number: the tails the window
    # left outside (capture), a neighbour's skirt removed, and windows that
    # followed each sample's drift.
    _cap = None if _fitc else r.get('_captured_fraction')
    if _cap is not None and np.isfinite(float(_cap)) and 0 < float(_cap) < 1:
        integ_label += f"  |  window holds {100 * float(_cap):.0f}% of the lineshape, corrected"
    _nbs = None if _fitc else r.get('_neighbour_share')
    if _nbs and np.isfinite(float(_nbs)) and float(_nbs) > 0:
        integ_label += f"  |  {100 * float(_nbs):.1f}% neighbour skirt removed"
    if r.get('_peak_align_shifts'):
        integ_label += "  |  window follows each sample's drift"
    if np.isfinite(lorentz_r2):
        integ_label += f"  |  R²={lorentz_r2:.4f}"

    rows = [
        [P(r['metabolite'], bold=True),
         P(f"ppm {r.get('ppm', 0):.4f}  |  HMDB {r.get('hmdb_id', '—')}", bold=True)],
        [P('Integration'),      P(integ_label)],
        [P('Annotation score'), P(f"{r.get('_ann_score', 0.0):.3f}")],
        [P('SNR'),              P(snr_s)],
        [P('Method'),
         P(_method_label(r))],
    ]
    for cl in conc_lines:
        rows.append([P('Concentration'), P(cl)])
    for cv in cv_lines:
        rows.append([P('CV%'), P(cv)])
    q_line = _quant_quality_line(r, ppm, all_int, samples_info)
    if q_line:
        rows.append([P('Quant. quality'), P(q_line, uni=True)])
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
def coverage_text(cov) -> str:
    """How much of the spectrum the run resolved, for the report's first
    page (processing.spectrum_coverage, ADR-065): the share of the detected
    peaks the final model explains and the share of the signal area under
    it. Said as not measured when the run kept no coverage."""
    try:
        peaks = int(cov['peaks'])
        explained = int(cov['explained'])
        area = float(cov['area_fraction'])
    except (TypeError, KeyError, ValueError):
        return ('Engine coverage: not measured -- run the quantification '
                'with the cascade on.')
    if peaks <= 0:
        return 'Engine coverage: no detected peak above the noise to explain.'
    share = int(round(100 * float(cov.get('explained_share', 0.5))))
    snr = float(cov.get('min_snr', 5.0))
    text = (f'Engine coverage: <b>{100 * explained / peaks:.0f}%</b> of the '
            f'detected peaks ({explained} of {peaks}, SNR &ge; {snr:g})')
    if np.isfinite(area):
        text += f' and <b>{100 * area:.0f}%</b> of the signal area'
    text += (f' are explained by the final model (a peak counts when the '
             f'model reaches {share}% of its height; the internal standard, '
             f'its 13C satellites and the excluded regions are left out).')
    n_orph = int(cov.get('n_orphans', peaks - explained) or 0)
    if n_orph:
        text += (f' The {n_orph} unexplained peak(s) may be compounds outside '
                 f'the library, signals away from their reference position or '
                 f'shapes the fit misplaced.')
    return text


def project_peak_count(quant_stage, app):
    """The median number of peaks a sample of this project shows on its own,
    or None: what the report matches the validation mixtures against.

    Not the length of the merged peak list. A line that moves between
    samples is kept there once per position, which in real extracts made
    the list 1.1-1.7 times longer than any sample and picked a mixture far
    more crowded than the project (processing.peak_count, ADR-089). None
    sends the report to the most crowded mixture and says the count is
    unknown, rather than quoting the merged list as if it were the count."""
    stage = getattr(quant_stage, 'peak_stage', None)
    ppm = getattr(app, 'ppm', None)
    if stage is None or ppm is None or not getattr(stage, 'peaks', None):
        return None
    try:
        data = app.corrected_intensities()
    except Exception:
        data = None
    if data is None:
        data = getattr(app, 'all_intensities', None)
    if data is None:
        return None
    # the regions detection applied, else the field a loaded project's
    # detection was given (as _quant_computation reads them)
    from processing.peak_count import per_sample_peak_count
    from processing.spectral_regions import parse_regions
    excluded = getattr(stage, '_excluded_regions', None)
    if excluded is None:
        try:
            excluded = parse_regions(stage.excluded_regions_var.get())
        except (ValueError, AttributeError):
            excluded = []
    try:
        settings = {'min_snr': float(stage.min_snr_var.get()),
                    'min_distance': float(stage.min_width_var.get()),
                    'detect_shoulders': bool(stage.detect_shoulders_var.get()),
                    'shoulder_curvature_k': float(stage.shoulder_k_var.get())}
    except Exception:   # a Tk variable left blank or not a number
        return None
    return per_sample_peak_count(ppm, data, excluded_regions=excluded,
                                 samples_info=getattr(app, 'samples_info', None),
                                 **settings)


def generate_pdf(output_path: str, quant_stage, app,
                 analytes_per_page: int = 5,
                 annotation_stage=None):
    ppm     = getattr(app, 'ppm', None)
    # The traces quantification integrated -- the baseline model applied,
    # alignment already in all_intensities -- not the traces as loaded.
    try:
        all_int = app.corrected_intensities()
    except Exception:
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

    results_sorted = quality_order(results, ppm, all_int, si)

    theoretical = dict(getattr(quant_stage, '_validation_theoretical', {}) or {})
    styles  = _make_styles()
    # The field the analysis ran at -- the one the analyst set for the
    # annotation, else the one the data declared -- picks the validation
    # the report quotes.
    project_mhz = None
    try:
        project_mhz = float(annotation_stage.spectrometer_var.get())
    except Exception:
        project_mhz = getattr(app, 'spectrometer_mhz', None)
    from data.quality_reference import (
        for_field, level_precision, tube_for_peaks)
    _vfield, _ventry, _vexact = for_field(project_mhz)
    # The mixture as crowded as this project, by the peaks detection found:
    # the number of compounds in a real sample is unknown, its peaks are not.
    n_peaks = project_peak_count(quant_stage, app)
    _vtube, _vbeyond = tube_for_peaks(_ventry, n_peaks)
    precision = {lv: level_precision(_ventry, lv, _vtube) for lv in (1, 2, 3)}
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

    # The resolution plot (RP-Method) of each group; without a run to read
    # it from, the group means, and the reason said.
    try:
        panels = resolution_panels(quant_stage, app, annotation_stage, project_mhz)
        _rp_error = None
    except Exception as exc:
        panels, _rp_error = [], exc
    if panels:
        story.append(Paragraph('Resolution Plot (RP-Method) — Spectrum vs. Resolved, '
                               'by Group', styles['h2']))
        for _g, _res in panels:
            story.append(KeepTogether([
                _fig_to_rl_image(_resolution_fig(_g, _res), width_cm=avail_w / cm,
                                 height_cm=6.2, dpi=130),
                Paragraph(resolved_text(_g, _res), styles['small']),
                Spacer(1, 0.15 * cm)]))
        story.append(Paragraph(
            'Above the axis the group mean spectrum (black) and the residual (red): the '
            'spectrum less the baselines the fits stood on, less everything resolved. '
            'Below, mirrored, each analyte as the curve that gave its number (hatched: an '
            'integral, the data it summed; dotted: another signal of a confirmed analyte, '
            'as its own number predicts it). Resolved: the part of the spectrum area '
            'above those baselines the curves account for, the internal standard, the '
            'reference and the excluded regions left out. Missing and over-claim: '
            'residual regions of at least the lone-singlet SNR, down a signal nothing '
            'explains, up a fit above the spectrum.', styles['small']))
    else:
        story.append(Paragraph('Spectral Overview — All Samples by Group', styles['h2']))
        wf_fig = _stacked_spectrum_fig(ppm, all_int, si, gc, width_in=13, height_in=4.5,
                                       ppm_range=(-0.5, 4.0))
        story.append(_fig_to_rl_image(wf_fig, width_cm=avail_w / cm,
                                       height_cm=6.5, dpi=130))
        from xml.sax.saxutils import escape
        story.append(Paragraph('Resolution plot not drawn: '
                               + escape(str(_rp_error or 'no group to draw')),
                               styles['small']))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f'Quantified metabolites: <b>{len(results_sorted)}</b>  |  '
        f'Ordered by quantification quality (Good, Warn, Poor), then by '
        f'concentration (decreasing)',
        styles['small']))
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(coverage_text(
        (getattr(quant_stage, '_quant_diagnostics', None) or {}).get('coverage')),
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
            crop_w_cm=crop_w_cm, crop_h_cm=crop_h_cm, unit=unit,
            precision=precision, rows=results)
        spacer = Spacer(1, 0.15 * cm)
        story.append(KeepTogether([block, spacer]))

    if _ventry is not None:
        from gui.widgets.pdf_validation import precision_footnote
        story.append(Paragraph(
            precision_footnote(_vfield, _ventry, _vtube, _vbeyond, n_peaks),
            styles['small']))
    from gui.widgets.pdf_validation import validation_section
    _val = validation_section(project_mhz, styles, avail_w, n_peaks=n_peaks)
    if _val:
        story.append(PageBreak())
        story.extend(_val)
    story.extend(_references_section(results_sorted, styles))
    doc.build(story)
