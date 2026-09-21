# gui/stages/_quant_visualization.py
"""
_QuantVisMixin — visualization methods for QuantificationStage.

Extracted from quantification_stage.py to isolate:
  • Tree selection / tooltip logic
  • Detail plot (_draw_detail): spectrum overlay, bar chart, quality card
"""
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from ._quant_helpers import (
    _fit_lorentz_mean, _fit_lorentz_mean_with_bl, _fit_lorentz_sample,
    _lorentz_amplitude, _lorentz_shape,
    _linear_baseline,
)


class _QuantVisMixin:
    """Mixin — mixed into QuantificationStage.  Uses self.* freely."""

    # ── Tree selection ────────────────────────────────────────────────────────

    def _on_tree_select(self, event=None):
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        r = next((x for x in self._quant_results
                  if x.get('hmdb_id', x['metabolite']) == iid), None)
        if r:
            try:
                self._draw_detail(r)
            except Exception as _e:
                import traceback, logging
                logging.getLogger(__name__).error(traceback.format_exc())
                print(f'[_draw_detail error] {_e}')
                traceback.print_exc()
            # Live-update QuantPeaksWindow if open
            qpw = getattr(self, '_quant_peaks_win', None)
            if qpw is not None:
                try:
                    if qpw.win.winfo_exists():
                        qpw.load_result(r)
                except Exception:
                    pass
            # Navigate PeakListWindow to the peak inside the integrated multiplet
            # window (matches Quant Peaks), not the bare competition anchor.
            peak_rank = (r.get('_display_peak_rank')
                         or (r.get('_peak') or {}).get('rank')
                         or r.get('_peak_rank'))
            if peak_rank is not None:
                try:
                    pl_win = getattr(self.app, '_peak_list_win', None)
                    if pl_win is not None and hasattr(pl_win, 'scroll_to_rank'):
                        pl_win.scroll_to_rank(peak_rank)
                except Exception:
                    pass

    def _on_tree_motion(self, event):
        """Show tooltip with quality reasons when hovering the Quality column."""
        tree = self._tree
        iid = tree.identify_row(event.y)
        col = tree.identify_column(event.x)
        col_name = ''
        if col and col.startswith('#'):
            try:
                col_idx = int(col[1:]) - 1
                all_cols = tree['columns']
                col_name = all_cols[col_idx] if 0 <= col_idx < len(all_cols) else ''
            except (ValueError, IndexError):
                pass
        if iid and col_name == 'status':
            if iid == self._tooltip_iid:
                return
            r = next((x for x in self._quant_results
                      if x.get('hmdb_id', x.get('metabolite')) == iid), None)
            if r:
                reasons = r.get('_status_reasons', [])
                text = '\n'.join(f'• {s}' for s in reasons) if reasons else '—'
                self._show_tooltip(event, text)
                self._tooltip_iid = iid
                return
        self._hide_tooltip()

    # ── Tree tooltip ──────────────────────────────────────────────────────────

    def _show_tooltip(self, event, text):
        self._hide_tooltip()
        x = self._tree.winfo_rootx() + event.x + 12
        y = self._tree.winfo_rooty() + event.y + 16
        self._tooltip_win = tw = tk.Toplevel(self._tree)
        tw.wm_overrideredirect(True)
        tk.Label(tw, text=text, justify=tk.LEFT,
                 background='#ffffcc', relief=tk.SOLID, borderwidth=1,
                 font=('Arial', 8), padx=6, pady=4).pack()
        self._clamp_tooltip(tw, x, y)

    def _hide_tooltip(self, event=None):
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None
        self._tooltip_iid = None

    # ── Card (matplotlib) tooltip ─────────────────────────────────────────────

    def _on_card_motion(self, event):
        if event.inaxes is not self._ax_card:
            self._hide_card_tooltip()
            return
        if event.x is None or event.y is None:
            return
        try:
            x_f, y_f = self._ax_card.transAxes.inverted().transform(
                (event.x, event.y))
        except Exception:
            return
        tip = None
        for (y_top, y_bot), text in self._card_row_tips.items():
            if y_bot <= y_f <= y_top:
                tip = text
                break
        if tip:
            self._show_card_tooltip(event, tip)
        else:
            self._hide_card_tooltip()

    def _clamp_tooltip(self, tw, x, y):
        """Clamp tooltip position to stay inside the app root window."""
        tw.update_idletasks()
        tw_w, tw_h = tw.winfo_width(), tw.winfo_height()
        root = self.app.root
        win_x, win_y = root.winfo_rootx(), root.winfo_rooty()
        win_w, win_h = root.winfo_width(), root.winfo_height()
        x = min(x, win_x + win_w - tw_w - 4)
        y = min(y, win_y + win_h - tw_h - 4)
        x = max(x, win_x + 4)
        y = max(y, win_y + 4)
        tw.wm_geometry(f'+{x}+{y}')

    def _show_card_tooltip(self, event, text):
        cw = self._cv_card.get_tk_widget()
        x = cw.winfo_rootx() + int(event.x) + 12
        y = cw.winfo_rooty() + (cw.winfo_height() - int(event.y)) + 4
        if self._card_tooltip_win and self._card_tooltip_win.winfo_exists():
            self._clamp_tooltip(self._card_tooltip_win, x, y)
            return
        self._card_tooltip_win = tw = tk.Toplevel(cw)
        tw.wm_overrideredirect(True)
        tk.Label(tw, text=text, justify=tk.LEFT,
                 background='#ffffcc', relief=tk.SOLID, borderwidth=1,
                 font=('Arial', 8), padx=6, pady=4).pack()
        self._clamp_tooltip(tw, x, y)

    def _hide_card_tooltip(self, event=None):
        if self._card_tooltip_win and self._card_tooltip_win.winfo_exists():
            self._card_tooltip_win.destroy()
        self._card_tooltip_win = None

    # ── Detail plot ───────────────────────────────────────────────────────────

    def _draw_detail(self, r):
        ppm_arr = self.app.ppm
        # Draw the SAME (baseline-corrected) matrix the integration used, so the
        # shaded area matches the reported value. Falls back to the raw spectrum
        # when a quantification has not run yet.
        # What quantification integrated: the project's baseline model
        # applied, alignment already in ``all_intensities``. Drawing the
        # traces as loaded put a window over one spectrum and a number from
        # another.
        try:
            all_int = self.app.corrected_intensities()
        except Exception:
            all_int = None
        if all_int is None:
            all_int = self.app.all_intensities
        si      = self.app.samples_info or []

        groups = {}
        n_s = all_int.shape[1] if (all_int is not None and all_int.ndim == 2) else 0
        for i, s in enumerate(si):
            lbl = s.get('label', 'Unknown')
            if str(s.get('type', lbl)).upper() != 'BLK':
                groups.setdefault(lbl, []).append(i)
        if not groups and n_s:
            groups = {'All': list(range(n_s))}

        _app_colors = getattr(self.app, 'group_colors', {})
        cmap = plt.cm.get_cmap('tab10', max(len(groups), 1))
        g_colors = {g: (_app_colors.get(g) or cmap(i))
                    for i, g in enumerate(groups)}

        lipid_regions = r.get('_lipid_regions') or []
        lipid_window = r.get('_lipid_window')
        if lipid_window:
            p = None
        else:
            p = r.get('_peak') or self._find_peak(r.get('ppm', 0), tol=0.05)
        integ_mode = r.get('_integ_mode', 'Area')
        lorentz_p  = r.get('_lorentz')

        # ── Spectrum overlay ─────────────────────────────────────────────
        ax1 = self._ax_spec
        ax1.clear()
        ax1.set_facecolor('#fafafa')
        align_shifts = r.get('_peak_align_shifts')   # list[float] per sample or None

        if ppm_arr is not None and all_int is not None and p:
            lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
            hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))

            # In Multiplet mode use the EXACT integration window. Match it via the
            # same ppm-membership mask the integrator uses (_sm_mult), not a linear
            # index approximation — int((ppm_max-mw)/step) truncates and can place
            # the shaded fill over a slightly different/wider range than what was
            # integrated, painting baseline outside the signal.
            _mw = r.get('_multiplet_window')
            if _mw and integ_mode == 'Multiplet' and _mw[0] is not None and _mw[1] is not None:
                _mw_lo, _mw_hi = min(_mw), max(_mw)
                _idx_mw = np.where((ppm_arr >= _mw_lo) & (ppm_arr <= _mw_hi))[0]
                if _idx_mw.size >= 2:
                    lo_i = int(_idx_mw.min())
                    hi_i = int(_idx_mw.max())

            pad  = max(8, (hi_i - lo_i))
            lo_e = max(0, lo_i - pad)
            hi_e = min(len(ppm_arr) - 1, hi_i + pad)
            x_view = ppm_arr[lo_e:hi_e + 1]
            x_reg  = ppm_arr[lo_i:hi_i + 1]

            # J-Resolve fractional deconvolution: fraction of this peak
            # attributed to this metabolite by NNLS (0 < frac ≤ 1).
            # When < 1, show raw in grey background + scaled contribution in colour.
            _jr_frac = r.get('_j_resolve_frac')
            _show_deconv = (_jr_frac is not None
                            and 0.0 < _jr_frac < 1.0
                            and r.get('_used_j_resolve'))

            if _show_deconv:
                # Grey background = raw full peak (context for the reader)
                _y_raw_mean = (np.mean(all_int[lo_e:hi_e + 1, :], axis=1)
                               if all_int.ndim > 1 else all_int[lo_e:hi_e + 1])
                ax1.plot(x_view, _y_raw_mean, color='#aaaaaa', lw=1.0,
                         alpha=0.45, ls='--', zorder=1, label='Raw (shared signal)')

            for g, idxs in groups.items():
                col = g_colors[g]
                ys  = []
                for i in idxs:
                    if i >= n_s:
                        continue
                    y_s = all_int[lo_e:hi_e + 1, i]
                    if align_shifts and i < len(align_shifts) and align_shifts[i] != 0:
                        delta = align_shifts[i]
                        # Interpolate sample onto reference x grid (aligned view)
                        x_src = x_view - delta   # where this sample's data lives
                        sort_idx = np.argsort(x_src)
                        y_aligned = np.interp(
                            np.sort(x_view),
                            x_src[sort_idx], y_s[sort_idx])
                        if x_view[0] > x_view[-1]:   # NMR: decreasing ppm
                            y_aligned = y_aligned[::-1]
                        if _show_deconv:
                            y_aligned = y_aligned * _jr_frac
                        ax1.plot(x_view, y_aligned, color=col, lw=0.6, alpha=0.35)
                        ys.append(y_aligned)
                    else:
                        y_plot = y_s * _jr_frac if _show_deconv else y_s
                        ax1.plot(x_view, y_plot, color=col, lw=0.6, alpha=0.35)
                        ys.append(y_plot)
                if ys:
                    gm = np.mean(ys, axis=0)
                    _shown = r.get('_snr_group') is not None and str(g) == str(r.get('_snr_group'))
                    ax1.plot(x_view, gm, color=col, lw=2.6 if _shown else 1.4,
                             alpha=0.95 if _shown else 0.7,
                             label=(f'{g}  (SNR, cosine read here)' if _shown else g))

                    if integ_mode == 'Lorentz':
                        s_idxs = [i for i in idxs if i < n_s]
                        # Only draw overlay when stored params exist and R² is valid
                        _lor_r2_viz = float((lorentz_p or {}).get('r2', float('nan')))
                        _lorentz_valid = (lorentz_p is not None
                                          and np.isfinite(_lor_r2_viz)
                                          and _lor_r2_viz >= 0.70)
                        if s_idxs and len(x_reg) >= 4 and _lorentz_valid:
                            y_gm_reg = np.mean(all_int[lo_i:hi_i+1, s_idxs], axis=1)

                            # ── Baseline ──────────────────────────────────────
                            if lorentz_p and 'bl_y0' in lorentz_p:
                                _bl_y0    = lorentz_p['bl_y0']
                                _bl_slope = lorentz_p['bl_slope']
                                _bl_x0    = lorentz_p.get('bl_x0', float(x_reg[0]))
                                cen_d     = lorentz_p['cen']
                                wid_d     = lorentz_p['wid']
                                A_l_gm, cen_d, wid_d = _fit_lorentz_mean(x_reg, y_gm_reg)
                            else:
                                A_l_gm, cen_d, wid_d, _bl_y0, _bl_slope = \
                                    _fit_lorentz_mean_with_bl(x_reg, y_gm_reg)
                                _bl_x0 = float(x_reg[0])

                            # Baseline drawn only over the integration region (x_reg),
                            # not extrapolated across x_view — prevents diagonal artefact.
                            y_bl_reg = _bl_y0 + _bl_slope * (x_reg - _bl_x0)
                            if g == list(groups.keys())[0]:
                                ax1.plot(x_reg, y_bl_reg,
                                         color='#ff6666', lw=1.0, ls='--',
                                         alpha=0.55, zorder=2,
                                         label='baseline')

                            max_shift = max(
                                abs(float(x_reg[-1] - x_reg[0])) * 0.15, 0.005)

                            # Fit range clipped to integration window only
                            x_fit    = np.linspace(float(min(x_reg)), float(max(x_reg)), 600)
                            y_bl_fit = _bl_y0 + _bl_slope * (x_fit - _bl_x0)

                            # Per-sample fits (drawn above baseline)
                            for si_idx in s_idxs:
                                y_s_full = all_int[lo_i:hi_i+1, si_idx]
                                res = _fit_lorentz_sample(
                                    x_reg, y_s_full, A_l_gm, cen_d, wid_d,
                                    max_shift_ppm=max_shift)
                                if res is not None:
                                    _, A_s, cen_s, wid_s = res
                                    y_s_fit = (A_s * _lorentz_shape(x_fit, cen_s, wid_s)
                                               + y_bl_fit)
                                    ax1.plot(x_fit, y_s_fit, color=col,
                                             lw=0.8, ls='--', alpha=0.25)

                            # Mean-spectrum fit
                            res_gm = _fit_lorentz_sample(
                                x_reg, y_gm_reg, A_l_gm, cen_d, wid_d,
                                max_shift_ppm=max_shift)
                            if res_gm is not None:
                                _, A_gm, cen_gm, wid_gm = res_gm
                            else:
                                A_gm, cen_gm, wid_gm = A_l_gm, cen_d, wid_d

                            y_gm_fit_pure = A_gm * _lorentz_shape(x_fit, cen_gm, wid_gm)
                            y_gm_fit      = y_gm_fit_pure + y_bl_fit

                            # Shaded integrated area (Lorentz above baseline)
                            ax1.fill_between(x_fit, y_bl_fit, y_gm_fit,
                                             color=col, alpha=0.12, zorder=1)

                            shift_hz = abs(cen_gm - cen_d) * 600
                            ax1.plot(x_fit, y_gm_fit, color=col,
                                     lw=2.5, ls='--', alpha=0.95,
                                     label=(f'{g} fit  γ={wid_gm*1000:.1f} mppm'
                                            f'  Δ={shift_hz:.1f} Hz'))

                            # HWHM markers on the first group only
                            if g == list(groups.keys())[0]:
                                hwhm_col = '#ff8c00'
                                _bl_cen = _bl_y0 + _bl_slope * (cen_gm - _bl_x0)
                                for xv_hw in (cen_gm - wid_gm, cen_gm + wid_gm):
                                    ax1.axvline(xv_hw, color=hwhm_col,
                                                lw=1.0, ls=':', alpha=0.85)
                                ax1.annotate(
                                    f'HWHM {wid_gm*1000:.1f} mδ',
                                    xy=(cen_gm + wid_gm,
                                        _bl_cen + float(A_gm) * 0.50),
                                    xytext=(8, 0), textcoords='offset points',
                                    fontsize=6, color=hwhm_col, va='center')

            mcr_data = r.get('_mcr')
            if r.get('_used_mcr') and mcr_data is not None:
                S_vec = mcr_data['S']
                m_lo  = mcr_data['lo_i']
                m_hi  = mcr_data['hi_i']
                x_mcr = ppm_arr[m_lo:m_hi+1]
                if len(x_mcr) == len(S_vec):
                    y_mean_reg = np.mean(all_int[m_lo:m_hi+1, :], axis=1)
                    scale = (np.max(y_mean_reg) / (np.max(S_vec) + 1e-30))
                    y_mcr = S_vec * scale
                    ax1.plot(x_mcr, y_mcr, color='#e63946', lw=2.0,
                             ls='-', zorder=5, label='MCR-ALS S (scaled)')
                    resid = y_mean_reg - y_mcr
                    ax1.plot(x_mcr, resid, color='#888888', lw=1.0,
                             ls=':', zorder=4, label='Residual')
                    ax1.axhline(0, color='#cccccc', lw=0.6, ls='-')
                    met = mcr_data.get('metrics', {})
                    lof = met.get('lof', float('nan'))
                    r2  = met.get('r2',  float('nan'))
                    ni  = met.get('n_iter', '?')
                    ax1.text(0.02, 0.97,
                             f'LOF={lof:.2f}%  R²={r2:.4f}  iter={ni}',
                             transform=ax1.transAxes, fontsize=7,
                             va='top', color='#333',
                             bbox=dict(fc='white', alpha=0.7, lw=0))

            # The pieces of the window the area was summed over: a foreign
            # peak between two lines is not filled, and the reader sees the
            # gap the number was cut around.
            _segs = r.get('_integ_segments') or []
            _in_seg = None
            if len(_segs) > 1:
                _in_seg = np.zeros(x_reg.shape, dtype=bool)
                for _a, _b in _segs:
                    _in_seg |= (x_reg >= min(_a, _b)) & (x_reg <= max(_a, _b))
            # The area the number came from: between the trace and the local
            # baseline the integration subtracted (the line joining the
            # window's ends, clipped at zero), as the PDF draws it.
            _use_bl = r.get('_local_baseline', True)
            _ord_r = np.argsort(x_reg)
            for g, idxs in groups.items():
                s_idxs = [i for i in idxs if i < n_s]
                if s_idxs:
                    y_g = np.mean(all_int[lo_i:hi_i+1, s_idxs], axis=1)
                    if _show_deconv:
                        y_g = y_g * _jr_frac
                    if _use_bl and x_reg.size >= 2:
                        _b = np.empty_like(y_g)
                        _b[_ord_r] = np.linspace(max(float(y_g[_ord_r][0]), 0.0),
                                                 max(float(y_g[_ord_r][-1]), 0.0),
                                                 x_reg.size)
                    else:
                        _b = np.zeros_like(y_g)
                    ax1.fill_between(x_reg, _b, y_g, where=_in_seg,
                                     alpha=0.14, color=g_colors.get(g, '#888'))

            if _show_deconv:
                ax1.text(0.02, 0.93,
                         f'JG-Deconv: {_jr_frac * 100:.1f}% of signal',
                         transform=ax1.transAxes, fontsize=7,
                         va='top', color='#1565C0',
                         bbox=dict(fc='white', alpha=0.75, lw=0))

            ppm_lo = float(ppm_arr[lo_i])
            ppm_hi = float(ppm_arr[hi_i])
            for xv in (ppm_lo, ppm_hi):
                ax1.axvline(xv, color='#1565C0', lw=1.0, ls='--', alpha=0.6)
            if _in_seg is not None:
                for _a, _b in _segs:
                    for xv in (_a, _b):
                        if ppm_lo < xv < ppm_hi:
                            ax1.axvline(xv, color='#1565C0', lw=0.8, ls=':',
                                        alpha=0.6)
            # Who is who: the signal's own lines in the peak marker the rest
            # of the software uses (red, dotted), a foreign peak between them
            # in light grey. A legacy row without line positions falls back
            # to the apex, which is only the whole signal for a singlet.
            _own_lines = r.get('_line_ppms')
            if _own_lines is None:
                _h0 = (r.get('_hits') or [{}])[0]
                _is_singlet = r.get('_is_singlet')
                if _is_singlet is None:
                    _is_singlet = ((_h0.get('multiplicity') or _h0.get('type')
                                    or '').strip().lower()
                                   in ('s', 'br s', 'brs', 'bs', ''))
                _own_lines = [p['ppm']] if _is_singlet else []
            for _lp in _own_lines:
                ax1.axvline(_lp, color='#e63946', lw=0.9, ls=':', alpha=0.7)
            for _fp in r.get('_foreign_inside') or ():
                ax1.axvline(_fp, color='#b0b0b0', lw=0.9, ls=':', alpha=0.9)
            for _rl in (r.get('_reference_lines')
                        or r.get('_missing_lines') or ()):
                ax1.axvline(_rl, color='#9e9e9e', lw=0.9, ls='--', alpha=0.9)

            v_lo = float(x_view.min())
            v_hi = float(x_view.max())
            ax1.set_xlim(v_hi, v_lo)

            # Clamp Y-axis to the integration window so out-of-window peaks
            # (visible only as context) don't dominate the scale.
            _y_in_win = (all_int[lo_i:hi_i + 1, :]
                         if all_int.ndim > 1 else all_int[lo_i:hi_i + 1])
            if _y_in_win.size > 0:
                _y_top = float(np.max(_y_in_win))
                ax1.set_ylim(bottom=0, top=max(_y_top * 1.25, 1e-3))

        elif ppm_arr is not None and all_int is not None and lipid_window:
            lo_ppm = float(min(lipid_window[0], lipid_window[1]))
            hi_ppm = float(max(lipid_window[0], lipid_window[1]))
            span = max(hi_ppm - lo_ppm, 1e-3)
            pad = max(span * 0.60, 0.03)
            view_lo = lo_ppm - pad
            view_hi = hi_ppm + pad
            m_view = (ppm_arr >= view_lo) & (ppm_arr <= view_hi)
            if np.sum(m_view) >= 3:
                x_view = ppm_arr[m_view]
                for g, idxs in groups.items():
                    col = g_colors[g]
                    ys = [all_int[m_view, i] for i in idxs if i < n_s]
                    for y_s in ys:
                        ax1.plot(x_view, y_s, color=col, lw=0.6, alpha=0.35)
                    if ys:
                        gm = np.mean(ys, axis=0)
                        ax1.plot(x_view, gm, color=col, lw=1.8, alpha=0.95, label=g)

                for lo_rg, hi_rg in lipid_regions:
                    lo_rg = float(min(lo_rg, hi_rg))
                    hi_rg = float(max(lo_rg, hi_rg))
                    ax1.axvspan(lo_rg, hi_rg, color='#1565C0', alpha=0.13, zorder=0)
                    ax1.axvline(lo_rg, color='#1565C0', lw=1.0, ls='--', alpha=0.6)
                    ax1.axvline(hi_rg, color='#1565C0', lw=1.0, ls='--', alpha=0.6)

                ax1.set_xlim(float(x_view.max()), float(x_view.min()))

        ax1.set_xlabel('δ (ppm)', fontsize=8)
        ax1.set_ylabel('Intensity', fontsize=8)
        name  = r['metabolite']
        _mw_t = r.get('_multiplet_window')
        if integ_mode == 'Multiplet' and _mw_t and _mw_t[0] is not None:
            _t_lo = min(float(_mw_t[0]), float(_mw_t[1]))
            _t_hi = max(float(_mw_t[0]), float(_mw_t[1]))
            ppm_s = f"{_t_lo:.4f}–{_t_hi:.4f}"
        elif p:
            ppm_s = f"{p['ppm']:.4f}"
        elif lipid_window:
            ppm_s = f"{min(lipid_window):.4f}-{max(lipid_window):.4f}"
        else:
            ppm_s = '?'
        ax1.set_title(f'{name}  {ppm_s} ppm  [{integ_mode}]',
                      fontsize=9, fontweight='bold')
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.2, ls=':')
        self._cv_spec.draw()

        # ── Bar chart ────────────────────────────────────────────────────
        ax2 = self._ax_bar
        ax2.clear()
        ax2.set_facecolor('#fafafa')
        norm_mode  = self.norm_var.get()
        concs_g    = r.get('_concs_by_group', {})
        g_names    = [g for g in groups if g in concs_g]
        if 'BLK' in concs_g and 'BLK' not in g_names:
            g_names = g_names + ['BLK']
        y_label    = ('Conc (mM)'  if norm_mode in ('IS', 'IS+PQN')
                      else 'Area %' if norm_mode == 'Total'
                      else 'Area (a.u.)')
        if g_names:
            means, sds, colors = [], [], []
            for g in g_names:
                vals = [c for c in concs_g[g] if np.isfinite(c) and c > 0]
                means.append(float(np.mean(vals)) if vals else 0)
                # Sample SD (ddof=1) so the ±SD error bar is consistent with the
                # CV% reported in the Groups table (which uses sample SD too).
                sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0)
                colors.append(g_colors.get(g, '#888'))
            x_pos  = np.arange(len(g_names))
            hatches = ['///' if g == 'BLK' else '' for g in g_names]
            for xi, (m, e, c, h) in enumerate(zip(means, sds, colors, hatches)):
                ax2.bar(xi, m, yerr=e if e > 0 else None, color=c, alpha=0.55,
                        capsize=5, error_kw={'linewidth': 1.5},
                        edgecolor='black', linewidth=0.8, hatch=h)
            rng = np.random.default_rng(42)
            for i, g in enumerate(g_names):
                vals = [v for v in concs_g.get(g, []) if np.isfinite(v)]
                if vals:
                    jitter = rng.uniform(-0.18, 0.18, len(vals))
                    ax2.scatter(x_pos[i] + jitter, vals,
                                color=colors[i], s=28, alpha=0.9,
                                edgecolors='black', lw=0.5, zorder=4)
            ax2.set_ylim(bottom=0)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(g_names, fontsize=8, rotation=20, ha='right')
            ax2.set_ylabel(y_label, fontsize=8)
            ax2.set_title('Mean ± SD per group', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No group data', ha='center', va='center',
                     transform=ax2.transAxes, color='gray', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, axis='y', alpha=0.2, ls=':')
        self._cv_bar.draw()

        # ── Quality card (top) ───────────────────────────────────────────
        ax3 = self._ax_card
        ax3.clear()
        ax3.axis('off')
        conc      = r.get('conc_mm', float('nan'))
        stat      = r.get('status', 'good')
        cv_groups = r.get('cv_by_group', {})
        integ_lbl = {'Area': 'Trapz', 'Lorentz': 'Lorentz Fit'}.get(integ_mode, integ_mode)
        if r.get('_used_mcr'):
            integ_lbl = 'MCR-ALS'
        bg = '#f0fff0' if stat == 'good' else '#fffde7' if stat == 'warn' else '#fff0f0'
        ax3.set_facecolor(bg)
        ax3.set_title('Signal Info', fontsize=8, fontweight='bold', pad=3)

        ann_score = r.get('_ann_score', 0.0)
        ann_score_col = ('#1a7a1a' if ann_score >= 0.8
                         else '#856404' if ann_score >= 0.5 else '#9a0000')

        _snr_global = r.get('snr', 0)
        _snr_col = ('#1a7a1a' if _snr_global > 10 else
                    '#856404' if _snr_global > 5 else '#9a0000')

        _mw_c = r.get('_multiplet_window')
        if integ_mode == 'Multiplet' and _mw_c and _mw_c[0] is not None:
            _c_lo = min(float(_mw_c[0]), float(_mw_c[1]))
            _c_hi = max(float(_mw_c[0]), float(_mw_c[1]))
            _ppm_label = 'Region'
            _ppm_value = f"{_c_lo:.4f}–{_c_hi:.4f}"
        else:
            _ppm_label = 'δ (ppm)'
            _ppm_value = f"{p['ppm']:.4f}" if p else '?'
        top_card = [
            ('Metabolite',   r['metabolite'],                                    '#333'),
            (_ppm_label,     _ppm_value,                                         '#333'),
            ('Width',        f"{p.get('width_ppm',0):.4f} ppm" if p else '?', '#333'),
            ('Annot. Score', f"{ann_score:.3f}",                                ann_score_col),
            ('Integration',  integ_lbl,                                          '#555'),
            ('Norm',         norm_mode,                                          '#555'),
            ('Mean SNR',     f"{_snr_global:.1f}",                              _snr_col),
        ]
        if r.get('_used_j_resolve') and r.get('_j_resolve_frac') is not None:
            _jr_f = r['_j_resolve_frac']
            _jr_col = '#1565C0'
            top_card.append(('JG-Deconv', f'{_jr_f * 100:.1f}% of signal', _jr_col))
        if r.get('_used_mcr') and r.get('_mcr'):
            mcr_m   = r['_mcr'].get('metrics', {})
            lof     = mcr_m.get('lof', float('nan'))
            r2      = mcr_m.get('r2',  float('nan'))
            ni      = mcr_m.get('n_iter', '?')
            lof_col = ('#1a7a1a' if (np.isfinite(lof) and lof < 5)
                       else '#856404' if (np.isfinite(lof) and lof < 15)
                       else '#9a0000')
            r2_col  = ('#1a7a1a' if (np.isfinite(r2) and r2 > 0.99)
                       else '#856404' if (np.isfinite(r2) and r2 > 0.95)
                       else '#9a0000')
            top_card.append(('LOF',   f'{lof:.2f}%' if np.isfinite(lof) else 'N/A', lof_col))
            top_card.append(('R²',    f'{r2:.4f}'   if np.isfinite(r2)  else 'N/A', r2_col))
            top_card.append(('Iters', str(ni),                                       '#555'))
        if np.isfinite(conc):
            if norm_mode in ('IS', 'IS+PQN'):
                top_card.append(('Conc (mM)', f'{conc:.4f}', '#1a1a2e'))
            elif norm_mode == 'Total':
                top_card.append(('Area %',    f'{conc:.4f}', '#1a1a2e'))
            else:
                top_card.append(('Area (a.u.)', f'{conc:.3e}', '#555'))

        _CARD_TIPS = {
            'Metabolite':   'Name from the annotation database (HMDB/BMRB).',
            'δ (ppm)':      'Chemical shift of the detected signal centre (ppm).',
            'Region':       'Multiplet integration window [lo–hi] ppm.\n'
                            'Area is integrated across this entire region,\n'
                            'covering all predicted multiplet lines.',
            'Width':        'Signal width at half-maximum (ppm).',
            'Annot. Score': 'Composite annotation score (0–1).\n'
                            'Combines multiplet pattern match + chemical shift\n'
                            'deviation + database confidence.',
            'Integration':  'Method used to compute signal area per sample.',
            'Norm':         'Normalisation applied before concentration calculation.',
            'Mean SNR':     'Mean signal-to-noise ratio across non-BLK samples.\n'
                            'SNR = signal max / noise level estimated at detection.',
            'LOF':          'Lack-of-Fit (MCR-ALS).\n'
                            'LOF = 100 × √(Σ(X−X̂)² / ΣX²)\n'
                            'Measures how well the model reconstructs the data.\n'
                            'Good: < 5%  |  Warn: 5–15%  |  Poor: > 15%',
            'R²':           'Coefficient of determination (MCR-ALS).\n'
                            'R² = 1 − SS_res / SS_tot\n'
                            'Good: > 0.99  |  Warn: 0.95–0.99  |  Poor: < 0.95',
            'Iters':        'Number of MCR-ALS iterations until convergence\n'
                            '(max 200). Fewer iterations = faster convergence.',
            'JG-Deconv':    'Fraction of this shared peak attributed to this metabolite\n'
                            'by NNLS deconvolution in JG-Deconv.\n'
                            'Spectrum and area are scaled by this fraction.',
            'Conc (mM)':    'Mean concentration across non-BLK samples (mM).\n'
                            'C = (A_met / nH_met) / (A_IS / nH_IS) × C_IS × dil',
            'Area %':       'Mean area as % of total spectrum area (non-BLK).',
            'Area (a.u.)':  'Mean raw trapezoid area (arbitrary units, no IS norm).',
        }
        self._card_row_tips = {}
        y = 0.97
        row_h = 0.11
        for key, val, col in top_card:
            ax3.text(0.04, y, f'{key}:', fontsize=7.5,
                     fontweight='bold', color='#555',
                     transform=ax3.transAxes, va='top')
            _display_val = val if key != 'Metabolite' or len(val) <= 28 else val[:25] + '…'
            ax3.text(0.50, y, _display_val, fontsize=7.5, color=col,
                     transform=ax3.transAxes, va='top', clip_on=True)
            tip = _CARD_TIPS.get(key)
            if tip:
                self._card_row_tips[(y, y - row_h)] = tip
            y -= row_h

        # ── Quality card (bottom) — per-group table ───────────────────────
        ax4 = self._ax_card_bot
        ax4.clear()
        ax4.axis('off')
        ax4.set_facecolor(bg)
        ax4.set_title('Groups', fontsize=7.5, fontweight='bold', pad=2)

        _conc_col_hdr = 'Conc (mM)' if norm_mode in ('IS', 'IS+PQN') else ('Area %' if norm_mode == 'Total' else 'Area')
        hdr_cols = ['Group', _conc_col_hdr, 'CV%', 'SNR', 'Quality']
        hdr_x    = [0.01, 0.22, 0.50, 0.65, 0.80]
        for hx, ht in zip(hdr_x, hdr_cols):
            ax4.text(hx, 0.97, ht, fontsize=6.5, fontweight='bold', color='#333',
                     transform=ax4.transAxes, va='top')
        ax4.plot([0, 1], [0.91, 0.91], color='#aaa', lw=0.6,
                 transform=ax4.transAxes, clip_on=False)

        row_y = 0.87
        concs_g = r.get('_concs_by_group', {})
        _app_colors = getattr(self.app, 'group_colors', {})
        blk_si = [i for i, s in enumerate(si)
                  if str(s.get('type', s.get('label', ''))).upper() == 'BLK']
        _global_snr = r.get('snr', 0) or 1
        _p_snr = p
        if _p_snr and ppm_arr is not None and all_int is not None:
            _lo = min(_p_snr.get('start_idx', 0), _p_snr.get('end_idx', 0))
            _hi = max(_p_snr.get('start_idx', 0), _p_snr.get('end_idx', 0))
            _peak_max_global = float(np.max(np.mean(all_int[_lo:_hi+1, :], axis=1))) if n_s else 1
            _noise_est = _peak_max_global / _global_snr if _global_snr > 0 else 1.0
        else:
            _noise_est = 1.0
        all_g = [g for g in groups] + (['BLK'] if 'BLK' in concs_g else [])
        for g in all_g:
            vals = concs_g.get(g, [])
            if not vals:
                continue
            g_mean = float(np.mean(vals))
            g_cv   = cv_groups.get(g, float('nan'))
            g_idxs = groups.get(g) if g != 'BLK' else blk_si
            if (g_idxs and _p_snr and ppm_arr is not None and all_int is not None
                    and _lo < _hi):
                _g_idxs_valid = [i for i in g_idxs if i < n_s]
                if _g_idxs_valid:
                    g_mean_spec = np.mean(all_int[_lo:_hi+1, :][:, _g_idxs_valid], axis=1)
                    g_snr = float(np.max(g_mean_spec)) / _noise_est if _noise_est > 0 else 0
                else:
                    g_snr = 0.0
            else:
                g_snr = r.get('snr', 0)
            g_cv_v = cv_groups.get(g, float('nan'))
            _g_crit = []
            if np.isfinite(g_cv_v):
                _g_crit.append(0 if g_cv_v < 20 else 1 if g_cv_v < 40 else 2)
            if g_snr > 0:
                _g_crit.append(0 if g_snr > 10 else 1 if g_snr > 5 else 2)
            _g_worst = max(_g_crit) if _g_crit else 1
            g_stat = ('Good' if _g_worst == 0 else 'Warn' if _g_worst == 1 else 'Poor')
            g_stat_col = ('#1a7a1a' if g_stat == 'Good'
                          else '#856404' if g_stat == 'Warn' else '#9a0000')
            g_col = _app_colors.get(g, '#333')
            if norm_mode in ('IS', 'IS+PQN'):
                conc_str = f'{g_mean:.4f}'
            elif norm_mode == 'Total':
                conc_str = f'{g_mean:.4f}'
            else:
                conc_str = f'{g_mean:.4e}'
            cv_str = f'{g_cv:.1f}%' if np.isfinite(g_cv) else 'N/A'
            row_vals = [g, conc_str, cv_str, f'{g_snr:.0f}', g_stat]
            row_cols = [g_col, '#1a1a2e', '#555', '#555', g_stat_col]
            for hx, rv, rc in zip(hdr_x, row_vals, row_cols):
                ax4.text(hx, row_y, rv, fontsize=6.5, color=rc,
                         transform=ax4.transAxes, va='top')
            row_y -= 0.13

        self._cv_card.draw()
