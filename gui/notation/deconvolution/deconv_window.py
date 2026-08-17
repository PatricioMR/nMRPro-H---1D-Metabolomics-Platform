# gui/notation/deconvolution/deconv_window.py
"""
Main Deconvolution Tool window.

Left  : DeconvListPanel  — hierarchical Treeview of regions + deconvolved children
Right : DeconvPlotPanel  — spectrum plot + quality card

Keyboard shortcuts:
  Ctrl+1/2/3  → Good / Undefined / Poor on selected child peak
  Ctrl+R      → Reset zoom / view
  Enter       → Deconvolve selected region
"""
import tkinter as tk
from tkinter import ttk, messagebox
from .deconv_list_panel import DeconvListPanel, STATUS_GOOD, STATUS_UNDEFINED, STATUS_POOR
from .deconv_plot_panel import DeconvPlotPanel


def _merge_overlapping_regions(regions):
    """
    Merge regions that share any PPM range.
    Input:  list of (start_ppm, end_ppm)  — start may be > end (NMR convention)
    Output: merged list of (lo, hi) with lo < hi
    """
    if not regions:
        return []

    # Normalise to (lo, hi) with lo < hi
    intervals = [(min(s, e), max(s, e)) for s, e in regions]
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:          # overlaps with last merged interval
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    return merged


class DeconvWindow:
    def __init__(self, parent, controller, notation_stage,
                 title="Deconvolution Tool"):
        self.parent         = parent
        self.controller     = controller
        self.notation_stage = notation_stage

        # Merge overlapping regions only when the controller has NO deconvolved
        # results yet.  Calling set_regions() invokes set_peaks() which clears
        # self.results — never do that once the user has already run deconvolution.
        if not controller.results:
            merged = _merge_overlapping_regions(controller.regions)
            if len(merged) < len(controller.regions):
                controller.set_regions(merged)

        self.window = tk.Toplevel(parent)
        self.window.title(
            f"{title}  —  {len(controller.regions)} region(s)")
        self.window.geometry("1300x720")
        # Removing transient to allow minimize/maximize in Windows
        # grab_set removed — allows main window interaction

        self._create_widgets()
        self._bind_shortcuts()
        self._center()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _create_widgets(self):
        main = ttk.Frame(self.window, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main,
            text=(f"🔍  {len(self.controller.regions)} overlapping region(s)  "
                  "— deconvolve to separate individual signals"),
            font=("Arial", 10, "italic")).pack(pady=2)

        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=4)

        self.list_panel = DeconvListPanel(
            content,
            self.controller,
            self._on_region_selected,
            self._on_apply,
            self._on_save,
            on_apply_all_callback=self._on_apply_all)

        self.plot_panel = DeconvPlotPanel(content, self.controller)
        self.list_panel.plot_panel = self.plot_panel

    def _bind_shortcuts(self):
        w = self.window
        w.bind('<Control-Key-1>', lambda e: self.list_panel.set_child_status(STATUS_GOOD))
        w.bind('<Control-Key-2>', lambda e: self.list_panel.set_child_status(STATUS_UNDEFINED))
        w.bind('<Control-Key-3>', lambda e: self.list_panel.set_child_status(STATUS_POOR))
        w.bind('<Control-r>',     lambda e: self._reset_view())
        w.bind('<Control-R>',     lambda e: self._reset_view())
        w.bind('<Control-t>',     lambda e: self._on_ctrl_t())
        w.bind('<Control-T>',     lambda e: self._on_ctrl_t())

    def _center(self):
        self.window.update_idletasks()
        w  = self.window.winfo_width()
        h  = self.window.winfo_height()
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        self.window.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def select_region(self, region_index, child_index=0):
        """Select a region/child and display it.
        If region_index is not in the tree (e.g. it was skipped because
        get_region_data returned < 3 pts), ask the user whether to create
        a new standalone region for that peak and navigate to it.
        """
        self.list_panel.refresh_list()
        tree  = self.list_panel.tree
        p_iid = f'region_{region_index}'

        if not tree.exists(p_iid):
            # Region not visible — ask user to create a custom-width region
            ctrl = self.controller
            meta = (ctrl._region_meta[region_index]
                    if region_index < len(ctrl._region_meta) else {})
            tgt_ppm = None
            if meta.get('peaks') and child_index < len(meta['peaks']):
                tgt_ppm = meta['peaks'][child_index]['ppm']
            elif meta:
                tgt_ppm = (meta['start'] + meta['end']) / 2

            # ── Dialog with spinbox for window width ──────────────────────
            import tkinter as tk
            from tkinter import ttk
            dlg = tk.Toplevel(self.window)
            dlg.title('Add to Deconvolution Tool')
            dlg.resizable(False, False)
            dlg.transient(self.window)
            dlg.grab_set()

            ppm_txt = f'{tgt_ppm:.4f} ppm' if tgt_ppm else 'unknown ppm'
            ttk.Label(dlg,
                text=f'Signal at {ppm_txt} is not in any overlap region.\n'
                      'Choose the window width (ppm) around this signal\n'
                      'to create a new deconvolution region.',
                justify='left', padding=10).grid(
                row=0, column=0, columnspan=2, sticky='w')

            ttk.Label(dlg, text='Window width (ppm):').grid(
                row=1, column=0, padx=10, pady=6, sticky='w')
            win_var = tk.DoubleVar(value=0.10)
            sp = ttk.Spinbox(dlg, from_=0.02, to=2.0, increment=0.02,
                             textvariable=win_var, width=8, format='%.2f')
            sp.grid(row=1, column=1, padx=10, pady=6)

            confirmed = tk.BooleanVar(value=False)

            def _ok():
                confirmed.set(True)
                dlg.destroy()

            def _cancel():
                dlg.destroy()

            bf = ttk.Frame(dlg)
            bf.grid(row=2, column=0, columnspan=2, pady=8)
            ttk.Button(bf, text='Create Region',
                       command=_ok).pack(side='left', padx=6)
            ttk.Button(bf, text='Cancel',
                       command=_cancel).pack(side='left', padx=6)

            dlg.wait_window()
            if not confirmed.get():
                return

            half = win_var.get() / 2
            center = tgt_ppm if tgt_ppm is not None else \
                (meta['start'] + meta['end']) / 2 if meta else 0
            lo = center - half
            hi = center + half
            region_index, child_index = ctrl.add_region(
                lo, hi, target_ppm=tgt_ppm)
            self.list_panel.refresh_list()
            p_iid = f'region_{region_index}'
            if not tree.exists(p_iid):
                # Still not visible — spectrum has no data here at all
                import tkinter.messagebox as mb
                mb.showwarning(
                    'No spectrum data in this region',
                    f'The spectrum has no data points between\n'
                    f'{lo:.4f} – {hi:.4f} ppm.\n\n'
                    f'This signal may be outside the acquisition window\n'
                    f'or in a zero-filled region. Check the spectral range\n'
                    f'in your Bruker acquisition parameters (SW, O1P).',
                    parent=self.window)
                return

        c_iid  = f'row_{region_index}_{child_index}'
        try:
            tree.item(p_iid, open=True)
            tree.see(p_iid)
            target = c_iid if tree.exists(c_iid) else p_iid
            tree.selection_set(target)
            tree.focus(target)
            tree.see(target)
            self._on_region_selected(
                region_index,
                child_index if tree.exists(c_iid) else None)
        except Exception as e:
            print(f'select_region error: {e}')

    def _on_region_selected(self, region_index, child_index=None):
        self.plot_panel.display_region(region_index, child_index)

    def _on_apply(self, region_index, params):
        """Run deconvolution in a background thread with a real progress window."""
        import threading, copy
        # Snapshot so cancel can restore
        # Note: controller._region_meta is a computed @property — no snapshot needed
        peaks_snapshot  = copy.deepcopy(self.controller.processor.peaks)
        results_snapshot = copy.deepcopy(self.controller.results)

        # ── Progress window ───────────────────────────────────────────────────
        pw = tk.Toplevel(self.window)
        pw.title("Deconvolving…")
        pw.geometry("380x160")
        pw.transient(self.window)
        pw.resizable(False, False)

        ttk.Label(pw, text=f"Region {region_index + 1}",
                  font=("Arial", 10, "bold")).pack(pady=(12, 4))

        self._prog_step = tk.StringVar(value="Starting…")
        ttk.Label(pw, textvariable=self._prog_step,
                  font=("Arial", 8), foreground="#444").pack()

        self._prog_var = tk.DoubleVar(value=0)
        bar = ttk.Progressbar(pw, variable=self._prog_var,
                               maximum=100, length=320, mode="determinate")
        bar.pack(pady=8)

        cancelled = [False]

        def _cancel():
            cancelled[0] = True
            # Restore state before this deconvolution attempt
            self.controller.processor.peaks = peaks_snapshot
            self.controller.results         = results_snapshot
            # _region_meta is a computed @property — no restore needed
            try:
                pw.destroy()
            except Exception:
                pass
            self.window.after(0, self.list_panel.refresh_list)

        ttk.Button(pw, text="Cancel", command=_cancel).pack(pady=4)
        pw.protocol("WM_DELETE_WINDOW", _cancel)

        # ── Progress callback (called from worker thread) ─────────────────────
        def _progress(step, pct):
            if cancelled[0]:
                raise InterruptedError("cancelled")
            try:
                self.window.after(0, lambda s=step, p=pct: (
                    self._prog_step.set(s),
                    self._prog_var.set(p)))
            except Exception:
                pass

        # ── Worker thread ─────────────────────────────────────────────────────
        def _worker():
            params['_progress_cb'] = _progress
            try:
                result = self.controller.deconvolve_peak(region_index, params)
            except InterruptedError:
                result = None
            except Exception as e:
                result = None
                print(f"Deconvolution error: {e}")
            finally:
                params.pop('_progress_cb', None)

            if cancelled[0]:
                return

            def _done():
                try:
                    pw.destroy()
                except Exception:
                    pass
                if result and result.get('peaks'):
                    self.plot_panel.display_region(region_index)
                    self.list_panel.refresh_list()
                    # Auto-expand parent in tree
                    iid = f'region_{region_index}'
                    if self.list_panel.tree.exists(iid):
                        self.list_panel.tree.item(iid, open=True)
                    # Warn if some groups were below SNR and zeroed out
                    low_groups = result.get('below_snr_groups', [])
                    if low_groups:
                        names = ', '.join(f'"{g}"' for g in low_groups)
                        messagebox.showwarning(
                            'Low-SNR Groups Detected',
                            f'The following group(s) had a signal below the SNR '
                            f'threshold in this region:\n\n  {names}\n\n'
                            f'Deconvolution was performed using the remaining '
                            f'group(s). The low-SNR group(s) have been zeroed out '
                            f'in the per-sample results.',
                            parent=self.window)
                else:
                    detail = ""
                    if result:
                        err = result.get('error', '')
                        bv  = result.get('baseline_validation', {})
                        if err:
                            detail = f"\n\nReason: {err}"
                        elif bv and not bv.get('valid', True):
                            detail = (f"\n\nBaseline not stable at region edges "
                                      f"({bv.get('reason','')}).\n"
                                      "Try widening the region or choosing a "
                                      "quieter region boundary.")
                    messagebox.showwarning(
                        "No Signals Found",
                        "No peaks could be separated in this region\n"
                        "with the current parameters.\n\n"
                        "Try lowering Min SNR or increasing Max Peaks."
                        + detail,
                        parent=self.window)

            self.window.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_apply_all(self, pending_indices, params):
        """
        Deconvolve all pending regions sequentially in a background thread.
        Shows a single progress window with per-region progress.
        Cancel restores the processor state to before the run.
        """
        import threading, copy

        # ── Snapshot state before starting ───────────────────────────────────
        # Deep-copy the peaks list so we can restore on cancel
        # Note: controller._region_meta is a computed @property — no snapshot needed
        peaks_snapshot   = copy.deepcopy(self.controller.processor.peaks)
        results_snapshot = copy.deepcopy(self.controller.results)

        n_total = len(pending_indices)

        # ── Progress window ───────────────────────────────────────────────────
        pw = tk.Toplevel(self.window)
        pw.title("Deconvolving All Regions…")
        pw.geometry("420x200")
        pw.transient(self.window)
        pw.resizable(False, False)

        ttk.Label(pw, text=f"Processing {n_total} region(s)",
                  font=("Arial", 10, "bold")).pack(pady=(12, 2))

        self._prog_step = tk.StringVar(value="Starting…")
        ttk.Label(pw, textvariable=self._prog_step,
                  font=("Arial", 8), foreground="#444").pack()

        # Overall bar (regions)
        self._prog_overall = tk.DoubleVar(value=0)
        ttk.Label(pw, text="Overall:", font=("Arial", 8)).pack(
            anchor=tk.W, padx=20)
        ttk.Progressbar(pw, variable=self._prog_overall,
                        maximum=n_total, length=360,
                        mode="determinate").pack(pady=2)

        # Per-region bar (optimisation iterations)
        self._prog_region = tk.DoubleVar(value=0)
        ttk.Label(pw, text="Current region:", font=("Arial", 8)).pack(
            anchor=tk.W, padx=20)
        ttk.Progressbar(pw, variable=self._prog_region,
                        maximum=100, length=360,
                        mode="determinate").pack(pady=2)

        cancelled = [False]

        def _cancel():
            cancelled[0] = True
            # Restore snapshot immediately
            self.controller.processor.peaks = peaks_snapshot
            self.controller.results         = results_snapshot
            # _region_meta is a computed @property — no restore needed
            try:
                pw.destroy()
            except Exception:
                pass
            # Refresh UI to show restored state
            self.window.after(0, self.list_panel.refresh_list)

        ttk.Button(pw, text="Cancel (restore previous state)",
                   command=_cancel).pack(pady=6)
        pw.protocol("WM_DELETE_WINDOW", _cancel)

        # ── Worker ────────────────────────────────────────────────────────────
        def _worker():
            for ri, region_index in enumerate(pending_indices):
                if cancelled[0]:
                    return

                self.window.after(0, lambda ri=ri, region_index=region_index: (
                    self._prog_step.set(
                        f"Region {region_index + 1}  ({ri + 1}/{n_total})"),
                    self._prog_overall.set(ri),
                    self._prog_region.set(0)))

                def _progress(step, pct, ri=ri):
                    if cancelled[0]:
                        raise InterruptedError("cancelled")
                    try:
                        self.window.after(0, lambda p=pct: (
                            self._prog_region.set(p)))
                    except Exception:
                        pass

                p = dict(params)
                p['_progress_cb'] = _progress
                try:
                    self.controller.deconvolve_peak(region_index, p)
                except InterruptedError:
                    _cancel()
                    return
                except Exception as e:
                    print(f"Region {region_index} error: {e}")
                finally:
                    p.pop('_progress_cb', None)

            if cancelled[0]:
                return

            # All done
            def _done():
                self._prog_overall.set(n_total)
                self._prog_step.set("Complete!")
                try:
                    pw.destroy()
                except Exception:
                    pass
                self.plot_panel.display_region(pending_indices[-1])
                self.list_panel.refresh_list()

            self.window.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_save(self):
        has_results = any(res.get('peaks')
                          for res in self.controller.results.values())
        if not has_results:
            messagebox.showwarning('Nothing to Save',
                'No signals deconvolved yet.\nSelect a signal and click Deconvolve.')
            return
        n_deconv  = sum(1 for r in self.controller.results.values() if r.get('peaks'))
        # Re-fit confirmed peaks to correct amplitudes before collecting them.
        # When the user marks some sub-peaks as Poor, the remaining ones carry
        # amplitudes from the original N-component fit. Re-fitting with only the
        # confirmed components gives the correct per-peak intensities.
        for pk_idx, res in self.controller.results.items():
            if res.get('peaks') and any(sp.get('status') == 'Poor'
                                        for sp in res['peaks']):
                self.controller.refit_confirmed_peaks(pk_idx)

        # Exclude sub-peaks marked as Poor — they are noise/artefacts
        sub_peaks = [p for r in self.controller.results.values()
                     for p in r.get('peaks', [])
                     if p.get('status') != 'Poor']
        n_poor = sum(1 for r in self.controller.results.values()
                     for p in r.get('peaks', [])
                     if p.get('status') == 'Poor')
        poor_note = f'\n({n_poor} Poor peak(s) excluded)' if n_poor else ''
        if not messagebox.askyesno('Confirm Save',
                f'Replace {n_deconv} signal(s) with {len(sub_peaks)} sub-signal(s)?'
                f'{poor_note}\n\nContinue?'):
            return
        peak_stage = self.notation_stage

        # PPMs a remover — os picos do controller que foram deconvolvidos
        remove_ppms = []
        for pk_idx, res in self.controller.results.items():
            if not res.get('peaks'): continue
            if pk_idx < len(self.controller.peaks):
                qp = self.controller.peaks[pk_idx]
                remove_ppms.append(float(qp['ppm']))
                for gppm in qp.get('_grouped_from', []):
                    remove_ppms.append(float(gppm))

        def _to_remove(p):
            """True if this peak should be replaced by sub-peaks.
            - Parent peaks whose ppm is in remove_ppms → remove
            - Sub-peaks whose parent_ppm is in remove_ppms → remove
              (old sub-peaks from a previous deconv of the same region)
            - Sub-peaks from OTHER regions → keep
            """
            pppm = float(p['ppm'])
            if p.get('is_sub_peak', False):
                parent = float(p.get('parent_ppm', pppm))
                return any(abs(parent - rp) < 0.0005 for rp in remove_ppms)
            return any(abs(pppm - rp) < 0.0005 for rp in remove_ppms)

        print(f"\n💾 Save: remove_ppms={[round(r,4) for r in remove_ppms]}")
        print(f"   stage.peaks count before: {len(peak_stage.peaks)}")
        print(f"   sub_peaks to add: {len(sub_peaks)}")

        # CRITICAL: modificar a lista IN-PLACE para que todas as referências
        # externas (PeakListWindow.self.peaks, etc.) vejam as alterações.
        # NÃO fazer peak_stage.peaks = kept (nova lista quebra referências).
        original_list = peak_stage.peaks
        kept = [p for p in original_list if not _to_remove(p)]
        kept.extend(sub_peaks)
        kept.sort(key=lambda p: abs(p['ppm']))
        for i, p in enumerate(kept, 1):
            p['rank'] = i

        original_list.clear()
        original_list.extend(kept)
        # peak_stage.peaks JÁ É original_list — não precisa de reatribuição
        # Sincronizar o processor também in-place
        if hasattr(peak_stage, 'processor') and peak_stage.processor:
            peak_stage.processor.peaks = original_list

        print(f"   stage.peaks count after: {len(original_list)}")
        print(f"   ppms after: {[round(p['ppm'],4) for p in original_list[:10]]}")

        # Refreshar PeakListWindow se estiver aberta (a referência é a mesma lista)
        try:
            plw = getattr(peak_stage.app, '_peak_list_win', None)
            if plw is not None and plw.win.winfo_exists():
                plw._populate()
        except Exception as e:
            print(f"   PeakList refresh: {e}")
        # Zero excluded samples
        app = getattr(peak_stage, 'app', None)
        raw = getattr(app, 'all_intensities', None)
        if raw is not None:
            for pk_idx in list(self.controller.results):
                excl = self.controller.get_excluded_samples(pk_idx)
                if not excl or pk_idx >= len(self.controller.peaks): continue
                pk = self.controller.peaks[pk_idx]
                lo = min(pk.get('start_ppm', pk['ppm']), pk.get('end_ppm', pk['ppm']))
                hi = max(pk.get('start_ppm', pk['ppm']), pk.get('end_ppm', pk['ppm']))
                mask = (app.ppm >= lo) & (app.ppm <= hi)
                for si in excl:
                    if si < raw.shape[1]: raw[mask, si] = 0.0
        # Restore mean
        _orig = getattr(self.controller.processor, '_original_mean_intensity', None)
        if _orig is not None:
            self.controller.processor.mean_intensity = _orig
            del self.controller.processor._original_mean_intensity
        # Redraw
        import numpy as np
        pm = getattr(peak_stage.app, 'plot_manager', None)
        if pm:
            try:
                from gui.stages.peak_detection_stage import _no_blk_group_mean
                mean = _no_blk_group_mean(peak_stage.app.all_intensities,
                                          peak_stage.app.samples_info)
            except Exception:
                mean = np.mean(peak_stage.app.all_intensities, axis=1)
            pm.force_redraw_peaks(original_list, mean, peak_stage.app.ppm)
        # Mark all saved results so the list panel shows the "saved" indicator
        for pk_idx in self.controller.results:
            if self.controller.results[pk_idx].get('peaks'):
                self.controller.results[pk_idx]['_saved'] = True

        # Não fechar — utilizador pode continuar a deconvolver
        poor_msg = f'  ({n_poor} Poor excluded)' if n_poor else ''
        messagebox.showinfo('Saved',
            f'Signal list updated: {len(original_list)} signals '
            f'({n_deconv} replaced, {len(sub_peaks)} sub-signals added{poor_msg})',
            parent=self.window)
        self.list_panel.refresh_list()

    def _reset_view(self):
        """Reset the plot zoom to the original window."""
        try:
            self.plot_panel.reset_view()
        except Exception:
            pass