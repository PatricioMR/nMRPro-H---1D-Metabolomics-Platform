# gui/widgets/pathway_window.py
"""
Pathway Analysis Window.

Data source priority:
  1. Local SQLite DB (databases/hmdb_pathways.db) — instant, offline.
     Build once with: python data/build_pathway_db.py
  2. HMDB REST API fallback — for IDs not in local DB (rate-limited).

Descriptive analysis only: pathways sorted by metabolite count.
"""
import os
import re
import json
import time
import sqlite3
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# ── Path to local pathway DB ──────────────────────────────────────────────────

from app_paths import databases_dir
_LOCAL_DB   = Path(databases_dir()) / 'hmdb_pathways.db'
HMDB_URL    = 'https://hmdb.ca/metabolites/{}.json'
SMPDB_URL   = 'https://smpdb.ca/view/{}'
RATE_DELAY  = 0.35
TIMEOUT     = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_hmdb(raw: str):
    m = re.search(r'\d+', raw or '')
    return f'HMDB{int(m.group()):07d}' if m else None


def _is_hmdb(raw: str) -> bool:
    return bool(raw) and str(raw).upper().startswith('HMDB')


_DRUG_KW    = {'action pathway', 'antihistamine', 'pharmacokinetics',
               'agonist', 'antagonist', 'antidepressant', 'antipsychotic',
               'statin', 'antibiotic', 'antifungal', 'antiviral',
               'inhibitor action', 'receptor action', 'drug metabolism',
               # broader: catch "Drug X metabolism", "Drug transport", etc.
               'drug ', ' drug', 'xenobiotic', 'pharmaceutical',
               'pharmacodynamic', 'pharmacological', 'medication',
               'biotransformation', 'bioavailability'}
_DISEASE_KW = {'deficiency', 'syndrome', 'disease', 'disorder',
               'type i', 'type ii', 'type iii', 'xanthinuria',
               'oncometabolite', 'inborn', 'acidemia', 'aciduria',
               'dystrophy', 'tumour', 'tumor', 'cancer', 'carcinoma'}

def _pathway_category(name: str, description: str = '', stored_cat: str = '') -> str:
    # Stored category from DB takes priority (set by builder from API or heuristic)
    if stored_cat in ('drug', 'disease', 'metabolic'):
        return stored_cat
    nd = (name + ' ' + description).lower()
    if any(k in nd for k in _DRUG_KW):
        return 'drug'
    if any(k in nd for k in _DISEASE_KW):
        return 'disease'
    return 'metabolic'


def _local_db_available() -> bool:
    return _LOCAL_DB.exists() and _LOCAL_DB.stat().st_size > 4096


def _migrate_local_db(con: sqlite3.Connection):
    """Add category column if missing, then backfill using keyword heuristic."""
    cols = {r[1] for r in con.execute('PRAGMA table_info(pathways)').fetchall()}
    if 'category' not in cols:
        con.execute("ALTER TABLE pathways ADD COLUMN category TEXT DEFAULT ''")
        con.commit()
    # Backfill rows where category is still empty
    rows = con.execute("SELECT id, name, description FROM pathways WHERE category = '' OR category IS NULL").fetchall()
    if rows:
        _DKW = {'drug', 'action pathway', 'pharmacokinetics', 'agonist',
                'antagonist', 'antidepressant', 'antipsychotic', 'statin',
                'antibiotic', 'antifungal', 'antiviral', 'xenobiotic',
                'pharmaceutical', 'pharmacodynamic', 'pharmacological',
                'medication', 'biotransformation', 'bioavailability',
                'anaesthetic', 'anesthetic', 'analgesic', 'sedative',
                'anticonvulsant', 'antiepileptic', 'antimicrobial',
                'antineoplastic', 'antihypertensive', 'antidiabetic',
                'anticoagulant', 'inhibitor action', 'receptor action'}
        _DISKW = {'deficiency', 'syndrome', 'disease', 'disorder', 'cancer',
                  'carcinoma', 'tumor', 'tumour', 'inborn', 'acidemia',
                  'aciduria', 'dystrophy', 'oncometabolite'}
        updates = []
        for pid, name, desc in rows:
            nd = (name + ' ' + (desc or '')).lower()
            if any(k in nd for k in _DKW):
                cat = 'drug'
            elif any(k in nd for k in _DISKW):
                cat = 'disease'
            else:
                cat = 'metabolic'
            updates.append((cat, pid))
        con.executemany("UPDATE pathways SET category = ? WHERE id = ?", updates)
        con.commit()


def _query_local(hmdb_id: str) -> list[dict] | None:
    """
    Query local DB for pathways of hmdb_id.
    Returns list of {name, smpdb_id, description, category}, or None if ID not in DB.
    """
    if not _local_db_available():
        return None
    try:
        con = sqlite3.connect(str(_LOCAL_DB))
        _migrate_local_db(con)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            'SELECT 1 FROM fetch_log WHERE hmdb_id = ?', (hmdb_id,))
        if cur.fetchone() is None:
            con.close()
            return None   # not in local DB → fall back to API
        # Include category; use '' if column doesn't exist (old DB schema)
        try:
            cur = con.execute(
                '''SELECT p.name, p.smpdb_id, p.description, p.category
                   FROM pathways p
                   JOIN metabolite_pathways mp ON mp.pathway_id = p.id
                   WHERE mp.hmdb_id = ?''',
                (hmdb_id,))
        except Exception:
            cur = con.execute(
                '''SELECT p.name, p.smpdb_id, p.description, '' AS category
                   FROM pathways p
                   JOIN metabolite_pathways mp ON mp.pathway_id = p.id
                   WHERE mp.hmdb_id = ?''',
                (hmdb_id,))
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return None


# ── Main window ───────────────────────────────────────────────────────────────

class PathwayAnalysisWindow:
    """
    Opens a Toplevel showing metabolic pathway membership for all quantified
    metabolites.  Uses the local pathway DB when available; falls back to the
    HMDB REST API for IDs not yet cached.
    """

    def __init__(self, parent_win, quant_results: list, app=None):
        self._quant_results = quant_results
        self._app           = app
        self._pathway_table = {}   # name → {smpdb_id, description, mets:[]}
        self._cancelled     = False
        self._n_done        = 0
        self._iid_map       = {}   # pathway_name → pw treeview iid
        self._met_iid_map   = {}   # met_name → met treeview iid

        self._use_local  = _local_db_available()
        self._n_api_hits = 0       # IDs that had to fall back to API

        self._win = tk.Toplevel(parent_win)
        self._win.title('Pathway Analysis')
        self._win.geometry('960x640')
        self._win.minsize(700, 420)
        self._win.protocol('WM_DELETE_WINDOW', self._on_close)

        self._build()
        self._start_fetch()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Top bar
        top = ttk.Frame(self._win)
        top.pack(fill=tk.X, padx=6, pady=4)

        self._status_var = tk.StringVar(value='Preparing…')
        ttk.Label(top, textvariable=self._status_var,
                  font=('Arial', 9)).pack(side=tk.LEFT)

        src = ('Local DB' if self._use_local else 'HMDB API')
        ttk.Label(top, text=f'[{src}]',
                  font=('Arial', 8), foreground='gray').pack(
                      side=tk.LEFT, padx=6)

        ttk.Label(top, text='Category:',
                  font=('Arial', 8)).pack(side=tk.LEFT, padx=(8, 2))
        self._cat_var = tk.StringVar(value='metabolic')
        cat_cb = ttk.Combobox(top, textvariable=self._cat_var, width=10,
                               state='readonly',
                               values=['metabolic', 'drug', 'disease', 'all'])
        cat_cb.pack(side=tk.LEFT)
        cat_cb.bind('<<ComboboxSelected>>', lambda _: self._refresh_pw_tree())

        self._btn_cancel = ttk.Button(top, text='Cancel',
                                       command=self._cancel)
        self._btn_cancel.pack(side=tk.RIGHT)

        self._btn_network = ttk.Button(top, text='Network Graph',
                                        command=self._show_network,
                                        state='disabled')
        self._btn_network.pack(side=tk.RIGHT, padx=4)

        self._btn_enrichment = ttk.Button(top, text='Enrichment Chart',
                                           command=self._show_enrichment,
                                           state='disabled')
        self._btn_enrichment.pack(side=tk.RIGHT, padx=4)

        # Paned
        paned = tk.PanedWindow(self._win, orient=tk.HORIZONTAL,
                               sashwidth=5, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: metabolite status list
        lf = ttk.LabelFrame(paned, text='Metabolites', padding=4)
        paned.add(lf, minsize=240)

        self._tree_met = ttk.Treeview(lf, columns=('name', 'status'),
                                       show='headings', selectmode='browse')
        self._tree_met.heading('name',   text='Metabolite')
        self._tree_met.heading('status', text='Status')
        self._tree_met.column('name',   width=170, stretch=True)
        self._tree_met.column('status', width=60,  stretch=False, anchor='center')
        sb_l = ttk.Scrollbar(lf, orient=tk.VERTICAL,
                              command=self._tree_met.yview)
        self._tree_met.configure(yscrollcommand=sb_l.set)
        sb_l.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_met.pack(fill=tk.BOTH, expand=True)

        # Right: pathway summary
        rf = ttk.LabelFrame(paned, text='Pathway Summary', padding=4)
        paned.add(rf, minsize=420)

        self._tree_pw = ttk.Treeview(
            rf, columns=('pathway', 'count', 'smpdb'),
            show='headings', selectmode='browse')
        self._tree_pw.heading('pathway', text='Pathway',
                               command=lambda: self._sort_by('pathway'))
        self._tree_pw.heading('count',   text='Count',
                               command=lambda: self._sort_by('count'))
        self._tree_pw.heading('smpdb',   text='SMPDB ID')
        self._tree_pw.column('pathway', width=420, stretch=True)
        self._tree_pw.column('count',   width=55,  stretch=False, anchor='center')
        self._tree_pw.column('smpdb',   width=100, stretch=False)

        sb_r = ttk.Scrollbar(rf, orient=tk.VERTICAL,
                              command=self._tree_pw.yview)
        self._tree_pw.configure(yscrollcommand=sb_r.set)
        sb_r.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_pw.pack(fill=tk.BOTH, expand=True)

        self._tree_pw.bind('<<TreeviewSelect>>', self._on_pw_select)
        self._tree_pw.bind('<Double-1>',          self._on_pw_dclick)

        self._detail_var = tk.StringVar(value='Select a pathway to see metabolites.')
        ttk.Label(rf, textvariable=self._detail_var,
                  wraplength=600, justify=tk.LEFT,
                  font=('Arial', 8), foreground='#444').pack(
                      anchor=tk.W, padx=2, pady=(4, 0))

        self._sort_col = 'count'
        self._sort_rev = True

        # Populate left tree with pending entries
        self._entries = []
        for r in self._quant_results:
            name   = r.get('metabolite', '?')
            raw_id = str(r.get('hmdb_id') or r.get('id') or '')
            if _is_hmdb(raw_id):
                status = '…'
                self._entries.append({'name': name, 'hmdb_id': raw_id})
            else:
                status = '—'
            iid = self._tree_met.insert('', 'end', values=(name, status))
            self._met_iid_map[name] = iid

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _start_fetch(self):
        n_hmdb = len(self._entries)
        n_skip = len(self._quant_results) - n_hmdb
        src    = 'local DB' if self._use_local else 'HMDB API'
        self._status_var.set(
            f'0 / {n_hmdb} queried via {src}  ·  {n_skip} skipped (no HMDB ID)')
        t = threading.Thread(target=self._fetch_worker, daemon=True)
        t.start()

    def _fetch_worker(self):
        for entry in self._entries:
            if self._cancelled:
                break
            hmdb_id = _norm_hmdb(entry['hmdb_id'])
            if not hmdb_id:
                self._win.after(0, self._on_error, entry, 'bad ID')
                continue

            # Try local DB first
            local = _query_local(hmdb_id)
            if local is not None:
                self._win.after(0, self._on_result, entry, local, 'local')
                continue   # no sleep needed for local queries

            # Fallback: HMDB REST API
            status, pathways = self._fetch_api(hmdb_id)
            self._win.after(0, self._on_result if status != 'error'
                            else self._on_error,
                            entry,
                            pathways if status != 'error' else status)
            time.sleep(RATE_DELAY)

        self._win.after(0, self._on_done)

    @staticmethod
    def _fetch_api(hmdb_id: str) -> tuple[str, list[dict]]:
        import urllib.request
        import urllib.error
        url = HMDB_URL.format(hmdb_id)
        req = urllib.request.Request(
            url, headers={'Accept': 'application/json',
                          'User-Agent': 'NMRAnalysis/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            pws = ((data.get('biological_properties') or {})
                   .get('pathways') or [])
            return ('ok' if pws else 'no_pathways'), pws
        except urllib.error.HTTPError as e:
            return ('not_found' if e.code == 404 else 'error'), []
        except Exception:
            return 'error', []

    # ── Callbacks (main thread) ───────────────────────────────────────────────

    def _on_result(self, entry: dict, pathways: list, source: str = 'api'):
        met_name = entry['name']
        status   = '✓' if pathways else '∅'
        self._set_met_status(met_name, status)

        if source == 'api':
            self._n_api_hits += 1

        for pw in pathways:
            pw_name = pw.get('name') or '(unnamed)'
            e = self._pathway_table.setdefault(pw_name, {
                'smpdb_id':    pw.get('smpdb_id', ''),
                'description': pw.get('description', ''),
                'category':    pw.get('category', ''),
                'mets':        [],
            })
            if met_name not in e['mets']:
                e['mets'].append(met_name)

        self._n_done += 1
        self._refresh_pw_tree()
        self._update_status()

    def _on_error(self, entry: dict, exc):
        code = '404' if '404' in str(exc) else ('⏱' if 'time' in str(exc).lower() else '✗')
        self._set_met_status(entry['name'], code)
        self._n_done += 1
        self._update_status()

    def _on_done(self):
        self._btn_cancel.config(text='Close', command=self._on_close)
        if self._pathway_table:
            self._btn_network.config(state='normal')
            self._btn_enrichment.config(state='normal')
        self._update_status(done=True)

    def _set_met_status(self, name: str, status: str):
        iid = self._met_iid_map.get(name)
        if iid:
            try:
                self._tree_met.set(iid, 'status', status)
            except Exception:
                pass

    def _update_status(self, done: bool = False):
        n_total = len(self._entries)
        n_pw    = len(self._pathway_table)
        src     = f'{n_total - self._n_api_hits} local · {self._n_api_hits} API'
        label   = 'Done' if done else 'Querying…'
        self._status_var.set(
            f'{label}  ·  {self._n_done} / {n_total} queried  '
            f'({src})  ·  {n_pw} pathways found')

    # ── Pathway treeview ──────────────────────────────────────────────────────

    def _filtered_pathway_table(self):
        cat = getattr(self, '_cat_var', None)
        cat = cat.get() if cat else 'all'
        if cat == 'all':
            return self._pathway_table
        return {k: v for k, v in self._pathway_table.items()
                if _pathway_category(k,
                                     v.get('description', ''),
                                     v.get('category', '')) == cat}

    def _refresh_pw_tree(self):
        visible = self._filtered_pathway_table()
        rows = sorted(
            visible.items(),
            key=lambda kv: (len(kv[1]['mets']) if self._sort_col == 'count'
                            else kv[0]),
            reverse=self._sort_rev,
        )
        # Hide rows not in current filter
        all_iids = set(self._iid_map.values())
        visible_iids = set()
        existing = set(self._iid_map.keys())
        for pw_name, info in rows:
            count = len(info['mets'])
            smpdb = info.get('smpdb_id', '')
            if pw_name in existing:
                self._tree_pw.item(self._iid_map[pw_name],
                                   values=(pw_name, count, smpdb))
            else:
                iid = self._tree_pw.insert('', 'end',
                                           values=(pw_name, count, smpdb))
                self._iid_map[pw_name] = iid
            visible_iids.add(self._iid_map[pw_name])

        # Detach hidden rows
        for iid in all_iids - visible_iids:
            try:
                self._tree_pw.detach(iid)
            except Exception:
                pass
        # Re-attach visible rows in sorted order
        for idx, (pw_name, _) in enumerate(rows):
            iid = self._iid_map.get(pw_name)
            if iid:
                try:
                    self._tree_pw.reattach(iid, '', 'end')
                    self._tree_pw.move(iid, '', idx)
                except Exception:
                    pass

        for idx, (pw_name, _) in enumerate(rows):
            iid = self._iid_map.get(pw_name)
            if iid:
                self._tree_pw.move(iid, '', idx)

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = (col == 'count')
        self._refresh_pw_tree()

    # ── Interactivity ─────────────────────────────────────────────────────────

    def _on_pw_select(self, _event=None):
        sel = self._tree_pw.selection()
        if not sel:
            return
        vals = self._tree_pw.item(sel[0], 'values')
        if not vals:
            return
        pw_name = vals[0]
        info    = self._pathway_table.get(pw_name, {})
        mets    = info.get('mets', [])
        desc    = (info.get('description', '') or '')[:200]
        detail  = f'{pw_name}  —  {len(mets)} metabolites: ' + ', '.join(sorted(mets))
        if desc:
            detail += f'\n{desc}…' if len(info.get('description','')) > 200 else f'\n{desc}'
        self._detail_var.set(detail)

    def _on_pw_dclick(self, _event=None):
        sel = self._tree_pw.selection()
        if not sel:
            return
        vals = self._tree_pw.item(sel[0], 'values')
        if not vals:
            return
        smpdb = (self._pathway_table.get(vals[0]) or {}).get('smpdb_id', '')
        if smpdb:
            webbrowser.open(SMPDB_URL.format(smpdb))

    # ── Enrichment lollipop chart (ORA) ──────────────────────────────────────

    def _compute_ora(self, hit_names: set) -> list[dict]:
        """
        Over-Representation Analysis (Fisher exact) for each pathway.
        hit_names  — set of metabolite names considered enriched.
        Background: metabolites in our local DB (N_bg) and pathway sizes from DB.
        Returns list of dicts sorted by p-value.
        """
        from scipy.stats import fisher_exact

        # Background size = all logged IDs in local DB
        N_bg = 380
        n    = len(hit_names)           # detected hits

        # Get pathway sizes within our background from local DB
        pw_sizes = {}
        if _local_db_available():
            try:
                con = sqlite3.connect(str(_LOCAL_DB))
                for pw_name in self._pathway_table:
                    cur = con.execute(
                        '''SELECT COUNT(DISTINCT mp.hmdb_id)
                           FROM metabolite_pathways mp
                           JOIN pathways p ON p.id = mp.pathway_id
                           WHERE p.name = ?''', (pw_name,))
                    pw_sizes[pw_name] = cur.fetchone()[0] or 0
                con.close()
            except Exception:
                pass

        results = []
        for pw_name, info in self._pathway_table.items():
            K  = pw_sizes.get(pw_name, len(info['mets']))  # pathway size in background
            k  = len([m for m in info['mets'] if m in hit_names])  # hits in pathway
            if k == 0:
                continue
            # 2×2 contingency table
            table = [[k,          n - k],
                     [K - k,      N_bg - n - (K - k)]]
            # clamp negatives (approximation artefact)
            table[1][1] = max(table[1][1], 0)
            _, p = fisher_exact(table, alternative='greater')
            expected = n * K / max(N_bg, 1)
            fold     = k / max(expected, 1e-9)
            results.append({
                'name':  pw_name,
                'k':     k,
                'K':     K,
                'p':     p,
                'fold':  fold,
                'smpdb': info.get('smpdb_id', ''),
            })

        # Benjamini-Hochberg FDR
        results.sort(key=lambda r: r['p'])
        m = len(results)
        for rank, r in enumerate(results, 1):
            r['fdr'] = min(r['p'] * m / rank, 1.0)

        return results

    def _show_enrichment(self):
        import math

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as mgridspec
            import matplotlib.colors as mcolors
            import numpy as np
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg, NavigationToolbar2Tk)
            from scipy.stats import wilcoxon
        except ImportError as e:
            from tkinter import messagebox
            messagebox.showerror('Missing dependency', str(e))
            return

        try:
            import networkx as nx
            HAS_NX = True
        except ImportError:
            HAS_NX = False

        # Collect non-BLK groups
        samples_info = []
        if self._app and hasattr(self._app, 'samples_info'):
            samples_info = [s for s in (self._app.samples_info or [])
                            if str(s.get('type', '')).upper() not in ('BLK', 'BLANK')]
        groups = sorted({s.get('label', '') for s in samples_info if s.get('label', '')})

        if len(groups) < 2:
            from tkinter import messagebox
            messagebox.showinfo('Pathway Activity',
                f'At least 2 groups required.\n'
                f'Groups found: {groups or ["none"]}')
            return

        # ── Window ────────────────────────────────────────────────────────────
        win = tk.Toplevel(self._win)
        win.title('Pathway Activity — Fold Change')
        win.geometry('1280x760')

        ctrl = ttk.Frame(win)
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(ctrl, text='Category:').pack(side=tk.LEFT)
        _init_cat = self._cat_var.get() if hasattr(self, '_cat_var') else 'metabolic'
        enr_cat_var = tk.StringVar(value=_init_cat)
        enr_cat_cb  = ttk.Combobox(ctrl, textvariable=enr_cat_var, width=10,
                                    state='readonly',
                                    values=['metabolic', 'drug', 'disease', 'all'])
        enr_cat_cb.pack(side=tk.LEFT, padx=(2, 10))

        def _get_visible_pws():
            _c = enr_cat_var.get()
            return {k: v for k, v in self._pathway_table.items()
                    if _c == 'all' or _pathway_category(k,
                                                         v.get('description', ''),
                                                         v.get('category', '')) == _c}

        ttk.Label(ctrl, text='Group A:').pack(side=tk.LEFT)
        ga_var = tk.StringVar(value=groups[0])
        ga_cb  = ttk.Combobox(ctrl, textvariable=ga_var, values=groups,
                               width=12, state='readonly')
        ga_cb.pack(side=tk.LEFT, padx=(2, 6))

        ttk.Label(ctrl, text='vs').pack(side=tk.LEFT)
        gb_var = tk.StringVar(value=groups[1])
        gb_cb  = ttk.Combobox(ctrl, textvariable=gb_var, values=groups,
                               width=12, state='readonly')
        gb_cb.pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(ctrl, text='Min mets:').pack(side=tk.LEFT)
        minm_var  = tk.IntVar(value=2)
        minm_spin = ttk.Spinbox(ctrl, from_=1, to=20, textvariable=minm_var, width=3)
        minm_spin.pack(side=tk.LEFT, padx=(2, 8))

        use_fdr_var = tk.BooleanVar(value=True)
        fdr_lbl_var = tk.StringVar(value='FDR <')
        def _toggle_fdr_label(*_):
            fdr_lbl_var.set('FDR <' if use_fdr_var.get() else 'p <')
        use_fdr_var.trace_add('write', _toggle_fdr_label)
        ttk.Checkbutton(ctrl, textvariable=fdr_lbl_var,
                        variable=use_fdr_var).pack(side=tk.LEFT)
        fdr_var = tk.DoubleVar(value=0.20)
        fdr_cb  = ttk.Combobox(ctrl, textvariable=fdr_var, width=5, state='readonly',
                                values=[0.05, 0.10, 0.20, 0.50, 1.00])
        fdr_cb.pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text='  Top N:').pack(side=tk.LEFT)
        top_var  = tk.IntVar(value=15)
        top_spin = ttk.Spinbox(ctrl, from_=1, to=100, textvariable=top_var, width=4)
        top_spin.pack(side=tk.LEFT, padx=(2, 8))

        if HAS_NX:
            ttk.Label(ctrl, text='Jaccard ≥').pack(side=tk.LEFT)
            jac_var = tk.DoubleVar(value=0.20)
            jac_cb  = ttk.Combobox(ctrl, textvariable=jac_var, width=5, state='readonly',
                                    values=[0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
            jac_cb.pack(side=tk.LEFT, padx=2)

        _CMAPS = ['RdPu_r', 'plasma', 'viridis', 'magma', 'YlOrRd_r']
        ttk.Label(ctrl, text='  Colors:').pack(side=tk.LEFT)
        cmap_var = tk.StringVar(value=_CMAPS[0])
        cmap_cb  = ttk.Combobox(ctrl, textvariable=cmap_var, values=_CMAPS,
                                 width=10, state='readonly')
        cmap_cb.pack(side=tk.LEFT, padx=(2, 8))

        # bg, txt_main, txt_dim, grid, stem, num_in_dot
        _BG_SCHEMES = {
            'Dark':       ('#1e1e2e', '#dddddd', '#aaaaaa', '#333355', '#444466', '#ffffff'),
            'Black':      ('#000000', '#e0e0e0', '#999999', '#222244', '#333355', '#ffffff'),
            'Slate':      ('#2d3748', '#e2e8f0', '#a0aec0', '#4a5568', '#5a6888', '#ffffff'),
            'White':      ('#ffffff', '#1a1a1a', '#555555', '#cccccc', '#aaaaaa', '#1a1a1a'),
            'Light gray': ('#f0f2f5', '#1a1a1a', '#555555', '#cccccc', '#bbbbbb', '#1a1a1a'),
        }
        ttk.Label(ctrl, text='  Background:').pack(side=tk.LEFT)
        bg_var = tk.StringVar(value='Dark')
        bg_cb  = ttk.Combobox(ctrl, textvariable=bg_var, values=list(_BG_SCHEMES),
                               width=10, state='readonly')
        bg_cb.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(ctrl, text='  Title:').pack(side=tk.LEFT)
        title_var   = tk.StringVar(value='')
        title_entry = ttk.Entry(ctrl, textvariable=title_var, width=28)
        title_entry.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(ctrl, text='Export Image',
                   command=lambda: _export_image()).pack(side=tk.RIGHT, padx=2)
        ttk.Button(ctrl, text='Export CSV',
                   command=lambda: _export_csv()).pack(side=tk.RIGHT, padx=2)

        COLOR_A  = '#5b9bd5'
        COLOR_B  = '#f4a261'
        FC_LINE  = math.log2(1.5)

        # ── Scroll container ──────────────────────────────────────────────────
        scroll_outer = tk.Frame(win)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        vbar      = ttk.Scrollbar(scroll_outer, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        tk_scroll = tk.Canvas(scroll_outer, yscrollcommand=vbar.set)
        tk_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=tk_scroll.yview)

        inner  = tk.Frame(tk_scroll)
        win_id = tk_scroll.create_window((0, 0), window=inner, anchor='nw')

        fig    = plt.figure(figsize=(11, 8))
        fig.patch.set_facecolor('#1e1e2e')
        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.get_tk_widget().pack()

        def _update_scroll(event=None):
            tk_scroll.configure(scrollregion=tk_scroll.bbox('all'))
            w = tk_scroll.winfo_width()
            if w > 1:
                tk_scroll.itemconfig(win_id, width=w)

        inner.bind('<Configure>', _update_scroll)
        tk_scroll.bind('<Configure>', _update_scroll)

        def _on_mousewheel(event):
            tk_scroll.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        win.bind('<MouseWheel>', _on_mousewheel)

        _st: dict = {'rows': [], 'ga': groups[0], 'gb': groups[1], 'met_fc': {}}

        def _recompute(*_):
            ga          = ga_var.get()
            gb          = gb_var.get()
            minm        = int(minm_var.get())
            visible_pws = _get_visible_pws()
            if ga == gb:
                return
            if not visible_pws:
                return

            met_fc: dict = {}
            for r in self._quant_results:
                nm  = r.get('metabolite', '')
                cbg = r.get('_concs_by_group') or {}
                va  = [x for x in (cbg.get(ga) or []) if x is not None and x > 0]
                vb  = [x for x in (cbg.get(gb) or []) if x is not None and x > 0]
                if not va or not vb:
                    continue
                ma, mb = float(np.mean(va)), float(np.mean(vb))
                if ma <= 0 or mb <= 0:
                    continue
                met_fc[nm] = math.log2(ma / mb)

            # Pathway DB sizes for coverage = n / K
            pw_sizes: dict = {}
            if _local_db_available():
                try:
                    _con = sqlite3.connect(str(_LOCAL_DB))
                    for _pw in visible_pws:
                        _cur = _con.execute(
                            '''SELECT COUNT(DISTINCT mp.hmdb_id)
                               FROM metabolite_pathways mp
                               JOIN pathways p ON p.id = mp.pathway_id
                               WHERE p.name = ?''', (_pw,))
                        pw_sizes[_pw] = _cur.fetchone()[0] or 0
                    _con.close()
                except Exception:
                    pass

            pw_rows = []
            for pw_name, info in visible_pws.items():
                mets_in = [m for m in info['mets'] if m in met_fc]
                if len(mets_in) < minm:
                    continue
                fcs     = [met_fc[m] for m in mets_in]
                mean_fc = float(np.mean(fcs))
                p       = 1.0
                if len(fcs) >= 3:
                    try:
                        _, p = wilcoxon(fcs, alternative='two-sided')
                        p = float(p)
                    except Exception:
                        pass
                K        = max(pw_sizes.get(pw_name, 0), len(mets_in))
                coverage = len(mets_in) / K if K > 0 else 1.0
                pw_rows.append({
                    'name':     pw_name,
                    'mean_fc':  mean_fc,
                    'n':        len(mets_in),
                    'K':        K,
                    'coverage': coverage,
                    'mets':     mets_in,
                    'fcs':      fcs,
                    'p':        p,
                    'smpdb':    info.get('smpdb_id', ''),
                })

            pw_rows.sort(key=lambda r: r['p'])
            m_tot = len(pw_rows)
            for rank, r in enumerate(pw_rows, 1):
                r['fdr'] = min(r['p'] * m_tot / rank, 1.0)

            _st.update({'rows': pw_rows, 'ga': ga, 'gb': gb, 'met_fc': met_fc})
            _draw()

        def _draw(*_):
            try:
                top_n = int(top_var.get())
            except (ValueError, tk.TclError):
                return
            fdr_thr  = float(fdr_var.get())
            use_fdr  = use_fdr_var.get()
            _p_key   = 'fdr' if use_fdr else 'p'
            _p_label = 'FDR' if use_fdr else 'p'
            ga, gb  = _st['ga'], _st['gb']
            CMAP    = plt.get_cmap(cmap_var.get())
            bg_color, txt_main, txt_dim, grid_col, stem_col, num_col = \
                _BG_SCHEMES.get(bg_var.get(), list(_BG_SCHEMES.values())[0])

            fig.clear()
            fig.patch.set_facecolor(bg_color)

            all_rows = _st.get('rows', [])
            sig = [r for r in all_rows if r[_p_key] <= fdr_thr]
            sig_a = sorted([r for r in sig if r['mean_fc'] > 0],
                           key=lambda r: r['mean_fc'], reverse=True)[:top_n]
            sig_b = sorted([r for r in sig if r['mean_fc'] < 0],
                           key=lambda r: r['mean_fc'])[:top_n]

            if not sig_a and not sig_b:
                ax_e = fig.add_subplot(111)
                ax_e.set_facecolor(bg_color)
                ax_e.text(0.5, 0.5,
                          f'No pathways with {_p_label} < {fdr_thr}\n\n'
                          f'Metabolites with log₂FC computed: {len(_st.get("met_fc", {}))}\n'
                          f'Pathways tested: {len(all_rows)}\n\n'
                          f'Relax {_p_label} threshold or reduce Min mets',
                          ha='center', va='center', color=txt_dim,
                          transform=ax_e.transAxes, fontsize=11)
                ax_e.axis('off')
                canvas.draw()
                return

            # Bidirectional: all sig sorted by FC desc (A on top, B on bottom)
            all_disp = sorted(sig_a + sig_b,
                              key=lambda r: r['mean_fc'], reverse=True)
            all_sig_ordered = all_disp   # for network
            pw_num = {r['name']: i + 1 for i, r in enumerate(all_disp)}

            all_fdrs = [r[_p_key] for r in all_disp]
            fdr_min  = max(min(all_fdrs), 1e-10)
            norm     = mcolors.LogNorm(vmin=fdr_min, vmax=1.0)

            # ── GridSpec ──────────────────────────────────────────────────────
            n_rows = max(len(all_disp), 1)
            fig_h  = min(max(8.0, n_rows * 0.48 + 2.5), 10.0)
            fig.set_size_inches(11, fig_h)
            if HAS_NX:
                gs     = mgridspec.GridSpec(1, 2, figure=fig,
                                            width_ratios=[1.5, 1], wspace=0.38)
                ax_l   = fig.add_subplot(gs[0, 0])
                ax_net = fig.add_subplot(gs[0, 1])
            else:
                gs     = mgridspec.GridSpec(1, 1, figure=fig)
                ax_l   = fig.add_subplot(gs[0])
                ax_net = None

            # ── Bidirectional lollipop ─────────────────────────────────────────
            ax_l.set_facecolor(bg_color)
            # disp: reversed → most negative (B) at bottom, most positive (A) at top
            disp      = list(reversed(all_disp))
            y_pos     = list(range(len(disp)))
            xs        = [r['mean_fc']  for r in disp]
            fdrs      = [r[_p_key]     for r in disp]
            coverages = [r.get('coverage', 1.0) for r in disp]
            dot_colors  = [CMAP(norm(f)) for f in fdrs]
            edge_colors = [COLOR_A if r['mean_fc'] > 0 else COLOR_B for r in disp]
            # Dot size = coverage (n/K); fallback to normalised count if K unknown
            has_k = any(r.get('K', 0) > r.get('n', 0) for r in disp)
            if has_k:
                sizes = [50 + 260 * c for c in coverages]
            else:
                ns    = [r['n'] for r in disp]
                max_n = max(ns) if ns else 1
                sizes = [50 + 260 * (n / max_n) for n in ns]
            label_strs = []
            for r in disp:
                nm   = r['name']
                n, K = r.get('n', 0), r.get('K', 0)
                tag  = f' [{n}/{K}]' if K > n else (f' [{n}]' if K == 0 else f' [{n}/{K}]')
                base = f"({pw_num[nm]}) {nm[:28]}{'…' if len(nm) > 28 else ''}"
                label_strs.append(base + tag)

            for y, x in zip(y_pos, xs):
                ax_l.plot([0, x], [y, y], color=stem_col, lw=1.1, zorder=1)
            ax_l.scatter(xs, y_pos, s=sizes, c=dot_colors,
                         edgecolors=edge_colors, linewidths=1.5,
                         alpha=0.93, zorder=3, clip_on=False)
            for y, x, r in zip(y_pos, xs, disp):
                ax_l.text(x, y, str(pw_num[r['name']]),
                          ha='center', va='center',
                          fontsize=6, color=num_col, fontweight='bold',
                          zorder=4, clip_on=False)

            max_abs = max(abs(x) for x in xs) if xs else 1.0
            ax_l.set_xlim(-max_abs * 1.18, max_abs * 1.18)
            ax_l.axvline(0,        color=txt_dim,   lw=1.0, alpha=0.7, zorder=2)
            ax_l.axvline( FC_LINE, color='#ffaa44', lw=0.9, ls='--', alpha=0.6)
            ax_l.axvline(-FC_LINE, color='#ffaa44', lw=0.9, ls='--', alpha=0.6)
            ax_l.set_ylim(-0.65, len(disp) - 0.35)
            ax_l.set_yticks(y_pos)
            ax_l.set_yticklabels(label_strs, fontsize=7.5, color=txt_main)
            ax_l.tick_params(axis='x', colors=txt_dim, labelsize=7.5)
            ax_l.spines[:].set_color(grid_col)
            ax_l.grid(axis='x', color=grid_col, ls='--', alpha=0.5, zorder=0)
            n_up_a = sum(1 for r in all_disp if r['mean_fc'] > 0)
            n_up_b = len(all_disp) - n_up_a
            ax_l.text(0.0, 1.005,
                      f'↑{ga}: {n_up_a}    ↑{gb}: {n_up_b}    ({_p_label} < {fdr_thr})',
                      transform=ax_l.transAxes, color=txt_main, fontsize=8,
                      fontweight='bold', va='bottom', ha='left', clip_on=False)
            size_legend = 'dot size = coverage (n/K)' if has_k else 'dot size ∝ hits'
            ax_l.set_xlabel(
                f'mean log₂FC  ({ga}/{gb})   dashed = ±1.5×   {size_legend}',
                color=txt_dim, fontsize=7.5)

            # ── Pathway network ───────────────────────────────────────────────
            if HAS_NX and ax_net is not None:
                jac_thr = float(jac_var.get()) if HAS_NX else 0.20
                G = nx.Graph()
                for r in all_sig_ordered:
                    G.add_node(r['name'],
                               n=r['n'], fdr=r[_p_key],
                               side='A' if r['mean_fc'] > 0 else 'B')

                for i, r1 in enumerate(all_sig_ordered):
                    for r2 in all_sig_ordered[i + 1:]:
                        s1, s2 = set(r1['mets']), set(r2['mets'])
                        u = s1 | s2
                        if not u:
                            continue
                        j = len(s1 & s2) / len(u)
                        if j >= jac_thr:
                            G.add_edge(r1['name'], r2['name'], weight=j)

                ax_net.set_facecolor(bg_color)
                ax_net.axis('off')

                if len(G.nodes) < 2:
                    ax_net.text(0.5, 0.5,
                                'No sufficient overlap\nbetween pathways\n'
                                f'(Jaccard < {jac_thr})',
                                ha='center', va='center', color=txt_dim,
                                transform=ax_net.transAxes, fontsize=9)
                else:
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except Exception:
                        pos = nx.spring_layout(G, seed=42, k=1.4)

                    node_list = list(G.nodes)
                    node_c    = [CMAP(norm(G.nodes[n]['fdr'])) for n in node_list]
                    node_ec   = [COLOR_A if G.nodes[n]['side'] == 'A'
                                 else COLOR_B for n in node_list]
                    max_n_net = max(G.nodes[n]['n'] for n in node_list)
                    node_sz   = [120 + 350 * (G.nodes[n]['n'] / max_n_net)
                                 for n in node_list]
                    edge_w    = [G[u][v]['weight'] for u, v in G.edges]

                    nx.draw_networkx_edges(G, pos, ax=ax_net,
                                           edge_color=txt_dim,
                                           width=[2.5 * w for w in edge_w],
                                           alpha=0.35)
                    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                           ax=ax_net,
                                           node_color=node_c,
                                           node_size=node_sz,
                                           edgecolors=node_ec,
                                           linewidths=2.0, alpha=0.95)
                    nx.draw_networkx_labels(
                        G, pos,
                        labels={n: str(pw_num[n]) for n in node_list},
                        ax=ax_net,
                        font_size=6.5, font_color=num_col, font_weight='bold')

                    fig.subplots_adjust(left=0.32, right=0.98, top=0.83, bottom=0.13)
                    fig.canvas.draw()
                    bbox_net = ax_net.get_position()
                    bbox_l   = ax_l.get_position()
                    cb_w = bbox_net.width * 0.55
                    cb_x = bbox_net.x0 + (bbox_net.width - cb_w) / 2
                    cb_h = 0.022
                    cb_y = bbox_l.y0 - cb_h - 0.012
                    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
                    sm.set_array([])
                    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
                    cax.set_facecolor(bg_color)
                    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
                    import math as _math
                    lo_exp = int(_math.floor(_math.log10(max(fdr_min, 1e-10))))
                    exps   = [e for e in range(lo_exp, 1)
                              if fdr_min <= 10**e <= 1.0]
                    import matplotlib.ticker as _mticker
                    cb.ax.xaxis.set_minor_locator(_mticker.NullLocator())
                    if len(exps) >= 2:
                        cb.ax.xaxis.set_major_locator(
                            _mticker.FixedLocator([10**e for e in exps]))
                        cb.ax.set_xticklabels(
                            [f'$10^{{{e}}}$' for e in exps])
                    else:
                        cb.ax.xaxis.set_major_locator(
                            _mticker.FixedLocator([fdr_min, 1.0]))
                        cb.ax.set_xticklabels(
                            [f'{fdr_min:.2g}', '1.0'])
                    cb.ax.tick_params(which='major', colors=txt_main,
                                      labelsize=7.5, length=3)
                    cb.ax.spines[:].set_color(txt_main)
                    cb.set_label('BH-FDR', color=txt_main, fontsize=8)

                ax_net.set_title(
                    f'Network  (Jaccard ≥ {jac_thr:.2f})\n'
                    f'↑{ga} blue  |  ↑{gb} orange',
                    color=txt_main, fontsize=13, fontweight='bold', y=1.06)

            # ── Left panel title — same y=1.06 keeps both titles at same height;
            #    stats annotation sits just below at y=1.005 (outside axes area) ─
            custom_title = title_var.get().strip()
            main_title = custom_title if custom_title else (
                f'Pathway Activity  —  {ga} vs {gb}\n'
                f'(mean log₂FC  Wilcoxon BH-FDR < {fdr_thr})')
            ax_l.set_title(main_title, color=txt_main, fontsize=13,
                           fontweight='bold', y=1.06)

            fig.subplots_adjust(left=0.32, right=0.98, top=0.83, bottom=0.13)
            dpi   = fig.get_dpi()
            w_px  = int(fig.get_figwidth()  * dpi)
            h_px  = int(fig.get_figheight() * dpi)
            canvas.get_tk_widget().configure(width=w_px, height=h_px)
            canvas.draw()
            canvas.get_tk_widget().update_idletasks()
            tk_scroll.configure(scrollregion=tk_scroll.bbox('all'))
            tk_scroll.yview_moveto(0.0)

        def _export_image():
            from tkinter import filedialog, messagebox as mb
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension='.png',
                filetypes=[('PNG', '*.png'), ('SVG', '*.svg'),
                           ('PDF', '*.pdf'), ('All files', '*.*')],
                initialfile='pathway_activity')
            if not path:
                return
            try:
                fig.savefig(path, dpi=150, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                mb.showinfo('Export Image', f'Saved to:\n{path}')
            except Exception as exc:
                mb.showerror('Export Image', str(exc))

        def _export_csv():
            import csv
            from tkinter import filedialog, messagebox as mb
            fdr_thr = float(fdr_var.get())
            ga, gb  = _st['ga'], _st['gb']
            met_fc  = _st.get('met_fc', {})

            filtered = sorted([r for r in _st.get('rows', []) if r['fdr'] <= fdr_thr],
                              key=lambda r: abs(r['mean_fc']), reverse=True)
            if not filtered:
                mb.showinfo('Export', f'No pathways with FDR < {fdr_thr}.')
                return

            path = filedialog.asksaveasfilename(
                parent=win, defaultextension='.csv',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
                initialfile='pathway_activity.csv')
            if not path:
                return

            rows_out = []
            for r in filtered:
                direction = f'enhanced in {ga}' if r['mean_fc'] > 0 else f'enhanced in {gb}'
                rows_out.append({
                    'pathway':        r['name'],
                    'direction':      direction,
                    'comparison':     f'{ga} vs {gb}',
                    'mean_log2fc':    round(r['mean_fc'], 4),
                    'n_metabolites':  r['n'],
                    'p_value':        f"{r['p']:.4e}",
                    'fdr':            f"{r['fdr']:.4e}",
                    'smpdb_id':       r['smpdb'],
                    'metabolites_fc': '; '.join(
                        f"{m}={met_fc.get(m, float('nan')):.3f}"
                        for m in r['mets']),
                })

            fieldnames = ['pathway', 'direction', 'comparison', 'mean_log2fc',
                          'n_metabolites', 'p_value', 'fdr', 'smpdb_id',
                          'metabolites_fc']
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows_out)
                mb.showinfo('Export', f'Saved {len(rows_out)} pathways to:\n{path}')
            except Exception as exc:
                mb.showerror('Export error', str(exc))

        enr_cat_cb.bind('<<ComboboxSelected>>', _recompute)
        ga_cb.bind('<<ComboboxSelected>>', _recompute)
        gb_cb.bind('<<ComboboxSelected>>', _recompute)
        minm_spin.config(command=_recompute)
        minm_spin.bind('<Return>', _recompute)
        minm_spin.bind('<FocusOut>', _recompute)
        fdr_cb.bind('<<ComboboxSelected>>', _draw)
        top_spin.config(command=_draw)
        top_spin.bind('<Return>', _draw)
        top_spin.bind('<FocusOut>', _draw)
        if HAS_NX:
            jac_cb.bind('<<ComboboxSelected>>', _draw)
        cmap_cb.bind('<<ComboboxSelected>>', _draw)
        bg_cb.bind('<<ComboboxSelected>>', _draw)
        title_entry.bind('<Return>', _draw)
        title_entry.bind('<FocusOut>', _draw)
        _recompute()

    # ── Network graph ─────────────────────────────────────────────────────────

    def _build_corr_matrix(self):
        """
        Build (names, corr_matrix) from _concs_per_sample in quant_results.
        Excludes BLK samples. Returns ([], None) if fewer than 4 usable samples.
        """
        import numpy as np

        # Identify non-BLK sample indices
        samples_info = []
        if self._app and hasattr(self._app, 'samples_info'):
            samples_info = self._app.samples_info or []

        if samples_info:
            keep_idx = [i for i, s in enumerate(samples_info)
                        if str(s.get('type', '')).upper() not in ('BLK', 'BLANK')]
        else:
            keep_idx = None  # keep all

        names, matrix = [], []
        for r in self._quant_results:
            concs = r.get('_concs_per_sample')
            if not concs:
                continue
            if keep_idx is not None:
                vals = [concs[i] for i in keep_idx if i < len(concs)]
            else:
                vals = list(concs)
            if len(vals) < 4:
                continue
            names.append(r.get('metabolite', '?'))
            matrix.append(vals)

        if len(names) < 3:
            return [], None

        X = np.array(matrix, dtype=float)   # (n_mets, n_samples)
        # replace NaN with column mean
        col_means = np.nanmean(X, axis=1, keepdims=True)
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.broadcast_to(col_means, X.shape)[nan_mask]

        corr = np.corrcoef(X)               # (n_mets, n_mets)
        return names, corr

    def _show_network(self):
        try:
            import networkx as nx
            import numpy as np
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg, NavigationToolbar2Tk)
        except ImportError as e:
            from tkinter import messagebox
            messagebox.showerror('Missing dependency',
                                 f'Install networkx:\n  pip install networkx\n\n{e}')
            return

        names, corr = self._build_corr_matrix()
        has_corr = corr is not None and len(names) >= 3

        win = tk.Toplevel(self._win)
        win.title('Metabolite Correlation Network' if has_corr else 'Pathway Network')
        win.geometry('1100x760')

        # ── Controls ─────────────────────────────────────────────────────────
        ctrl = ttk.Frame(win)
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        if has_corr:
            ttk.Label(ctrl, text='|r| ≥').pack(side=tk.LEFT)
            r_var = tk.DoubleVar(value=0.70)
            r_slider = ttk.Scale(ctrl, from_=0.3, to=0.99, variable=r_var,
                                 orient=tk.HORIZONTAL, length=150)
            r_slider.pack(side=tk.LEFT, padx=2)
            r_lbl = ttk.Label(ctrl, text='0.70', width=4)
            r_lbl.pack(side=tk.LEFT, padx=(0, 8))

            ttk.Label(ctrl, text='Min links:').pack(side=tk.LEFT)
            min_deg_var = tk.IntVar(value=2)
            min_deg_slider = ttk.Scale(ctrl, from_=1, to=10, variable=min_deg_var,
                                       orient=tk.HORIZONTAL, length=100)
            min_deg_slider.pack(side=tk.LEFT, padx=2)
            deg_lbl = ttk.Label(ctrl, text='2', width=2)
            deg_lbl.pack(side=tk.LEFT, padx=(0, 8))

        # top-N pathways slider (used for node colouring)
        max_n = min(len(self._pathway_table), 40)
        default_n = min(20, max_n)
        ttk.Label(ctrl, text='Top pathways (colour):').pack(side=tk.LEFT)
        n_var = tk.IntVar(value=default_n)
        n_slider = ttk.Scale(ctrl, from_=1, to=max(max_n, 1), variable=n_var,
                             orient=tk.HORIZONTAL, length=140)
        n_slider.pack(side=tk.LEFT, padx=2)
        n_lbl = ttk.Label(ctrl, text=str(default_n), width=3)
        n_lbl.pack(side=tk.LEFT)

        # ── Figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7.5))
        fig.patch.set_facecolor('#1e1e2e')
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        # Pre-compute pathway membership for node colouring
        # met_name → list of pathway names
        met_to_pws = {}
        for pw_name, info in self._pathway_table.items():
            for met in info.get('mets', []):
                met_to_pws.setdefault(met, []).append(pw_name)

        def _draw(*_):
            ax.clear()
            ax.set_facecolor('#1e1e2e')

            top_n = int(n_var.get())
            n_lbl.config(text=str(top_n))

            # Top-N pathways → colour palette for nodes
            sorted_pws = [pw for pw, _ in sorted(
                self._pathway_table.items(),
                key=lambda kv: len(kv[1]['mets']), reverse=True)][:top_n]
            pw_palette = {pw: cm.tab20(i / max(len(sorted_pws), 1))
                          for i, pw in enumerate(sorted_pws)}

            def _node_color(met):
                pws = met_to_pws.get(met, [])
                for pw in pws:
                    if pw in pw_palette:
                        return pw_palette[pw]
                return '#888888'

            if has_corr:
                r_thr   = round(r_var.get(), 2)
                min_deg = int(min_deg_var.get())
                r_lbl.config(text=f'{r_thr:.2f}')
                deg_lbl.config(text=str(min_deg))

                G = nx.Graph()
                for nm in names:
                    G.add_node(nm)

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        r = corr[i, j]
                        if abs(r) >= r_thr:
                            G.add_edge(names[i], names[j], weight=abs(r), r=float(r))

                # Remove nodes below min degree
                low_deg = [n for n, d in G.degree() if d < min_deg]
                G.remove_nodes_from(low_deg)

                if not G.edges:
                    ax.text(0.5, 0.5,
                            f'No edges with |r| ≥ {r_thr} and ≥ {min_deg} connections\n'
                            'Lower threshold or min links',
                            ha='center', va='center', color='#aaaaaa',
                            transform=ax.transAxes, fontsize=11)
                    canvas.draw()
                    return

                try:
                    pos = nx.kamada_kawai_layout(G, weight='weight')
                except Exception:
                    pos = nx.spring_layout(G, seed=42, weight='weight', k=1.2)

                node_colors = [_node_color(n) for n in G.nodes]
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                       node_color=node_colors,
                                       node_size=220, alpha=0.92)

                pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['r'] > 0]
                neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['r'] < 0]
                pos_w = [G[u][v]['weight'] for u, v in pos_edges]
                neg_w = [G[u][v]['weight'] for u, v in neg_edges]

                if pos_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                                           edge_color='#4caf50', alpha=0.65,
                                           width=[3.0 * w for w in pos_w])
                if neg_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                                           edge_color='#ef5350', alpha=0.65,
                                           width=[3.0 * w for w in neg_w])

                nx.draw_networkx_labels(G, pos, ax=ax,
                                        font_size=6.5, font_color='#e0e0e0')

                n_pos = len(pos_edges)
                n_neg = len(neg_edges)
                ax.set_title(
                    f'Correlation Network  |r| ≥ {r_thr}  '
                    f'({n_pos} positive, {n_neg} negative)',
                    color='#e0e0e0', fontsize=10, pad=8)

                # Legend
                from matplotlib.lines import Line2D
                legend = [
                    Line2D([0], [0], color='#4caf50', lw=2.5, label='Positive r'),
                    Line2D([0], [0], color='#ef5350', lw=2.5, label='Negative r'),
                ]
                ax.legend(handles=legend, loc='lower right',
                          facecolor='#2a2a3e', labelcolor='white', fontsize=8)

            else:
                # Fallback: bipartite pathway network (no per-sample data)
                pw_items = sorted(self._pathway_table.items(),
                                  key=lambda kv: len(kv[1]['mets']),
                                  reverse=True)[:top_n]
                G = nx.Graph()
                pw_nodes, met_nodes = [], []
                for pw_name, info in pw_items:
                    G.add_node(pw_name, ntype='pathway', count=len(info['mets']))
                    pw_nodes.append(pw_name)
                    for met in info['mets']:
                        if met not in G:
                            G.add_node(met, ntype='metabolite')
                            met_nodes.append(met)
                        G.add_edge(met, pw_name)

                try:
                    pos = nx.kamada_kawai_layout(G)
                except Exception:
                    pos = nx.spring_layout(G, seed=42, k=1.5)

                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2,
                                       edge_color='#888888', width=0.8)
                nx.draw_networkx_nodes(G, pos, nodelist=met_nodes, ax=ax,
                                       node_color=[_node_color(n) for n in met_nodes],
                                       node_size=150, alpha=0.9)
                nx.draw_networkx_nodes(G, pos, nodelist=pw_nodes, ax=ax,
                                       node_size=[300 + 120 * G.nodes[n]['count']
                                                  for n in pw_nodes],
                                       node_color='#f4a261', alpha=0.95)
                nx.draw_networkx_labels(G, pos,
                                        labels={n: n for n in met_nodes},
                                        ax=ax, font_size=6.5, font_color='#e0e0e0')
                nx.draw_networkx_labels(G, pos,
                                        labels={n: (n if len(n) <= 28 else n[:26] + '…')
                                                for n in pw_nodes},
                                        ax=ax, font_size=7.5, font_color='#ffd580',
                                        font_weight='bold')
                ax.set_title(f'Pathway Network — top {top_n} pathways',
                             color='#e0e0e0', fontsize=10, pad=8)

            ax.axis('off')
            canvas.draw()

        if has_corr:
            r_slider.config(command=_draw)
            min_deg_slider.config(command=_draw)
        n_slider.config(command=_draw)
        _draw()

    # ── Close / cancel ────────────────────────────────────────────────────────

    def _cancel(self):
        self._cancelled = True
        self._btn_cancel.config(state='disabled')

    def _on_close(self):
        self._cancelled = True
        try:
            self._win.destroy()
        except Exception:
            pass
