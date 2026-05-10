# gui/app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.1
matplotlib.rcParams['agg.path.chunksize'] = 10000

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_all_samples
from data.bruker_loader import load_bruker_dataset
from gui._bruker_import_dialog import _BrukerImportDialog
from gui._sample_editor import SampleEditorDialog
from plotting.spectrum import plot_spectrum, plot_multiple_spectra
from gui.stages.visualization_stage import VisualizationStage
from gui.stages.peak_detection_stage import PeakDetectionStage
from gui.stages.annotation_stage import AnnotationStage
from gui.stages.quantification_stage import QuantificationStage
from gui.stages.fullspectrum_mcr_stage import FullSpectrumMCRStage
from gui.stages.j_resolve_stage import JResolveStage
from gui.stages.statistics_stage import StatisticsStage
from gui._plot_settings import _PlotSettingsMixin
from gui._group_editor import _GroupEditorMixin


class App(_PlotSettingsMixin, _GroupEditorMixin):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NMR Metabolomics Viewer")
        self.screen_width  = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.withdraw()
        self.root.minsize(1000, 600)

        # Flags de controlo do splash
        self._splash_min_done  = False   # 3 s passados
        self._splash_load_done = False   # UI totalmente construída
        self._splash_ref       = None    # referência ao Toplevel do splash

        # ---------- Data shared between stages ----------
        self.current_file = None
        self.samples_info = []
        self.all_intensities = None
        self.ppm = None
        self._ppm_calibration = None
        self.eligible_overrides = {}   # {f'{met_id}__{peak_rank}': bool}

        self.group_colors = {
            'CTRL': '#2E86AB', 'G1': '#A23B72', 'G2': '#F18F01',
            'G3': '#17BEBB',   'BLK': '#C73E1D', 'Unknown': '#593C8F'
        }
        self.group_dilution_factors = {}

        self.show_legend = tk.BooleanVar(value=True)
        self._project_mgr = None
        self.plot_settings = {
            'background': '#FFFFFF', 'grid_style': '--',
            'line_width': 0.5, 'transparency': 0.8, 'font_size': 10
        }
        self.axes_settings = {
            'x_label': 'δ (ppm)', 'y_label': 'Intensity',
            'x_lim': None, 'y_lim': None
        }

        self.plot_manager   = None
        self._peak_list_win = None
        self._deconv_win    = None
        self._annot_win     = None
        self.btn_import     = None
        self.file_label     = None
        self.btn_select_all = None
        self.btn_clear_all  = None
        self.btn_update     = None
        self.active_stage   = None
        self.stages         = {}

        # 1. Mostrar splash imediatamente (antes de construir UI)
        self._show_splash()

        # 2. Construir UI no próximo ciclo do mainloop (splash já visível)
        self.root.after(50, self._build_ui)

    # ---------- Splash screen ----------
    def _show_splash(self):
        splash = tk.Toplevel(self.root)
        splash.overrideredirect(True)
        splash.attributes('-topmost', True)
        splash.attributes('-transparentcolor', 'black')  # black = transparente

        try:
            from PIL import Image, ImageTk
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'utils', 'Logo_nMRPipe.png')
            img = Image.open(logo_path).convert('RGBA')
            img = img.resize((300, 300), Image.LANCZOS)
            self._splash_logo = ImageTk.PhotoImage(img)
            logo_ok = True
        except Exception:
            self._splash_logo = None
            logo_ok = False

        w, h = 300, 360
        sw, sh = self.screen_width, self.screen_height
        splash.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        splash.configure(bg='black')

        if logo_ok:
            tk.Label(splash, image=self._splash_logo,
                     bg='black', borderwidth=0).pack()

        tk.Label(splash,
                 text="nMRPipe v0.1 Pre-Alpha",
                 font=('Segoe UI', 13, 'bold'),
                 bg='black', fg='white').pack(pady=(8, 0))

        self._splash_ref = splash
        splash.update()

        # Minimum display time: 3 seconds
        self.root.after(3000, self._on_splash_timer)

    def _on_splash_timer(self):
        """Called after the 3-second minimum splash duration."""
        self._splash_min_done = True
        self._maybe_finish_splash()

    def _maybe_finish_splash(self):
        """Close splash and show main window only when BOTH conditions are met:
        - 3 s minimum elapsed  (_splash_min_done)
        - UI fully built       (_splash_load_done)
        """
        if not (self._splash_min_done and self._splash_load_done):
            return
        if self._splash_ref is not None:
            try:
                if self._splash_ref.winfo_exists():
                    self._splash_ref.destroy()
            except Exception:
                pass
            self._splash_ref = None
        self.root.deiconify()
        self.root.state('zoomed')

    # ---------- UI construction (runs during splash) ----------
    def _build_ui(self):
        """Build the entire UI. Called via root.after(50) so the splash is
        already visible when construction starts. Sets _splash_load_done when
        complete and triggers _maybe_finish_splash."""
        try:
            self.create_menu()
            self.create_stage_toolbar()
            self.create_main_layout()
            self.initialize_stages()
        finally:
            self._splash_load_done = True
            self._maybe_finish_splash()

    def initialize_stages(self):
        """Create all stage instances."""
        self.stages['visualization'] = VisualizationStage(self.left_panel, self)
        self.stages['peak_detection'] = PeakDetectionStage(self.left_panel, self)
        self.stages['annotation'] = AnnotationStage(
            self.left_panel, self,
            peak_stage=self.stages['peak_detection'])

        self.stages['quantification'] = QuantificationStage(
            self.left_panel, self,
            peak_stage=self.stages['peak_detection'],
            annotation_stage=self.stages['annotation'])

        self.stages['fullspectrum_mcr'] = FullSpectrumMCRStage(
            self.left_panel, self,
            annotation_stage=self.stages['annotation'])

        self.stages['j_resolve'] = JResolveStage(
            self.left_panel, self,
            annotation_stage=self.stages['annotation'])

        self.stages['statistics'] = StatisticsStage(self.left_panel, self)

        # Activate visualization by default
        self.switch_stage('visualization')

    # ---------- Menu ----------
    def create_menu(self):
        """Create a custom frame-based menu bar so the background matches the app theme.
        Using native tk.Menu (root.config(menu=...)) on Windows 11 renders the bar
        in white via Win32 regardless of the bg parameter — a Frame avoids that."""
        _mb_bg     = ttk.Style().lookup('TFrame', 'background') or '#f0f0f0'
        _mb_fg     = '#1a1a1a'
        _mb_act_bg = '#d0d8e8'
        _font      = ('Segoe UI', 9)

        # Propagate colours to all dropdown menus
        self.root.option_add('*Menu.background',       _mb_bg)
        self.root.option_add('*Menu.foreground',       _mb_fg)
        self.root.option_add('*Menu.activeBackground', _mb_act_bg)
        self.root.option_add('*Menu.activeForeground', _mb_fg)
        self.root.option_add('*Menu.relief',           'flat')

        # Frame that replaces the native Win32 menu bar
        mb_frame = tk.Frame(self.root, bg=_mb_bg, bd=0)
        mb_frame.pack(side=tk.TOP, fill=tk.X)

        def _menu(parent):
            return tk.Menu(parent, tearoff=0, bg=_mb_bg, fg=_mb_fg,
                           activebackground=_mb_act_bg, activeforeground=_mb_fg,
                           relief='flat', bd=1, font=_font)

        def _btn(label, menu):
            b = tk.Button(mb_frame, text=label,
                          bg=_mb_bg, fg=_mb_fg,
                          activebackground=_mb_act_bg, activeforeground=_mb_fg,
                          relief='flat', bd=0, padx=10, pady=3, font=_font,
                          cursor='arrow')
            def _post(event=None, _b=b, _m=menu):
                _m.tk_popup(_b.winfo_rootx(),
                            _b.winfo_rooty() + _b.winfo_height())
            b.config(command=_post)
            b.bind('<Enter>', lambda e, w=b: w.config(bg=_mb_act_bg))
            b.bind('<Leave>', lambda e, w=b: w.config(bg=_mb_bg))
            b.pack(side=tk.LEFT)

        # ── File ─────────────────────────────────────────────────────────────
        file_menu = _menu(mb_frame)
        file_menu.add_command(label='Save Project…',  command=self._save_project_as,
                              accelerator='Ctrl+Shift+S')
        file_menu.add_command(label='Open Project…',  command=self._open_project,
                              accelerator='Ctrl+O')
        file_menu.add_separator()
        file_menu.add_command(label='Save',           command=self._save,
                              accelerator='Ctrl+S')
        file_menu.add_separator()
        file_menu.add_command(label='📂 Import CSV/TXT…',    command=self.import_data)
        file_menu.add_command(label='📁 Import Bruker NMR…', command=self.import_bruker)
        _btn('File', file_menu)

        # ── Edit ─────────────────────────────────────────────────────────────
        edit_menu = _menu(mb_frame)

        groups_menu = _menu(edit_menu)
        groups_menu.add_command(label="Edit Groups and Colors...",
                                command=self.open_group_editor)
        edit_menu.add_cascade(label="Groups", menu=groups_menu)

        edit_menu.add_separator()
        edit_menu.add_command(label="Axes Settings...",    command=self.open_axes_settings)
        edit_menu.add_separator()
        edit_menu.add_command(label="Show/Hide Peak Areas", command=self.toggle_peak_areas)
        edit_menu.add_separator()

        plot_settings_menu = _menu(edit_menu)
        plot_settings_menu.add_command(label="Background Color...",
                                       command=self.set_background_color)

        grid_menu = _menu(plot_settings_menu)
        for _sty, _lbl in [('--','Dashed'),('-','Solid'),(':','Dotted'),('none','None')]:
            grid_menu.add_command(label=_lbl,
                                  command=lambda s=_sty: self.set_grid_style(s))
        plot_settings_menu.add_cascade(label="Grid Style", menu=grid_menu)

        lw_menu = _menu(plot_settings_menu)
        for _w in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
            lw_menu.add_command(label=str(_w),
                                command=lambda v=_w: self.set_line_width(v))
        plot_settings_menu.add_cascade(label="Line Width", menu=lw_menu)

        tr_menu = _menu(plot_settings_menu)
        for _a in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            tr_menu.add_command(label=str(_a),
                                command=lambda v=_a: self.set_transparency(v))
        plot_settings_menu.add_cascade(label="Transparency", menu=tr_menu)

        fs_menu = _menu(plot_settings_menu)
        for _s in [8, 9, 10, 11, 12, 14]:
            fs_menu.add_command(label=str(_s),
                                command=lambda v=_s: self.set_font_size(v))
        plot_settings_menu.add_cascade(label="Font Size", menu=fs_menu)

        plot_settings_menu.add_separator()
        plot_settings_menu.add_command(label="Reset to Default",
                                       command=self.reset_plot_settings)
        edit_menu.add_cascade(label="Plot Settings", menu=plot_settings_menu)

        edit_menu.add_separator()
        edit_menu.add_command(label="🔍 Debug Peak Flow", command=self.open_peak_debugger)
        edit_menu.add_separator()
        edit_menu.add_command(label="⚙ Annotation Parameters…",
                              command=self._open_annotation_params)
        _btn('Edit', edit_menu)

        # ── View ─────────────────────────────────────────────────────────────
        view_menu = _menu(mb_frame)
        view_menu.add_checkbutton(label="Show Legend", variable=self.show_legend,
                                  command=self.toggle_legend)
        _btn('View', view_menu)

        # ── Validation ────────────────────────────────────────────────────────
        ttk.Button(mb_frame, text='✓ Validation',
                   command=self._open_validation).pack(side=tk.LEFT, padx=2)

    # ---------- Stage Toolbar ----------
    def create_stage_toolbar(self):
        stage_frame = ttk.Frame(self.root)
        stage_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        style = ttk.Style()
        style.configure('Stage.TButton', font=('Arial', 10))
        style.configure('ActiveStage.TButton', font=('Arial', 10, 'bold'))

        self.btn_visualization = ttk.Button(
            stage_frame, text="1. 📊 Visualization",
            command=lambda: self.switch_stage('visualization'),
            style='ActiveStage.TButton'
        )
        self.btn_visualization.pack(side=tk.LEFT, padx=2)

        self.btn_peak_detection = ttk.Button(
            stage_frame, text="2. 🔍 Peak Detection",
            command=lambda: self.switch_stage('peak_detection'),
            style='Stage.TButton'
        )
        self.btn_peak_detection.pack(side=tk.LEFT, padx=2)

        self.btn_annotation = ttk.Button(
            stage_frame, text="3. 🧪 Annotation",
            command=lambda: self.switch_stage('annotation'),
            style='Stage.TButton'
        )
        self.btn_annotation.pack(side=tk.LEFT, padx=2)

        self.btn_mcr_analysis = ttk.Button(
            stage_frame, text="4. 🔀 J-Resolve",
            command=lambda: self.switch_stage('j_resolve'),
            style='Stage.TButton'
        )
        self.btn_mcr_analysis.pack(side=tk.LEFT, padx=2)

        self.btn_quantification = ttk.Button(
            stage_frame, text="5. ⚗ Quantification",
            command=lambda: self.switch_stage('quantification'),
            style='Stage.TButton'
        )
        self.btn_quantification.pack(side=tk.LEFT, padx=2)

        self.btn_statistics = ttk.Button(
            stage_frame, text="6. 📊 Statistics",
            command=lambda: self.switch_stage('statistics'),
            style='Stage.TButton'
        )
        self.btn_statistics.pack(side=tk.LEFT, padx=2)

    def switch_stage(self, stage_name):
        """Switch between analysis stages."""
        if self.active_stage == stage_name:
            return

        # Deactivate current stage
        if self.active_stage and self.active_stage in self.stages:
            self.stages[self.active_stage].on_deactivate()

        for name, btn in {
            'visualization':   self.btn_visualization,
            'peak_detection':  self.btn_peak_detection,
            'annotation':      self.btn_annotation,
            'j_resolve':       self.btn_mcr_analysis,
            'quantification':  self.btn_quantification,
            'statistics':      self.btn_statistics,
        }.items():
            btn.configure(
                style='ActiveStage.TButton'
                if name == stage_name else 'Stage.TButton')

        # Activate new stage
        self.active_stage = stage_name
        self.stages[stage_name].on_activate()

    # ---------- Main Layout ----------
    def create_main_layout(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left panel (for stages)
        self.left_panel = ttk.Frame(self.main_container, width=320, relief=tk.SUNKEN, borderwidth=2)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2), pady=0)
        self.left_panel.pack_propagate(False)

        # Right panel (plot)
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100, tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.plot_settings['background'])
        self.figure.patch.set_facecolor(self.plot_settings['background'])

        # Bottom button bar — packed BEFORE canvas so tkinter reserves space.
        _btn_bar = ttk.Frame(self.right_panel)
        _btn_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)

        self.btn_peak_list = ttk.Button(
            _btn_bar, text="📋 Peak List",
            command=self.open_peak_list, state="disabled")
        self.btn_peak_list.pack(side=tk.LEFT, padx=4)

        self.btn_deconv_shortcut = ttk.Button(
            _btn_bar, text="🔀 Deconvolution Tool",
            command=self.open_deconv_shortcut, state="disabled")
        self.btn_deconv_shortcut.pack(side=tk.LEFT, padx=4)

        self.btn_annot_shortcut = ttk.Button(
            _btn_bar, text="🔬 Annotation Results",
            command=self.open_annot_shortcut, state="disabled")
        self.btn_annot_shortcut.pack(side=tk.LEFT, padx=4)

        # Create canvas first
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas.draw()

        # Build toolbar with canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        self.toolbar.update()

        # Ensure Toolbar is at Bottom and Canvas is at Top expanding
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialise the shared InteractivePlotManager once.
        # All stages use self.app.plot_manager — no per-stage instance.
        try:
            from plotting.interactive import InteractivePlotManager
            self.plot_manager = InteractivePlotManager(
                self.figure, self.ax, self.canvas, self)
            print('✅ Shared InteractivePlotManager initialised')
        except Exception as e:
            print(f'❌ PlotManager: {e}')

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="gray")
        self.status_label.pack(side=tk.LEFT)
        self.samples_count_label = ttk.Label(status_frame, text="", foreground="gray")
        self.samples_count_label.pack(side=tk.RIGHT)

    # ── Project management ───────────────────────────────────────────────────

    def _init_project_manager(self):
        from gui.project_manager import ProjectManager
        self._project_mgr = ProjectManager(self)
        self.root.bind('<Control-s>', lambda e: self._save())
        self.root.bind('<Control-S>', lambda e: self._save())
        self.root.bind('<Control-o>', lambda e: self._open_project())
        self.root.bind('<Control-O>', lambda e: self._open_project())

    def _save_project_as(self):
        if self._project_mgr:
            self._project_mgr.save_project()

    def _open_project(self):
        if self._project_mgr:
            self._project_mgr.open_project()

    def _save(self):
        if self._project_mgr:
            self._project_mgr.save_or_saveas()

    # ---------- Data Import ----------
    def import_data(self):
        """Import NMR spectrum data from CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select spectrum file",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            # Update status
            self.status_label.config(text="Loading file...", foreground="blue")
            self.root.update()

            # Load all samples using the robust loader
            self.ppm, self.all_intensities, sample_names, sample_labels = load_all_samples(filepath)

            # CRITICAL: Create samples_info for BLK identification
            self.samples_info = []
            for i, (name, label) in enumerate(zip(sample_names, sample_labels)):
                _type = 'BLK' if str(label).upper() == 'BLK' else 'Sample'
                self.samples_info.append({
                    'index': i,
                    'name': name,
                    'label': label,
                    'type': _type,
                    'display': f"{name} ({label})"
                })

            print(f"\n📊 Loaded {len(self.samples_info)} samples:")
            for s in self.samples_info:
                print(f"   Sample {s['index']}: {s['name']} - label: '{s['label']}'")

            # Gerar cores automáticas para grupos que não estão em group_colors
            self._assign_group_colors()

            # Clear existing checkboxes in visualization stage
            if 'visualization' in self.stages:
                self.stages['visualization'].refresh_sample_list()
                # Enable buttons in visualization stage
                if hasattr(self.stages['visualization'], 'sample_vars'):
                    self.btn_select_all.config(state="normal")
                    self.btn_clear_all.config(state="normal")
                    self.btn_update.config(state="normal")

            self.current_file = filepath
            if self.file_label:
                self.file_label.config(text=os.path.basename(filepath))

            # Update the plot
            self.update_plot()

            # Update status
            self.status_label.config(text="File loaded successfully", foreground="green")
            self.samples_count_label.config(text=f"{len(self.samples_info)} samples")

            self._open_sample_editor()

        except Exception as e:
            self.status_label.config(text="Error loading file", foreground="red")
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def import_bruker(self):
        """Import raw Bruker NMR data from a folder hierarchy."""
        root_folder = filedialog.askdirectory(
            title="Select Bruker dataset folder (contains one sub-folder per sample)"
        )
        if not root_folder:
            return

        opts = _BrukerImportDialog(self.root).result
        if opts is None:
            return

        try:
            self.status_label.config(text="Loading Bruker data…", foreground="blue")
            self.root.update()

            self.ppm, self.all_intensities, sample_names, sample_labels = \
                load_bruker_dataset(
                    root_folder,
                    ppm_min=opts['ppm_min'],
                    ppm_max=opts['ppm_max'],
                    n_points=opts['n_points'],
                    expno=opts['expno'],
                    procno=opts['procno'],
                )

            self.samples_info = []
            for i, (name, label) in enumerate(zip(sample_names, sample_labels)):
                _type = 'BLK' if str(label).upper() in ('BLK', 'BLANK') else 'Sample'
                self.samples_info.append({
                    'index': i,
                    'name': name,
                    'label': label,
                    'type': _type,
                    'display': f"{name} ({label})"
                })

            self._assign_group_colors()

            if 'visualization' in self.stages:
                self.stages['visualization'].refresh_sample_list()
                if hasattr(self.stages['visualization'], 'sample_vars'):
                    self.btn_select_all.config(state="normal")
                    self.btn_clear_all.config(state="normal")
                    self.btn_update.config(state="normal")

            self.current_file = root_folder
            if self.file_label:
                self.file_label.config(text=os.path.basename(root_folder))

            self.update_plot()
            self.status_label.config(text="Bruker data loaded", foreground="green")
            self.samples_count_label.config(text=f"{len(self.samples_info)} samples")

            self._open_sample_editor()

        except Exception as e:
            self.status_label.config(text="Error loading Bruker data", foreground="red")
            messagebox.showerror("Bruker Import Error", str(e))
            import traceback
            traceback.print_exc()

    def _open_sample_editor(self):
        """Abre o diálogo per-sample após qualquer import e aplica as alterações."""
        SampleEditorDialog(
            self.root,
            self.samples_info,
            self.group_colors,
            self.group_dilution_factors,
        )
        # Após fechar, cores e labels já foram actualizados in-place pelo diálogo.
        # Regenerar cores para grupos novos que possam ter sido introduzidos.
        self._assign_group_colors()
        if 'visualization' in self.stages:
            self.stages['visualization'].refresh_sample_list()
        self.update_plot()

    # ---------- Plot Update ----------
    def update_plot(self):
        """Update plot based on current stage and settings."""
        if self.ppm is None or self.all_intensities is None:
            return

        # If in visualization stage, use sample selection
        if self.active_stage == 'visualization':
            stage = self.stages['visualization']
            if hasattr(stage, 'sample_vars') and stage.sample_vars:
                selected = [i for i, var in enumerate(stage.sample_vars) if var.get()]

                if not selected:
                    self.ax.clear()
                    self.ax.set_xlabel(self.axes_settings['x_label'], fontsize=self.plot_settings['font_size'])
                    self.ax.set_ylabel(self.axes_settings['y_label'], fontsize=self.plot_settings['font_size'])
                    if self.plot_settings['grid_style'] != 'none':
                        self.ax.grid(True, linestyle=self.plot_settings['grid_style'], alpha=0.7, linewidth=0.5)
                    self.ax.set_facecolor(self.plot_settings['background'])
                    self.ax.text(0.5, 0.5, 'No samples selected', transform=self.ax.transAxes,
                                 ha='center', va='center', fontsize=14, color='gray')
                    self.canvas.draw()
                    return

                if len(selected) == 1:
                    idx = selected[0]
                    sample = self.samples_info[idx]
                    plot_spectrum(self.ax, self.ppm, self.all_intensities[:, idx],
                                  title=f"Sample: {sample['name']} - Group: {sample['label']}",
                                  color='black', label=sample['display'],
                                  show_legend=self.show_legend.get(),
                                  settings=self.plot_settings)
                else:
                    samples = [self.samples_info[i] for i in selected]
                    names = [s['name'] for s in samples]
                    labels = [s['label'] for s in samples]
                    intensities = self.all_intensities[:, selected]
                    plot_multiple_spectra(self.ax, self.ppm, intensities, names, labels,
                                          self.group_colors, show_legend=self.show_legend.get(),
                                          settings=self.plot_settings)
            else:
                # Fallback: plot all samples
                plot_multiple_spectra(self.ax, self.ppm, self.all_intensities,
                                      [s['name'] for s in self.samples_info],
                                      [s['label'] for s in self.samples_info],
                                      self.group_colors, show_legend=self.show_legend.get(),
                                      settings=self.plot_settings)
        else:
            # In peak_detection / annotation stages, plot all samples
            plot_multiple_spectra(self.ax, self.ppm, self.all_intensities,
                                  [s['name'] for s in self.samples_info],
                                  [s['label'] for s in self.samples_info],
                                  self.group_colors, show_legend=self.show_legend.get(),
                                  settings=self.plot_settings)

        self.apply_axes_settings()
        self.status_label.config(text=f"Showing data")
        self.canvas.draw()

    def _open_validation(self):
        """Open the Validation window."""
        qs = self.stages.get('quantification')
        if qs is None or not getattr(qs, '_quant_results', None):
            messagebox.showinfo(
                'Validation',
                'Run quantification first (Step 5).',
                parent=self.root)
            return
        from gui.widgets.validation_window import ValidationWindow
        ValidationWindow(self.root, qs, self)

    def _open_resolving_method(self):
        """Open the Overlap Resolving Method window."""
        ann = self.stages.get('annotation')
        metabolites = getattr(ann, '_metabolites', None)
        if not metabolites and ann and getattr(ann, '_last_results_args', None):
            # Project was loaded from disk — reconstruct the annotation results window
            try:
                self.open_annot_shortcut()
                metabolites = getattr(ann, '_metabolites', None)
            except Exception:
                pass
        if not metabolites:
            messagebox.showinfo(
                'Resolving Method',
                'Run annotation search and review results first.\n'
                'Mark peaks as Good (Ctrl+1) in the Annotation Results window.',
                parent=self.root)
            return
        from gui.widgets.resolving_method_window import ResolvingMethodWindow
        ResolvingMethodWindow(
            self.root,
            metabolites=metabolites,
            peaks=ann.peaks,
            ppm=self.ppm,
            all_intensities=self.all_intensities,
            samples_info=getattr(self, 'samples_info', []),
            app=self)

    def open_peak_debugger(self):
        """Open the peak flow debugger for the current stage."""
        if hasattr(self, 'stages') and 'peak_detection' in self.stages:
            notation_stage = self.stages['peak_detection']
            try:
                from ParseHMDBNames import open_debugger
                open_debugger(notation_stage)
            except ImportError:
                messagebox.showerror("Error",
                                     "Debug module not found.\n"
                                     "Please ensure debug_peak_flow.py exists in the project root.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open debugger:\n{str(e)}")
        else:
            messagebox.showinfo("Info", "Please switch to Metabolites Notation stage first.")

    def _open_annotation_params(self):
        """Open Annotation Parameters dialog via the active AnnotationResultsWindow."""
        win = getattr(self, '_annot_win', None)
        if win is None or not getattr(win, 'win', None):
            messagebox.showinfo(
                "Annotation Parameters",
                "No annotation results window open.\n"
                "Run annotation first.")
            return
        try:
            win.win.deiconify()
            win.win.lift()
        except Exception:
            pass
        win._open_param_dialog()

    def toggle_peak_areas(self):
        """Show or hide peak highlight areas from the Edit menu."""
        # Delegate to PeakDetectionStage which owns the peak list
        stage = self.stages.get('peak_detection')
        if stage is None or not stage.peaks:
            return
        stage.toggle_peaks()

    def open_peak_list(self):
        """Open the Peak List window — singleton: reutiliza se já estiver aberta."""
        stage = self.stages.get('peak_detection')
        if stage is None:
            return
        # Also accept peaks injected by MCR Analysis (K=1 regions)
        if not stage.peaks:
            return
        # Reutilizar janela existente se estiver aberta — atualizar dados
        if self._peak_list_win is not None:
            try:
                w = self._peak_list_win.win
                if w.winfo_exists():
                    # Atualizar referência de peaks e repopular (pode ter mudado)
                    self._peak_list_win.peaks = stage.peaks
                    self._peak_list_win._populate()
                    w.deiconify()
                    w.lift()
                    w.focus_force()
                    return
            except Exception:
                pass
        from gui.widgets.peak_list_window import PeakListWindow
        self._peak_list_win = PeakListWindow(
            self.root, stage.peaks,
            self.ppm, self.all_intensities, app=self)
        # Limpar referência quando a janela fechar
        self._peak_list_win.win.protocol(
            'WM_DELETE_WINDOW',
            lambda: (self._peak_list_win.win.destroy(),
                     setattr(self, '_peak_list_win', None)))

    def open_deconv_shortcut(self):
        """Open Deconvolution window.

        After annotation: opens MCRDeconvWindow (MCR results).
        Before annotation: opens legacy DeconvWindow (peak fitting).
        """
        ann_stage  = self.stages.get('annotation')
        pk_stage   = self.stages.get('peak_detection')
        fs_stage   = self.stages.get('fullspectrum_mcr')

        # Priority 1: Full-spectrum MCR results
        _fs_regions = getattr(fs_stage, '_mcr_regions', [])
        # Priority 2: annotation-stage MCR peaks
        _peaks   = pk_stage.peaks if pk_stage else []
        _ann_mcr = [p for p in _peaks
                    if p.get('mcr_annotation', {}).get('K', 1) >= 2]

        if _fs_regions:
            _source_peaks = _fs_regions
            _source_stage = fs_stage
        elif _ann_mcr:
            _source_peaks = _peaks
            _source_stage = ann_stage
        else:
            _source_peaks = None
            _source_stage = None

        if _source_peaks:
            # Singleton MCR window
            if self._deconv_win is not None:
                try:
                    w = self._deconv_win.win
                    if w.winfo_exists():
                        w.deiconify(); w.lift(); w.focus_force()
                        return
                except Exception:
                    pass
            from gui.widgets.mcr_deconv_window import MCRDeconvWindow
            self._deconv_win = MCRDeconvWindow(
                self.root, _source_peaks, self, ann_stage=_source_stage)
        else:
            # Legacy peak-detection deconvolution tool
            if self._deconv_win is not None:
                try:
                    w = self._deconv_win.window
                    if w.winfo_exists():
                        w.deiconify(); w.lift(); w.focus_force()
                        return
                except Exception:
                    pass
            if pk_stage:
                pk_stage.open_deconv_tool()

    def open_annot_shortcut(self):
        """Open Annotation Results — singleton."""
        # Reutilizar janela existente
        if self._annot_win is not None:
            try:
                w = self._annot_win.win
                if w.winfo_exists():
                    w.deiconify()
                    w.lift()
                    w.focus_force()
                    return
            except Exception:
                pass
        stage = self.stages.get('annotation')
        if stage and hasattr(stage, '_last_results_args'):
            args = stage._last_results_args
            if not args:
                return
            pk_stage = self.stages.get('peak_detection')

            # Progress dialog while AnnotationResultsWindow initialises
            import tkinter.ttk as _ttk
            _pdlg = tk.Toplevel(self.root)
            _pdlg.title('Building Results…')
            _pdlg.resizable(False, False)
            _pdlg.grab_set()
            self.root.update_idletasks()
            _px = self.root.winfo_rootx() + self.root.winfo_width()  // 2 - 175
            _py = self.root.winfo_rooty() + self.root.winfo_height() // 2 - 60
            _pdlg.geometry(f'350x120+{_px}+{_py}')
            _ttk.Label(_pdlg, text='Processing annotation results…',
                       font=('Arial', 9, 'bold')).pack(pady=(12, 4))
            _pb = _ttk.Progressbar(_pdlg, orient='horizontal', length=310,
                                    mode='determinate', maximum=100)
            _pb.pack(padx=20)
            _msg_var = tk.StringVar(value='Initialising…')
            _ttk.Label(_pdlg, textvariable=_msg_var,
                       font=('Arial', 8), foreground='gray').pack(pady=(4, 8))
            _pdlg.update()

            def _progress_cb(step, total, msg):
                try:
                    _pb['value'] = int(step / max(total, 1) * 100)
                    _msg_var.set(msg)
                    _pdlg.update_idletasks()
                except Exception:
                    pass

            from gui.widgets.annotation_results_window import AnnotationResultsWindow
            _saved_mets = getattr(stage, '_metabolites', None)
            self._annot_win = AnnotationResultsWindow(
                self.root,
                args['results'],
                pk_stage.peaks if pk_stage else [],
                self.ppm,
                self.all_intensities,
                db_path=args.get('db_path'),
                elapsed=args.get('elapsed', 0),
                app=self,
                saved_quality=dict(getattr(stage, '_quality_marks', {})),
                progress_cb=_progress_cb,
                saved_metabolites=_saved_mets)
            try:
                _pdlg.destroy()
            except Exception:
                pass

            # Keep the live reference updated so Resolving Method sees
            # the same metabolite dicts that the user is annotating.
            stage._metabolites = self._annot_win.metabolites
            self._annot_win.win.protocol(
                'WM_DELETE_WINDOW',
                lambda: self._annot_win.win.withdraw())

    def run(self):
        self._init_project_manager()
        self.root.mainloop()