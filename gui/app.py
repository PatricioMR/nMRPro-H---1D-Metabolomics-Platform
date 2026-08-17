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

from processing.baseline_model import BaselineModel, BaselineCache
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
from gui.stages.cosy_annotation_stage import CosyAnnotationStage
from gui.stages.cosy_quantification_stage import CosyQuantificationStage
from gui.stages.statistics_stage import StatisticsStage
from gui._plot_settings import _PlotSettingsMixin
from gui._group_editor import _GroupEditorMixin
from app_paths import APP_VERSION


class App(_PlotSettingsMixin, _GroupEditorMixin):
    def _set_app_icon(self, win):
        """Apply the app icon (utils/nmrpipe.ico — same as the .exe/installer)
        to *win*. Using iconbitmap(default=...) makes it the icon for the root
        AND every Toplevel/dialog opened afterwards, so the whole app is
        consistent. Falls back to the PNG via iconphoto on non-Windows."""
        from app_paths import asset_path
        try:
            ico = asset_path('utils', 'nmrpipe.ico')
            if os.path.isfile(ico):
                win.iconbitmap(default=ico)
                return
        except Exception:
            pass
        try:
            from PIL import Image, ImageTk
            png = asset_path('utils', 'Logo_nMRPipe.png')
            self._app_icon_img = ImageTk.PhotoImage(Image.open(png))
            win.iconphoto(True, self._app_icon_img)
        except Exception:
            pass

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_VERSION)
        self._set_app_icon(self.root)
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
        # The project's single baseline decision. all_intensities stays raw —
        # every step reads corrected_intensities() instead, so detection,
        # quantification, statistics and MCR can never disagree about whether
        # (or how) the baseline was removed.
        self.baseline_model = BaselineModel()
        self._bl_cache = BaselineCache()
        self.baseline_corrected  = False
        self.alignment_corrected = False
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

        self.cosy_data      = None   # dict com f2_ppm, f1_ppm, data, meta (2D)
        self._2d_params     = {      # controlos de contorno — editáveis pelo left panel
            'base_mult': 5.0,        # múltiplo do nível de ruído para o 1º contorno
            'n_levels':  12,         # número de contornos positivos
            'show_neg':  False,      # contornos negativos (ruído) — off por omissão
            'show_diag': True,       # diagonal (experimentos HH)
            'show_proj': True,       # projeções 1D nas margens
        }
        self._2d_bg     = None       # background cache p/ blitting do crosshair
        self._2d_fig    = None       # figura dedicada do visualizador 2D
        self._2d_canvas = None       # FigureCanvasTkAgg do visualizador 2D
        self._2d_ax     = None       # eixo principal (contour)
        self._2d_ax_top = None       # projeção F2 (topo)
        self._2d_ax_left = None      # projeção F1 (esquerda)
        self._2d_ax_read = None      # canto: leitura de ppm
        self._2d_cids   = []         # matplotlib event connection IDs (modo 2D)
        self._2d_drag   = None       # estado do drag activo (zoom/pan)
        self._2d_cross  = None       # (vline, hline) do crosshair

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
            from app_paths import asset_path
            logo_path = asset_path('utils', 'Logo_nMRPipe.png')
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
                 text=APP_VERSION,
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

        self.stages['cosy_annotation'] = CosyAnnotationStage(
            self.left_panel, self)

        self.stages['cosy_quantification'] = CosyQuantificationStage(
            self.left_panel, self)

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

        edit_menu.add_command(label="Groups and Samples…",
                              command=self._open_sample_editor)

        edit_menu.add_separator()
        edit_menu.add_command(label="⚙ Annotation Parameters…",
                              command=self._open_annotation_params)
        _btn('Edit', edit_menu)

        # ── View ─────────────────────────────────────────────────────────────
        # Display/appearance controls: legend, axes, peak-area overlay, plot style.
        view_menu = _menu(mb_frame)
        view_menu.add_checkbutton(label="Show Legend", variable=self.show_legend,
                                  command=self.toggle_legend)
        view_menu.add_separator()
        view_menu.add_command(label="Axes Settings...",     command=self.open_axes_settings)
        view_menu.add_command(label="Show/Hide Signal Areas", command=self.toggle_peak_areas)
        view_menu.add_separator()

        plot_settings_menu = _menu(view_menu)
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
        view_menu.add_cascade(label="Plot Settings", menu=plot_settings_menu)

        _btn('View', view_menu)


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
            stage_frame, text="2. 🔍 Signal Detection",
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
            stage_frame, text="4. 🔀 JG-Deconv",
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

    @property
    def is_cosy(self):
        """True quando há um espectro 2D COSY/homonuclear carregado."""
        cd = getattr(self, 'cosy_data', None)
        if cd is None:
            return False
        return cd.get('meta', {}).get('exp_type', '') in (
            'COSY', 'TOCSY', 'NOESY', 'ROESY', '2D')

    @property
    def is_lipidomics(self):
        """True quando a anotação está em modo Lipidomics (integração por região)."""
        ann = (getattr(self, 'stages', {}) or {}).get('annotation')
        var = getattr(ann, 'internal_library_mode_var', None)
        try:
            return var is not None and var.get() == 'Lipidomics'
        except Exception:
            return False

    def switch_stage(self, stage_name):
        """Switch between analysis stages.

        Em modo COSY: Peak Detection e JG-Deconv (MCR) ficam bloqueados, e o
        botão Annotation roteia para a stage de anotação COSY.
        Em modo Lipidomics: JG-Deconv e Full-Spectrum MCR ficam bloqueados
        (integração por região não usa deteção de picos nem deconvolução).
        """
        # Bloqueios em modo COSY
        if self.is_cosy and stage_name in ('peak_detection', 'j_resolve'):
            return
        # Bloqueios em modo Lipidomics
        if self.is_lipidomics and stage_name in ('j_resolve', 'fullspectrum_mcr'):
            from tkinter import messagebox
            messagebox.showinfo(
                'Indisponível em Lipidomica',
                'Esta etapa está desativada em modo Lipidomica.\n\n'
                'A lipidomica usa integração por região do espectro — não há '
                'deteção de picos, deconvolução de sobreposição (JG-Deconv) nem '
                'MCR de espectro completo.')
            return
        # Routing consoante o tipo de dados (COSY usa stages dedicadas)
        if self.is_cosy:
            if stage_name == 'annotation':
                stage_name = 'cosy_annotation'
            elif stage_name == 'quantification':
                stage_name = 'cosy_quantification'
        else:
            if stage_name == 'cosy_annotation':
                stage_name = 'annotation'
            elif stage_name == 'cosy_quantification':
                stage_name = 'quantification'

        if self.active_stage == stage_name:
            return

        # Deactivate current stage
        if self.active_stage and self.active_stage in self.stages:
            self.stages[self.active_stage].on_deactivate()

        # Os botões da toolbar mapeiam para as stages 1D e COSY equivalentes
        annot_active = stage_name in ('annotation', 'cosy_annotation')
        quant_active = stage_name in ('quantification', 'cosy_quantification')
        for name, btn in {
            'visualization':   self.btn_visualization,
            'peak_detection':  self.btn_peak_detection,
            'annotation':      self.btn_annotation,
            'j_resolve':       self.btn_mcr_analysis,
            'quantification':  self.btn_quantification,
            'statistics':      self.btn_statistics,
        }.items():
            is_active = (name == stage_name or
                         (name == 'annotation' and annot_active) or
                         (name == 'quantification' and quant_active))
            btn.configure(
                style='ActiveStage.TButton' if is_active else 'Stage.TButton')

        # Activate new stage
        self.active_stage = stage_name
        self.stages[stage_name].on_activate()

    def _update_stage_availability(self):
        """Activa/desactiva botões de stage conforme 1D vs COSY vs Lipidomica."""
        cosy = self.is_cosy
        lipid = self.is_lipidomics
        # Peak Detection e JG-Deconv não se aplicam a COSY
        try:
            self.btn_peak_detection.configure(
                state='disabled' if cosy else 'normal')
        except Exception:
            pass
        # JG-Deconv não se aplica a COSY nem a Lipidomica
        try:
            self.btn_mcr_analysis.configure(
                state='disabled' if (cosy or lipid) else 'normal')
        except Exception:
            pass
        self._ensure_nav_lipid_tooltip()

    def _ensure_nav_lipid_tooltip(self):
        """Attach a one-time tooltip on the JG-Deconv nav button explaining the
        disabled state in Lipidomica (shown only while the button is disabled)."""
        if getattr(self, '_nav_lipid_tip_attached', False):
            return
        btn = getattr(self, 'btn_mcr_analysis', None)
        if btn is None:
            return

        def _show(_e=None):
            try:
                if str(btn.cget('state')) != 'disabled' or not self.is_lipidomics:
                    return
            except Exception:
                return
            self._hide_nav_lipid_tooltip()
            try:
                tw = tk.Toplevel(btn)
                tw.wm_overrideredirect(True)
                tw.wm_geometry(
                    f'+{btn.winfo_rootx() + 10}'
                    f'+{btn.winfo_rooty() + btn.winfo_height() + 2}')
                tk.Label(tw,
                         text=('Desativado em modo Lipidomica:\n'
                               'integração por região, sem deconvolução JG.'),
                         justify=tk.LEFT, background='#ffffe0', relief=tk.SOLID,
                         borderwidth=1, font=('Arial', 8), padx=4, pady=2).pack()
                self._nav_lipid_tip_win = tw
            except Exception:
                self._nav_lipid_tip_win = None

        btn.bind('<Enter>', _show, add='+')
        btn.bind('<Leave>', self._hide_nav_lipid_tooltip, add='+')
        self._nav_lipid_tip_attached = True

    def _hide_nav_lipid_tooltip(self, _e=None):
        tw = getattr(self, '_nav_lipid_tip_win', None)
        if tw is not None:
            try: tw.destroy()
            except Exception: pass
            self._nav_lipid_tip_win = None

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
            _btn_bar, text="📋 Signal List",
            command=self.open_peak_list, state="disabled")
        self.btn_peak_list.pack(side=tk.LEFT, padx=4)

        self.btn_deconv_shortcut = ttk.Button(
            _btn_bar, text="🔀 MCR Deconvolution",
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
        """Choose import format, then load CSV/TXT or Bruker NMR data."""
        choice = self._ask_import_format()
        if choice == 'csv':
            self._import_csv()
        elif choice == 'bruker':
            self.import_bruker()

    def _ask_import_format(self):
        """Show a small format-chooser dialog. Returns 'csv', 'bruker', or None."""
        import tkinter as tk
        result = [None]
        dlg = tk.Toplevel(self.root)
        dlg.title('Import Data')
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        from tkinter import ttk
        ttk.Label(dlg, text='Select import format:',
                  font=('Arial', 10, 'bold')).pack(padx=20, pady=(16, 8))
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(padx=20, pady=(0, 16))

        def _pick(fmt):
            result[0] = fmt
            dlg.destroy()

        ttk.Button(btn_frame, text='📄  CSV / TXT file',
                   width=22, command=lambda: _pick('csv')).pack(side='left', padx=6)
        ttk.Button(btn_frame, text='📁  Bruker NMR folder',
                   width=22, command=lambda: _pick('bruker')).pack(side='left', padx=6)
        dlg.bind('<Escape>', lambda e: dlg.destroy())
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f'+{(sw-w)//2}+{(sh-h)//2}')
        self.root.wait_window(dlg)
        return result[0]

    def _import_csv(self):
        """Import NMR spectrum data from CSV/TXT file."""
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

            self.cosy_data = None   # CSV é sempre 1D
            self._update_stage_availability()

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
                    'correction_factor': 1.0,
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
        """Import raw Bruker NMR data from a folder hierarchy.

        Detecta automaticamente se o experimento é 1D ou 2D a partir do
        PULPROG e da presença do ficheiro 2rr — o visualizador adapta-se.
        """
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

            from data.bruker_loader import detect_experiment_type, load_bruker_2d
            exp_info = detect_experiment_type(
                root_folder, opts['expno'], opts['procno'])

            if exp_info['ndim'] == 2:
                # ── Caminho 2D ────────────────────────────────────────────────
                f2_ppm, f1_ppm, data_2d, meta = load_bruker_2d(
                    exp_info['expno_dir'], opts['procno'])

                self.cosy_data = {
                    'f2_ppm': f2_ppm, 'f1_ppm': f1_ppm,
                    'data': data_2d, 'meta': meta,
                }
                # Limpar dados 1D para que update_plot saiba que estamos em 2D
                self.ppm             = None
                self.all_intensities = None

                # Lista de amostras: uma entrada para o experimento 2D
                name = os.path.basename(root_folder.rstrip('/\\'))
                self.samples_info = [{
                    'index': 0, 'name': name, 'label': 'Sample',
                    'type': 'Sample', 'correction_factor': 1.0,
                    'display': f"{name} ({meta['exp_type']})",
                }]
                self._assign_group_colors()

                self.current_file = root_folder
                if self.file_label:
                    self.file_label.config(
                        text=f"{os.path.basename(root_folder)} [{meta['exp_type']}]")

                self._update_stage_availability()
                if self.active_stage != 'visualization':
                    self.switch_stage('visualization')
                else:
                    self.stages['visualization'].refresh_sample_list()
                self.update_plot()

                self.status_label.config(
                    text=(f"2D {meta['exp_type']} carregado  "
                          f"({data_2d.shape[0]}×{data_2d.shape[1]})"),
                    foreground="green")
                self.samples_count_label.config(
                    text=f"2D · {meta['exp_type']}")

            else:
                # ── Caminho 1D (comportamento original) ───────────────────────
                self.cosy_data = None   # garantir que 2D anterior não persiste
                self._update_stage_availability()

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
                        'index': i, 'name': name, 'label': label,
                        'type': _type, 'correction_factor': 1.0,
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
                self.samples_count_label.config(
                    text=f"{len(self.samples_info)} samples")

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
    # ── Baseline ──────────────────────────────────────────────────────────────

    def corrected_intensities(self):
        """Intensities with the project's baseline model applied.

        This is what every processing step should read.  ``all_intensities``
        stays as loaded, so the correction is recomputed rather than baked in
        and can be changed or undone at any point.  Caching and the "is any
        work needed" decision live in processing/baseline_model.py.
        """
        return self._bl_cache.corrected(
            self.ppm, self.all_intensities, getattr(self, 'baseline_model', None),
            already_corrected=getattr(self, 'baseline_corrected', False))

    def invalidate_baseline_cache(self):
        """Drop the cached correction — call after changing the model or data."""
        self._bl_cache.invalidate()

    def update_plot(self):
        """Update plot based on current stage and settings."""
        # Dados 2D têm precedência quando estamos na visualização
        if self.cosy_data is not None and self.active_stage == 'visualization':
            self._show_2d_viewer()
            return

        # Saímos do modo 2D (1D activo) → restaurar canvas principal
        self._hide_2d_viewer()

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

    # ── Visualizador 2D dedicado (estilo TopSpin) ─────────────────────────────
    # Canvas próprio sobreposto à área direita (padrão do j_resolve_stage):
    #   • contour central multi-cor
    #   • projeção F2 no topo, projeção F1 à esquerda
    #   • crosshair com leitura de ppm
    #   • scroll = threshold | left-drag = zoom | right-drag = pan | dblclick = reset

    def _show_2d_viewer(self):
        """Constrói (se preciso) e mostra o visualizador 2D, escondendo o 1D."""
        self._build_2d_canvas()
        # Esconder canvas/toolbar principais
        try:
            self.toolbar.pack_forget()
        except Exception:
            pass
        try:
            self.canvas.get_tk_widget().pack_forget()
        except Exception:
            pass
        self._2d_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True)
        self._draw_2d_spectrum(preserve_view=False)

    def _hide_2d_viewer(self):
        """Esconde o visualizador 2D e restaura o canvas principal."""
        if self._2d_canvas is None:
            return
        try:
            self._2d_canvas.get_tk_widget().pack_forget()
        except Exception:
            pass
        # Re-packing na ordem correcta (BOTTOM antes de TOP)
        try:
            self.toolbar.pack_forget()
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception:
            pass

    def _build_2d_canvas(self):
        """Cria a figura/eixos do visualizador 2D uma única vez."""
        if self._2d_fig is not None:
            return
        from matplotlib.gridspec import GridSpec
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._2d_fig = plt.Figure(facecolor='#fafafa')
        gs = GridSpec(2, 2, figure=self._2d_fig,
                      width_ratios=[1, 8], height_ratios=[1, 8],
                      wspace=0.03, hspace=0.03,
                      left=0.11, right=0.98, top=0.94, bottom=0.08)
        self._2d_ax      = self._2d_fig.add_subplot(gs[1, 1])
        self._2d_ax_top  = self._2d_fig.add_subplot(gs[0, 1], sharex=self._2d_ax)
        self._2d_ax_left = self._2d_fig.add_subplot(gs[1, 0], sharey=self._2d_ax)
        self._2d_ax_read = self._2d_fig.add_subplot(gs[0, 0])
        self._2d_ax_read.axis('off')

        self._2d_canvas = FigureCanvasTkAgg(
            self._2d_fig, master=self.right_panel)
        self._bind_2d_events()

    def _draw_2d_spectrum(self, preserve_view=False):
        """Desenha o contour 2D + projeções no visualizador dedicado."""
        if self._2d_fig is None:
            self._build_2d_canvas()

        cd      = self.cosy_data
        f2_ppm  = cd['f2_ppm']
        f1_ppm  = cd['f1_ppm']
        data    = cd['data']
        meta    = cd['meta']

        p         = self._2d_params
        base_mult = float(p.get('base_mult', 3.0))
        n_lev     = max(4, int(p.get('n_levels', 12)))
        show_neg  = bool(p.get('show_neg',  True))
        show_diag = bool(p.get('show_diag', True))
        show_proj = bool(p.get('show_proj', True))

        ax   = self._2d_ax
        axt  = self._2d_ax_top
        axl  = self._2d_ax_left

        _cur_xlim = ax.get_xlim() if preserve_view else None
        _cur_ylim = ax.get_ylim() if preserve_view else None
        # Preservar também o zoom de intensidade das projeções
        _cur_top_ylim  = axt.get_ylim() if preserve_view else None
        _cur_left_xlim = axl.get_xlim() if preserve_view else None

        ax.clear(); axt.clear(); axl.clear()
        self._2d_cross = None   # crosshair invalidado pelo clear
        ax.set_facecolor(self.plot_settings['background'])

        dmax = float(np.percentile(np.abs(data), 99.9))
        if dmax < 1e-10:
            ax.text(0.5, 0.5, 'Espectro 2D vazio', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)
            self._2d_canvas.draw()
            return

        # Ruído robusto via mediana de |data| (maioria dos pontos = ruído)
        noise = float(np.median(np.abs(data)))
        if noise < dmax * 1e-4:
            noise = dmax * 1e-4
        base = noise * base_mult
        if base >= dmax:
            base = dmax * 0.05
        factor = (dmax / base) ** (1.0 / (n_lev - 1))
        levels = [base * (factor ** i) for i in range(n_lev)]

        # ── Contour multi-cor (gradiente low→high) ────────────────────────────
        from matplotlib.colors import LogNorm
        try:
            ax.contour(f2_ppm, f1_ppm, data,
                       levels=levels, cmap='jet',
                       norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                       linewidths=0.6, alpha=0.95)
            if show_neg:
                ax.contour(f2_ppm, f1_ppm, data,
                           levels=[-v for v in reversed(levels)],
                           colors='#9e9e9e', linewidths=0.5,
                           linestyles='--', alpha=0.6)
        except Exception as e:
            ax.text(0.5, 0.5, f'Erro no contour:\n{e}', transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=9)
            self._2d_canvas.draw()
            return

        # ── Projeções 1D (skyline = máximo) ───────────────────────────────────
        if show_proj:
            proj_f2 = data.max(axis=0)   # sobre F1 → perfil em F2 (topo)
            proj_f1 = data.max(axis=1)   # sobre F2 → perfil em F1 (esquerda)
            axt.plot(f2_ppm, proj_f2, color='#1a5276', lw=0.6)
            axt.fill_between(f2_ppm, proj_f2, 0, color='#1a5276', alpha=0.12)
            axl.plot(proj_f1, f1_ppm, color='#1a5276', lw=0.6)
            axl.fill_betweenx(f1_ppm, proj_f1, 0, color='#1a5276', alpha=0.12)
            axt.set_visible(True); axl.set_visible(True)
        else:
            axt.set_visible(False); axl.set_visible(False)

        # ── Limites (convenção NMR: ppm alto à esquerda/topo) ────────────────
        if preserve_view and _cur_xlim is not None:
            ax.set_xlim(_cur_xlim); ax.set_ylim(_cur_ylim)
        else:
            ax.set_xlim(float(f2_ppm[0]),  float(f2_ppm[-1]))
            ax.set_ylim(float(f1_ppm[-1]), float(f1_ppm[0]))

        # Projeção F1 (esquerda): intensidade cresce para fora (esquerda)
        axl.set_xlim(axl.get_xlim()[::-1])

        # Restaurar zoom de intensidade das projeções (não resetar no redraw)
        if preserve_view and show_proj:
            if _cur_top_ylim is not None:
                axt.set_ylim(_cur_top_ylim)
            if _cur_left_xlim is not None:
                axl.set_xlim(_cur_left_xlim)

        # ── Diagonal (homonuclear HH) ─────────────────────────────────────────
        exp_type = meta.get('exp_type', '')
        if show_diag and exp_type in ('COSY', 'TOCSY', 'NOESY', 'ROESY', '2D'):
            ppm_hi = min(float(f2_ppm[0]),  float(f1_ppm[0]))
            ppm_lo = max(float(f2_ppm[-1]), float(f1_ppm[-1]))
            ax.plot([ppm_hi, ppm_lo], [ppm_hi, ppm_lo],
                    color='#888888', lw=0.6, ls='--', alpha=0.35, zorder=0)

        fs = self.plot_settings['font_size']
        ax.set_xlabel('δ F2 (ppm)', fontsize=fs)
        ax.set_ylabel('')                       # escala F1 vai para a projeção
        # Main: ticks F2 em baixo; sem labels de F1 (ficam na projeção esquerda)
        ax.tick_params(labelsize=max(7, fs - 2), labelleft=False, left=False)

        # Projeção topo (F2): sem ticks/labels
        axt.tick_params(labelbottom=False, labelleft=False,
                        bottom=False, left=False, top=False, right=False)
        for sp in axt.spines.values():
            sp.set_visible(False)

        # Projeção esquerda (F1): mantém a escala ppm de F1 no lado de fora
        axl.tick_params(labelbottom=False, bottom=False, top=False, right=False,
                        labelleft=True, left=True, labelsize=max(7, fs - 2))
        axl.set_ylabel('δ F1 (ppm)', fontsize=fs)
        for side in ('top', 'right', 'bottom'):
            axl.spines[side].set_visible(False)
        axl.spines['left'].set_visible(True)

        axt.set_title(meta.get('pulprog', ''), fontsize=max(7, fs - 1),
                      color='#555')

        # ── Canto superior esquerdo: sequência + núcleos ─────────────────────
        self._2d_ax_read.clear()
        self._2d_ax_read.axis('off')
        self._2d_ax_read.text(
            0.5, 0.95,
            f"{meta.get('exp_type', '2D')}\n"
            f"F2: {meta.get('nuc_f2', '1H')}\n"
            f"F1: {meta.get('nuc_f1', '1H')}",
            transform=self._2d_ax_read.transAxes,
            ha='center', va='top', fontsize=9, color='#222',
            family='monospace', linespacing=1.6)

        # Coordenadas do cursor — dentro do gráfico central, canto sup. esquerdo
        self._2d_coord_txt = ax.text(
            0.012, 0.985, '', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color='#e91e63',
            family='monospace', linespacing=1.5, animated=True, zorder=11,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.7))

        if self.plot_settings['grid_style'] != 'none':
            ax.grid(True, linestyle=self.plot_settings['grid_style'],
                    alpha=0.3, linewidth=0.4)

        # Crosshair como artistas 'animated' (excluídos do draw normal → blitting)
        self._2d_cross = (
            ax.axvline(ax.get_xlim()[0], color='#e91e63', lw=0.6,
                       alpha=0.8, animated=True, visible=False, zorder=9),
            ax.axhline(ax.get_ylim()[0], color='#e91e63', lw=0.6,
                       alpha=0.8, animated=True, visible=False, zorder=9),
        )

        self._2d_bg = None       # invalidar cache; recapturado no draw_event
        self._2d_canvas.draw()

    # ── Interacção 2D estilo TopSpin ──────────────────────────────────────────

    def _bind_2d_events(self):
        """Liga os handlers de rato ao canvas 2D (idempotente)."""
        if self._2d_cids or self._2d_canvas is None:
            return
        c = self._2d_canvas
        self._2d_cids = [
            c.mpl_connect('scroll_event',         self._on_2d_scroll),
            c.mpl_connect('button_press_event',   self._on_2d_press),
            c.mpl_connect('motion_notify_event',  self._on_2d_motion),
            c.mpl_connect('button_release_event', self._on_2d_release),
            c.mpl_connect('draw_event',           self._on_2d_draw),
        ]

    def _on_2d_draw(self, event):
        """Captura o background após cada draw completo (para blitting)."""
        try:
            self._2d_bg = self._2d_canvas.copy_from_bbox(self._2d_fig.bbox)
        except Exception:
            self._2d_bg = None

    def _unbind_2d_events(self):
        """Remove os handlers do canvas 2D."""
        if self._2d_canvas is not None:
            for cid in self._2d_cids:
                try:
                    self._2d_canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._2d_cids = []
        self._2d_drag = None

    def _on_2d_scroll(self, event):
        """Scroll: threshold no contour; intensidade nas projeções."""
        # up → amplificar (picos mais altos); down → reduzir
        factor = 0.8 if event.button == 'up' else 1.25

        if event.inaxes is self._2d_ax_top:
            # Projeção F2: intensidade no eixo Y (base ancorada em 0)
            _, y1 = self._2d_ax_top.get_ylim()
            self._2d_ax_top.set_ylim(0, y1 * factor)
            self._2d_canvas.draw_idle()
            return

        if event.inaxes is self._2d_ax_left:
            # Projeção F1: intensidade no eixo X (invertido → cresce p/ esquerda;
            # base ancorada em 0)
            x0, _ = self._2d_ax_left.get_xlim()
            self._2d_ax_left.set_xlim(x0 * factor, 0)
            self._2d_canvas.draw_idle()
            return

        if event.inaxes != self._2d_ax:
            return
        mult = float(self._2d_params.get('base_mult', 3.0))
        mult = mult * 1.3 if event.button == 'up' else mult / 1.3
        mult = max(0.5, min(200.0, mult))
        self._2d_params['base_mult'] = mult
        self._draw_2d_spectrum(preserve_view=True)
        self.status_label.config(
            text=f"Contour threshold ×{mult:.2f} noise", foreground="blue")

    def _on_2d_press(self, event):
        """Zoom 2D (main) / zoom só F2 (projeção topo) / zoom só F1 (projeção
        esquerda) / pan (right) / double-click reset."""
        inax = event.inaxes
        if inax not in (self._2d_ax, self._2d_ax_top, self._2d_ax_left):
            return

        cd = self.cosy_data
        f2_ppm, f1_ppm = cd['f2_ppm'], cd['f1_ppm']

        # ── Double-click → reset (total no main, por-eixo nas projeções) ──────
        if event.dblclick and event.button == 1:
            self._2d_drag = None
            if inax is self._2d_ax_top:
                self._2d_ax.set_xlim(float(f2_ppm[0]), float(f2_ppm[-1]))
                self._2d_canvas.draw_idle()
                self.status_label.config(text="F2 full", foreground="gray")
            elif inax is self._2d_ax_left:
                self._2d_ax.set_ylim(float(f1_ppm[-1]), float(f1_ppm[0]))
                self._2d_canvas.draw_idle()
                self.status_label.config(text="F1 full", foreground="gray")
            else:
                self._draw_2d_spectrum(preserve_view=False)
                self.status_label.config(text="Full view", foreground="gray")
            return

        if event.button == 1 and event.xdata is not None \
                and event.ydata is not None:
            import matplotlib.patches as _patches
            from matplotlib.transforms import blended_transform_factory

            if inax is self._2d_ax:
                # Zoom rectângulo 2D
                rect = _patches.Rectangle(
                    (event.xdata, event.ydata), 0, 0,
                    fill=True, facecolor='#2196F3', edgecolor='#1565C0',
                    alpha=0.18, lw=0.8, zorder=10)
                self._2d_ax.add_patch(rect)
                self._2d_drag = {'mode': 'zoom', 'x0': event.xdata,
                                 'y0': event.ydata, 'rect': rect}

            elif inax is self._2d_ax_top:
                # Zoom só F2 (arrasto horizontal na projeção do topo)
                trans = blended_transform_factory(
                    self._2d_ax_top.transData, self._2d_ax_top.transAxes)
                rect = _patches.Rectangle(
                    (event.xdata, 0), 0, 1, transform=trans,
                    fill=True, facecolor='#2196F3', edgecolor='#1565C0',
                    alpha=0.18, lw=0.8, zorder=10)
                self._2d_ax_top.add_patch(rect)
                self._2d_drag = {'mode': 'zoomx', 'x0': event.xdata,
                                 'rect': rect}

            elif inax is self._2d_ax_left:
                # Zoom só F1 (arrasto vertical na projeção da esquerda)
                trans = blended_transform_factory(
                    self._2d_ax_left.transAxes, self._2d_ax_left.transData)
                rect = _patches.Rectangle(
                    (0, event.ydata), 1, 0, transform=trans,
                    fill=True, facecolor='#2196F3', edgecolor='#1565C0',
                    alpha=0.18, lw=0.8, zorder=10)
                self._2d_ax_left.add_patch(rect)
                self._2d_drag = {'mode': 'zoomy', 'y0': event.ydata,
                                 'rect': rect}

        elif event.button == 3 and inax is self._2d_ax:
            self._2d_drag = {
                'mode': 'pan',
                'xlim0': self._2d_ax.get_xlim(),
                'ylim0': self._2d_ax.get_ylim(),
                'x0px': event.x, 'y0px': event.y}

    def _on_2d_motion(self, event):
        """Crosshair + leitura ppm; actualiza zoom rect ou pan se activo."""
        drag = self._2d_drag

        if drag and drag['mode'] == 'zoom':
            if event.xdata is None or event.ydata is None:
                return
            x0, y0 = drag['x0'], drag['y0']
            rect = drag['rect']
            rect.set_x(min(x0, event.xdata)); rect.set_y(min(y0, event.ydata))
            rect.set_width(abs(event.xdata - x0))
            rect.set_height(abs(event.ydata - y0))
            self._2d_canvas.draw_idle()
            return

        if drag and drag['mode'] == 'zoomx':
            if event.xdata is None:
                return
            x0 = drag['x0']
            rect = drag['rect']
            rect.set_x(min(x0, event.xdata))
            rect.set_width(abs(event.xdata - x0))
            self._2d_canvas.draw_idle()
            return

        if drag and drag['mode'] == 'zoomy':
            if event.ydata is None:
                return
            y0 = drag['y0']
            rect = drag['rect']
            rect.set_y(min(y0, event.ydata))
            rect.set_height(abs(event.ydata - y0))
            self._2d_canvas.draw_idle()
            return

        if drag and drag['mode'] == 'pan':
            if event.x is None or event.y is None:
                return
            bbox = self._2d_ax.get_window_extent()
            if bbox.width == 0 or bbox.height == 0:
                return
            xlim0, ylim0 = drag['xlim0'], drag['ylim0']
            dx = (event.x - drag['x0px']) * (xlim0[1] - xlim0[0]) / bbox.width
            dy = (event.y - drag['y0px']) * (ylim0[1] - ylim0[0]) / bbox.height
            self._2d_ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
            self._2d_ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
            self._2d_canvas.draw_idle()
            return

        # Sem drag → crosshair via blitting (rápido) + readout na status bar
        if event.inaxes != self._2d_ax or event.xdata is None:
            return
        if self._2d_cross is None or self._2d_bg is None:
            return   # background ainda não capturado; próximo draw resolve
        vline, hline = self._2d_cross
        vline.set_xdata([event.xdata, event.xdata]); vline.set_visible(True)
        hline.set_ydata([event.ydata, event.ydata]); hline.set_visible(True)
        self._2d_coord_txt.set_text(
            f"F2: {event.xdata:.3f}\nF1: {event.ydata:.3f}")
        self._2d_canvas.restore_region(self._2d_bg)
        self._2d_ax.draw_artist(vline)
        self._2d_ax.draw_artist(hline)
        self._2d_ax.draw_artist(self._2d_coord_txt)
        self._2d_canvas.blit(self._2d_fig.bbox)
        self.status_label.config(
            text=f"F2: {event.xdata:.3f} ppm    F1: {event.ydata:.3f} ppm",
            foreground="#333")

    def _on_2d_release(self, event):
        """Conclui o zoom rectângulo; termina o pan."""
        drag = self._2d_drag
        self._2d_drag = None
        if not drag:
            return

        try:
            drag['rect'].remove()
        except Exception:
            pass

        mode = drag['mode']
        if mode == 'zoom':
            if event.xdata is None or event.ydata is None:
                self._2d_canvas.draw_idle()
                return
            x0, y0 = drag['x0'], drag['y0']
            x1, y1 = event.xdata, event.ydata
            if abs(x1 - x0) < 1e-3 or abs(y1 - y0) < 1e-3:
                self._2d_canvas.draw_idle()
                return
            self._2d_ax.set_xlim(max(x0, x1), min(x0, x1))
            self._2d_ax.set_ylim(min(y0, y1), max(y0, y1))
            self._2d_canvas.draw_idle()

        elif mode == 'zoomx':
            if event.xdata is None:
                self._2d_canvas.draw_idle()
                return
            x0, x1 = drag['x0'], event.xdata
            if abs(x1 - x0) < 1e-3:
                self._2d_canvas.draw_idle()
                return
            self._2d_ax.set_xlim(max(x0, x1), min(x0, x1))
            self._2d_canvas.draw_idle()

        elif mode == 'zoomy':
            if event.ydata is None:
                self._2d_canvas.draw_idle()
                return
            y0, y1 = drag['y0'], event.ydata
            if abs(y1 - y0) < 1e-3:
                self._2d_canvas.draw_idle()
                return
            self._2d_ax.set_ylim(min(y0, y1), max(y0, y1))
            self._2d_canvas.draw_idle()

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
        """Open Annotation Parameters dialog.

        If an AnnotationResultsWindow is active, delegates to it (enables
        live recompute).  Otherwise opens a standalone pre-annotation dialog
        that persists params on the annotation stage for the next run.
        """
        win = getattr(self, '_annot_win', None)
        if win is not None and getattr(win, 'win', None):
            try:
                win.win.deiconify()
                win.win.lift()
            except Exception:
                pass
            win._open_param_dialog()
            return

        # ── Pre-annotation: standalone dialog ────────────────────────────────
        ann_stage = self.stages.get('annotation')
        if ann_stage is None:
            messagebox.showinfo("Annotation Parameters",
                                "Annotation stage not initialised.",
                                parent=self.root)
            return

        _DEFAULTS = {
            'good_score_thresh': 0.80, 'bad_score_thresh': 0.50,
            'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50, 'dj_neutral': 0.00,
            'dppm_gold_hz': 0.50, 'dppm_gold': 0.05,
            'fp_coverage_pct': 50.0, 'nnls_min_frac': 0.03,
            'anchor_clean_lo': 0.35, 'anchor_clean_hi': 2.50,
            'anchor_snr_bdl': 3.0,
            'w_dj': 0.57, 'w_cosine_ns': 0.14, 'w_ppm_ns': 0.10,
            'w_found_ns': 0.05, 'w_rri_ns': 0.09, 'w_reverse_ns': 0.05,
            'w_cosine_s': 0.32, 'w_ppm_s': 0.27,
            'w_found_s': 0.13, 'w_rri_s': 0.18, 'w_reverse_s': 0.10,
        }
        current = dict(_DEFAULTS)
        if ann_stage._annotation_params:
            current.update(ann_stage._annotation_params)

        dlg = tk.Toplevel(self.root)
        dlg.title("Annotation Parameters")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        nb = ttk.Notebook(dlg)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        _vars = {}

        def _tab(title, rows):
            frm = ttk.Frame(nb, padding=10)
            nb.add(frm, text=title)
            for r, (lbl, key, tip) in enumerate(rows):
                ttk.Label(frm, text=lbl, anchor=tk.W).grid(
                    row=r, column=0, sticky=tk.W, padx=(0, 10), pady=3)
                var = tk.StringVar(value=str(current[key]))
                _vars[key] = var
                ttk.Entry(frm, textvariable=var, width=10).grid(
                    row=r, column=1, sticky=tk.W, pady=3)
                if tip:
                    ttk.Label(frm, text=tip, foreground='gray',
                              font=('Arial', 8)).grid(
                        row=r, column=2, sticky=tk.W, padx=(6, 0))

        _tab("Score & Quality", [
            ("Good score threshold",   'good_score_thresh', "Rules 3/4/5 — min score → Good"),
            ("Bad score threshold",    'bad_score_thresh',  "Rule 1 — singlets only"),
            ("Fingerprint coverage %", 'fp_coverage_pct',   "≥2 DB peaks: min coverage %"),
            ("NNLS min fraction",      'nnls_min_frac',     "< this → downgrade to yellow"),
        ])
        _tab("ΔJ Criteria (Hz)", [
            ("Gold standard ΔJ", 'dj_gold_hz', "ΔJ < this → s_dj=1.0 (binary)"),
            ("Bad ΔJ threshold", 'dj_bad_hz',  "ΔJ ≥ this → Bad (non-singlet gate)"),
            ("s_dj neutral",     'dj_neutral', "s_dj when ΔJ unmeasurable"),
        ])
        _tab("Δppm Criteria", [
            ("Gold standard Δppm (Hz)",  'dppm_gold_hz', "Max Δppm in Hz (field-independent)"),
            ("Gold standard Δppm (ppm)", 'dppm_gold',    "Max Δppm in ppm"),
        ])
        _tab("RANSY Anchor", [
            ("Clean ratio low",   'anchor_clean_lo',  "Min area ratio → clean"),
            ("Clean ratio high",  'anchor_clean_hi',  "Max area ratio → clean"),
            ("SNR BDL threshold", 'anchor_snr_bdl',   "SNR < this → below detection limit"),
        ])
        _tab("Score Weights", [
            ("w_dj (non-singlet)",      'w_dj',         "ΔJ score weight"),
            ("w_cosine (non-singlet)",  'w_cosine_ns',  ""),
            ("w_ppm (non-singlet)",     'w_ppm_ns',     ""),
            ("w_found (non-singlet)",   'w_found_ns',   ""),
            ("w_rri (non-singlet)",     'w_rri_ns',     ""),
            ("w_reverse (non-singlet)", 'w_reverse_ns', "Reverse-match weight"),
            ("w_cosine (singlet)",      'w_cosine_s',   ""),
            ("w_ppm (singlet)",         'w_ppm_s',      ""),
            ("w_found (singlet)",       'w_found_s',    ""),
            ("w_rri (singlet)",         'w_rri_s',      ""),
            ("w_reverse (singlet)",     'w_reverse_s',  "Reverse-match weight"),
        ])

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, padx=8, pady=(0, 8))

        def _apply():
            errs, new_p = [], {}
            for key, var in _vars.items():
                try:
                    new_p[key] = float(var.get())
                except ValueError:
                    errs.append(key)
            if errs:
                messagebox.showerror("Invalid value",
                                     f"Not a number: {', '.join(errs)}",
                                     parent=dlg)
                return
            ann_stage._annotation_params = new_p
            dlg.destroy()

        def _reset():
            for key, var in _vars.items():
                var.set(str(_DEFAULTS.get(key, current[key])))

        ttk.Button(bf, text="Reset Defaults", command=_reset).pack(side=tk.LEFT)
        ttk.Button(bf, text="Apply",  command=_apply).pack(side=tk.RIGHT)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=6)

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
        """Open the unified MCR Deconvolution window (singleton).

        Gathers every peak/region that already has 'mcr_annotation' data
        (full-spectrum MCR regions take priority, then annotation-stage
        peaks) and opens MCRWindow. The generic Lorentzian/Gaussian/Onion
        Deconvolution Tool no longer lives behind this button — it is
        still reachable from the Peak List's right-click
        "View in Deconvolution Tool".
        """
        ann_stage  = self.stages.get('annotation')
        pk_stage   = self.stages.get('peak_detection')
        fs_stage   = self.stages.get('fullspectrum_mcr')

        # Priority 1: Full-spectrum MCR results
        _fs_regions = getattr(fs_stage, '_mcr_regions', [])
        # Priority 2: annotation-stage MCR peaks
        _peaks   = pk_stage.peaks if pk_stage else []
        _ann_mcr = [p for p in _peaks if p.get('mcr_annotation')]

        if _fs_regions:
            _source_peaks = _fs_regions
            _source_stage = fs_stage
        elif _ann_mcr:
            _source_peaks = _ann_mcr
            _source_stage = ann_stage
        else:
            _source_peaks = None
            _source_stage = None

        if not _source_peaks:
            from tkinter import messagebox
            messagebox.showinfo(
                'MCR Deconvolution',
                'No MCR deconvolution available yet.\n\n'
                'Run Annotation or Full-Spectrum MCR first, or use the '
                'Signal List → right-click → "View in Deconvolution Tool" '
                'for manual signal-fitting.')
            return

        # Singleton MCR window
        if self._deconv_win is not None:
            try:
                w = self._deconv_win.win
                if w.winfo_exists():
                    w.deiconify(); w.lift(); w.focus_force()
                    return
            except Exception:
                pass
        from gui.widgets.mcr_window import MCRWindow
        items = [{'kind': 'region', 'met': None, 'mp': None, 'peak': p}
                 for p in _source_peaks]
        self._deconv_win = MCRWindow(
            self.root, items, self, ann_stage=_source_stage)

    def open_annot_shortcut(self):
        """Open Annotation Results — singleton.

        Em modo COSY abre a janela de resultados COSY (cross-peaks + analitos),
        não a janela de anotação 1D.
        """
        if self.is_cosy:
            self._open_cosy_results()
            return

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

    def _open_cosy_results(self):
        """Abre/reutiliza a janela de resultados COSY após a anotação."""
        stage = self.stages.get('cosy_annotation')
        # Reutilizar janela existente
        win = getattr(stage, '_results_win', None)
        if win is not None and getattr(win, 'win', None) is not None:
            try:
                if win.win.winfo_exists():
                    win.win.deiconify(); win.win.lift(); win.win.focus_force()
                    return
            except Exception:
                pass
        # Reconstruir a partir dos dados guardados
        ca = getattr(self, 'cosy_annotation', None)
        if not ca:
            messagebox.showinfo(
                'COSY', 'Run COSY annotation first (Annotation stage).')
            return
        from gui.widgets.cosy_results_window import CosyResultsWindow
        stage._results_win = CosyResultsWindow(
            self, self.cosy_data, ca['crosspeaks'], ca['results'])

    def _on_closing(self):
        """Clean shutdown — prompt on unsaved changes, then tear down so the
        process actually exits (autosave 'after' jobs and TkAgg figure managers
        otherwise keep the interpreter alive after the window closes)."""
        mgr = getattr(self, '_project_mgr', None)
        if mgr is not None and getattr(mgr, '_dirty', False):
            resp = messagebox.askyesnocancel(
                'Unsaved Changes',
                'Save changes to the current project before closing?')
            if resp is None:          # Cancel → abort close
                return
            if resp:                  # Yes → save; abort close if save failed
                if not mgr.save_or_saveas():
                    return
        # Stop autosave + cancel its pending Tk 'after' callback
        try:
            if mgr is not None:
                mgr._stop_autosave()
        except Exception:
            pass
        # Close matplotlib figures (TkAgg managers can outlive the root)
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self._init_project_manager()
        self.root.protocol('WM_DELETE_WINDOW', self._on_closing)
        self.root.mainloop()
        # Guarantee the process terminates even if a stray non-daemon thread or
        # lingering Tk/matplotlib resource would otherwise keep it alive.
        sys.exit(0)