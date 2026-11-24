"""Temporary: build only part of the UI to find what crashes Tk on Linux.

    python bisect_app.py <steps> [stages]
steps: how many of menu, toolbar, layout, stages to build.
stages: with steps=4, how many stages to create (in initialize_stages order).
Exit by timeout (124) = survived; 139 = crashed.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ['NMRPRO_NO_UPDATE_CHECK'] = '1'
import gui.app as A  # noqa: E402

steps = int(sys.argv[1])
n_stages = int(sys.argv[2]) if len(sys.argv) > 2 else 99

ORDER = [
    ('visualization', lambda s: A.VisualizationStage(s.left_panel, s)),
    ('peak_detection', lambda s: A.PeakDetectionStage(s.left_panel, s)),
    ('annotation', lambda s: A.AnnotationStage(
        s.left_panel, s, peak_stage=s.stages['peak_detection'])),
    ('quantification', lambda s: A.QuantificationStage(
        s.left_panel, s, peak_stage=s.stages['peak_detection'],
        annotation_stage=s.stages['annotation'])),
    ('fullspectrum_mcr', lambda s: A.FullSpectrumMCRStage(
        s.left_panel, s, annotation_stage=s.stages['annotation'])),
    ('j_resolve', lambda s: A.JResolveStage(
        s.left_panel, s, annotation_stage=s.stages['annotation'])),
    ('cosy_annotation', lambda s: A.CosyAnnotationStage(s.left_panel, s)),
    ('cosy_quantification', lambda s: A.CosyQuantificationStage(s.left_panel, s)),
    ('statistics', lambda s: A.StatisticsStage(s.left_panel, s)),
]


def initialize_stages(self):
    for name, make in ORDER[:n_stages]:
        print('stage', name, flush=True)
        self.stages[name] = make(self)
    if 'visualization' in self.stages:
        self.switch_stage('visualization')


def build(self):
    try:
        parts = [self.create_menu, self.create_stage_toolbar,
                 self.create_main_layout, lambda: initialize_stages(self)]
        for p in parts[:steps]:
            p()
    finally:
        self._splash_load_done = True
        self._maybe_finish_splash()
        print('built', flush=True)


A.App._build_ui = build
app = A.App()
app.root.mainloop()
