# plotting/interactive.py
"""
Interactive plot manager — pan, zoom, peak overlay.

Peak fills:
  - x : start_ppm → end_ppm  (data coordinates)
  - y : 0 → peak height × 1.15  (data coordinates, 15% above peak top)
  - zorder 1 → behind the spectrum line (zorder 2)
  - overlapping peaks : light red  (#ffaaaa, alpha 0.45)
  - normal peaks      : alternating Set2/Set3 colours, alpha 0.25
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class InteractivePlotManager:
    def __init__(self, figure, ax, canvas, app):
        self.figure = figure
        self.ax     = ax
        self.canvas = canvas
        self.app    = app

        self.zoomed        = False
        self.original_xlim = None
        self.original_ylim = None


        # Peak overlay
        self._peak_collection  = None   # PatchCollection (normal peaks)
        self._over_collection  = None   # PatchCollection (overlapping peaks)
        self._peak_labels      = []
        self._peak_centerlines = []     # dotted vertical lines at peak centres
        self._peaks_drawn      = False

        # Navegação (pan/zoom) tratada pela NavigationToolbar2Tk nativa.
        # Este manager trata apenas do peak overlay.

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def reset_zoom(self):
        if self.zoomed and self.original_xlim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoomed = False
            self.canvas.draw_idle()

    def zoom_to_peak(self, peak):
        margin = max(0.2, abs(peak.get('start_ppm', peak['ppm']) -
                               peak.get('end_ppm',   peak['ppm'])) * 3)
        self.ax.set_xlim(peak['ppm'] + margin, peak['ppm'] - margin)
        self.canvas.draw_idle()


    # ── Peak overlay ──────────────────────────────────────────────────────────

    def draw_peak_overlay(self, peaks, mean_intensity, ppm_array):
        """
        Toggle highlight fills on/off.

        Each fill is a rectangle in DATA coordinates:
          x : peak['start_ppm'] → peak['end_ppm']
          y : 0 → peak['height'] * 1.15   (15 % above peak top)

        zorder=1 → drawn behind spectrum line (which is at zorder 2 by default)

        Colour logic:
          peak['overlapping'] == True  → light red  (#ffaaaa)
          otherwise                    → alternating Set2 / Set3 palette
        """
        if self._peaks_drawn:
            self._remove_peak_overlay()
            self.canvas.draw_idle()
            return

        if not peaks:
            return

        print(f"[draw_peak_overlay] drawing {len(peaks)} peak(s)")

        palette_a = plt.cm.Set2(np.linspace(0, 1, 8))
        palette_b = plt.cm.Set3(np.linspace(0, 1, 8))
        overlap_color = np.array([1.0, 0.67, 0.67, 1.0])   # light red

        normal_patches   = []
        normal_colors    = []
        overlap_patches  = []

        normal_idx = 0  # separate counter so alternation ignores overlap peaks

        for peak in peaks:
            try:
                x_start = float(peak['start_ppm'])
                x_end   = float(peak['end_ppm'])
                # Ensure x_start > x_end (ppm axis decreases left→right)
                if x_start < x_end:
                    x_start, x_end = x_end, x_start

                # Height: 15 % above the peak maximum intensity
                y_top = float(peak['height']) * 1.15

                # Rectangle vertices in data coords
                verts = [
                    (x_end,   0),
                    (x_start, 0),
                    (x_start, y_top),
                    (x_end,   y_top),
                    (x_end,   0),
                ]
                poly = Polygon(verts, closed=True)

                if peak.get('overlapping', False):
                    overlap_patches.append(poly)
                else:
                    color = (palette_a[normal_idx % 8]
                             if normal_idx % 2 == 0
                             else palette_b[normal_idx % 8])
                    normal_patches.append(poly)
                    normal_colors.append(color)
                    normal_idx += 1
            except Exception as _e:
                print(f"[draw_peak_overlay] skipping peak fill: {_e}  peak={peak}")
                continue

        # Normal peaks — one PatchCollection, zorder=1 (behind spectrum)
        if normal_patches:
            pc = PatchCollection(
                normal_patches,
                facecolors=normal_colors,
                alpha=0.30,
                edgecolors='none',
                zorder=1,
            )
            self.ax.add_collection(pc)
            self._peak_collection = pc

        # Overlapping peaks — separate collection, same zorder
        if overlap_patches:
            oc = PatchCollection(
                overlap_patches,
                facecolors=[overlap_color] * len(overlap_patches),
                alpha=0.45,
                edgecolors='none',
                zorder=1,
            )
            self.ax.add_collection(oc)
            self._over_collection = oc

        # Rank labels — all peaks, number above each apex
        for p in peaks:
            try:
                txt = self.ax.text(
                    p['ppm'],
                    float(p['height']) * 1.18,
                    str(p.get('rank', '?')),
                    ha='center', va='bottom',
                    fontsize=6, color='#333',
                    fontweight='bold',
                    clip_on=True,
                    zorder=3,
                )
                self._peak_labels.append(txt)
            except Exception:
                continue

        # Centre dotted lines — one axvline per peak, 50 % opacity, red
        self._peak_centerlines = []
        for peak in peaks:
            try:
                line = self.ax.axvline(
                    float(peak['ppm']),
                    color='red',
                    linewidth=0.9,
                    linestyle=':',
                    alpha=0.50,
                    zorder=2,
                )
                self._peak_centerlines.append(line)
            except Exception:
                continue

        self._peaks_drawn = True
        self.canvas.draw_idle()

    def _remove_peak_overlay(self):
        for coll in (self._peak_collection, self._over_collection):
            if coll is not None:
                try:
                    coll.remove()
                except Exception:
                    pass
        self._peak_collection = None
        self._over_collection = None

        for txt in self._peak_labels:
            try:
                txt.remove()
            except Exception:
                pass
        self._peak_labels = []

        for line in self._peak_centerlines:
            try:
                line.remove()
            except Exception:
                pass
        self._peak_centerlines = []

        self._peaks_drawn = False

    def force_redraw_peaks(self, peaks, mean_intensity, ppm_array):
        """Remove overlay and redraw — used after deconvolution."""
        self._remove_peak_overlay()
        self.draw_peak_overlay(peaks, mean_intensity, ppm_array)