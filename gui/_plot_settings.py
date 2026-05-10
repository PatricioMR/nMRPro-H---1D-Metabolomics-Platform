# gui/_plot_settings.py
"""Mixin with plot/axes settings methods for App."""
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import numpy as np


class _PlotSettingsMixin:

    # ---------- Axes Settings Dialog ----------
    def open_axes_settings(self):
        if self.ppm is None:
            messagebox.showinfo("Info", "Please load data first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Axes Settings (NMR: ppm decreases left to right)")
        dialog.geometry("420x400")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_set()

        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Labels
        ttk.Label(main_frame, text="Axis Labels", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                   pady=5, sticky=tk.W)
        ttk.Label(main_frame, text="X Label:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        x_label_var = tk.StringVar(value=self.axes_settings['x_label'])
        x_label_entry = ttk.Entry(main_frame, textvariable=x_label_var, width=30)
        x_label_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(main_frame, text="Y Label:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        y_label_var = tk.StringVar(value=self.axes_settings['y_label'])
        y_label_entry = ttk.Entry(main_frame, textvariable=y_label_var, width=30)
        y_label_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Separator(main_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Ranges with NMR explanation
        ttk.Label(main_frame, text="Axis Ranges (NMR: ppm decreases left to right)",
                  font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Left value (max ppm) → Right value (min ppm)",
                  font=("Arial", 9, "italic"), foreground="blue").grid(row=5, column=0, columnspan=2, pady=2,
                                                                       sticky=tk.W)

        # X range (ppm)
        ttk.Label(main_frame, text="X Min (left):").grid(row=6, column=0, padx=5, pady=2, sticky=tk.W)
        x_min_var = tk.StringVar()
        if self.axes_settings['x_lim'] is not None:
            x_min_var.set(f"{self.axes_settings['x_lim'][1]:.4f}")  # Max value (left)
        x_min_entry = ttk.Entry(main_frame, textvariable=x_min_var, width=15)
        x_min_entry.grid(row=6, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(main_frame, text="X Max (right):").grid(row=7, column=0, padx=5, pady=2, sticky=tk.W)
        x_max_var = tk.StringVar()
        if self.axes_settings['x_lim'] is not None:
            x_max_var.set(f"{self.axes_settings['x_lim'][0]:.4f}")  # Min value (right)
        x_max_entry = ttk.Entry(main_frame, textvariable=x_max_var, width=15)
        x_max_entry.grid(row=7, column=1, padx=5, pady=2, sticky=tk.W)

        # Y range (normal)
        ttk.Label(main_frame, text="Y Min (bottom):").grid(row=8, column=0, padx=5, pady=2, sticky=tk.W)
        y_min_var = tk.StringVar()
        if self.axes_settings['y_lim'] is not None:
            y_min_var.set(f"{self.axes_settings['y_lim'][0]:.4f}")
        y_min_entry = ttk.Entry(main_frame, textvariable=y_min_var, width=15)
        y_min_entry.grid(row=8, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(main_frame, text="Y Max (top):").grid(row=9, column=0, padx=5, pady=2, sticky=tk.W)
        y_max_var = tk.StringVar()
        if self.axes_settings['y_lim'] is not None:
            y_max_var.set(f"{self.axes_settings['y_lim'][1]:.4f}")
        y_max_entry = ttk.Entry(main_frame, textvariable=y_max_var, width=15)
        y_max_entry.grid(row=9, column=1, padx=5, pady=2, sticky=tk.W)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=10, column=0, columnspan=2, pady=15)

        def apply_settings():
            # Update labels
            self.axes_settings['x_label'] = x_label_var.get()
            self.axes_settings['y_label'] = y_label_var.get()

            # Parse X range (NMR: min is right, max is left)
            x_left_str = x_min_var.get().strip()  # MAX (left)
            x_right_str = x_max_var.get().strip()  # MIN (right)

            if x_left_str and x_right_str:
                try:
                    x_left = float(x_left_str)  # Max (left)
                    x_right = float(x_right_str)  # Min (right)

                    if x_left <= x_right:
                        messagebox.showerror("Error",
                                             "For NMR spectra, the left value (max ppm) must be greater than the right value (min ppm).")
                        return

                    # Store as (min, max) internally
                    self.axes_settings['x_lim'] = (x_right, x_left)  # (min, max)

                except ValueError:
                    messagebox.showerror("Error", "Invalid number format for X range.")
                    return
            else:
                self.axes_settings['x_lim'] = None

            # Parse Y range (normal: min < max)
            y_min_str = y_min_var.get().strip()
            y_max_str = y_max_var.get().strip()
            if y_min_str and y_max_str:
                try:
                    y_min = float(y_min_str)
                    y_max = float(y_max_str)
                    if y_min >= y_max:
                        messagebox.showerror("Error", "Y min must be less than Y max.")
                        return
                    self.axes_settings['y_lim'] = (y_min, y_max)
                except ValueError:
                    messagebox.showerror("Error", "Invalid number format for Y range.")
                    return
            else:
                self.axes_settings['y_lim'] = None

            self.apply_axes_settings()
            dialog.destroy()

        def reset_to_data():
            if self.ppm is not None and self.all_intensities is not None:
                x_min_var.set(f"{np.max(self.ppm):.4f}")  # Max ppm goes to left field
                x_max_var.set(f"{np.min(self.ppm):.4f}")  # Min ppm goes to right field
                y_min_var.set(f"{np.min(self.all_intensities):.4f}")
                y_max_var.set(f"{np.max(self.all_intensities):.4f}")

        ttk.Button(btn_frame, text="OK", command=apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Data Range", command=reset_to_data).pack(side=tk.LEFT, padx=5)

    def apply_axes_settings(self):
        if hasattr(self, 'ax'):
            self.ax.set_xlabel(self.axes_settings['x_label'], fontsize=self.plot_settings['font_size'])
            self.ax.set_ylabel(self.axes_settings['y_label'], fontsize=self.plot_settings['font_size'])

            if self.axes_settings['x_lim'] is not None:
                x_min, x_max = self.axes_settings['x_lim']
                self.ax.set_xlim(x_min, x_max)

            if self.axes_settings['y_lim'] is not None:
                self.ax.set_ylim(self.axes_settings['y_lim'])

            self.canvas.draw_idle()

    def toggle_legend(self):
        if hasattr(self, 'ax'):
            legend = self.ax.get_legend()
            if legend:
                legend.set_visible(self.show_legend.get())
                self.canvas.draw_idle()

    def set_background_color(self):
        color = colorchooser.askcolor(color=self.plot_settings['background'], title="Choose Background Color")
        if color[1]:
            self.plot_settings['background'] = color[1]
            self.ax.set_facecolor(color[1])
            self.figure.patch.set_facecolor(color[1])
            self.canvas.draw_idle()

    def set_grid_style(self, style):
        self.plot_settings['grid_style'] = style
        if hasattr(self, 'ax'):
            if style == 'none':
                self.ax.grid(False)
            else:
                self.ax.grid(True, linestyle=style, alpha=0.7, linewidth=0.5)
            self.canvas.draw_idle()

    def set_line_width(self, width):
        self.plot_settings['line_width'] = width
        self.update_plot()

    def set_transparency(self, alpha):
        self.plot_settings['transparency'] = alpha
        self.update_plot()

    def set_font_size(self, size):
        self.plot_settings['font_size'] = size
        if hasattr(self, 'ax'):
            self.ax.xaxis.label.set_size(size)
            self.ax.yaxis.label.set_size(size)
            self.ax.title.set_size(size + 1)
            self.ax.tick_params(labelsize=size - 1)
            self.canvas.draw_idle()

    def reset_plot_settings(self):
        self.plot_settings = {
            'background': '#FFFFFF',
            'grid_style': '--',
            'line_width': 0.5,
            'transparency': 0.8,
            'font_size': 10
        }
        if hasattr(self, 'ax'):
            self.ax.set_facecolor(self.plot_settings['background'])
            self.figure.patch.set_facecolor(self.plot_settings['background'])
            self.ax.grid(True, linestyle=self.plot_settings['grid_style'], alpha=0.7, linewidth=0.5)
            self.ax.xaxis.label.set_size(self.plot_settings['font_size'])
            self.ax.yaxis.label.set_size(self.plot_settings['font_size'])
            self.ax.tick_params(labelsize=self.plot_settings['font_size'] - 1)
            self.update_plot()
