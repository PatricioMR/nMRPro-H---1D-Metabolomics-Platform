# gui/notation/deconvolution/deconv_params_frame.py
"""
Bottom panel with deconvolution parameters and buttons.
"""
import tkinter as tk
from tkinter import ttk


class DeconvParamsFrame:
    def __init__(self, parent, controller, on_apply, on_save):
        self.parent = parent
        self.controller = controller
        self.on_apply = on_apply
        self.on_save = on_save

        self.create_widgets()

    def create_widgets(self):
        """Create parameters and buttons."""
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Parameters frame
        params_frame = ttk.LabelFrame(self.frame, text="Parameters", padding=5)
        params_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Min SNR
        ttk.Label(params_frame, text="Min SNR:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.min_snr_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params_frame, textvariable=self.min_snr_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        # Max peaks
        ttk.Label(params_frame, text="Max signals:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.max_peaks_var = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.max_peaks_var, width=8).grid(row=1, column=1,
                                                                                                 padx=5, pady=2)

        # Buttons frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(side=tk.RIGHT, padx=5)

        self.apply_btn = ttk.Button(btn_frame, text="Apply Deconvolution",
                                    command=self.on_apply, state="disabled")
        self.apply_btn.pack(side=tk.TOP, pady=2)

        self.save_btn = ttk.Button(btn_frame, text="Save Results",
                                   command=self.on_save, state="disabled")
        self.save_btn.pack(side=tk.TOP, pady=2)

    def get_params(self):
        """Return current parameters as dictionary."""
        return {
            'min_snr': self.min_snr_var.get(),
            'max_peaks': self.max_peaks_var.get()
        }
