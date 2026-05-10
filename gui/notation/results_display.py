# gui/notation/results_display.py
"""
Results display module.
"""
import tkinter as tk
from tkinter import ttk


class ResultsDisplay:
    """Handles display of search and deconvolution results."""

    def __init__(self, app):
        self.app = app
        print("✅ ResultsDisplay initialized")

    def show_deconv_results(self, parent, results):
        """Show deconvolution results in a new window."""
        win = tk.Toplevel(parent)
        win.title("Deconvolution Results")
        win.geometry("600x400")

        main_frame = ttk.Frame(win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Summary
        summary = f"Region: {results['region_ppm'][0]:.3f} - {results['region_ppm'][-1]:.3f} ppm\n"
        summary += f"Peaks found: {len(results['peaks'])}\n"
        summary += f"BIC: {results['bic']:.1f}"

        ttk.Label(main_frame, text=summary, font=("Arial", 10)).pack(anchor=tk.W, pady=5)

        # Create treeview
        columns = ('#', 'ppm', 'height', 'width', 'snr', 'bic')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=10)

        tree.heading('#', text='#')
        tree.heading('ppm', text='PPM')
        tree.heading('height', text='Height')
        tree.heading('width', text='Width')
        tree.heading('snr', text='SNR')
        tree.heading('bic', text='BIC')

        for i, p in enumerate(results['peaks'], 1):
            tree.insert('', tk.END, values=(
                i,
                f"{p['ppm']:.4f}",
                f"{p['height']:.2f}",
                f"{p['width']:.4f}",
                f"{p['snr']:.1f}",
                f"{p['bic']:.1f}"
            ))

        tree.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Button(main_frame, text="Close", command=win.destroy).pack(pady=5)

        # Center window
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f'{width}x{height}+{x}+{y}')