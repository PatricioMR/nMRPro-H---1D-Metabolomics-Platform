# gui/widgets/filtered_results_window.py
"""
Results window with filtering options before export.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import csv


class FilteredResultsWindow:
    """
    Window to display and filter database results before export.
    """

    def __init__(self, parent, results, callback_export, title="HMDB Results"):
        """
        Args:
            parent: parent window
            results: list of match dictionaries
            callback_export: function to call with filtered results for export
            title: window title
        """
        self.parent = parent
        self.all_results = results
        self.filtered_results = results.copy()
        self.callback_export = callback_export

        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1200x700")
        self.window.transient(parent)

        self.create_widgets()
        self.update_display()

        # Center window
        self.center_window()

    def center_window(self):
        """Center window on screen."""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        """Create all widgets."""
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== FILTER SECTION ==========
        filter_frame = ttk.LabelFrame(main_frame, text="🔍 Export Filters", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)

        # Filter row 1
        row1 = ttk.Frame(filter_frame)
        row1.pack(fill=tk.X, pady=2)

        # Confidence level filter
        ttk.Label(row1, text="Confidence Level:").pack(side=tk.LEFT, padx=5)
        self.level_var = tk.StringVar(value="all")
        ttk.Radiobutton(row1, text="All Levels", variable=self.level_var,
                        value="all", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row1, text="Level 1 Only (Unique)", variable=self.level_var,
                        value="level1", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row1, text="Level 2 Only (Overlap)", variable=self.level_var,
                        value="level2", command=self.apply_filters).pack(side=tk.LEFT, padx=2)

        # Filter row 2
        row2 = ttk.Frame(filter_frame)
        row2.pack(fill=tk.X, pady=2)

        # Region type filter
        ttk.Label(row2, text="Region Type:").pack(side=tk.LEFT, padx=5)
        self.region_var = tk.StringVar(value="all")
        ttk.Radiobutton(row2, text="All Regions", variable=self.region_var,
                        value="all", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row2, text="Unique Regions Only", variable=self.region_var,
                        value="unique", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row2, text="Overlapping Regions Only", variable=self.region_var,
                        value="overlap", command=self.apply_filters).pack(side=tk.LEFT, padx=2)

        # Filter row 3
        row3 = ttk.Frame(filter_frame)
        row3.pack(fill=tk.X, pady=2)

        # Match type filter
        ttk.Label(row3, text="Match Type:").pack(side=tk.LEFT, padx=5)
        self.match_var = tk.StringVar(value="all")
        ttk.Radiobutton(row3, text="All Types", variable=self.match_var,
                        value="all", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row3, text="Multiplets Only", variable=self.match_var,
                        value="multiplet", command=self.apply_filters).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row3, text="Signals Only", variable=self.match_var,
                        value="peak", command=self.apply_filters).pack(side=tk.LEFT, padx=2)

        # Filter row 4 - Summary
        row4 = ttk.Frame(filter_frame)
        row4.pack(fill=tk.X, pady=5)

        self.summary_label = ttk.Label(row4, text="", font=("Arial", 9, "bold"))
        self.summary_label.pack(side=tk.LEFT, padx=5)

        # Reset filters button
        ttk.Button(row4, text="↺ Reset Filters", command=self.reset_filters).pack(side=tk.RIGHT, padx=5)

        # ========== RESULTS SECTION ==========
        results_frame = ttk.LabelFrame(main_frame, text="📋 Filtered Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create Treeview
        self.create_treeview(results_frame)

        # ========== EXPORT SECTION ==========
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=5)

        # Export button
        ttk.Button(export_frame, text="💾 Export Filtered Results",
                   command=self.export_filtered).pack(side=tk.LEFT, padx=5)

        # Export all button
        ttk.Button(export_frame, text="📦 Export All Results",
                   command=self.export_all).pack(side=tk.LEFT, padx=5)

        # Close button
        ttk.Button(export_frame, text="❌ Close",
                   command=self.window.destroy).pack(side=tk.RIGHT, padx=5)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status_label.pack(anchor=tk.W, pady=2)

    def create_treeview(self, parent):
        """Create the results treeview."""
        # Scrollbars
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL)

        # Columns
        columns = ('hmdb_id', 'name', 'match_type', 'level', 'ppm', 'protons',
                   'type', 'interval', 'overlaps', 'peak_ppm', 'peak_rank')

        self.tree = ttk.Treeview(
            parent,
            columns=columns,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            height=15
        )

        # Configure headings
        self.tree.heading('hmdb_id', text='HMDB ID')
        self.tree.heading('name', text='Metabolite')
        self.tree.heading('match_type', text='Type')
        self.tree.heading('level', text='Level')
        self.tree.heading('ppm', text='δ (ppm)')
        self.tree.heading('protons', text='H')
        self.tree.heading('type', text='Multiplicity')
        self.tree.heading('interval', text='HMDB Interval')
        self.tree.heading('overlaps', text='Overlaps')
        self.tree.heading('peak_ppm', text='Signal δ (ppm)')
        self.tree.heading('peak_rank', text='Signal #')

        # Column widths
        self.tree.column('hmdb_id', width=100)
        self.tree.column('name', width=150)
        self.tree.column('match_type', width=70)
        self.tree.column('level', width=60)
        self.tree.column('ppm', width=80)
        self.tree.column('protons', width=40)
        self.tree.column('type', width=80)
        self.tree.column('interval', width=150)
        self.tree.column('overlaps', width=60)
        self.tree.column('peak_ppm', width=80)
        self.tree.column('peak_rank', width=60)

        # Configure scrollbars
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Configure tags
        self.tree.tag_configure('level1', background='#e6ffe6')  # Light green
        self.tree.tag_configure('level2', background='#fff3e6')  # Light orange
        self.tree.tag_configure('level3', background='#ffe6e6')  # Light red

    def apply_filters(self):
        """Apply selected filters to results."""
        self.filtered_results = []

        for r in self.all_results:
            include = True

            # Filter by confidence level
            overlaps = r.get('overlap_count', 0)
            if self.level_var.get() == 'level1' and overlaps != 0:
                include = False
            elif self.level_var.get() == 'level2' and overlaps == 0:
                include = False

            # Filter by region type
            if self.region_var.get() == 'unique' and overlaps != 0:
                include = False
            elif self.region_var.get() == 'overlap' and overlaps == 0:
                include = False

            # Filter by match type
            match_type = r.get('match_type', '')
            if self.match_var.get() != 'all' and match_type != self.match_var.get():
                include = False

            if include:
                self.filtered_results.append(r)

        self.update_display()

    def reset_filters(self):
        """Reset all filters to default."""
        self.level_var.set('all')
        self.region_var.set('all')
        self.match_var.set('all')
        self.filtered_results = self.all_results.copy()
        self.update_display()

    def update_display(self):
        """Update the treeview with filtered results."""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Calculate statistics
        total = len(self.all_results)
        filtered = len(self.filtered_results)
        level1 = sum(1 for r in self.all_results if r.get('overlap_count', 0) == 0)
        level2 = sum(1 for r in self.all_results if r.get('overlap_count', 0) > 0)

        # Update summary
        self.summary_label.config(
            text=f"Showing {filtered} of {total} matches | "
                 f"Level 1 (Unique): {level1} | Level 2 (Overlap): {level2}"
        )

        # Add filtered results to tree
        for r in self.filtered_results:
            overlaps = r.get('overlap_count', 0)

            # Determine level
            if overlaps == 0:
                level = "Level 1"
                tag = 'level1'
            else:
                level = "Level 2"
                tag = 'level2'

            # Format values based on match type
            if r.get('match_type') == 'multiplet':
                interval = f"{r.get('ppm_start', 0):.3f} - {r.get('ppm_end', 0):.3f}"
                ppm = f"{r.get('ppm_center', 0):.3f}"
                protons = r.get('protons', '')
                mult_type = r.get('type', '')
            else:
                interval = ''
                ppm = f"{r.get('ppm', 0):.3f}"
                protons = ''
                mult_type = 'peak'

            self.tree.insert('', tk.END, values=(
                r.get('hmdb_id', ''),
                r.get('name', ''),
                r.get('match_type', ''),
                level,
                ppm,
                protons,
                mult_type,
                interval,
                overlaps,
                f"{r.get('peak_ppm', 0):.3f}",
                r.get('peak_rank', '')
            ), tags=(tag,))

        self.status_label.config(text=f"Displaying {filtered} results")

    def export_filtered(self):
        """Export only filtered results."""
        if not self.filtered_results:
            messagebox.showwarning("Export", "No filtered results to export!")
            return

        self.callback_export(self.filtered_results)

    def export_all(self):
        """Export all results."""
        self.callback_export(self.all_results)