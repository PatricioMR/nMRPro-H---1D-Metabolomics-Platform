# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# plotting/spectrum.py
import matplotlib.pyplot as plt


def plot_spectrum(ax, ppm, intensity, title=None, color='black', label=None, show_legend=True, settings=None):
    """
    Plot NMR spectrum on the given axis.

    Args:
        ax: matplotlib axis
        ppm: array of ppm values
        intensity: array of intensity values
        title: plot title (optional)
        color: line color
        label: legend label (optional)
        show_legend: whether to show legend
        settings: dictionary with plot settings (background, grid_style, line_width, transparency, font_size)
    """
    if settings is None:
        settings = {'background': '#FFFFFF', 'grid_style': '--', 'line_width': 0.5,
                    'transparency': 0.8, 'font_size': 10}

    ax.clear()

    # Apply background color
    ax.set_facecolor(settings['background'])

    # Plot spectrum
    ax.plot(ppm, intensity, color=color, linewidth=settings['line_width'],
            alpha=settings['transparency'], label=label)

    # Set labels with font size
    ax.set_xlabel("δ (ppm)", fontsize=settings['font_size'])
    ax.set_ylabel("Intensity", fontsize=settings['font_size'])
    ax.invert_xaxis()  # NMR standard: decreasing ppm from left to right

    # Grid
    if settings['grid_style'] != 'none':
        ax.grid(True, linestyle=settings['grid_style'], alpha=0.7, linewidth=0.5)
    else:
        ax.grid(False)

    # Tick parameters
    ax.tick_params(labelsize=settings['font_size'] - 1)

    if title:
        ax.set_title(title, fontsize=settings['font_size'] + 1, fontweight='bold')

    if label and show_legend:
        legend = ax.legend(fontsize=settings['font_size'] - 1, framealpha=0.9)
        legend.set_draggable(True)  # Make legend draggable


def plot_multiple_spectra(ax, ppm, intensities_matrix, sample_names, sample_labels,
                          group_colors=None, show_legend=True, settings=None):
    """
    Plot multiple spectra on the same axis.

    Args:
        ax: matplotlib axis
        ppm: array of ppm values
        intensities_matrix: matrix of intensities (rows=ppm, columns=samples)
        sample_names: list of sample names
        sample_labels: list of sample labels/groups
        group_colors: dictionary mapping group names to colors
        show_legend: whether to show legend
        settings: dictionary with plot settings
    """
    if settings is None:
        settings = {'background': '#FFFFFF', 'grid_style': '--', 'line_width': 0.5,
                    'transparency': 0.8, 'font_size': 10}

    ax.clear()

    # Apply background color
    ax.set_facecolor(settings['background'])

    # Default colors if none provided
    if group_colors is None:
        group_colors = {
            'CTRL': '#2E86AB',
            'G1': '#A23B72',
            'G2': '#F18F01',
            'G3': '#17BEBB',
            'BLK': '#C73E1D',
            'Desconhecido': '#593C8F'
        }

    # Plot each spectrum
    for i in range(intensities_matrix.shape[1]):
        # Get color based on group
        group = sample_labels[i]
        color = group_colors.get(group, f'C{i}')

        label = f"{sample_names[i]} ({sample_labels[i]})"
        ax.plot(ppm, intensities_matrix[:, i], color=color,
                linewidth=settings['line_width'],
                alpha=settings['transparency'],
                label=label)

    # Set labels with font size
    ax.set_xlabel("δ (ppm)", fontsize=settings['font_size'])
    ax.set_ylabel("Intensity", fontsize=settings['font_size'])
    ax.invert_xaxis()

    # Grid
    if settings['grid_style'] != 'none':
        ax.grid(True, linestyle=settings['grid_style'], alpha=0.7, linewidth=0.5)
    else:
        ax.grid(False)

    # Tick parameters
    ax.tick_params(labelsize=settings['font_size'] - 1)

    # Create legend if requested
    if show_legend:
        # Legenda dentro do eixo (upper right) — não desaparece ao
        # redimensionar ou ao voltar do Quantification stage.
        # Apenas os grupos únicos aparecem (evita duplicados por amostra).
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            # Extrair só o grupo do label "Nome (Grupo)"
            grp = l.split('(')[-1].rstrip(')') if '(' in l else l
            if grp not in seen:
                seen[grp] = h
        legend = ax.legend(
            seen.values(), seen.keys(),
            loc='upper right',
            fontsize=settings['font_size'] - 1,
            framealpha=0.85,
            handlelength=1.2)
        legend.set_draggable(True)