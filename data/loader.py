# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# data/loader.py
import pandas as pd
import numpy as np


def detect_separator(filepath):
    """
    Detect automatically the separator of the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        elif ';' in first_line:
            return ';'
        elif ' ' in first_line:
            return ' '
        else:
            return '\t'  # default


def load_all_samples(filepath):
    """
    Load all samples from the file.
    Robust version with automatic separator detection.

    Returns:
        ppm: array of ppm values
        intensities: 2D array of intensities (rows=ppm, columns=samples)
        sample_names: list of sample names
        sample_labels: list of sample labels/groups
    """
    print(f"Loading all samples: {filepath}")

    try:
        # Detect separator
        separator = detect_separator(filepath)
        print(f"Detected separator: '{separator}'")

        # Read all lines
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("File too short. Must have at least 3 lines (2 headers + data).")

        # Process headers
        header_line1 = lines[0].strip().split(separator)
        header_line2 = lines[1].strip().split(separator)

        # Check number of columns
        if len(header_line1) < 2:
            raise ValueError(f"File must have at least 2 columns. Found: {len(header_line1)}")

        # Process data lines
        data_lines = lines[2:]

        ppm_list = []
        intensities_list = []

        for i, line in enumerate(data_lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split(separator)
            if len(parts) < 2:
                print(f"Line {i + 3} ignored: too few columns - {parts}")
                continue

            try:
                ppm_value = float(parts[0])
                ppm_list.append(ppm_value)

                intensities = []
                for j in range(1, len(parts)):
                    try:
                        val = float(parts[j]) if parts[j] else 0.0
                        intensities.append(val)
                    except ValueError:
                        print(f"Non-numeric value at line {i + 3}, column {j + 1}: '{parts[j]}' - using 0")
                        intensities.append(0.0)

                # Pad with zeros if needed
                while len(intensities) < len(header_line1) - 1:
                    intensities.append(0.0)

                intensities_list.append(intensities)

            except ValueError as e:
                print(f"Error converting line {i + 3}: {line} - {e}")
                continue

        if len(ppm_list) == 0:
            raise ValueError("No valid numeric data found in file.")

        ppm = np.array(ppm_list)
        intensities = np.array(intensities_list)

        # Sample names and labels
        sample_names = header_line1[1:] if len(header_line1) > 1 else []
        sample_labels = header_line2[1:] if len(header_line2) > 1 else []

        # Ensure consistent length
        while len(sample_names) < intensities.shape[1]:
            sample_names.append(f"Sample_{len(sample_names) + 1}")
        while len(sample_labels) < intensities.shape[1]:
            sample_labels.append("Unknown")

        return ppm, intensities, sample_names, sample_labels

    except Exception as e:
        print(f"Detailed error loading file: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_sample_info(filepath):
    """
    Return information about available samples.
    """
    print(f"Getting sample info: {filepath}")

    try:
        separator = detect_separator(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return []

        header_line1 = lines[0].strip().split(separator)
        header_line2 = lines[1].strip().split(separator) if len(lines) > 1 else []

        sample_names = header_line1[1:] if len(header_line1) > 1 else []
        sample_labels = header_line2[1:] if len(header_line2) > 1 else []

        # Ensure both lists have the same length
        while len(sample_labels) < len(sample_names):
            sample_labels.append("Unknown")

        samples_info = []
        for i, (name, label) in enumerate(zip(sample_names, sample_labels)):
            samples_info.append({
                'index': i,
                'name': name,
                'label': label,
                'type': 'BLK' if label.upper() == 'BLK' else 'Sample',
                'correction_factor': 1.0,
                'display': f"{name} ({label})"
            })

        return samples_info

    except Exception as e:
        print(f"Error getting sample info: {e}")
        return []