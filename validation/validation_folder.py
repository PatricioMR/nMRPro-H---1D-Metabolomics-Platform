# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Where each bench writes inside the validation folder (the one folder that
holds every exported run: ``G:/My Drive/Mestrado/NMR/nMRPro-H_Validation``).
Every export takes that folder and writes only into its own subfolder, which
it replaces whole; the workbook, figure, version and LEIA-ME of a bench live
inside its subfolder."""
from __future__ import annotations

import os

SUBFOLDERS = {
    'quality': '01_Tabela_PDF',            # validation.export_validation
    'composition_600': '02_Composicoes_600MHz',  # validation.composition_bench --field 600
    'composition_500': '03_Composicoes_500MHz',  # validation.composition_bench --field 500
    'ablation': '04_Ablacao',              # validation.ablation csv --export-dir
    'steps': '05_Etapas_quantificacao',    # validation.quant_steps export
    'external': '06_Comparacao_externa',   # comparisons with other software (by hand)
    'documents': '07_Documentos',          # methods and manuscript material (by hand)
    'archive': '99_Arquivo',               # earlier versions' exports
}


def subfolder(dest: str, key: str) -> str:
    """The path of bench ``key`` inside the validation folder ``dest``."""
    return os.path.join(dest, SUBFOLDERS[key])
