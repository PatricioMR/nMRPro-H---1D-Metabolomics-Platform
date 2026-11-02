# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""One short line per reason, for the quantification table's tooltip.

The engine's reasons (``_status_reasons``) are written to be audited: what
was measured, against what, and what the bench says about rows of that
kind -- a median 394 characters per row on a real study, up to 2 000. On
hover that is a wall. The tooltip says what decided the row, one line per
reason, the deciding one first; the full text stays in the row, the
quantification log and the report.
"""
from __future__ import annotations

import re

MAX_LINES = 4

# (pattern on the full reason, short line). The first match wins; the
# groups of the pattern fill the line. A reason no pattern knows is cut at
# its first clause.
_SHORT = (
    (r'Not quantified: lost the competition',
     'Lost its peaks to analytes with better evidence.'),
    (r'Not quantified: the joint fit with (\d+) overlapping',
     'Joint fit gave it zero: the shapes of {0} overlapping analytes '
     'explain its region.'),
    (r'Not quantified: its signals overlap others, and the joint fit.*?'
     r'listed at ([\d., ]+?) ppm',
     'Joint fit held its signal at {0} ppm at the edge of its reach: '
     'mark it Good to quantify it anyway (one dot).'),
    (r'Not quantified: its signals overlap others, and the joint fit.*?'
     r'centres on a signal at ([\d.]+-[\d.]+) ppm',
     'Joint fit centred on its signal at {0} ppm, not the elected one.'),
    (r'Not quantified: its signals overlap others, and the joint fit',
     'Joint fit could not place its shape on its peak.'),
    (r'Not quantified: complex multiplet \(m\*\)',
     'Complex multiplet (m*): no free peak confirms it.'),
    (r'Not quantified: a singlet on its own.*?SNR ([\d.]+)',
     'Lone singlet (SNR {0}): no other signal confirms it.'),
    (r'Not quantified: a singlet on its own',
     'Lone singlet: no other signal confirms it.'),
    (r'Not quantified: the amount this signal implies is contradicted',
     'Its other signals are missing at the amount this peak implies.'),
    (r'Not quantified: ambiguous identity',
     'Ties with another candidate for the same peak: mark one Good.'),
    (r'Not quantified: every signal .*? sits (\d+) mppm from the drift',
     'Its peaks sit {0} mppm off the study\'s shift.'),
    (r'Identified by the cascade but not quantified',
     'Found by the cascade; its signals do not agree on one amount.'),
    (r'Lost peak (\S+) to (.+?) on (.+?)(?: \(|\.|$)',
     'Lost peak {0} to {1} ({2}).'),
    (r'Added by the cascade \(step (\d+)\)',
     'Added by the cascade (step {0}).'),
    (r'Won its peak as a complex multiplet',
     'Complex multiplet (m*): one dot.'),
    (r'Measured by fitting its shape together with (\d+) other',
     'Joint fit with {0} overlapping analytes.'),
    (r'Upper bound: quantified from its fitted shape .*?explains (\d+)%',
     'Upper bound: its shape explains {0}% of a shared region.'),
    (r'Quantified from its fitted shape, which explains (\d+)%',
     'Shape fit explains {0}% of its region.'),
    (r'Shared window resolved by shape with (.+?):',
     'Shared window, split by shape with {0}.'),
    (r'Shared window, not split',
     'Shared window, not split: an upper bound.'),
    (r'Its share of the shared window could not be resolved',
     'Shared window could not be split.'),
    (r'The accepted shapes leave (\d+)% of the window unexplained',
     '{0}% of its window is unexplained.'),
    (r'The band flanking this window reads (\d+)x',
     'Crowded surroundings ({0}x noise): check the region.'),
    (r'Most of this analyte\'s signal lies where .*?\((\d+)%\)',
     '{0}% of its signal lies under quantified analytes: an estimate.'),
    (r'Identified by the remainder of',
     'Identified from what a shared window left unexplained.'),
    (r'(?:The signal this amount was measured on|The peak this analyte '
     r'was matched to) stands ([\d.]+) mppm',
     '{0} mppm off the study\'s shift: two dots at most.'),
)
_SHORT = tuple((re.compile(p, re.S), t) for p, t in _SHORT)

# Detail of another reason, said on its own line: the analytes the window
# was offered to, where a neighbour was placed. Left to the full text.
_DETAIL = re.compile(
    r'^(The remainder was offered to|.+ was placed where annotation expects'
    r'|.+: a shape this signal\'s own shape makes|.+ \(listed\): the '
    r'reference lists it)', re.S)

_PREFIX = 'Before the cascade: '
_NAMES_SHOWN = 3


def _names(text: str) -> str:
    names = [n.strip() for n in text.split(',') if n.strip()]
    more = len(names) - _NAMES_SHOWN
    return ', '.join(names[:_NAMES_SHOWN]) + (f' +{more}' if more > 0 else '')


def short_reason(reason: str) -> str | None:
    """The tooltip line for one engine reason, or None for a detail."""
    text = (reason or '').strip()
    before = text.startswith(_PREFIX)
    if before:
        text = text[len(_PREFIX):]
    if _DETAIL.match(text):
        return None
    for pat, tpl in _SHORT:
        m = pat.search(text)
        if m:
            parts = [_names(g) if ',' in g else g for g in m.groups()]
            line = tpl.format(*parts)
            break
    else:
        head = re.split(r' — |: | -- ', text, maxsplit=1)[0]
        line = head if len(head) <= 90 else head[:87].rstrip() + '…'
    return f'Before the cascade: {line}' if before else line


_DECISIVE = ('Not quantified', 'Identified by the cascade but not quantified')


def short_reasons(reasons, limit: int = MAX_LINES, *,
                  not_quantified: bool = False) -> list[str]:
    """Short lines for a row's reasons: the one that decided it first (a
    "Not quantified" reason), duplicates and details left out, at most
    ``limit`` lines and a count of the rest.

    ``not_quantified`` rows whose reasons never say why -- the joint fit's
    zero, before it was named -- open with that they have no amount."""
    reasons = [str(r) for r in reasons or ()]
    lines = []
    if not_quantified and not any(r.startswith(_DECISIVE) for r in reasons):
        lines.append('Not quantified: no amount in any sample.')
    for r in sorted(reasons, key=lambda s: not s.startswith(_DECISIVE)):
        line = short_reason(r)
        if line and line not in lines:
            lines.append(line)
    if len(lines) > limit:
        rest = len(lines) - (limit - 1)
        lines = lines[:limit - 1] + [f'(+{rest} more)']
    return lines
