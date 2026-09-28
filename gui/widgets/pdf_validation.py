"""The report's validation page: what the confidence levels were measured to
mean, at the field the project was acquired at or the nearest one measured.

The figures come from ``data/quality_reference.json`` (written by
``validation.quality_reference``); nothing here is computed from the
user's own data. The page says so, says what the bench was, and says what
it cannot tell the reader.
"""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from data.quality_reference import for_field

_DARK = colors.HexColor('#1a3a5c')
_LIGHT = colors.HexColor('#f5f5f5')
_GRID = colors.HexColor('#bdbdbd')
_NOTE = colors.HexColor('#fff8e1')

DOTS = {3: '●●●', 2: '●●○', 1: '●○○'}

MEANING = {
    3: 'Identity confirmed by the compound’s own signals: a multiplet '
       'complete with its coupling inside the gate, or its signals agreeing '
       'on one amount in the ratio of their protons; nothing of its listing '
       'missing at that amount.',
    2: 'Identity supported, not confirmed line by line: most of its listing '
       'covered, a singlet moving with the study’s drift, a cascade '
       'identification whose signals agree, or an amount taken from what a '
       'neighbour’s window left.',
    1: 'A candidate: identified by shape alone, a single listed signal with '
       'nothing to corroborate it, a signal of its listing missing at the '
       'measured amount, or its measured signal off where this study puts '
       'its compounds. Check before use.',
}


def _pct(v):
    return '—' if v is None else f'{100 * v:.0f}%'


def _err(d):
    if d.get('error_median') is None:
        return '—'
    return f"{100 * d['error_median']:.0f}% [{100 * d['error_p90']:.0f}%]"


def _table(rows, widths, first_number_col=1):
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), _DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.3, _GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (first_number_col, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def field_note(project_mhz, measured, exact) -> str:
    """What the reader should make of a validation measured at ``measured``
    MHz for a project at ``project_mhz``."""
    if exact:
        return (f'This project was acquired at {project_mhz:.0f} MHz, the '
                f'field this validation was measured at.')
    if project_mhz and project_mhz > measured:
        return (f'This project was acquired at {project_mhz:.0f} MHz; the '
                f'validation was measured at {measured} MHz. A higher field '
                f'separates lines further, so these figures are expected to '
                f'be conservative for this project — that has not been '
                f'measured.')
    if project_mhz:
        return (f'This project was acquired at {project_mhz:.0f} MHz; the '
                f'validation was measured at {measured} MHz. A lower field '
                f'overlaps lines more, so these figures may be optimistic for '
                f'this project.')
    return (f'The acquisition field of this project is not recorded; the '
            f'validation was measured at {measured} MHz.')


def validation_section(project_mhz, styles, avail_w, doc=None) -> list:
    """Flowables for the validation page; ``[]`` when nothing is measured."""
    measured, entry, exact = for_field(project_mhz, doc)
    if entry is None:
        return []
    tubes = sorted(entry['tubes'].items(), key=lambda t: int(t[0]))
    names = [f'{k} analytes' for k, _v in tubes]
    d = entry.get('design', {})
    out = [Paragraph(f'Validation of the confidence levels '
                     f'({measured} MHz)', styles['h2'])]
    out.append(Paragraph(
        'Every quantified compound in this report carries one to three dots. '
        'The dots are not a score out of three: each level was measured on '
        'synthetic mixtures of known composition, analysed with the same '
        'settings as this report. The tables give, for each level, how often '
        'a compound reported at it was really present (<b>identity '
        'precision</b>) and how far its amount was from the true one '
        '(<b>amount error</b>, median [90th percentile]). Mixtures of 18, 50 '
        'and 100 compounds show how both change as a spectrum fills.',
        styles['small']))
    out.append(Spacer(1, 0.2 * cm))

    uni = styles.get('cell_uni', styles['small'])
    cell = styles.get('cell', styles['small'])
    head = ['Level', 'What it means'] + [
        f'Identity precision\n{n}' for n in names]
    rows = [head]
    for lv in (3, 2, 1):
        rows.append([Paragraph(DOTS[lv], uni), Paragraph(MEANING[lv], cell)]
                    + [_pct(t['levels'][str(lv)]['precision'])
                       for _k, t in tubes])
    w0, w1 = 1.3 * cm, avail_w * 0.42
    rest = (avail_w - w0 - w1) / max(len(tubes), 1)
    out.append(_table(rows, [w0, w1] + [rest] * len(tubes),
                      first_number_col=2))
    out.append(Spacer(1, 0.25 * cm))

    rows = [['Level'] + [f'Amount error\n{n}' for n in names]]
    for lv in (3, 2, 1):
        rows.append([Paragraph(DOTS[lv], uni)]
                    + [_err(t['levels'][str(lv)]) for _k, t in tubes])
    rest = (avail_w - w0) / max(len(tubes), 1)
    out.append(_table(rows, [w0] + [rest] * len(tubes)))
    out.append(Spacer(1, 0.25 * cm))

    rows = [['Mixture', 'Compounds\nfound', 'Found at\n●● or more',
             'Precision at\n●● or more', 'False rows\nper analysis',
             'Amount error,\nall found']]
    for k, t in tubes:
        rows.append([f'{k} analytes', _pct(t['found']), _pct(t['recall_2plus']),
                     _pct(t['precision_2plus']), f"{t['false_per_run']:.1f}",
                     _err(t)])
    out.append(_table(rows, [avail_w * f for f in
                             (0.14, 0.15, 0.17, 0.18, 0.16, 0.20)]))
    out.append(Spacer(1, 0.25 * cm))

    note = Table([[Paragraph(field_note(project_mhz, measured, exact),
                             styles['small'])]], colWidths=[avail_w])
    note.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), _NOTE),
                              ('BOX', (0, 0), (-1, -1), 0.4, _GRID),
                              ('LEFTPADDING', (0, 0), (-1, -1), 6),
                              ('RIGHTPADDING', (0, 0), (-1, -1), 6)]))
    out.append(note)
    out.append(Spacer(1, 0.2 * cm))

    conc = d.get('concentration_range_mm') or ['?', '?']
    out.append(Paragraph(
        f"<b>How it was measured.</b> {d.get('samples', '?')} spectra in "
        f"{d.get('groups', '?')} groups at {float(d.get('field_mhz', measured)):g} MHz, "
        f"linewidth {d.get('linewidth_hz', '?')} Hz, noise set by the "
        f"{d.get('noise', '?')}, internal standard "
        f"{d.get('internal_standard', '?')}. The {d.get('reference_analytes', '?')} "
        f"compounds of a mouse skin extract, then compounds drawn at random "
        f"from the reference database at {conc[0]}–{conc[1]} mM "
        f"(log-uniform), each drawn {d.get('seeds', '?')} times; figures "
        f"pooled over the draws. Software {entry.get('software', '?')}, "
        f"database {entry.get('database', '?')}, measured "
        f"{entry.get('date', '?')}.", styles['small']))
    out.append(Spacer(1, 0.1 * cm))
    out.append(Paragraph(
        '<b>What it cannot tell.</b> The mixtures were built from the '
        'reference database’s own measured line shapes, so a compound '
        'whose real spectrum differs from its reference entry, a matrix '
        'effect, an uneven baseline or phase error is not in these figures: '
        'on real samples identity precision can only be lower. The figures '
        'describe the method, not any single compound of this report.',
        styles['small']))
    return out
