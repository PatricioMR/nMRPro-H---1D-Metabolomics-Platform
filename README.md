# nMRPipe

Desktop application (Python / Tkinter) for 1H NMR analysis in metabolomics and
lipidomics: spectrum import, peak detection, metabolite annotation against
HMDB/BMRB databases, overlap deconvolution (**JG-Deconv**), and quantification
with between-group statistics and PDF reporting.

---

## Features

- Multi-sample spectrum import (CSV/TXT) with blank (BLK) identification.
- Peak detection with baseline correction and SNR estimation.
- Metabolite annotation against local SQLite databases (per-biofluid HMDB, BMRB).
- **JG-Deconv** overlap deconvolution for shared peaks.
- Quantification with proton correction, normalization, and overlap correction.
- Between-group statistics and PDF report export.
- Saveable projects with auto-save every 5 minutes.

---

## Architecture

```
main.py            entry point
gui/               Tkinter UI and workflow stages
  stages/          visualization, peak_detection, annotation,
                   j_resolve (JG-Deconv), quantification, statistics, MCR
  widgets/         auxiliary windows (results, PDF, pathways, …)
data/              loaders, DB access (HMDB/BMRB), external API
processing/        DSP: baseline, detection, alignment, integration, quality
plotting/          spectrum rendering
controllers/       business logic / orchestration
utils/             helpers, async workers
databases/         SQLite databases (read-only at runtime)
tests/             unit / integration / smoke (pytest)
```

Scientific conventions: the δ (ppm) axis is **always inverted**; units are shown
on every plot; parsing/detection errors are surfaced to the user, never
silently swallowed.

---

## Workflow

1. Import spectra, define groups/samples.
2. Detect peaks (baseline + SNR).
3. Annotate against the biofluid database.
4. JG-Deconv — resolve overlapping peaks, apply fractions to quantification.
5. Quantify (with optional JG-Deconv correction).
6. Statistics + report export.

---

## How it works

### Annotation

Annotation matches detected peaks to database entries and assigns a confidence
to each candidate.

1. **Predicted multiplet pattern.** For every database entry (chemical shift,
   multiplicity, J-coupling, proton count) the expected line positions and
   relative intensities are built — from explicit DB line positions when
   available, otherwise from Pascal coefficients with line spacing
   `J(Hz) / spectrometer(MHz)`. The field strength comes from the value the
   user sets in the Annotation stage (no hard-coded fallback).

2. **Per-candidate scoring.** Each candidate combines several independent
   signals into a `composite_score` (0–1):
   - **chemical-shift score** — proximity of the detected peak to the DB shift,
     within a tolerance window;
   - **multiplet score** — how well the observed line pattern (positions and
     intensity ratios) matches the predicted multiplet;
   - **J agreement (ΔJ)** — difference between the measured and database
     J-coupling; a multiplet is *J-validated* only when ΔJ was actually
     measured and is within tolerance (default 0.3 Hz), which prevents
     unconfirmed candidates from anchoring matches in crowded spectra;
   - **reverse coverage** — proton-weighted fraction of the metabolite's
     expected lines that are actually observed.

3. **Quality assignment.** The enriched `composite_score` drives the
   annotation quality: candidates above the *good* threshold (0.80) are
   accepted, those below the *bad* threshold (0.50) are rejected, and
   competition between candidates at the same peak is resolved by score.
   The metabolite-level score is the maximum across its matched peaks — the
   same value shown downstream in Quantification.

### JG-Deconv (J-Guided Overlap Deconvolution)

When several metabolites share a peak, JG-Deconv estimates each one's
fractional contribution so the area can be split correctly. Signal areas are
linear and additive, which makes subtraction and least-squares valid.

1. **Anchors (clean concentrations).** A concentration anchor is taken only
   from a peak unique to one analyte: `conc = area / effective_nH`, where
   `effective_nH` is the proton count weighted by the integrated fraction of
   the multiplet inside the detected window. Weak single-line probes
   (`eff_nH < 0.45`) are rejected; with ≥2 probes a fold-range > 2.5 flags the
   anchor as low-confidence. A bootstrap CI is reported.

2. **Contribution subtraction.** For each shared peak, known contributions
   (`conc × eff_nH`) are subtracted from the observed area. Mass is conserved:
   if the known contributions exceed the observed area they are renormalized to
   100 % and the peak is flagged as over-predicted.

3. **Iterative cascade.** A shared peak left with a single unknown becomes a new
   anchor (`residual_area / eff_nH`); estimates accumulate and the median is
   used (order-independent, convergent). The cascade repeats until no peak can
   be further resolved.

4. **Global NNLS (optional).** A non-negative least-squares system
   `A·x = b` — where `A[i,j]` is the effective_nH of metabolite *j* at peak *i*
   and `b` the observed areas minus known contributions — solves remaining
   overlaps simultaneously. Quality thresholds (coverage, cross-validation CV)
   are auto-calibrated from the anchor metabolites rather than fixed constants.

5. **Per-sample fractions.** The *structure* of the solution (who anchors what)
   is fixed on the mean spectrum, but magnitudes are recomputed per sample, so
   the fractions track real group differences instead of being frozen at the
   mean ratio. Mass is conserved per peak per sample.

Outputs are fractional areas `{peak: {metabolite: fraction}}` (scalar and
per-sample) that Quantification consumes when overlap correction is enabled.

### Quantification

Quantification turns areas into concentrations per sample and group.

1. **Integration** of each metabolite's region, in one of several modes:
   trapezoidal, whole-multiplet (integrates the full DB multiplet span), or
   Lorentzian fitting (per-sample line fit).

2. **Clean-line selection** for multiplets: individual lines contaminated by
   overlap are detected (via JG-Deconv own-fraction when available — clean
   ≥ 0.70, marginal ≥ 0.40 — or a robust intensity-ratio deviation otherwise)
   and excluded; the remaining clean area is scaled back to the full-multiplet
   equivalent.

3. **Proton correction** (`area / nH`) and **normalization**: internal standard
   (IS), PQN, or both.

4. **Overlap correction.** When enabled, the per-sample JG-Deconv fraction
   multiplies each shared peak's area so only the analyte's share is counted.

5. **Reporting.** SNR and uncertainty (CV by group, confidence bars) are always
   reported; detection is per-sample, so a metabolite absent in a sample is not
   fabricated by interpolation.

---

## Requirements

- Python 3.12+ (developed and tested on 3.14)
- Dependencies:

```bash
pip install numpy scipy matplotlib pandas reportlab requests pillow
```

`sqlite3` and `tkinter` ship with the standard CPython installation.

### Databases

Active SQLite databases live in `databases/` and are **read-only** at runtime.
Schemas are contracts — changes require explicit migration; queries are always
parameterized.

---

## Run

```bash
python main.py
```

---

## Tests

```bash
pytest
```

`processing/` logic is covered by unit tests in `tests/unit/`; full pipelines
(load → process → annotate) live in `tests/integration/` and run against a real
test database (not mocked).
