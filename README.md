# HIV Vaccine BCR Reanalysis

Independent computational reanalysis of published IAVI
G002 B-cell receptor repertoire data.

## Research question

How do participant-balanced longitudinal IGHV1-2*02
CDRH2 lineage-modal-state transition profiles differ
between the two G002 immunization regimens?

## Study design

Primary analysis:
- 15 participants
- 332 prospectively matched lineages
- Week-8 to Week-24 comparison
- Participant-level statistical inference
- Sampling and probe-ascertainment sensitivities

Secondary analysis:
- Group-2 Week-24 lineage-detection benchmark
- 2,739 Week-8 lineages from 17 participants
- Biological baseline versus IgBERT and AbLang-2

The primary analysis is exploratory. Observed lineage
modal-state changes do not establish mutation timing,
fixation, antibody function, or vaccine causality.

## Project structure

scripts/         Source analysis and figure scripts
data/external/   Original-data acquisition instructions
data/processed/  Derived-data documentation
results/         Frozen summary-output location
figures/         Manuscript figure copies
docs/            Workflow, provenance and review notes
MANIFEST.tsv     SHA256 file inventory

## Source data

Willis et al., Science (2025).
DOI: 10.1126/science.adr8382

Original investigators' repository:
https://github.com/SchiefLab/G002-and-G003

Original code archive:
https://doi.org/10.5281/zenodo.15425848

## Reproducibility status

This repository is a private publication-release draft.

The copied scripts and figure files are checksum verified.
End-to-end reproducibility is not yet certified.
See REPRODUCE.md and docs/RELEASE_CHECKLIST.md.

No original participant-level source data are bundled.

## Licensing

The independently authored Python analysis scripts in
`scripts/` are licensed under the BSD 2-Clause License.
See `LICENSE`.

The original G002 data and SchiefLab source software
remain subject to their own applicable terms.

The license does not automatically authorize
redistribution of third-party data, models, or figures.
The repository remains a private release draft.
