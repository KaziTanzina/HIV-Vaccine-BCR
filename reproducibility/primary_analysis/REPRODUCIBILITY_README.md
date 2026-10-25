# HIV vaccine BCR reanalysis — reproducibility package

STATUS: PRIVATE PREPUBLICATION REVIEW.

Authors:
Kazi Tanzina Begum
Kazi Abdul Gofur Rafea

Computational code developer:
Kazi Tanzina Begum.

The research was independently developed without
external grant funding.

## Primary scientific analysis

Original study:
Willis et al., Science (2025)
DOI: 10.1126/science.adr8382

Original source repository:
https://github.com/SchiefLab/G002-and-G003

Original archived source:
https://doi.org/10.5281/zenodo.15425848

Primary analysis:
Participant-balanced comparison of observed
Week-8 to Week-24 IGHV1-2*02 CDRH2 positional
lineage-modal-state acquisitions.

Participants: 15
Group 1: 7
Group 2: 8

CDRH2 positions:
51, 52, 53, 54, 55, 57, 58

## Existing computational dependencies

Original G002 final_df.feather
    |
Week-8 prospective reclustering
    |
Longitudinal cell/lineage matching (script 156)
    |
Mutation validation (script 171)
    |
Transition construction (script 173)
    |
Participant profile (script 177)
    |
Global centering (script 178)
    |
Regional analysis (script 189)
    |
Position sensitivity (script 191)
    |
Figure 1A participant plot reconstruction

Week-8 reclustering script 152:
Present in staged scripts.

The frozen intermediate inputs are available in
the research workspace but are not included here.

The exact source-to-output execution chain has
not yet been independently tested from a clean
public-data download.

Some steps may require additional upstream
intermediates beyond the scripts staged here.

## Centering

The participant-level profile contains 65
complete-case IGHV1-2*02 positions.

Global centering subtracts each participant's
mean across those positions before calculating
the participant-balanced group differences.

The centered contrasts were independently
verified against frozen tables 201 and 212.

See:
documentation/CENTERING_PROVENANCE.md

## Figure 1A

The reconstruction script uses frozen participant
profile table 198.

It reproduces raw participant fractions and group
means, but its output is not certified as a
pixel-identical reconstruction of the manuscript
figure.

The original manuscript figure is unchanged.

## Data and distribution restrictions

No original third-party sequence-level dataset
or participant-level source table is included.

The publication permissions and disclosure risk
for processed data remain to be reviewed.

This is not an authorized public data release.

## Code release status

These are selected source scripts from the
independent research project, not the complete
historical execution environment.

They have passed Python syntax inspection but
have not been validated as a clean-room pipeline
in this staging step.

The antibody-language-model benchmarks require
additional model-specific resources and are not
reproduced by these primary-analysis scripts.

No GitHub publication or release has occurred.
