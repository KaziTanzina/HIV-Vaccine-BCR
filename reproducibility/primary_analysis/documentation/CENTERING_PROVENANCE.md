# Figure 1A — Centering provenance

Status: PRIVATE reproducibility documentation.

Verified source: participant-level frozen table 198.

Participant count: 15.
Group 1: 7.
Group 2: 8.

Complete-case positions used: 65.

Definition:

For each participant, calculate the mean mutation
fraction across the complete-case position set.

For each position, subtract that participant's
overall mean from its position-specific fraction.

Compute the group difference using equal weighting
of participants within each group.

Equivalently:

Centered contrast at position p =
raw Group2-minus-Group1 contrast at position p
minus the global Group2-minus-Group1 mean.

Global group mean difference: -0.0037847628238568687

The reconstructed contrasts agree with tables
201 and 212 within 1e-8.

Seven-position maximum error: 1.1102230246251565e-16
All comparable 201 positions maximum error:
5.551115123125783e-17

SHA256:
198: c2a68095cc5dffc072555e6d4d1878fe7c1ccbd214f8b1dc27c5797f00c5b85a
201: de24498777a307a2089f702320f6740800c42223eb966e577f711cfd97c97029
212: d6471318c2d1ce93e7bb57aea37ede91d574fcf9ccc472ee1d1fc28a432cadc5
178: 10e85b94720e9dd0613c48814802d3e38d125c74bc38b91e49d257ddda01aba9

Interpretation:

The previously observed constant contrast offset
comes from the reference centering operation,
not an independent biological result.

Script 191 additionally centers over the remaining
selected positions in its leave-one-position-out test.
That test statistic is distinct from its original
position-difference reference column.

The private Step 4AU figure plots raw participant
fractions. It is not established as an exact replica
of the frozen manuscript Figure 1A.

Original manuscript and publication figure unchanged.
No new statistical hypothesis testing performed.
No public data or code release authorized.
