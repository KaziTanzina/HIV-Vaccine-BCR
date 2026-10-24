# Reproduction workflow

Status: documentation and dependency review pending.

Run scripts from the repository root after obtaining
the original published data under applicable conditions.

## Primary CDRH2 analysis

The source-code dependency order is:

152  Week-8 baseline reconstruction
156  Longitudinal lineage matching
171  Mutation-call validation
173  Longitudinal mutation transitions
177  Participant-position profiles
178  Centered comparisons
188  V-position-to-region mapping
189  Region-level analysis
190  Leave-one-participant-out sensitivity
191  Leave-one-position-out sensitivity
192  Week-8 baseline sensitivity

See docs/PRIMARY_CDRH2_CHAIN_REVIEW.md.

## Secondary benchmark

207  Frozen Group-2 outcome
209  Biological baseline features
210  Baseline participant-held-out model
212  IgBERT sequence cleaning
214  IgBERT embedding generation
217  Biological baseline plus IgBERT
571  Frozen AbLang-2 model evaluation

See docs/GROUP2_BENCHMARK_CHAIN_REVIEW.md.

## Known limitations

The AbLang-2 embedding-generation source has not
been identified as a standalone script.

Its historical execution log and saved input/output
SHA256 checks were verified separately.

Historical AbLang-2 environment:
PyTorch 2.5.1; AbLang-2 0.1.1.

The current pfizer-vaccine environment differs and is
missing some dependencies used in historical scripts.

Additional upstream dependencies and exact execution
instructions require validation before publication.

Do not claim end-to-end reproduction at this stage.
