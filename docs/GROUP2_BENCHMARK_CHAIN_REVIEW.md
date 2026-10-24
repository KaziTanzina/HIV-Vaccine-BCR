# Group-2 benchmark reproducibility

STATUS: PRIVATE REVIEW ONLY.

207 — Generates the saved Group-2 Week-24
      recovery-outcome overlap table.

209 — Constructs the frozen biological baseline
      feature table.

210 — Evaluates the baseline using participant-held-out
      predictions.

212_phase4a_igbert_gap_cleanup — Generates cleaned
      input sequences for IgBERT.

214 — Generates IgBERT embedding arrays.
217 — Evaluates the baseline-plus-IgBERT model.

571_phaseG_ablang2_frozen_lopo — Evaluates frozen
      AbLang-2 representations.

IMPORTANT UNRESOLVED ITEM:
The generator of the frozen AbLang-2 embeddings,
embedding index, and metadata has not yet been
identified by the filename-based source search.

The files themselves exist in the original local
research workspace but are not included here.

This document describes identified source-code
relationships. It does not establish historical
execution provenance or end-to-end reproducibility.

The benchmark is secondary to the primary CDRH2
biological analysis. The frozen interpretation is
modest discrimination without measured average-
precision improvement from the language models.

No participant-level data or model weights are
included in this private review package.

Do not publicly release this folder as-is.
