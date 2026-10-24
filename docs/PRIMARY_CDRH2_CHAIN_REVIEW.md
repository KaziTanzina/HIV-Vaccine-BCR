# Frozen primary CDRH2 workflow

STATUS: PRIVATE REPRODUCIBILITY REVIEW ONLY.

The following is the source-code dependency order
identified from the existing analysis scripts:

152 — Reconstruct Week-8 baseline lineages
156 — Match prospective lineages to follow-up cells
171 — Validate V-region mutation calls
173 — Construct Week-8-to-Week-24 transitions
177 — Construct supported participant-position profiles
178 — Calculate centered positional comparisons
188 — Construct V-position-to-region mapping
189 — Test regional steering
190 — Leave-one-participant-out sensitivity
191 — Leave-one-position-out sensitivity
192 — Week-8 baseline sensitivity

These relationships were established by inspection
of source-code inputs and output declarations.

This review did not rerun the scripts or establish
a fully tested end-to-end reproduction workflow.

Original G002 source data are NOT included.
Frozen participant-level records are NOT included.
The published source data must be obtained through
the original investigators' repository under its
applicable access and reuse conditions.

The manuscript's final figure files are preserved
separately. Exact figure generation has not been
certified by this code-staging step.

Additional Group-2 AI benchmarking dependencies
and model resources require separate release
documentation.

Do not publish this private package as-is.
