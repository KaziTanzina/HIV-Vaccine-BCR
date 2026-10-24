from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()

SOURCE = ROOT / (
    "scripts/156_match_frozen_prime_week8_to_later_cg28v2.py"
)
SUPPORT = ROOT / (
    "results/manuscript/phaseF_support_review/"
    "549_cell_support_20261022_201852_290342/"
    "participant_position_cell_support.tsv"
)
EARLY = ROOT / (
    "data/processed/g002_g003/"
    "g002_groups1_2_week8_eod_prime_reclustered_n3.feather"
)
RAW = ROOT / "data/external/g002_g003/g002/final_df.feather"

REFERENCE = ROOT / (
    "results/bcr/g002_g003/"
    "171_g002_prime_later_cg28v2_cell_matches.tsv"
)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT = ROOT / "results/bcr/g002_g003" / (
    f"phaseH_probe_pair_sensitivity_{stamp}"
)

def stop(message):
    raise SystemExit("STOP: " + message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("=" * 76)
print("PHASE H — PAIRED WEEK-24 PROBE RECOVERY")
print("SINGLE BOUNDED ASCERTAINMENT SENSITIVITY ANALYSIS")
print("=" * 76, flush=True)

for path in [SOURCE, SUPPORT, EARLY, RAW]:
    if not path.is_file():
        stop(f"Missing source: {path}")

source_sha = sha(SOURCE)
early_sha = sha(EARLY)
raw_sha = sha(RAW)

# 1. Recover and freeze the original biological participants.

p = pd.read_csv(SUPPORT, sep="\t")

required = {
    "pubID", "group",
    "minimum_cells_per_visit", "eligible_lineages"
}
if not required.issubset(p.columns):
    stop("Participant-support columns differ")

p["group"] = pd.to_numeric(p["group"], errors="coerce")
p["minimum_cells_per_visit"] = pd.to_numeric(
    p["minimum_cells_per_visit"], errors="coerce"
)

PRIMARY_IDS = sorted(
    p.loc[
        p["group"].eq(1)
        & p["minimum_cells_per_visit"].eq(1)
        & p["eligible_lineages"].gt(0),
        "pubID"
    ].dropna().unique().tolist()
)

expected_ids = sorted([
    "G002-509", "G002-616", "G002-619",
    "G002-678", "G002-758", "G002-813",
    "G002-852"
])

if PRIMARY_IDS != expected_ids:
    stop(
        "Primary Group-1 cohort changed: "
        + repr(PRIMARY_IDS)
    )

print("\nFROZEN PRIMARY GROUP-1 PARTICIPANTS")
print(", ".join(PRIMARY_IDS))

# 2. Make an exact, controlled derivative of Script 156.

original = SOURCE.read_text(encoding="utf-8")
modified = original

def change(old, new, description):
    global modified

    count = modified.count(old)

    if count != 1:
        stop(
            f"Source mismatch for {description}: "
            f"expected one anchor, found {count}"
        )

    modified = modified.replace(old, new, 1)
    print("PASS:", description)

change(
    'outdir = Path("results/bcr/g002_g003")',
    f"outdir = Path({str(OUT)!r})",
    "Versioned result directory"
)

change(
    "early = pd.read_feather(early_file)\n",
    "early = pd.read_feather(early_file)\n"
    f"PRIMARY_IDS = {PRIMARY_IDS!r}\n"
    "early = early.loc[\n"
    "    early['pubID'].isin(PRIMARY_IDS)\n"
    "].copy()\n",
    "Restrict baseline to the original seven participants"
)

change(
    'full["group"].isin([1, 2])',
    'full["group"].eq(1)',
    "Group 1 only"
)

change(
    'full["weeks"].isin([16, 24])',
    'full["weeks"].eq(24)',
    "Week 24 only"
)

change(
    '    & (full["probe_set"] == "Cg28v2")\n',
    '    & full["probe_set"].isin(["eODGT8", "Cg28v2"])\n'
    '    & full["pubID"].isin(PRIMARY_IDS)\n',
    "Both Week-24 probes in the same participants"
)

change(
    "lineages = pd.DataFrame(lineages)\n",
    "lineages = pd.DataFrame(lineages)\n"
    "lineages.to_csv(\n"
    "    outdir / '606_baseline_lineage_index.tsv',\n"
    "    sep='\\t', index=False\n"
    ")\n",
    "Save unchanged baseline-lineage index"
)

change(
    "    matches.append({\n",
    "    matches.append({\n"
    '        "probe_set": cell["probe_set"],\n',
    "Retain probe labels in cell assignments"
)

compile(modified, str(OUT / "matching_core.py"), "exec")

if sha(SOURCE) != source_sha:
    stop("Original Script 156 changed")

OUT.mkdir(parents=True, exist_ok=False)
DERIVED = OUT / "matching_core.py"
DERIVED.write_text(modified, encoding="utf-8")

print("\nMATCHING RULES")
print("Original CDR3-distance and compatibility blocks untouched.")
print("Original unique/ambiguous assignment rules untouched.")
print("Original Script 156 unchanged.")
print("\nRUNNING FROZEN MATCHING LOGIC", flush=True)

subprocess.run(
    [sys.executable, "-u", str(DERIVED)],
    cwd=ROOT,
    check=True
)

# 3. Read matches and original baseline-lineage definitions.

MATCH = OUT / (
    "171_g002_prime_later_cg28v2_cell_matches.tsv"
)
INDEX = OUT / "606_baseline_lineage_index.tsv"

if not MATCH.is_file() or not INDEX.is_file():
    stop("Expected matching outputs missing")

m = pd.read_csv(MATCH, sep="\t")
lineages = pd.read_csv(INDEX, sep="\t")

for col in [
    "pubID", "weeks", "probe_set", "cellid",
    "match_status", "assigned_lineage"
]:
    if col not in m:
        stop("Missing match column: " + col)

for col in ["pubID", "lineage", "v_h"]:
    if col not in lineages:
        stop("Missing lineage column: " + col)

if set(m["pubID"]) != set(PRIMARY_IDS):
    stop("Unexpected matched participants")

if set(m["probe_set"]) != {"eODGT8", "Cg28v2"}:
    stop("Both probe populations not present")

if not m["weeks"].eq(24).all():
    stop("Non-Week-24 observations present")

if m["cellid"].duplicated().any():
    stop("Duplicate cell IDs across probe populations")

# Verify counts against the original feasibility output.

counts = (
    m.groupby(["pubID", "probe_set"])
    .size()
    .unstack(fill_value=0)
)

if (
    int(counts["eODGT8"].sum()) != 3144
    or int(counts["Cg28v2"].sum()) != 2202
):
    stop("Week-24 probe cell counts changed")

print("\nPASS: Seven participants and both probe pools verified")
print("eOD-GT8 cells:", int(counts["eODGT8"].sum()))
print("Cg28v2 cells:", int(counts["Cg28v2"].sum()))

# 4. Confirm the Cg28v2 arm reproduces frozen assignments.

parity = "NOT CHECKED — reference file unavailable"

if REFERENCE.is_file():
    ref = pd.read_csv(REFERENCE, sep="\t")

    needed = {
        "cellid", "pubID", "group", "weeks",
        "match_status", "assigned_lineage"
    }

    if not needed.issubset(ref.columns):
        stop("Frozen reference columns differ")

    ref = ref.loc[
        ref["pubID"].isin(PRIMARY_IDS)
        & ref["group"].eq(1)
        & ref["weeks"].eq(24),
        ["cellid", "match_status", "assigned_lineage"]
    ].copy()

    current = m.loc[
        m["probe_set"].eq("Cg28v2"),
        ["cellid", "match_status", "assigned_lineage"]
    ].copy()

    if (
        ref["cellid"].duplicated().any()
        or current["cellid"].duplicated().any()
    ):
        stop("Duplicate IDs in assignment parity audit")

    check = current.merge(
        ref,
        on="cellid",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_new", "_frozen")
    )

    if not check["_merge"].eq("both").all():
        stop("Cg28v2 cell population differs from frozen source")

    statuses_equal = (
        check["match_status_new"]
        .astype("string")
        .fillna("<NA>")
        .eq(
            check["match_status_frozen"]
            .astype("string")
            .fillna("<NA>")
        )
    )

    lineages_equal = (
        check["assigned_lineage_new"]
        .astype("string")
        .fillna("<NA>")
        .eq(
            check["assigned_lineage_frozen"]
            .astype("string")
            .fillna("<NA>")
        )
    )

    if not (statuses_equal & lineages_equal).all():
        stop("Cg28v2 assignments disagree with frozen matching")

    parity = f"PASS: {len(check)} identical Cg28v2 cell assignments"

print("\nFROZEN ASSIGNMENT REPLAY")
print(parity)

# 5. Use original IGHV1-2*02 Week-8 target lineages.
# All competing lineages were retained during matching.

target = lineages.loc[
    lineages["v_h"].eq("IGHV1-2*02")
].copy()

if not set(target["pubID"]) == set(PRIMARY_IDS):
    stop("Target baseline lineage coverage differs")

target_ids = set(target["lineage"].astype(str))

m["assigned_lineage"] = (
    m["assigned_lineage"].astype("string")
)

# 6. Paired lineage detection and equal-cell-depth sensitivity.

REPS = 1000
rng = np.random.default_rng(606)
records = []
simulation = []

for participant in PRIMARY_IDS:
    baseline_ids = set(
        target.loc[
            target["pubID"].eq(participant),
            "lineage"
        ].astype(str)
    )

    denominator = len(baseline_ids)

    if denominator == 0:
        stop("No target lineages: " + participant)

    probe_data = {
        probe: m.loc[
            m["pubID"].eq(participant)
            & m["probe_set"].eq(probe)
        ].reset_index(drop=True)
        for probe in ["eODGT8", "Cg28v2"]
    }

    depths = {
        probe: len(frame)
        for probe, frame in probe_data.items()
    }

    depth = min(depths.values())

    if depth < 3:
        stop("Insufficient paired depth: " + participant)

    recovered = {}

    for probe, frame in probe_data.items():
        recovered[probe] = set(
            frame.loc[
                frame["match_status"].eq("unique"),
                "assigned_lineage"
            ].dropna().astype(str)
        ) & baseline_ids

    eod = recovered["eODGT8"]
    cg = recovered["Cg28v2"]

    records.append({
        "pubID": participant,
        "baseline_target_lineages": denominator,
        "eODGT8_cells": depths["eODGT8"],
        "Cg28v2_cells": depths["Cg28v2"],
        "equalized_cells_per_probe": depth,
        "eODGT8_detected_lineages": len(eod),
        "Cg28v2_detected_lineages": len(cg),
        "detected_under_both": len(eod & cg),
        "eODGT8_only": len(eod - cg),
        "Cg28v2_only": len(cg - eod),
        "detected_under_either": len(eod | cg),
        "raw_eODGT8_fraction": len(eod) / denominator,
        "raw_Cg28v2_fraction": len(cg) / denominator,
        "raw_Cg28v2_minus_eODGT8": (
            len(cg) - len(eod)
        ) / denominator
    })

    for repetition in range(REPS):
        balanced_counts = {}

        for probe, frame in probe_data.items():
            selected = (
                frame
                if len(frame) == depth
                else frame.iloc[
                    rng.choice(
                        len(frame),
                        size=depth,
                        replace=False
                    )
                ]
            )

            detected = set(
                selected.loc[
                    selected["match_status"].eq("unique"),
                    "assigned_lineage"
                ].dropna().astype(str)
            ) & baseline_ids

            balanced_counts[probe] = len(detected)

        simulation.append({
            "pubID": participant,
            "repetition": repetition,
            "equalized_cells": depth,
            "eODGT8_lineages": balanced_counts["eODGT8"],
            "Cg28v2_lineages": balanced_counts["Cg28v2"],
            "Cg28v2_minus_eODGT8_fraction": (
                balanced_counts["Cg28v2"]
                - balanced_counts["eODGT8"]
            ) / denominator
        })

participant = pd.DataFrame(records)
sim = pd.DataFrame(simulation)

sim_summary = (
    sim.groupby("pubID")
    ["Cg28v2_minus_eODGT8_fraction"]
    .agg(["mean", "min", "max"])
    .reset_index()
)

average_by_draw = (
    sim.groupby("repetition")
    ["Cg28v2_minus_eODGT8_fraction"]
    .mean()
)

raw_equal_weight_mean = float(
    participant["raw_Cg28v2_minus_eODGT8"].mean()
)

balanced_mean = float(average_by_draw.mean())
balanced_range = np.quantile(
    average_by_draw, [0.025, 0.975]
)

participant.to_csv(
    OUT / "606_participant_probe_recovery.tsv",
    sep="\t", index=False
)

sim_summary.to_csv(
    OUT / "606_equal_depth_participant_summary.tsv",
    sep="\t", index=False
)

sim.to_csv(
    OUT / "606_equal_depth_resampling.tsv",
    sep="\t", index=False
)

print("\n" + "=" * 76)
print("PAIRED PROBE-SPECIFIC LINEAGE RECOVERY")
print("=" * 76)

print(
    participant[[
        "pubID",
        "baseline_target_lineages",
        "eODGT8_detected_lineages",
        "Cg28v2_detected_lineages",
        "detected_under_both",
        "eODGT8_only",
        "Cg28v2_only",
        "raw_Cg28v2_minus_eODGT8"
    ]].to_string(index=False)
)

print("\nDESCRIPTIVE PARTICIPANT-EQUAL SUMMARY")
print(
    "Raw mean Cg28v2 minus eOD-GT8 recovery fraction:",
    round(raw_equal_weight_mean, 6)
)

print(
    "Equal-cell-depth mean Cg28v2 minus eOD-GT8:",
    round(balanced_mean, 6)
)

print(
    "Equal-depth Monte Carlo 2.5%-97.5% range:",
    tuple(np.round(balanced_range, 6))
)

print("\nINTERPRETATION LIMITS")
print(
    "Resampling range measures sensitivity to the observed "
    "cell selection, not a participant-population confidence interval."
)
print(
    "This analysis assesses probe-associated ascertainment, "
    "not causal probe effects or biological mutation acquisition."
)
print(
    "It is restricted to seven previously selected "
    "Group-1 participants."
)
print(
    "Group 2 has no corresponding dual-probe Week-24 comparison."
)

# 7. Integrity checks and manifest.

for path, original_digest in [
    (SOURCE, source_sha),
    (EARLY, early_sha),
    (RAW, raw_sha)
]:
    if sha(path) != original_digest:
        stop("Frozen source changed: " + str(path))

manifest = {
    "analysis": "Paired within-visit probe-recovery sensitivity",
    "participants": PRIMARY_IDS,
    "week": 24,
    "group": 1,
    "probes": ["eODGT8", "Cg28v2"],
    "target_v_gene": "IGHV1-2*02",
    "source_matching_script_sha256": source_sha,
    "baseline_file_sha256": early_sha,
    "original_data_sha256": raw_sha,
    "reference_assignment_parity": parity,
    "equal_depth_repetitions": REPS,
    "new_hypothesis_tests": False,
    "cdrh2_mutation_analysis_changed": False,
    "model_training": False
}

(OUT / "606_analysis_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("\nSOURCE INTEGRITY: PASS")
print("Matching logic: unchanged")
print("Original Script 156: unchanged")
print("Original biological analysis: unchanged")
print("Saved output:", OUT)

print(
    "\nPHASE H PAIRED PROBE "
    "SENSITIVITY ANALYSIS COMPLETE"
)
