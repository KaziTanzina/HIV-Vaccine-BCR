from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / "results/bcr/g002_g003"

TRANSITIONS = BASE / (
    "193_g002_prime_week24_v_mutation_transitions.tsv"
)

PROFILE = BASE / (
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv"
)

POSITIONS = [51, 52, 53, 54, 55, 57, 58]
THRESHOLDS = [1, 2, 3]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("=" * 76)
print("PHASE F — STEP 1")
print("CDRH2 LONGITUDINAL CELL SUPPORT AND DENOMINATORS")
print("=" * 76)

print("\nSCIENTIFIC QUESTION")
print(
    "How much cell-level evidence supports the original "
    "seven-position CDRH2 acquisition profiles?"
)

# -----------------------------------------------------
# 1. VERIFY FROZEN INPUTS
# -----------------------------------------------------

for path in [TRANSITIONS, PROFILE]:
    if not path.is_file():
        raise SystemExit(f"Missing source: {path}")

    print("\nSource:", path.relative_to(ROOT))
    print("SHA256:", sha256(path))

before = {
    path: sha256(path)
    for path in [TRANSITIONS, PROFILE]
}

df = pd.read_csv(
    TRANSITIONS,
    sep="\t",
    low_memory=False
)

profile = pd.read_csv(
    PROFILE,
    sep="\t"
)

required = {
    "pubID",
    "group",
    "lineage",
    "chain",
    "v_gene",
    "position",
    "transition",
    "w8_aa",
    "w24_aa",
    "germline_aa",
    "w8_cells",
    "w24_cells",
}

missing = required - set(df.columns)

if missing:
    raise SystemExit(
        f"Missing transition columns: {sorted(missing)}"
    )

profile_required = {
    "pubID",
    "group",
    *[str(p) for p in POSITIONS],
}

missing_profile = profile_required - set(profile.columns)

if missing_profile:
    raise SystemExit(
        f"Missing profile columns: {sorted(missing_profile)}"
    )

# -----------------------------------------------------
# 2. RECONSTRUCT THE ORIGINAL ELIGIBLE POPULATION
# -----------------------------------------------------

df["position"] = pd.to_numeric(
    df["position"],
    errors="coerce"
)

df["group"] = pd.to_numeric(
    df["group"],
    errors="coerce"
)

profile["pubID"] = profile["pubID"].astype(str)
df["pubID"] = df["pubID"].astype(str)

profile["group"] = pd.to_numeric(
    profile["group"],
    errors="raise"
).astype(int)

if len(profile) != 15:
    raise SystemExit("Expected 15 original participants.")

if profile["pubID"].duplicated().any():
    raise SystemExit("Duplicate participant in profile.")

if profile["group"].value_counts().to_dict() != {
    1: 7,
    2: 8,
}:
    raise SystemExit("Original group counts mismatch.")

participants = set(profile["pubID"])

eligible = df[
    (df["chain"] == "heavy")
    & (df["v_gene"] == "IGHV1-2*02")
    & (df["position"].isin(POSITIONS))
    & (df["transition"] != "unresolved")
    & df["w24_aa"].notna()
    & (df["w8_aa"] == df["germline_aa"])
    & df["pubID"].isin(participants)
].copy()

eligible["acquired"] = (
    eligible["w24_aa"] != eligible["germline_aa"]
).astype(int)

for column in ["w8_cells", "w24_cells"]:
    eligible[column] = pd.to_numeric(
        eligible[column],
        errors="raise"
    )

    if (
        eligible[column].isna().any()
        or (eligible[column] < 1).any()
        or (eligible[column] % 1 != 0).any()
    ):
        raise SystemExit(
            f"Invalid assessable cell count: {column}"
        )

eligible["w8_cells"] = eligible["w8_cells"].astype(int)
eligible["w24_cells"] = eligible["w24_cells"].astype(int)

# A lineage-position must be one observation.
keys = ["pubID", "position", "lineage"]

if eligible.duplicated(keys).any():
    raise SystemExit(
        "Duplicate eligible lineage-position records. "
        "Review eligibility before proceeding."
    )

# Check group membership against the frozen profile.
group_map = profile.set_index("pubID")["group"]

if not (
    eligible["group"].astype(int).to_numpy()
    == eligible["pubID"].map(group_map).to_numpy()
).all():
    raise SystemExit("Participant group mismatch.")

print("\nORIGINAL ELIGIBLE POPULATION")
print("Participants:", eligible["pubID"].nunique())
print("Lineages:", eligible["lineage"].nunique())
print("Eligible lineage-position records:", len(eligible))
print("Acquisition records:", int(eligible["acquired"].sum()))

# -----------------------------------------------------
# 3. VERIFY ALL 105 ORIGINAL PROFILE FRACTIONS
# -----------------------------------------------------

counts = (
    eligible.groupby(
        ["pubID", "position"]
    )
    .agg(
        eligible_lineages=("lineage", "nunique"),
        acquired_lineages=("acquired", "sum")
    )
)

counts["fraction"] = (
    counts["acquired_lineages"]
    / counts["eligible_lineages"]
)

verified = 0

for _, participant in profile.iterrows():
    for pos in POSITIONS:

        key = (participant["pubID"], pos)

        if key not in counts.index:
            raise SystemExit(
                f"Missing original profile support: {key}"
            )

        actual = float(counts.loc[key, "fraction"])
        expected = float(participant[str(pos)])

        if not np.isclose(
            actual, expected,
            rtol=0,
            atol=1e-8
        ):
            raise SystemExit(
                f"Original profile mismatch: {key}, "
                f"observed={actual}, expected={expected}"
            )

        verified += 1

print("\nORIGINAL PROFILE VERIFICATION")
print(f"Matching participant-position fractions: {verified}/105")

# -----------------------------------------------------
# 4. DESCRIBE CELL SUPPORT
# -----------------------------------------------------

print("\nASSESSABLE CELL-COUNT DISTRIBUTION")

for group in [1, 2]:
    g = eligible[eligible["group"] == group]

    print(f"\nGROUP {group}")
    print("Eligible lineage-position records:", len(g))

    for visit, column in [
        ("Week 8", "w8_cells"),
        ("Week 24", "w24_cells"),
    ]:
        values = g[column]

        print(
            visit,
            "median=", values.median(),
            "Q1=", values.quantile(0.25),
            "Q3=", values.quantile(0.75),
            "maximum=", values.max(),
        )

    for k in THRESHOLDS:
        passed = (
            (g["w8_cells"] >= k)
            & (g["w24_cells"] >= k)
        )

        print(
            f"At least {k} cells at BOTH visits:",
            int(passed.sum()),
            f"({100 * passed.mean():.2f}%)"
        )

# -----------------------------------------------------
# 5. BUILD PARTICIPANT-POSITION SUPPORT AT EACH DEPTH
# -----------------------------------------------------

records = []

for k in THRESHOLDS:

    subset = eligible[
        (eligible["w8_cells"] >= k)
        & (eligible["w24_cells"] >= k)
    ]

    grouped = (
        subset.groupby(["pubID", "position"])
        .agg(
            eligible_lineages=("lineage", "nunique"),
            acquired_lineages=("acquired", "sum")
        )
    )

    for _, participant in profile.iterrows():

        pubid = participant["pubID"]
        group = int(participant["group"])

        for pos in POSITIONS:

            key = (pubid, pos)

            if key in grouped.index:
                n = int(
                    grouped.loc[key, "eligible_lineages"]
                )
                a = int(
                    grouped.loc[key, "acquired_lineages"]
                )
                fraction = a / n
            else:
                n = 0
                a = 0
                fraction = np.nan

            records.append({
                "minimum_cells_per_visit": k,
                "pubID": pubid,
                "group": group,
                "position": pos,
                "eligible_lineages": n,
                "acquired_lineages": a,
                "acquisition_fraction": fraction,
            })

support = pd.DataFrame(records)

summary = (
    support.groupby(
        ["minimum_cells_per_visit", "group", "position"]
    )
    .agg(
        participants_with_support=(
            "eligible_lineages",
            lambda x: int((x > 0).sum())
        ),
        eligible_lineages=("eligible_lineages", "sum"),
        acquired_lineages=("acquired_lineages", "sum"),
        mean_participant_fraction=(
            "acquisition_fraction",
            "mean"
        ),
    )
    .reset_index()
)

print("\nPARTICIPANT-POSITION ELIGIBILITY")
print(
    "Important: participants with zero eligible lineages "
    "have a missing fraction, not a zero fraction."
)

print(summary.to_string(index=False))

# -----------------------------------------------------
# 6. DETERMINE WHETHER FULL SEVEN-POSITION
#    PARTICIPANT PROFILES REMAIN AVAILABLE
# -----------------------------------------------------

print("\nCOMPLETE SEVEN-POSITION PROFILES")

for k in THRESHOLDS:

    s = support[
        support["minimum_cells_per_visit"] == k
    ]

    complete = (
        s.groupby(["pubID", "group"])["eligible_lineages"]
        .apply(lambda x: bool((x > 0).all()))
        .reset_index(name="complete")
    )

    for group in [1, 2]:
        g = complete[complete["group"] == group]

        print(
            f"Minimum {k}, Group {group}: "
            f"{int(g['complete'].sum())}/{len(g)} "
            "participants complete"
        )

    print(
        "A group-profile statistic is not calculated "
        "in this step."
    )

# -----------------------------------------------------
# 7. DESCRIBE ORIGINAL DENOMINATOR HETEROGENEITY
# -----------------------------------------------------

print("\nORIGINAL ELIGIBLE-LINEAGE DENOMINATORS")

original_support = support[
    support["minimum_cells_per_visit"] == 1
]

for group in [1, 2]:
    g = original_support[
        original_support["group"] == group
    ]

    print(
        f"Group {group}: "
        f"minimum={g['eligible_lineages'].min()}, "
        f"median={g['eligible_lineages'].median()}, "
        f"maximum={g['eligible_lineages'].max()}"
    )

# -----------------------------------------------------
# 8. SAVE DESCRIPTIVE OUTPUT ONLY
# -----------------------------------------------------

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

outdir = ROOT / (
    "results/manuscript/phaseF_support_review/"
    f"549_cell_support_{stamp}"
)

outdir.mkdir(parents=True, exist_ok=False)

support_path = outdir / (
    "participant_position_cell_support.tsv"
)

summary_path = outdir / (
    "group_position_cell_support_summary.tsv"
)

support.to_csv(
    support_path,
    sep="\t",
    index=False,
    na_rep="NA"
)

summary.to_csv(
    summary_path,
    sep="\t",
    index=False,
    na_rep="NA"
)

for path, digest in before.items():
    if sha256(path) != digest:
        raise RuntimeError(
            f"Frozen input changed: {path}"
        )

print("\nOUTPUTS")
print(support_path.resolve())
print(summary_path.resolve())

print("\nINTEGRITY")
print("Original 105 fractions reproduced: PASS")
print("Frozen transition data unchanged: PASS")
print("Frozen participant profile unchanged: PASS")
print("No permutation tests performed: PASS")
print("No manuscript modifications: PASS")

print("\nPHASE F STEP 1 COMPLETE")
