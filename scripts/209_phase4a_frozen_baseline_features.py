from pathlib import Path

import numpy as np
import pandas as pd


week8_path = Path(
    "data/processed/g002_g003/"
    "g002_groups1_2_week8_eod_prime_reclustered_n3.feather"
)

cohort_path = Path(
    "results/bcr/g002_g003/"
    "243_phase4a_group2_week24_medoid_outcome_overlap.tsv"
)

outdir = Path(
    "results/bcr/g002_g003"
)


week8 = pd.read_feather(
    week8_path
)

cohort = pd.read_csv(
    cohort_path,
    sep="\t",
)


print(
    "PHASE 4A FROZEN BASELINE FEATURE MATRIX"
)


required_week8 = [
    "pubID",
    "group",
    "prospective_lineage",
    "v_mutation_heavy",
    "v_mutation_light",
    "cottrell_focused_v_common_score",
]


print(
    "\nWEEK-8 SOURCE COLUMN CHECK"
)

for col in required_week8:

    print(
        col,
        "present=",
        col in week8.columns,
    )


missing = [
    col
    for col in required_week8
    if col not in week8.columns
]

if missing:

    raise ValueError(
        "Missing required Week-8 columns: "
        + ", ".join(missing)
    )


required_cohort = [
    "prospective_lineage",
    "pubID",
    "group",
    "week",
    "assessable",
    "detected",
    "early_cells",
]


print(
    "\nCOHORT COLUMN CHECK"
)

for col in required_cohort:

    print(
        col,
        "present=",
        col in cohort.columns,
    )


missing = [
    col
    for col in required_cohort
    if col not in cohort.columns
]

if missing:

    raise ValueError(
        "Missing required cohort columns: "
        + ", ".join(missing)
    )


print(
    "\nCOHORT CHECK"
)

print(
    "rows:",
    len(cohort),
)

print(
    "lineages:",
    cohort[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    cohort[
        "pubID"
    ].nunique(),
)

print(
    "groups:",
    sorted(
        cohort[
            "group"
        ].unique()
    ),
)

print(
    "weeks:",
    sorted(
        cohort[
            "week"
        ].unique()
    ),
)


if not (
    cohort["group"] == 2
).all():

    raise ValueError(
        "Cohort contains rows outside Group 2."
    )


if not (
    cohort["week"] == 24
).all():

    raise ValueError(
        "Cohort contains rows outside Week 24."
    )


if not (
    cohort["assessable"] == True
).all():

    raise ValueError(
        "Cohort contains non-assessable rows."
    )


week8_group2 = week8[
    week8["group"] == 2
].copy()


print(
    "\nGROUP-2 WEEK-8 SOURCE"
)

print(
    "cells:",
    len(week8_group2),
)

print(
    "lineages:",
    week8_group2[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    week8_group2[
        "pubID"
    ].nunique(),
)


for col in [
    "v_mutation_heavy",
    "v_mutation_light",
    "cottrell_focused_v_common_score",
]:

    week8_group2[col] = pd.to_numeric(
        week8_group2[col],
        errors="coerce",
    )


features = (
    week8_group2
    .groupby(
        [
            "pubID",
            "prospective_lineage",
        ],
        as_index=False,
    )
    .agg(
        week8_cells=(
            "prospective_lineage",
            "size",
        ),
        week8_heavy_shm=(
            "v_mutation_heavy",
            "mean",
        ),
        week8_light_shm=(
            "v_mutation_light",
            "mean",
        ),
        cottrell_mean=(
            "cottrell_focused_v_common_score",
            "mean",
        ),
    )
)


features[
    "log2_week8_cells"
] = np.log2(
    features[
        "week8_cells"
    ]
)


features[
    "week8_heavy_shm_pct"
] = (
    features[
        "week8_heavy_shm"
    ]
    * 100.0
)


features[
    "week8_light_shm_pct"
] = (
    features[
        "week8_light_shm"
    ]
    * 100.0
)


features = features[
    [
        "pubID",
        "prospective_lineage",
        "week8_cells",
        "log2_week8_cells",
        "week8_heavy_shm_pct",
        "week8_light_shm_pct",
        "cottrell_mean",
    ]
]


print(
    "\nRECONSTRUCTED FEATURE TABLE"
)

print(
    "rows:",
    len(features),
)

print(
    "lineages:",
    features[
        "prospective_lineage"
    ].nunique(),
)


cohort_small = cohort[
    [
        "prospective_lineage",
        "pubID",
        "group",
        "week",
        "assessable",
        "detected",
        "early_cells",
    ]
].copy()


merged = cohort_small.merge(
    features,
    on=[
        "pubID",
        "prospective_lineage",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)


print(
    "\nCOHORT / BASELINE FEATURE JOIN"
)

print(
    merged["_merge"]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print(
    "\nEARLY CELL COUNT VALIDATION"
)

early_numeric = pd.to_numeric(
    merged["early_cells"],
    errors="coerce",
)


count_match = (
    early_numeric
    == merged[
        "week8_cells"
    ]
)


print(
    "comparable rows:",
    int(
        count_match.notna().sum()
    ),
)

print(
    "matching counts:",
    int(
        count_match.sum()
    ),
)

print(
    "match fraction:",
    float(
        count_match.mean()
    ),
)


feature_cols = [
    "log2_week8_cells",
    "week8_heavy_shm_pct",
    "week8_light_shm_pct",
    "cottrell_mean",
]


print(
    "\nFEATURE COMPLETENESS"
)

for col in feature_cols:

    print(
        col,
        "nonmissing=",
        int(
            merged[col]
            .notna()
            .sum()
        ),
        "fraction=",
        float(
            merged[col]
            .notna()
            .mean()
        ),
    )


print(
    "\nFEATURE RANGES"
)

print(
    merged[
        feature_cols
    ]
    .describe()
    .to_string()
)


duplicate_lineages = (
    merged[
        "prospective_lineage"
    ]
    .duplicated()
    .sum()
)


print(
    "\nDUPLICATE LINEAGES:",
    duplicate_lineages,
)


if duplicate_lineages != 0:

    raise ValueError(
        "Duplicate prospective lineages "
        "in final feature matrix."
    )


if not (
    merged["_merge"] == "both"
).all():

    raise ValueError(
        "Some prediction cohort lineages "
        "lack reconstructed baseline features."
    )


if not count_match.all():

    mismatch = merged[
        ~count_match
    ][
        [
            "pubID",
            "prospective_lineage",
            "early_cells",
            "week8_cells",
        ]
    ]

    print(
        "\nCELL COUNT MISMATCH EXAMPLES"
    )

    print(
        mismatch.head(25)
        .to_string(
            index=False
        )
    )

    raise ValueError(
        "Reconstructed Week-8 cell counts "
        "do not exactly match outcome early_cells."
    )


if merged[
    feature_cols
].isna().any().any():

    raise ValueError(
        "Missing values remain in "
        "the four baseline features."
    )


final = merged[
    [
        "pubID",
        "prospective_lineage",
        "group",
        "week",
        "assessable",
        "detected",
        "week8_cells",
        "log2_week8_cells",
        "week8_heavy_shm_pct",
        "week8_light_shm_pct",
        "cottrell_mean",
    ]
].copy()


final.to_csv(
    outdir /
    "246_phase4a_group2_week24_frozen_baseline_features.tsv",
    sep="\t",
    index=False,
)


audit = pd.DataFrame(
    [
        {
            "cohort_lineages":
            len(cohort),

            "feature_lineages":
            len(features),

            "joined_lineages":
            len(final),

            "participants":
            final[
                "pubID"
            ].nunique(),

            "cell_count_match_fraction":
            count_match.mean(),

            "complete_log2_week8_cells":
            final[
                "log2_week8_cells"
            ].notna().mean(),

            "complete_week8_heavy_shm_pct":
            final[
                "week8_heavy_shm_pct"
            ].notna().mean(),

            "complete_week8_light_shm_pct":
            final[
                "week8_light_shm_pct"
            ].notna().mean(),

            "complete_cottrell_mean":
            final[
                "cottrell_mean"
            ].notna().mean(),
        }
    ]
)


audit.to_csv(
    outdir /
    "247_phase4a_group2_week24_frozen_baseline_feature_audit.tsv",
    sep="\t",
    index=False,
)


print(
    "\nFINAL FROZEN BASELINE MATRIX"
)

print(
    "rows:",
    len(final),
)

print(
    "participants:",
    final[
        "pubID"
    ].nunique(),
)

print(
    "detected:",
    int(
        final[
            "detected"
        ].sum()
    ),
)

print(
    "not detected:",
    int(
        (
            ~final[
                "detected"
            ].astype(bool)
        ).sum()
    ),
)


print(
    "\nSaved:"
)

print(
    outdir /
    "246_phase4a_group2_week24_frozen_baseline_features.tsv"
)

print(
    outdir /
    "247_phase4a_group2_week24_frozen_baseline_feature_audit.tsv"
)
