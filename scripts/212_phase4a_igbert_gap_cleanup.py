from pathlib import Path

import pandas as pd


input_path = Path(
    "results/bcr/g002_g003/"
    "252_phase4a_group2_week24_igbert_input_readiness.tsv"
)

outdir = Path(
    "results/bcr/g002_g003"
)


df = pd.read_csv(
    input_path,
    sep="\t",
)


heavy_col = "sequence_alignment_aa_heavy"
light_col = "sequence_alignment_aa_light"


required = [
    "prospective_lineage",
    "pubID_cohort",
    "group_cohort",
    "detected",
    "representative_cellid",
    heavy_col,
    light_col,
]


print("PHASE 4A IGBERT GAP CLEANUP AUDIT")

print("\nREQUIRED COLUMN CHECK")

for col in required:
    print(
        col,
        "present=",
        col in df.columns,
    )

    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )


print("\nINPUT COHORT")

print(
    "rows:",
    len(df),
)

print(
    "lineages:",
    df[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    df[
        "pubID_cohort"
    ].nunique(),
)

print(
    "detected:",
    int(
        df["detected"]
        .astype(bool)
        .sum()
    ),
)


allowed = set(
    "ACDEFGHIKLMNPQRSTVWYXBZUO"
)


def invalid_characters(sequence, allow_gap=False):

    sequence = str(
        sequence
    ).upper()

    valid = set(
        allowed
    )

    if allow_gap:
        valid.add("-")

    invalid = sorted(
        set(sequence)
        - valid
    )

    return "".join(
        invalid
    )


df["heavy_gap_count"] = (
    df[heavy_col]
    .astype(str)
    .str.count("-")
)

df["light_gap_count"] = (
    df[light_col]
    .astype(str)
    .str.count("-")
)


df[
    "heavy_other_invalid_before"
] = (
    df[heavy_col]
    .astype(str)
    .apply(
        lambda x:
        invalid_characters(
            x,
            allow_gap=True,
        )
    )
)


df[
    "light_other_invalid_before"
] = (
    df[light_col]
    .astype(str)
    .apply(
        lambda x:
        invalid_characters(
            x,
            allow_gap=True,
        )
    )
)


print("\nGAP COUNTS")

print(
    "lineages with heavy gaps:",
    int(
        (
            df["heavy_gap_count"]
            > 0
        ).sum()
    ),
)

print(
    "lineages with light gaps:",
    int(
        (
            df["light_gap_count"]
            > 0
        ).sum()
    ),
)

print(
    "lineages with any gaps:",
    int(
        (
            (
                df["heavy_gap_count"]
                > 0
            )
            |
            (
                df["light_gap_count"]
                > 0
            )
        ).sum()
    ),
)

print(
    "total heavy gap characters:",
    int(
        df[
            "heavy_gap_count"
        ].sum()
    ),
)

print(
    "total light gap characters:",
    int(
        df[
            "light_gap_count"
        ].sum()
    ),
)


print("\nNON-GAP INVALID CHARACTERS BEFORE CLEANING")

print(
    "heavy rows:",
    int(
        (
            df[
                "heavy_other_invalid_before"
            ]
            != ""
        ).sum()
    ),
)

print(
    "light rows:",
    int(
        (
            df[
                "light_other_invalid_before"
            ]
            != ""
        ).sum()
    ),
)


if (
    (
        df[
            "heavy_other_invalid_before"
        ]
        != ""
    ).any()
    or
    (
        df[
            "light_other_invalid_before"
        ]
        != ""
    ).any()
):

    print(
        "\nUnexpected non-gap characters found."
    )

    print(
        df.loc[
            (
                df[
                    "heavy_other_invalid_before"
                ]
                != ""
            )
            |
            (
                df[
                    "light_other_invalid_before"
                ]
                != ""
            ),
            [
                "prospective_lineage",
                "heavy_other_invalid_before",
                "light_other_invalid_before",
            ],
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    raise ValueError(
        "Stopping because invalid "
        "characters other than '-' exist."
    )


df[
    "igbert_heavy_sequence"
] = (
    df[heavy_col]
    .astype(str)
    .str.upper()
    .str.replace(
        "-",
        "",
        regex=False,
    )
)


df[
    "igbert_light_sequence"
] = (
    df[light_col]
    .astype(str)
    .str.upper()
    .str.replace(
        "-",
        "",
        regex=False,
    )
)


df[
    "heavy_length_before"
] = (
    df[heavy_col]
    .astype(str)
    .str.len()
)

df[
    "light_length_before"
] = (
    df[light_col]
    .astype(str)
    .str.len()
)

df[
    "heavy_length_after"
] = (
    df[
        "igbert_heavy_sequence"
    ]
    .str.len()
)

df[
    "light_length_after"
] = (
    df[
        "igbert_light_sequence"
    ]
    .str.len()
)


df[
    "heavy_invalid_after"
] = (
    df[
        "igbert_heavy_sequence"
    ]
    .apply(
        invalid_characters
    )
)

df[
    "light_invalid_after"
] = (
    df[
        "igbert_light_sequence"
    ]
    .apply(
        invalid_characters
    )
)


df[
    "igbert_ready_after_cleanup"
] = (
    (
        df[
            "heavy_length_after"
        ]
        > 0
    )
    &
    (
        df[
            "light_length_after"
        ]
        > 0
    )
    &
    (
        df[
            "heavy_invalid_after"
        ]
        == ""
    )
    &
    (
        df[
            "light_invalid_after"
        ]
        == ""
    )
)


df[
    "estimated_token_length_after"
] = (
    df[
        "heavy_length_after"
    ]
    +
    df[
        "light_length_after"
    ]
    + 3
)


print("\nAFTER GAP REMOVAL")

print(
    df[
        "igbert_ready_after_cleanup"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print(
    "\nready fraction:",
    df[
        "igbert_ready_after_cleanup"
    ].mean(),
)


print("\nLENGTH AFTER CLEANUP")

print(
    df[
        [
            "heavy_length_after",
            "light_length_after",
            "estimated_token_length_after",
        ]
    ]
    .describe()
    .to_string()
)


modified = df[
    (
        df["heavy_gap_count"]
        > 0
    )
    |
    (
        df["light_gap_count"]
        > 0
    )
].copy()


print("\nMODIFIED LINEAGES")

print(
    "lineages:",
    len(modified),
)

if len(modified) > 0:

    print(
        modified[
            [
                "prospective_lineage",
                "pubID_cohort",
                "heavy_gap_count",
                "light_gap_count",
                "heavy_length_before",
                "heavy_length_after",
                "light_length_before",
                "light_length_after",
            ]
        ]
        .to_string(
            index=False
        )
    )


if not df[
    "igbert_ready_after_cleanup"
].all():

    raise ValueError(
        "Some sequences remain invalid "
        "after removing alignment gaps."
    )


clean = df[
    [
        "prospective_lineage",
        "pubID_cohort",
        "group_cohort",
        "representative_cellid",
        "igbert_heavy_sequence",
        "igbert_light_sequence",
        "heavy_length_after",
        "light_length_after",
        "estimated_token_length_after",
    ]
].copy()


clean = clean.rename(
    columns={
        "pubID_cohort":
        "pubID",
        "group_cohort":
        "group",
    }
)


print("\nFINAL LABEL-BLIND IGBERT INPUT")

print(
    "rows:",
    len(clean),
)

print(
    "lineages:",
    clean[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    clean[
        "pubID"
    ].nunique(),
)

print(
    "duplicate lineages:",
    int(
        clean[
            "prospective_lineage"
        ].duplicated().sum()
    ),
)


clean.to_csv(
    outdir /
    "253_phase4a_group2_week24_igbert_clean_sequences.tsv",
    sep="\t",
    index=False,
)


audit_cols = [
    "prospective_lineage",
    "pubID_cohort",
    "representative_cellid",
    "heavy_gap_count",
    "light_gap_count",
    "heavy_length_before",
    "heavy_length_after",
    "light_length_before",
    "light_length_after",
    "heavy_invalid_after",
    "light_invalid_after",
    "estimated_token_length_after",
    "igbert_ready_after_cleanup",
]


df[
    audit_cols
].to_csv(
    outdir /
    "254_phase4a_group2_week24_igbert_gap_cleanup_audit.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")

print(
    outdir /
    "253_phase4a_group2_week24_igbert_clean_sequences.tsv"
)

print(
    outdir /
    "254_phase4a_group2_week24_igbert_gap_cleanup_audit.tsv"
)
