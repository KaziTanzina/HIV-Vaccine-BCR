from pathlib import Path

import pandas as pd


outdir = Path(
    "results/bcr/g002_g003"
)

medoid_path = (
    outdir /
    "240_phase4a_week8_lineage_medoid_sequences.tsv"
)

outcome_path = (
    outdir /
    "172_g002_prime_lineage_detection_by_visit.tsv"
)


medoids = pd.read_csv(
    medoid_path,
    sep="\t",
)

outcomes = pd.read_csv(
    outcome_path,
    sep="\t",
)


print(
    "PHASE 4A PREDICTION COHORT AUDIT"
)


print("\nMEDOID TABLE")

print(
    "rows:",
    len(medoids),
)

print(
    "lineages:",
    medoids[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    medoids[
        "pubID"
    ].nunique(),
)

print(
    "groups:"
)

print(
    medoids[
        "group"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nOUTCOME TABLE")

print(
    "rows:",
    len(outcomes),
)

print(
    "columns:",
    len(outcomes.columns),
)


print("\nOUTCOME COLUMNS")

for col in outcomes.columns:

    print(col)


interesting = [
    col
    for col in outcomes.columns
    if any(
        key in col.lower()
        for key in [
            "pub",
            "group",
            "week",
            "lineage",
            "detect",
            "assess",
        ]
    )
]


column_rows = []


for col in interesting:

    s = outcomes[col]

    column_rows.append({
        "column":
        col,
        "dtype":
        str(s.dtype),
        "nonmissing":
        int(s.notna().sum()),
        "unique_nonmissing":
        int(
            s.dropna()
            .astype(str)
            .nunique()
        ),
    })


column_audit = pd.DataFrame(
    column_rows
)


print(
    "\nRELEVANT OUTCOME COLUMN AUDIT"
)

print(
    column_audit.to_string(
        index=False
    )
)


for col in [
    "group",
    "week",
    "weeks",
]:

    if col in outcomes.columns:

        print(
            "\nVALUE COUNTS:",
            col,
        )

        print(
            outcomes[col]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )


if (
    "prospective_lineage"
    in outcomes.columns
):

    overlap = set(
        medoids[
            "prospective_lineage"
        ].astype(str)
    ).intersection(
        set(
            outcomes[
                "prospective_lineage"
            ].dropna().astype(str)
        )
    )

    print(
        "\nMEDOID / OUTCOME LINEAGE OVERLAP"
    )

    print(
        "medoid lineages:",
        medoids[
            "prospective_lineage"
        ].nunique(),
    )

    print(
        "outcome lineages:",
        outcomes[
            "prospective_lineage"
        ].nunique(),
    )

    print(
        "overlap:",
        len(overlap),
    )


group2_medoids = medoids[
    medoids["group"] == 2
].copy()


print(
    "\nGROUP-2 MEDOIDS"
)

print(
    "lineages:",
    group2_medoids[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    group2_medoids[
        "pubID"
    ].nunique(),
)


week_col = None

if "week" in outcomes.columns:

    week_col = "week"

elif "weeks" in outcomes.columns:

    week_col = "weeks"


if (
    "group" in outcomes.columns
    and week_col is not None
):

    g2w24 = outcomes[
        (pd.to_numeric(
            outcomes["group"],
            errors="coerce",
        ) == 2)
        &
        (pd.to_numeric(
            outcomes[week_col],
            errors="coerce",
        ) == 24)
    ].copy()


    print(
        "\nGROUP-2 WEEK-24 OUTCOME ROWS"
    )

    print(
        "rows:",
        len(g2w24),
    )


    if "pubID" in g2w24.columns:

        print(
            "participants:",
            g2w24[
                "pubID"
            ].nunique(),
        )


    if (
        "prospective_lineage"
        in g2w24.columns
    ):

        print(
            "lineages:",
            g2w24[
                "prospective_lineage"
            ].nunique(),
        )


        joined = (
            group2_medoids.merge(
                g2w24,
                on=[
                    "pubID",
                    "group",
                    "prospective_lineage",
                ],
                how="left",
                indicator=True,
                suffixes=(
                    "_week8",
                    "_outcome",
                ),
            )
        )


        print(
            "\nGROUP-2 MEDOID / WEEK-24 JOIN"
        )

        print(
            joined[
                "_merge"
            ]
            .value_counts()
            .to_string()
        )


        detection_columns = [
            col
            for col in g2w24.columns
            if (
                "detect"
                in col.lower()
                or "assess"
                in col.lower()
            )
        ]


        print(
            "\nDETECTION / ASSESSABILITY COLUMNS"
        )

        for col in detection_columns:

            print(
                "\n",
                col,
                sep="",
            )

            print(
                g2w24[col]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )


        joined.to_csv(
            outdir /
            "243_phase4a_group2_week24_medoid_outcome_overlap.tsv",
            sep="\t",
            index=False,
        )


column_audit.to_csv(
    outdir /
    "242_phase4a_prediction_outcome_column_audit.tsv",
    sep="\t",
    index=False,
)


print(
    "\nFIRST 10 OUTCOME ROWS"
)

print(
    outcomes.head(10)
    .to_string(
        index=False
    )
)


print("\nSaved:")

print(
    outdir /
    "242_phase4a_prediction_outcome_column_audit.tsv"
)

if (
    "group" in outcomes.columns
    and week_col is not None
    and "prospective_lineage"
    in outcomes.columns
):

    print(
        outdir /
        "243_phase4a_group2_week24_medoid_outcome_overlap.tsv"
    )
