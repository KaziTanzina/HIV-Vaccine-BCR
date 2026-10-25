from pathlib import Path
import pandas as pd


outdir = Path("results/bcr/g002_g003")

early = pd.read_feather(
    "data/processed/g002_g003/"
    "g002_groups1_2_week8_eod_prime_reclustered_n3.feather"
)

full = pd.read_feather(
    "data/external/g002_g003/g002/final_df.feather"
)

matches = pd.read_csv(
    outdir / "171_g002_prime_later_cg28v2_cell_matches.tsv",
    sep="\t",
)


if "prospective_lineage" not in early.columns:
    early["prospective_lineage"] = (
        early["pubID"].astype(str)
        + "::"
        + early["cluster"].astype(str)
    )


w24 = matches[
    (matches["weeks"] == 24)
    & (matches["match_status"] == "unique")
].copy()

tracked = set(
    w24["assigned_lineage"]
)

early = early[
    early["prospective_lineage"].isin(tracked)
].copy()

early["visit"] = "week8"
early["lineage"] = early["prospective_lineage"]


w24 = w24.merge(
    full[
        [
            "cellid",
            "v_call_top_heavy",
            "v_call_top_light",
            "v_sequence_alignment_aa_heavy",
            "v_germline_alignment_aa_heavy",
            "v_sequence_alignment_aa_light",
            "v_germline_alignment_aa_light",
            "v_mutation_aa_heavy",
            "v_mutation_aa_light",
        ]
    ],
    on="cellid",
    how="left",
    validate="one_to_one",
)

w24["visit"] = "week24"
w24["lineage"] = w24["assigned_lineage"]


def call_mutations(seq, germ):

    mutations = []
    assessable = 0
    position = 0

    for s, g in zip(str(seq), str(germ)):

        if g == "-":
            continue

        position += 1

        if g == "X":
            continue

        if s in ["-", "X"]:
            continue

        assessable += 1

        if s != g:
            mutations.append(
                f"{position}:{g}>{s}"
            )

    return mutations, assessable


rows = []


for visit, df in [
    ("week8", early),
    ("week24", w24),
]:

    for _, row in df.iterrows():

        for chain in ["heavy", "light"]:

            seq = row[
                f"v_sequence_alignment_aa_{chain}"
            ]

            germ = row[
                f"v_germline_alignment_aa_{chain}"
            ]

            mutations, assessable = call_mutations(
                seq,
                germ,
            )

            if assessable == 0:
                local_fraction = None
            else:
                local_fraction = (
                    len(mutations)
                    / assessable
                )

            sadie_fraction = row[
                f"v_mutation_aa_{chain}"
            ]

            rows.append({
                "visit": visit,
                "cellid": row["cellid"],
                "lineage": row["lineage"],
                "chain": chain,
                "v_gene": row[
                    f"v_call_top_{chain}"
                ],
                "assessable_aa": assessable,
                "mutation_count": len(mutations),
                "mutations": ";".join(mutations),
                "local_mutation_fraction":
                local_fraction,
                "sadie_mutation_fraction":
                sadie_fraction,
                "abs_difference":
                abs(
                    local_fraction
                    - sadie_fraction
                ),
                "sequence_has_gap":
                "-" in str(seq),
                "germline_has_gap":
                "-" in str(germ),
            })


validation = pd.DataFrame(rows)


summary = (
    validation.groupby(
        ["visit", "chain"]
    )
    .agg(
        cells=("cellid", "size"),
        median_assessable_aa=(
            "assessable_aa",
            "median",
        ),
        median_mutation_count=(
            "mutation_count",
            "median",
        ),
        median_abs_difference=(
            "abs_difference",
            "median",
        ),
        max_abs_difference=(
            "abs_difference",
            "max",
        ),
        gap_cells=(
            "sequence_has_gap",
            "sum",
        ),
    )
    .reset_index()
)


print("MUTATION CALL VALIDATION")

print(
    summary.to_string(
        index=False
    )
)


print("\nCELLS WITH ABSOLUTE DIFFERENCE > 0.01")

different = validation[
    validation["abs_difference"] > 0.01
]

print(
    different[
        [
            "visit",
            "cellid",
            "chain",
            "v_gene",
            "assessable_aa",
            "mutation_count",
            "local_mutation_fraction",
            "sadie_mutation_fraction",
            "abs_difference",
            "sequence_has_gap",
            "germline_has_gap",
        ]
    ].to_string(
        index=False
    )
)


validation.to_csv(
    outdir /
    "190_g002_prime_v_mutation_call_validation.tsv",
    sep="\t",
    index=False,
)

summary.to_csv(
    outdir /
    "191_g002_prime_v_mutation_call_validation_summary.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "190_g002_prime_v_mutation_call_validation.tsv"
)
print(
    outdir /
    "191_g002_prime_v_mutation_call_validation_summary.tsv"
)
