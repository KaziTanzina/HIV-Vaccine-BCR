from pathlib import Path
import pandas as pd


outdir = Path("results/bcr/g002_g003")

full = pd.read_feather(
    "data/external/g002_g003/g002/final_df.feather"
)

validation = pd.read_csv(
    outdir /
    "190_g002_prime_v_mutation_call_validation.tsv",
    sep="\t",
)

differences = pd.read_csv(
    outdir /
    "201_g002_prime_ighv1_2_02_centered_position_differences.tsv",
    sep="\t",
)

transitions = pd.read_csv(
    outdir /
    "193_g002_prime_week24_v_mutation_transitions.tsv",
    sep="\t",
)


valid = validation[
    (validation["chain"] == "heavy")
    & (validation["v_gene"] == "IGHV1-2*02")
    & (validation["sequence_has_gap"] == False)
    & (validation["germline_has_gap"] == False)
    & (validation["abs_difference"] <= 0.001)
].copy()


cols = [
    "cellid",
    "v_sequence_start_heavy",
    "v_sequence_end_heavy",
    "fwr1_start_heavy",
    "fwr1_end_heavy",
    "cdr1_start_heavy",
    "cdr1_end_heavy",
    "fwr2_start_heavy",
    "fwr2_end_heavy",
    "cdr2_start_heavy",
    "cdr2_end_heavy",
    "fwr3_start_heavy",
    "fwr3_end_heavy",
    "cdr3_start_heavy",
    "cdr3_end_heavy",
    "fwr4_start_heavy",
    "fwr4_end_heavy",
]

cells = full[
    full["cellid"].isin(valid["cellid"])
][cols].copy()


regions = [
    ("FWR1", "fwr1_start_heavy", "fwr1_end_heavy"),
    ("CDRH1", "cdr1_start_heavy", "cdr1_end_heavy"),
    ("FWR2", "fwr2_start_heavy", "fwr2_end_heavy"),
    ("CDRH2", "cdr2_start_heavy", "cdr2_end_heavy"),
    ("FWR3", "fwr3_start_heavy", "fwr3_end_heavy"),
    ("CDRH3", "cdr3_start_heavy", "cdr3_end_heavy"),
    ("FWR4", "fwr4_start_heavy", "fwr4_end_heavy"),
]


positions = (
    pd.to_numeric(
        differences["position"]
    )
    .astype(int)
    .tolist()
)


rows = []


for _, cell in cells.iterrows():

    v_start = int(
        cell["v_sequence_start_heavy"]
    )

    v_end = int(
        cell["v_sequence_end_heavy"]
    )

    for position in positions:

        codon_start = (
            v_start
            + 3 * (position - 1)
        )

        codon_end = (
            codon_start + 2
        )

        if codon_end > v_end:
            continue

        region_name = None

        for region, start_col, end_col in regions:

            start = cell[start_col]
            end = cell[end_col]

            if pd.isna(start) or pd.isna(end):
                continue

            if (
                codon_start >= start
                and codon_end <= end
            ):
                region_name = region
                break

        rows.append({
            "cellid": cell["cellid"],
            "position": position,
            "codon_start": codon_start,
            "codon_end": codon_end,
            "region": region_name,
        })


mapping = pd.DataFrame(rows)


counts = (
    mapping.dropna(
        subset=["region"]
    )
    .groupby(
        ["position", "region"]
    )
    .size()
    .reset_index(
        name="cells"
    )
)


totals = (
    mapping.groupby("position")
    .size()
    .reset_index(
        name="assessable_cells"
    )
)


best = (
    counts.sort_values(
        ["position", "cells"],
        ascending=[True, False],
    )
    .drop_duplicates("position")
    .merge(
        totals,
        on="position",
        how="right",
    )
)


best["region_agreement"] = (
    best["cells"]
    / best["assessable_cells"]
)


germline = (
    transitions[
        (transitions["chain"] == "heavy")
        & (transitions["v_gene"] == "IGHV1-2*02")
    ][
        ["position", "germline_aa"]
    ]
    .drop_duplicates()
)


differences["position"] = pd.to_numeric(
    differences["position"]
).astype(int)


annotated = differences.merge(
    germline,
    on="position",
    how="left",
)

annotated = annotated.merge(
    best[
        [
            "position",
            "region",
            "cells",
            "assessable_cells",
            "region_agreement",
        ]
    ],
    on="position",
    how="left",
)


print("COORDINATE-BASED REGION MAPPING")
print("valid cells:", len(cells))
print("positions:", len(annotated))


print("\nPOSITIONS BY REGION")

print(
    annotated.groupby(
        "region",
        dropna=False,
    )
    .size()
    .to_string()
)


print("\nREGION AGREEMENT")

print(
    annotated[
        "region_agreement"
    ]
    .describe()
    .to_string()
)


print("\nTOP CENTERED DIFFERENCES")

print(
    annotated
    .sort_values(
        "centered_difference_group2_minus_group1",
        key=abs,
        ascending=False,
    )
    .head(20)
    .to_string(
        index=False
    )
)


annotated.to_csv(
    outdir /
    "209_g002_prime_ighv1_2_02_coordinate_region_mapping.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "209_g002_prime_ighv1_2_02_coordinate_region_mapping.tsv"
)
