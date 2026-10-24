from pathlib import Path

import pandas as pd
from sadie.airr import LinkedAirrTable
from sadie.cluster import Cluster
from sklearn.metrics import adjusted_rand_score


input_path = Path(
    "data/external/g002_g003/g002/final_df.feather"
)

reference_path = Path(
    "data/external/g002_g003/g002/"
    "author_prime_clusters/"
    "final_df_prime_pad_somaticFalse_CDR3_"
    "link_average_group_v_call_cluster_n3.feather"
)

processed_path = Path(
    "data/processed/g002_g003/"
    "g002_groups1_2_week8_eod_prime_"
    "reclustered_n3.feather"
)

result_path = Path(
    "results/bcr/g002_g003/"
    "163_g002_prime_week8_recluster_validation.tsv"
)

changed_path = Path(
    "results/bcr/g002_g003/"
    "164_g002_prime_week8_recluster_changed_cells.tsv"
)


df = pd.read_feather(input_path)

df["group"] = pd.to_numeric(
    df["group"],
    errors="coerce",
)

df["weeks"] = pd.to_numeric(
    df["weeks"],
    errors="coerce",
)


week8 = df[
    df["group"].isin([1, 2])
    &
    (df["weeks"] == 8)
    &
    (df["probe_set"] == "eODGT8")
    &
    (df["is_vrc01_class"] == True)
    &
    (df["top_c_call"] == "IGHG")
].copy()


print("WEEK-8 INPUT")
print("cells:", len(week8))
print(
    "participants:",
    week8["pubID"].nunique(),
)

print("\nBY GROUP")
print(
    week8.groupby("group")
    .agg(
        cells=("cellid", "size"),
        participants=("pubID", "nunique"),
    )
    .to_string()
)


all_clustered = []


for pubid, participant_df in week8.groupby(
    "pubID",
    sort=True,
):

    work = participant_df.drop(
        columns=[
            "cluster",
            "is_centroid",
        ],
        errors="ignore",
    ).copy()

    airr = LinkedAirrTable(
        work,
        key_column="cellid",
    )

    model = Cluster(
        airr,
        linkage="average",
        groupby=[
            "v_call_top_heavy",
            "v_call_top_light",
            "junction_aa_length_heavy",
            "top_c_call",
            "is_vrc01_class",
        ],
        lookup=[
            "cdr3_aa_heavy",
            "cdr3_aa_light",
        ],
        pad_somatic=False,
    )

    clustered = pd.DataFrame(
        model.cluster(3)
    )

    all_clustered.append(
        clustered
    )


early = pd.concat(
    all_clustered,
    ignore_index=True,
)


early["prospective_lineage"] = (
    early["pubID"].astype(str)
    + "::"
    + early["cluster"].astype(str)
)


print("\nPROSPECTIVE WEEK-8 CLUSTERING")
print("cells:", len(early))
print(
    "participants:",
    early["pubID"].nunique(),
)
print(
    "lineages:",
    early["prospective_lineage"].nunique(),
)


processed_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

early.to_feather(
    processed_path
)


reference = pd.read_feather(
    reference_path
)

reference["group"] = pd.to_numeric(
    reference["group"],
    errors="coerce",
)

reference["weeks"] = pd.to_numeric(
    reference["weeks"],
    errors="coerce",
)


reference = reference[
    reference["group"].isin([1, 2])
    &
    (reference["weeks"] == 8)
    &
    (reference["probe_set"] == "eODGT8")
    &
    (reference["is_vrc01_class"] == True)
    &
    (reference["top_c_call"] == "IGHG")
].copy()


reference["author_lineage"] = (
    reference["pubID"].astype(str)
    + "::"
    + reference["cluster"].astype(str)
)


prospective_ids = set(
    early["cellid"]
)

reference_ids = set(
    reference["cellid"]
)

common_ids = (
    prospective_ids
    & reference_ids
)


print("\nCELL-SET AGREEMENT")
print(
    "prospective:",
    len(prospective_ids),
)
print(
    "author:",
    len(reference_ids),
)
print(
    "common:",
    len(common_ids),
)
print(
    "prospective-only:",
    len(
        prospective_ids
        - reference_ids
    ),
)
print(
    "author-only:",
    len(
        reference_ids
        - prospective_ids
    ),
)


comparison = (
    early[
        early["cellid"].isin(
            common_ids
        )
    ][
        [
            "cellid",
            "pubID",
            "group",
            "prospective_lineage",
        ]
    ]
    .merge(
        reference[
            reference["cellid"].isin(
                common_ids
            )
        ][
            [
                "cellid",
                "author_lineage",
            ]
        ],
        on="cellid",
        how="inner",
        validate="one_to_one",
    )
)


ari = adjusted_rand_score(
    comparison[
        "prospective_lineage"
    ],
    comparison[
        "author_lineage"
    ],
)


def membership_map(
    dataframe,
    lineage_column,
):

    mapping = {}

    for _, g in dataframe.groupby(
        lineage_column
    ):

        members = frozenset(
            g["cellid"].tolist()
        )

        for cellid in members:
            mapping[cellid] = members

    return mapping


prospective_map = membership_map(
    comparison,
    "prospective_lineage",
)

author_map = membership_map(
    comparison,
    "author_lineage",
)


identical = [
    cellid
    for cellid in common_ids
    if prospective_map[cellid]
    == author_map[cellid]
]

changed = [
    cellid
    for cellid in common_ids
    if prospective_map[cellid]
    != author_map[cellid]
]


print("\nPARTITION AGREEMENT")
print("ARI:", ari)

print(
    "cells with identical lineage membership:",
    len(identical),
)

print(
    "cells with different lineage membership:",
    len(changed),
)

print(
    "percent identical:",
    100
    * len(identical)
    / len(common_ids),
)


rows = []

for group in [1, 2]:

    ids = set(
        comparison.loc[
            comparison["group"] == group,
            "cellid",
        ]
    )

    same = sum(
        prospective_map[cellid]
        == author_map[cellid]
        for cellid in ids
    )

    rows.append({
        "group": group,
        "cells": len(ids),
        "identical_membership_cells": same,
        "different_membership_cells": (
            len(ids) - same
        ),
        "percent_identical": (
            100 * same / len(ids)
            if ids
            else float("nan")
        ),
    })


summary = pd.DataFrame(
    rows
)

summary["overall_ari"] = ari


print("\nARM-SPECIFIC AGREEMENT")
print(
    summary.to_string(
        index=False
    )
)


result_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    result_path,
    sep="\t",
    index=False,
)


if changed:

    changed_df = comparison[
        comparison["cellid"].isin(
            changed
        )
    ].copy()

    changed_df.to_csv(
        changed_path,
        sep="\t",
        index=False,
    )


print("\nSaved:")
print(processed_path)
print(result_path)

if changed:
    print(changed_path)
