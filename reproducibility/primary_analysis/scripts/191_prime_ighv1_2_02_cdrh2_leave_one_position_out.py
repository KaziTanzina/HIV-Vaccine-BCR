from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


outdir = Path("results/bcr/g002_g003")

profile = pd.read_csv(
    outdir /
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv",
    sep="\t",
)

mapping = pd.read_csv(
    outdir /
    "209_g002_prime_ighv1_2_02_coordinate_region_mapping.tsv",
    sep="\t",
)

differences = pd.read_csv(
    outdir /
    "201_g002_prime_ighv1_2_02_centered_position_differences.tsv",
    sep="\t",
)


cdrh2 = (
    mapping[
        mapping["region"] == "CDRH2"
    ][
        [
            "position",
            "germline_aa",
        ]
    ]
    .drop_duplicates()
    .copy()
)

cdrh2["position"] = (
    cdrh2["position"]
    .astype(int)
    .astype(str)
)

cdrh2_positions = [
    p for p in cdrh2["position"]
    if p in profile.columns
]


difference_map = dict(
    zip(
        differences["position"]
        .astype(int)
        .astype(str),
        differences[
            "centered_difference_group2_minus_group1"
        ],
    )
)

germline_map = dict(
    zip(
        cdrh2["position"],
        cdrh2["germline_aa"],
    )
)


groups = profile["group"].to_numpy()

n = len(profile)

n1 = int(
    (groups == 1).sum()
)


rows = []


for removed_position in cdrh2_positions:

    positions = [
        p for p in cdrh2_positions
        if p != removed_position
    ]

    values = profile[
        positions
    ].to_numpy(
        dtype=float
    )


    participant_mean = values.mean(
        axis=1
    )

    centered = (
        values
        - participant_mean[:, None]
    )


    mean1 = centered[
        groups == 1
    ].mean(axis=0)

    mean2 = centered[
        groups == 2
    ].mean(axis=0)


    observed = np.mean(
        (mean2 - mean1) ** 2
    )


    perm_stats = []


    for group1_idx in combinations(
        range(n),
        n1,
    ):

        mask1 = np.zeros(
            n,
            dtype=bool,
        )

        mask1[
            list(group1_idx)
        ] = True

        m1 = centered[
            mask1
        ].mean(axis=0)

        m2 = centered[
            ~mask1
        ].mean(axis=0)

        perm_stats.append(
            np.mean(
                (m2 - m1) ** 2
            )
        )


    perm_stats = np.array(
        perm_stats
    )

    p_value = (
        perm_stats >= observed
    ).mean()


    rows.append({
        "removed_position":
        int(removed_position),
        "removed_germline_aa":
        germline_map.get(
            removed_position
        ),
        "removed_original_centered_difference":
        difference_map.get(
            removed_position
        ),
        "remaining_positions":
        len(positions),
        "participants":
        n,
        "permutations":
        len(perm_stats),
        "observed_centered_difference":
        observed,
        "permutation_p":
        p_value,
    })


results = pd.DataFrame(rows)

results = results.sort_values(
    "permutation_p",
    ascending=False,
)


print("CDRH2 LEAVE-ONE-POSITION-OUT")

print(
    results.to_string(
        index=False
    )
)


print("\nSUMMARY")

print(
    "positions tested:",
    len(results)
)

print(
    "p < 0.05:",
    (results["permutation_p"] < 0.05).sum()
)

print(
    "p < 0.01:",
    (results["permutation_p"] < 0.01).sum()
)

print(
    "minimum p:",
    results["permutation_p"].min()
)

print(
    "maximum p:",
    results["permutation_p"].max()
)


results.to_csv(
    outdir /
    "212_g002_prime_ighv1_2_02_cdrh2_leave_one_position_out.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "212_g002_prime_ighv1_2_02_cdrh2_leave_one_position_out.tsv"
)
