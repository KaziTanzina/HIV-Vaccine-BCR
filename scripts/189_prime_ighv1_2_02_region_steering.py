from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


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


position_region = (
    mapping[
        [
            "position",
            "region",
        ]
    ]
    .dropna()
    .drop_duplicates()
)

position_region["position"] = (
    position_region["position"]
    .astype(int)
    .astype(str)
)


groups = profile["group"].to_numpy()

n = len(profile)

n1 = int(
    (groups == 1).sum()
)


rows = []


for region in [
    "FWR1",
    "CDRH1",
    "FWR2",
    "CDRH2",
    "FWR3",
    "CDRH3",
]:

    positions = (
        position_region[
            position_region["region"] == region
        ]["position"]
        .tolist()
    )

    positions = [
        p for p in positions
        if p in profile.columns
    ]

    if len(positions) == 0:
        continue


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
        "region": region,
        "positions": len(positions),
        "participants": n,
        "group1_participants":
        int((groups == 1).sum()),
        "group2_participants":
        int((groups == 2).sum()),
        "permutations": len(perm_stats),
        "observed_centered_difference":
        observed,
        "permutation_p": p_value,
    })


results = pd.DataFrame(rows)


results["fdr"] = multipletests(
    results["permutation_p"],
    method="fdr_bh",
)[1]


results = results.sort_values(
    "permutation_p"
)


print("REGION-LEVEL STEERING")

print(
    results.to_string(
        index=False
    )
)


results.to_csv(
    outdir /
    "210_g002_prime_ighv1_2_02_region_steering.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "210_g002_prime_ighv1_2_02_region_steering.tsv"
)
