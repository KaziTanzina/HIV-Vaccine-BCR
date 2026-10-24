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


cdrh2_positions = (
    mapping[
        mapping["region"] == "CDRH2"
    ]["position"]
    .astype(int)
    .astype(str)
    .tolist()
)

cdrh2_positions = [
    p for p in cdrh2_positions
    if p in profile.columns
]


print("CDRH2 POSITIONS")
print(cdrh2_positions)


rows = []


for removed in profile["pubID"]:

    x = profile[
        profile["pubID"] != removed
    ].copy()

    values = x[
        cdrh2_positions
    ].to_numpy(
        dtype=float
    )

    groups = x["group"].to_numpy()


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


    n = len(x)

    n1 = int(
        (groups == 1).sum()
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


    removed_group = int(
        profile.loc[
            profile["pubID"] == removed,
            "group",
        ].iloc[0]
    )


    rows.append({
        "removed_pubID": removed,
        "removed_group": removed_group,
        "remaining_participants": n,
        "group1_participants":
        int((groups == 1).sum()),
        "group2_participants":
        int((groups == 2).sum()),
        "cdrh2_positions":
        len(cdrh2_positions),
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


print("\nCDRH2 LEAVE-ONE-PARTICIPANT-OUT")

print(
    results.to_string(
        index=False
    )
)


print("\nSUMMARY")

print(
    "runs:",
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
    "211_g002_prime_ighv1_2_02_cdrh2_leave_one_out.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "211_g002_prime_ighv1_2_02_cdrh2_leave_one_out.tsv"
)
