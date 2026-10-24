from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


outdir = Path("results/bcr/g002_g003")

transitions = pd.read_csv(
    outdir /
    "193_g002_prime_week24_v_mutation_transitions.tsv",
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
    .unique()
    .tolist()
)


x = transitions[
    (transitions["chain"] == "heavy")
    & (transitions["v_gene"] == "IGHV1-2*02")
    & (transitions["position"].isin(cdrh2_positions))
    & (transitions["transition"] != "unresolved")
].copy()


x["week8_mutated"] = (
    x["w8_aa"] != x["germline_aa"]
).astype(int)


participant = (
    x.groupby(
        [
            "pubID",
            "group",
            "position",
        ]
    )
    .agg(
        assessable_lineages=(
            "lineage",
            "nunique",
        ),
        mutated_lineages=(
            "week8_mutated",
            "sum",
        ),
    )
    .reset_index()
)


participant["mutation_fraction"] = (
    participant["mutated_lineages"]
    / participant["assessable_lineages"]
)


profile = participant.pivot(
    index="pubID",
    columns="position",
    values="mutation_fraction",
)


group_map = (
    participant[
        ["pubID", "group"]
    ]
    .drop_duplicates()
    .set_index("pubID")["group"]
)


profile["group"] = group_map


complete_positions = [
    p for p in cdrh2_positions
    if p in profile.columns
    and profile[p].notna().all()
]


profile = profile[
    ["group"] + complete_positions
].dropna()


values = profile[
    complete_positions
].to_numpy(
    dtype=float
)

groups = profile[
    "group"
].to_numpy()


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


difference = mean2 - mean1


observed = np.mean(
    difference ** 2
)


n = len(profile)

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


print("WEEK-8 CDRH2 BASELINE PROFILE")
print("participants:", n)

print(
    "group 1:",
    (groups == 1).sum()
)

print(
    "group 2:",
    (groups == 2).sum()
)

print(
    "CDRH2 positions:",
    len(complete_positions)
)

print(
    "positions:",
    complete_positions
)

print(
    "permutations:",
    len(perm_stats)
)


print("\nPARTICIPANT WEEK-8 MEAN MUTATION FRACTION")

for group in [1, 2]:

    z = participant_mean[
        groups == group
    ]

    print(
        "group",
        group,
        "median=",
        np.median(z),
        "mean=",
        np.mean(z),
    )


print(
    "\nobserved centered mean squared difference:",
    observed
)

print(
    "exact permutation p:",
    p_value
)


position_results = pd.DataFrame({
    "position": complete_positions,
    "centered_mean_group1": mean1,
    "centered_mean_group2": mean2,
    "centered_difference_group2_minus_group1":
    difference,
})


position_results = position_results.sort_values(
    "centered_difference_group2_minus_group1",
    key=abs,
    ascending=False,
)


print("\nWEEK-8 POSITION DIFFERENCES")

print(
    position_results.to_string(
        index=False
    )
)


profile.reset_index().to_csv(
    outdir /
    "213_g002_prime_ighv1_2_02_week8_cdrh2_profile.tsv",
    sep="\t",
    index=False,
)


position_results.to_csv(
    outdir /
    "214_g002_prime_ighv1_2_02_week8_cdrh2_position_differences.tsv",
    sep="\t",
    index=False,
)


summary = pd.DataFrame([{
    "participants": n,
    "group1_participants":
    int((groups == 1).sum()),
    "group2_participants":
    int((groups == 2).sum()),
    "positions":
    len(complete_positions),
    "permutations":
    len(perm_stats),
    "observed_centered_mean_squared_difference":
    observed,
    "permutation_p":
    p_value,
}])


summary.to_csv(
    outdir /
    "215_g002_prime_ighv1_2_02_week8_cdrh2_baseline_test.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "213_g002_prime_ighv1_2_02_week8_cdrh2_profile.tsv"
)

print(
    outdir /
    "214_g002_prime_ighv1_2_02_week8_cdrh2_position_differences.tsv"
)

print(
    outdir /
    "215_g002_prime_ighv1_2_02_week8_cdrh2_baseline_test.tsv"
)
