from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


outdir = Path("results/bcr/g002_g003")

df = pd.read_csv(
    outdir / "193_g002_prime_week24_v_mutation_transitions.tsv",
    sep="\t",
)


x = df[
    (df["chain"] == "heavy")
    & (df["v_gene"] == "IGHV1-2*02")
    & (df["transition"] != "unresolved")
    & df["w24_aa"].notna()
    & (df["w8_aa"] == df["germline_aa"])
].copy()

x["acquired"] = (
    x["w24_aa"] != x["germline_aa"]
).astype(int)


support = (
    x.groupby(
        ["position", "germline_aa"]
    )
    .agg(
        acquired_lineages=("acquired", "sum"),
        acquired_participants=(
            "pubID",
            lambda z:
            x.loc[
                z.index[
                    x.loc[z.index, "acquired"] == 1
                ],
                "pubID",
            ].nunique(),
        ),
    )
    .reset_index()
)

support = support[
    (support["acquired_lineages"] >= 5)
    & (support["acquired_participants"] >= 3)
].copy()


keep = set(
    support["position"]
)

x = x[
    x["position"].isin(keep)
].copy()


participant = (
    x.groupby(
        ["group", "pubID", "position"]
    )
    .agg(
        eligible_lineages=("lineage", "nunique"),
        acquired_lineages=("acquired", "sum"),
    )
    .reset_index()
)

participant["mutation_fraction"] = (
    participant["acquired_lineages"]
    / participant["eligible_lineages"]
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


position_cols = [
    c for c in profile.columns
    if c != "group"
    and profile[c].notna().all()
]

profile = profile[
    ["group"] + position_cols
].dropna()


values = profile[
    position_cols
].to_numpy()

groups = profile["group"].to_numpy()


mean1 = values[
    groups == 1
].mean(axis=0)

mean2 = values[
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

indices = range(n)


for group1_idx in combinations(
    indices,
    n1,
):

    group1_idx = set(
        group1_idx
    )

    mask1 = np.array([
        i in group1_idx
        for i in indices
    ])

    m1 = values[
        mask1
    ].mean(axis=0)

    m2 = values[
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


print("GLOBAL IGHV1-2*02 MUTATION PROFILE")
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
    "positions:",
    len(position_cols)
)

print(
    "permutations:",
    len(perm_stats)
)

print(
    "observed mean squared difference:",
    observed
)

print(
    "exact permutation p:",
    p_value
)


position_results = pd.DataFrame({
    "position": position_cols,
    "mean_group1": mean1,
    "mean_group2": mean2,
    "difference_group2_minus_group1":
    difference,
})

position_results = position_results.sort_values(
    "difference_group2_minus_group1",
    key=abs,
    ascending=False,
)


print("\nLARGEST POSITION DIFFERENCES")

print(
    position_results
    .head(20)
    .to_string(
        index=False
    )
)


profile.reset_index().to_csv(
    outdir /
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv",
    sep="\t",
    index=False,
)

position_results.to_csv(
    outdir /
    "199_g002_prime_ighv1_2_02_global_position_differences.tsv",
    sep="\t",
    index=False,
)


summary = pd.DataFrame([{
    "participants": n,
    "group1_participants":
    int((groups == 1).sum()),
    "group2_participants":
    int((groups == 2).sum()),
    "positions": len(position_cols),
    "permutations": len(perm_stats),
    "observed_mean_squared_difference":
    observed,
    "permutation_p": p_value,
}])

summary.to_csv(
    outdir /
    "200_g002_prime_ighv1_2_02_global_steering_test.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv"
)
print(
    outdir /
    "199_g002_prime_ighv1_2_02_global_position_differences.tsv"
)
print(
    outdir /
    "200_g002_prime_ighv1_2_02_global_steering_test.tsv"
)
