from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


outdir = Path("results/bcr/g002_g003")

df = pd.read_csv(
    outdir /
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv",
    sep="\t",
)


position_cols = [
    c for c in df.columns
    if c not in ["pubID", "group"]
]


values = df[position_cols].to_numpy(
    dtype=float
)

groups = df["group"].to_numpy()


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


n = len(df)

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


print("CENTERED IGHV1-2*02 MUTATION PROFILE")
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


print("\nPARTICIPANT MEAN MUTATION FRACTION")

for group in [1, 2]:

    x = participant_mean[
        groups == group
    ]

    print(
        "group",
        group,
        "median=",
        np.median(x),
        "mean=",
        np.mean(x),
    )


print(
    "\nobserved centered mean squared difference:",
    observed
)

print(
    "exact permutation p:",
    p_value
)


positions = pd.DataFrame({
    "position": position_cols,
    "centered_mean_group1": mean1,
    "centered_mean_group2": mean2,
    "centered_difference_group2_minus_group1":
    difference,
})

positions = positions.sort_values(
    "centered_difference_group2_minus_group1",
    key=abs,
    ascending=False,
)


print("\nLARGEST CENTERED POSITION DIFFERENCES")

print(
    positions
    .head(20)
    .to_string(
        index=False
    )
)


summary = pd.DataFrame([{
    "participants": n,
    "group1_participants":
    int((groups == 1).sum()),
    "group2_participants":
    int((groups == 2).sum()),
    "positions": len(position_cols),
    "permutations": len(perm_stats),
    "observed_centered_mean_squared_difference":
    observed,
    "permutation_p": p_value,
}])


positions.to_csv(
    outdir /
    "201_g002_prime_ighv1_2_02_centered_position_differences.tsv",
    sep="\t",
    index=False,
)

summary.to_csv(
    outdir /
    "202_g002_prime_ighv1_2_02_centered_steering_test.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "201_g002_prime_ighv1_2_02_centered_position_differences.tsv"
)
print(
    outdir /
    "202_g002_prime_ighv1_2_02_centered_steering_test.tsv"
)
