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

validation = pd.read_csv(
    outdir / "190_g002_prime_v_mutation_call_validation.tsv",
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
        ]
    ],
    on="cellid",
    how="left",
    validate="one_to_one",
)

w24["lineage"] = w24["assigned_lineage"]


validation["valid"] = (
    ~validation["sequence_has_gap"]
    & ~validation["germline_has_gap"]
    & (validation["abs_difference"] <= 0.001)
)


def make_states(df, visit, chain):

    valid_ids = set(
        validation[
            (validation["visit"] == visit)
            & (validation["chain"] == chain)
            & (validation["valid"])
        ]["cellid"]
    )

    x = df[
        df["cellid"].isin(valid_ids)
    ].copy()

    rows = []

    for _, cell in x.iterrows():

        seq = str(
            cell[
                f"v_sequence_alignment_aa_{chain}"
            ]
        )

        germ = str(
            cell[
                f"v_germline_alignment_aa_{chain}"
            ]
        )

        gene = cell[
            f"v_call_top_{chain}"
        ]

        for i, (g, aa) in enumerate(
            zip(germ, seq),
            start=1,
        ):

            if g == "X" or aa == "X":
                continue

            rows.append({
                "lineage": cell["lineage"],
                "cellid": cell["cellid"],
                "chain": chain,
                "v_gene": gene,
                "position": i,
                "germline_aa": g,
                "aa": aa,
            })

    residues = pd.DataFrame(rows)

    state_rows = []

    for keys, g in residues.groupby(
        [
            "lineage",
            "chain",
            "v_gene",
            "position",
            "germline_aa",
        ]
    ):

        counts = g["aa"].value_counts()

        if (
            len(counts) > 1
            and counts.iloc[0] == counts.iloc[1]
        ):
            consensus = None
        else:
            consensus = counts.index[0]

        state_rows.append({
            "lineage": keys[0],
            "chain": keys[1],
            "v_gene": keys[2],
            "position": keys[3],
            "germline_aa": keys[4],
            "cells_assessable": len(g),
            "consensus_aa": consensus,
            "consensus_fraction":
            counts.iloc[0] / len(g),
        })

    return pd.DataFrame(state_rows)


w8_states = []

w24_states = []


for chain in ["heavy", "light"]:

    w8_states.append(
        make_states(
            early,
            "week8",
            chain,
        )
    )

    w24_states.append(
        make_states(
            w24,
            "week24",
            chain,
        )
    )


w8_states = pd.concat(
    w8_states,
    ignore_index=True,
)

w24_states = pd.concat(
    w24_states,
    ignore_index=True,
)


w8_states = w8_states.rename(
    columns={
        "cells_assessable": "w8_cells",
        "consensus_aa": "w8_aa",
        "consensus_fraction": "w8_consensus_fraction",
    }
)

w24_states = w24_states.rename(
    columns={
        "cells_assessable": "w24_cells",
        "consensus_aa": "w24_aa",
        "consensus_fraction": "w24_consensus_fraction",
    }
)


transitions = w8_states.merge(
    w24_states,
    on=[
        "lineage",
        "chain",
        "v_gene",
        "position",
        "germline_aa",
    ],
    how="inner",
    validate="one_to_one",
)


info = (
    early.groupby("lineage")
    .agg(
        pubID=("pubID", "first"),
        group=("group", "first"),
    )
    .reset_index()
)

transitions = transitions.merge(
    info,
    on="lineage",
    how="left",
    validate="many_to_one",
)


def transition_type(row):

    if pd.isna(row["w8_aa"]) or pd.isna(row["w24_aa"]):
        return "unresolved"

    g = row["germline_aa"]
    a = row["w8_aa"]
    b = row["w24_aa"]

    if a == g and b == g:
        return "stable_germline"

    if a == g and b != g:
        return "acquired_from_germline"

    if a != g and b == a:
        return "preexisting_mutation"

    if a != g and b == g:
        return "reverted_to_germline"

    if a != b:
        return "changed_mutation"

    return "other"


transitions["transition"] = transitions.apply(
    transition_type,
    axis=1,
)


transitions["mutation"] = None

x = (
    transitions["transition"]
    == "acquired_from_germline"
)

transitions.loc[x, "mutation"] = (
    transitions.loc[x, "chain"]
    + "::"
    + transitions.loc[x, "v_gene"]
    + "::"
    + transitions.loc[x, "position"].astype(str)
    + "::"
    + transitions.loc[x, "germline_aa"]
    + ">"
    + transitions.loc[x, "w24_aa"]
)


print("TRANSITION TABLE")
print("rows:", len(transitions))
print(
    "lineages:",
    transitions["lineage"].nunique(),
)
print(
    "participants:",
    transitions["pubID"].nunique(),
)


print("\nTRANSITION COUNTS")

print(
    transitions.groupby(
        ["chain", "transition"]
    )
    .size()
    .reset_index(
        name="rows"
    )
    .to_string(
        index=False
    )
)


print("\nLINEAGES RETAINED AFTER QC")

for group in [1, 2]:

    for chain in ["heavy", "light"]:

        x = transitions[
            (transitions["group"] == group)
            & (transitions["chain"] == chain)
        ]

        print(
            "group",
            group,
            chain,
            "lineages=",
            x["lineage"].nunique(),
            "participants=",
            x["pubID"].nunique(),
        )


acquired = transitions[
    transitions["transition"]
    == "acquired_from_germline"
].copy()


print("\nACQUIRED MUTATIONS")

print(
    acquired.groupby("chain")
    .agg(
        mutation_events=("mutation", "size"),
        unique_mutations=("mutation", "nunique"),
        lineages=("lineage", "nunique"),
        participants=("pubID", "nunique"),
    )
    .to_string()
)


transitions.to_csv(
    outdir /
    "193_g002_prime_week24_v_mutation_transitions.tsv",
    sep="\t",
    index=False,
)

acquired.to_csv(
    outdir /
    "194_g002_prime_week24_acquired_v_mutations.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(
    outdir /
    "193_g002_prime_week24_v_mutation_transitions.tsv"
)
print(
    outdir /
    "194_g002_prime_week24_acquired_v_mutations.tsv"
)
