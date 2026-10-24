from pathlib import Path
import pandas as pd


early_file = Path(
    "data/processed/g002_g003/"
    "g002_groups1_2_week8_eod_prime_reclustered_n3.feather"
)

full_file = Path(
    "data/external/g002_g003/g002/final_df.feather"
)

outdir = Path("results/bcr/g002_g003")
outdir.mkdir(parents=True, exist_ok=True)


early = pd.read_feather(early_file)
full = pd.read_feather(full_file)

full["group"] = pd.to_numeric(full["group"], errors="coerce")
full["weeks"] = pd.to_numeric(full["weeks"], errors="coerce")

if "prospective_lineage" not in early.columns:
    early["prospective_lineage"] = (
        early["pubID"].astype(str)
        + "::"
        + early["cluster"].astype(str)
    )


later = full[
    full["group"].isin([1, 2])
    & full["weeks"].isin([16, 24])
    & (full["probe_set"] == "Cg28v2")
    & (full["is_vrc01_class"] == True)
    & (full["top_c_call"] == "IGHG")
].copy()


print("EARLY")
print("cells:", len(early))
print("participants:", early["pubID"].nunique())
print("lineages:", early["prospective_lineage"].nunique())

print("\nLATER")
print(
    later.groupby(["group", "weeks"])
    .agg(
        cells=("cellid", "size"),
        participants=("pubID", "nunique"),
    )
    .to_string()
)


def lev(a, b):
    a = str(a)
    b = str(b)

    prev = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        curr = [i]

        for j, y in enumerate(b, 1):
            curr.append(
                min(
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (x != y),
                )
            )

        prev = curr

    return prev[-1]


lineages = []

for lineage, g in early.groupby("prospective_lineage"):

    row = g.iloc[0]

    h_j = g["j_call_top_heavy"].dropna().astype(str).mode()
    l_j = g["j_call_top_light"].dropna().astype(str).mode()

    lineages.append({
        "lineage": lineage,
        "pubID": row["pubID"],
        "group": row["group"],
        "v_h": row["v_call_top_heavy"],
        "v_l": row["v_call_top_light"],
        "len_h": row["junction_aa_length_heavy"],
        "len_l": row["junction_aa_length_light"],
        "vrc01": row["is_vrc01_class"],
        "j_h": h_j.iloc[0] if len(h_j) else None,
        "j_l": l_j.iloc[0] if len(l_j) else None,
        "early_cells": len(g),
    })

lineages = pd.DataFrame(lineages)


matches = []

for _, cell in later.iterrows():

    cand = lineages[
        (lineages["pubID"] == cell["pubID"])
        & (lineages["v_h"] == cell["v_call_top_heavy"])
        & (lineages["v_l"] == cell["v_call_top_light"])
        & (lineages["len_h"] == cell["junction_aa_length_heavy"])
        & (lineages["len_l"] == cell["junction_aa_length_light"])
        & (lineages["vrc01"] == cell["is_vrc01_class"])
        & (lineages["j_h"] == str(cell["j_call_top_heavy"]))
        & (lineages["j_l"] == str(cell["j_call_top_light"]))
    ]

    compatible = []

    for _, c in cand.iterrows():

        g = early[
            early["prospective_lineage"] == c["lineage"]
        ]

        d = []

        for _, e in g.iterrows():
            d.append(
                lev(
                    cell["cdr3_aa_heavy"],
                    e["cdr3_aa_heavy"],
                )
                +
                lev(
                    cell["cdr3_aa_light"],
                    e["cdr3_aa_light"],
                )
            )

        mean_d = sum(d) / len(d)

        if mean_d <= 4:
            compatible.append(
                (c["lineage"], mean_d)
            )

    if len(compatible) == 0:
        status = "unmatched"
        assigned = None
        best_d = None

    elif len(compatible) == 1:
        status = "unique"
        assigned = compatible[0][0]
        best_d = compatible[0][1]

    else:
        status = "ambiguous"
        assigned = None
        best_d = min(x[1] for x in compatible)

    matches.append({
        "cellid": cell["cellid"],
        "pubID": cell["pubID"],
        "group": cell["group"],
        "weeks": cell["weeks"],
        "match_status": status,
        "assigned_lineage": assigned,
        "compatible_lineages": len(compatible),
        "mean_distance": best_d,
    })


matches = pd.DataFrame(matches)

print("\nMATCH STATUS")
print(
    matches.groupby(
        ["group", "weeks", "match_status"]
    )
    .size()
    .reset_index(name="cells")
    .to_string(index=False)
)


unique = matches[
    matches["match_status"] == "unique"
].copy()


visit_rows = []

for _, lin in lineages.iterrows():

    for week in [16, 24]:

        depth = len(
            later[
                (later["pubID"] == lin["pubID"])
                & (later["weeks"] == week)
            ]
        )

        if depth == 0:
            detected = pd.NA
            matched_cells = pd.NA
        else:
            x = unique[
                (unique["assigned_lineage"] == lin["lineage"])
                & (unique["weeks"] == week)
            ]

            detected = len(x) > 0
            matched_cells = len(x)

        visit_rows.append({
            "prospective_lineage": lin["lineage"],
            "pubID": lin["pubID"],
            "group": lin["group"],
            "week": week,
            "assessable": depth > 0,
            "detected": detected,
            "matched_later_cells": matched_cells,
            "participant_later_depth": depth if depth > 0 else pd.NA,
            "early_cells": lin["early_cells"],
        })


visits = pd.DataFrame(visit_rows)


print("\nLINEAGE DETECTION")

for group in [1, 2]:
    for week in [16, 24]:

        x = visits[
            (visits["group"] == group)
            & (visits["week"] == week)
            & (visits["assessable"] == True)
        ]

        print(
            "group",
            group,
            "week",
            week,
            "assessable=",
            len(x),
            "detected=",
            (x["detected"] == True).sum(),
        )


participant = (
    later.groupby(["group", "weeks", "pubID"])
    .size()
    .reset_index(name="later_cg28v2_cells")
)

matched = (
    unique.groupby(["group", "weeks", "pubID"])
    .agg(
        matched_cells=("cellid", "size"),
        detected_lineages=("assigned_lineage", "nunique"),
    )
    .reset_index()
)

participant = participant.merge(
    matched,
    on=["group", "weeks", "pubID"],
    how="left",
)

participant[
    ["matched_cells", "detected_lineages"]
] = participant[
    ["matched_cells", "detected_lineages"]
].fillna(0).astype(int)


summary = (
    matches.groupby(["group", "weeks", "match_status"])
    .size()
    .reset_index(name="cells")
)


matches.to_csv(
    outdir / "171_g002_prime_later_cg28v2_cell_matches.tsv",
    sep="\t",
    index=False,
)

visits.to_csv(
    outdir / "172_g002_prime_lineage_detection_by_visit.tsv",
    sep="\t",
    index=False,
)

summary.to_csv(
    outdir / "173_g002_prime_matching_summary.tsv",
    sep="\t",
    index=False,
)

participant.to_csv(
    outdir / "174_g002_prime_matching_participant_summary.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")
print(outdir / "171_g002_prime_later_cg28v2_cell_matches.tsv")
print(outdir / "172_g002_prime_lineage_detection_by_visit.tsv")
print(outdir / "173_g002_prime_matching_summary.tsv")
print(outdir / "174_g002_prime_matching_participant_summary.tsv")
