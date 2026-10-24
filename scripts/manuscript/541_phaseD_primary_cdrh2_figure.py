from pathlib import Path
from datetime import datetime
import hashlib
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / "results/bcr/g002_g003"
OUTDIR = ROOT / "figures/bcr/g002_g003"

PROFILE = BASE / (
    "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv"
)

REGION = BASE / (
    "210_g002_prime_ighv1_2_02_region_steering.tsv"
)

EXPECTED_PROFILE_SHA = (
    "c2a68095cc5dffc072555e6d4d1878fe7c1ccbd214f8b1dc27c5797f00c5b85a"
)

POSITIONS = [51, 52, 53, 54, 55, 57, 58]

EXPECTED = {
    (51, 1): 47.5,
    (51, 2): 16.1,
    (54, 1): 10.6,
    (54, 2): 29.1,
    (55, 1): 0.0,
    (55, 2): 23.4,
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

STEM = f"figure2_cdrh2_exploratory_profile_{STAMP}"

PNG = OUTDIR / f"{STEM}.png"
PDF = OUTDIR / f"{STEM}.pdf"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("=" * 76)
print("HIV VACCINE — PHASE D")
print("STEP 2: REVISED PRIMARY CDRH2 FIGURE")
print("=" * 76)

print("\nSCIENTIFIC QUESTION")
print(
    "Can frozen participant-level acquisition profiles "
    "and the global regional comparison be shown "
    "without overstating position-specific significance?"
)

print("\n1. VERIFY ORIGINAL INPUTS")

for path in [PROFILE, REGION]:
    if not path.is_file():
        raise SystemExit(f"Missing frozen input: {path}")

    print(path.relative_to(ROOT))
    print("SHA256:", sha256(path))

before = {
    PROFILE: sha256(PROFILE),
    REGION: sha256(REGION),
}

if before[PROFILE] != EXPECTED_PROFILE_SHA:
    raise SystemExit(
        "STOP: Frozen participant profile checksum mismatch."
    )

profile = pd.read_csv(PROFILE, sep="\t")
region = pd.read_csv(REGION, sep="\t")

required = {"pubID", "group"} | {
    str(p) for p in POSITIONS
}

if not required.issubset(profile.columns):
    raise SystemExit(
        "STOP: Missing frozen participant-profile columns."
    )

if len(profile) != 15 or profile["pubID"].duplicated().any():
    raise SystemExit(
        "STOP: Expected exactly 15 distinct participants."
    )

profile["group"] = pd.to_numeric(
    profile["group"], errors="raise"
).astype(int)

group_counts = profile["group"].value_counts().to_dict()

if group_counts != {1: 7, 2: 8}:
    raise SystemExit(
        f"STOP: Unexpected group counts: {group_counts}"
    )

for pos in POSITIONS:
    name = str(pos)
    profile[name] = pd.to_numeric(
        profile[name], errors="raise"
    )

values = profile[[str(p) for p in POSITIONS]].to_numpy(
    dtype=float
)

if not np.isfinite(values).all():
    raise SystemExit("STOP: Missing profile values.")

if np.any((values < 0) | (values > 1)):
    raise SystemExit(
        "STOP: Profile fractions outside [0,1]."
    )

cdrh2 = region.loc[region["region"] == "CDRH2"]

if len(cdrh2) != 1:
    raise SystemExit(
        "STOP: Expected one original CDRH2 result."
    )

result = cdrh2.iloc[0]

pvalue = float(result["permutation_p"])
qvalue = float(result["fdr"])
statistic = float(result["observed_centered_difference"])

if not math.isclose(
    pvalue,
    0.0017094017094017094,
    rel_tol=0,
    abs_tol=1e-10,
):
    raise SystemExit("STOP: Primary p-value mismatch.")

if not math.isclose(
    qvalue,
    0.010256410256410256,
    rel_tol=0,
    abs_tol=1e-10,
):
    raise SystemExit("STOP: Primary q-value mismatch.")

if int(result["positions"]) != 7:
    raise SystemExit("STOP: Unexpected position count.")

print("\n2. ORIGINAL PARTICIPANT-LEVEL SUMMARIES")

means = {}

for group in [1, 2]:
    subset = profile.loc[profile["group"] == group]

    means[group] = np.array([
        100 * subset[str(pos)].mean()
        for pos in POSITIONS
    ])

    print(f"\nGroup {group}: n={len(subset)}")

    for pos, mean in zip(POSITIONS, means[group]):
        print(f"  V{pos}: {mean:.4f}%")

for (pos, group), expected in EXPECTED.items():

    mean = 100 * profile.loc[
        profile["group"] == group, str(pos)
    ].mean()

    if round(mean, 1) != expected:
        raise SystemExit(
            f"STOP: Frozen manuscript mean mismatch "
            f"at V{pos}, group {group}."
        )

print("\nManuscript's six descriptive percentages: PASS")

print("\n3. ORIGINAL GLOBAL REGIONAL COMPARISON")
print(f"CDRH2 statistic: {statistic:.9f}")
print(f"Exact participant permutation p: {pvalue:.9f}")
print(f"Six-region BH q: {qvalue:.9f}")

# -------------------------------------------------------
# CREATE FIGURE
# -------------------------------------------------------

print("\n4. DRAWING REVISED FIGURE")

BLUE = "#2166AC"
ORANGE = "#D97723"
GRAY = "#B9C0C6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(14.0, 6.2),
    gridspec_kw={"width_ratios": [1.5, 1]},
)

fig.subplots_adjust(
    left=0.075,
    right=0.975,
    top=0.81,
    bottom=0.27,
    wspace=0.30,
)

fig.suptitle(
    "Exploratory CDRH2 mutation-acquisition profiles "
    "after boosting",
    fontsize=16,
    fontweight="bold",
    y=0.955,
)

# -------------------------------------------------------
# PANEL A: PARTICIPANT PROFILES
# -------------------------------------------------------

rng = np.random.default_rng(20261022)

colors = {1: BLUE, 2: ORANGE}
offsets = {1: -0.105, 2: 0.105}
x = np.array(POSITIONS, dtype=float)

for group in [1, 2]:

    subset = profile.loc[
        profile["group"] == group
    ].sort_values("pubID")

    color = colors[group]
    shift = offsets[group]

    # Each small dot is one participant's eligible-lineage
    # acquisition fraction at the specified position.
    for pos in POSITIONS:

        observed = (
            100 * subset[str(pos)].to_numpy(dtype=float)
        )

        jitter = rng.uniform(
            -0.045,
            0.045,
            len(observed),
        )

        ax_a.scatter(
            np.full(len(observed), pos + shift) + jitter,
            observed,
            s=18,
            color=color,
            alpha=0.29,
            edgecolors="none",
            zorder=2,
        )

    # Equally weighted participant group means.
    ax_a.plot(
        x + shift,
        means[group],
        color=color,
        marker="o",
        markersize=7.5,
        linewidth=2.4,
        label=f"Group {group} (n={len(subset)})",
        zorder=4,
    )

ax_a.set_xticks(POSITIONS)
ax_a.set_xticklabels([f"V{pos}" for pos in POSITIONS])

ax_a.set_xlabel("V-relative CDRH2 residue position")
ax_a.set_ylabel(
    "Observed acquisition fraction "
    "of eligible lineages (%)"
)

ax_a.set_ylim(-4, 104)
ax_a.set_xlim(50.45, 58.45)

ax_a.grid(
    axis="y",
    color="0.88",
    linewidth=0.7,
    zorder=0,
)

ax_a.legend(
    loc="upper right",
    frameon=False,
    fontsize=10,
)

ax_a.set_title(
    "A. Participant-level positional profiles",
    loc="left",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

# -------------------------------------------------------
# PANEL B: REGIONAL BH-ADJUSTED RESULTS
# -------------------------------------------------------

region_order = [
    "CDRH2",
    "FWR1",
    "FWR3",
    "CDRH3",
    "FWR2",
    "CDRH1",
]

r = region.set_index("region").reindex(region_order)

if r["fdr"].isna().any():
    raise SystemExit(
        "STOP: Regional result missing one of six regions."
    )

q = pd.to_numeric(r["fdr"], errors="raise").to_numpy()

if not np.isfinite(q).all() or np.any((q <= 0) | (q > 1)):
    raise SystemExit("STOP: Invalid regional q-values.")

strength = -np.log10(q)
y = np.arange(len(region_order))

bar_colors = [
    BLUE if name == "CDRH2" else GRAY
    for name in region_order
]

ax_b.barh(
    y,
    strength,
    color=bar_colors,
    height=0.64,
)

ax_b.set_yticks(y)
ax_b.set_yticklabels(region_order)
ax_b.invert_yaxis()

ax_b.axvline(
    -np.log10(0.05),
    linestyle="--",
    linewidth=1.2,
    color="0.35",
)

ax_b.set_xlim(
    0,
    max(2.5, float(strength.max()) + 0.42),
)

ax_b.set_xlabel(
    r"$-\log_{10}$(six-region BH-adjusted q)"
)

ax_b.set_title(
    "B. Regional participant-level comparison",
    loc="left",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

ax_b.text(
    float(strength[0]) + 0.055,
    0,
    f"q={qvalue:.4f}",
    fontsize=9,
    fontweight="bold",
    va="center",
)

ax_b.text(
    0.97,
    0.06,
    f"CDRH2 exact permutation p = {pvalue:.6f}\n"
    f"BH q across six regions = {qvalue:.6f}",
    transform=ax_b.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox={
        "facecolor": "white",
        "edgecolor": "0.80",
        "boxstyle": "round,pad=0.35",
    },
)

# -------------------------------------------------------
# FIGURE QUALIFICATIONS
# -------------------------------------------------------

fig.text(
    0.075,
    0.155,
    "G002 Week 8 to Week 24  |  "
    "15 participants (Group 1: 7; Group 2: 8)  |  "
    "Points: individual participants; lines: equal-weight "
    "group means.",
    fontsize=9.5,
)

fig.text(
    0.075,
    0.115,
    "Eligible lineages: germline Week-8 modal state and "
    "resolved Week-24 state. Positions selected using pooled "
    "acquisition-support thresholds.",
    fontsize=9.1,
)

fig.text(
    0.075,
    0.075,
    "Exploratory analysis. The permutation p-value tests "
    "the overall centered regional profile, not individual "
    "positions. Sampling-qualified association; "
    "no causal or functional inference.",
    fontsize=9.1,
)

# -------------------------------------------------------
# SAVE WITHOUT OVERWRITING
# -------------------------------------------------------

if PNG.exists() or PDF.exists():
    raise SystemExit(
        "STOP: New figure filename already exists."
    )

OUTDIR.mkdir(parents=True, exist_ok=True)

fig.savefig(
    PNG,
    dpi=300,
    facecolor="white",
)

fig.savefig(
    PDF,
    facecolor="white",
)

plt.close(fig)

print("\n5. SAVED FIGURE FILES")

for path in [PNG, PDF]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Figure was not saved correctly: {path}"
        )

    print(path.resolve())
    print("SHA256:", sha256(path))

print("\n6. FROZEN SOURCE INTEGRITY")

for path, digest in before.items():
    if sha256(path) != digest:
        raise RuntimeError(
            f"Frozen source changed: {path}"
        )

print("Participant-profile table unchanged: PASS")
print("Regional statistical table unchanged: PASS")
print("Original Figure 2 files untouched: PASS")
print("No statistical tests recomputed: PASS")
print("No manuscript edited: PASS")

print("\nPHASE D STEP 2 COMPLETE")
