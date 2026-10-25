from pathlib import Path
import csv
import hashlib
import json
import math
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "projects/ai_hiv_vaccine_design"
BASE = ROOT / "results/bcr/g002_g003"
OUT = Path(__file__).resolve().parent

PROFILE = BASE / "198_g002_prime_ighv1_2_02_participant_mutation_profile.tsv"
REFERENCE = BASE / "212_g002_prime_ighv1_2_02_cdrh2_leave_one_position_out.tsv"
TRANSITIONS = BASE / "193_g002_prime_week24_v_mutation_transitions.tsv"
GENERATOR = ROOT / "scripts/177_prime_ighv1_2_02_global_steering.py"

POSITIONS = ("51", "52", "53", "54", "55", "57", "58")

print("STEP 4AU — FIGURE 1A PRIVATE REPRODUCTION", flush=True)
print("=" * 70, flush=True)

def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

for path in (PROFILE, REFERENCE, TRANSITIONS, GENERATOR):
    if not path.is_file():
        raise RuntimeError(f"Missing frozen source: {path}")

participants = read_table(PROFILE)
reference = read_table(REFERENCE)

if len(participants) != 15:
    raise RuntimeError("Expected exactly 15 participants.")

if len({r["pubID"] for r in participants}) != 15:
    raise RuntimeError("Duplicate participant identifier.")

groups = {
    group: sorted(
        [r for r in participants if r["group"].strip() == group],
        key=lambda r: r["pubID"]
    )
    for group in ("1", "2")
}

if (len(groups["1"]), len(groups["2"])) != (7, 8):
    raise RuntimeError("Participant-group counts do not match.")

if len(reference) != 7:
    raise RuntimeError("Expected seven reference rows.")

ref = {
    r["removed_position"]: float(
        r["removed_original_centered_difference"]
    )
    for r in reference
}

if set(ref) != set(POSITIONS):
    raise RuntimeError("Frozen reference positions differ.")

print("\n1. PARTICIPANT-LEVEL DATA")
print("-" * 70)

summary = []
points = {}

for pos in POSITIONS:
    values = {}

    for group in ("1", "2"):
        x = [float(r[pos]) for r in groups[group]]

        if not all(math.isfinite(v) and 0 <= v <= 1 for v in x):
            raise RuntimeError(f"Invalid fraction at position {pos}")

        values[group] = x
        points[(pos, group)] = x

    g1 = statistics.mean(values["1"])
    g2 = statistics.mean(values["2"])

    summary.append({
        "position": pos,
        "group1_participants": 7,
        "group2_participants": 8,
        "group1_mean": g1,
        "group2_mean": g2,
        "group2_minus_group1": g2 - g1,
    })

    print(
        f"Position {pos}: "
        f"G1={g1:.9f}, G2={g2:.9f}, "
        f"Difference={g2-g1:+.9f}"
    )

print("\n2. FROZEN REFERENCE VERIFICATION")
print("-" * 70)

raw = {
    row["position"]: row["group2_minus_group1"]
    for row in summary
}

raw_center = statistics.mean(raw.values())
reference_center = statistics.mean(ref.values())

errors = [
    abs(
        (raw[pos] - raw_center)
        - (ref[pos] - reference_center)
    )
    for pos in POSITIONS
]

max_error = max(errors)
offset = reference_center - raw_center

print("Maximum centered-shape error:", max_error)
print("Absolute reference offset:", offset)

if max_error > 1e-8:
    raise RuntimeError(
        "Frozen positional profile mismatch. "
        "No aggregate or figure generated."
    )

print("Frozen centered-shape consistency: PASS")

print("\n3. SAVE SEVEN-POSITION AGGREGATE")
print("-" * 70)

aggregate = OUT / "figure1a_aggregate_from_198.tsv"

if aggregate.exists():
    raise RuntimeError("Aggregate exists. No overwrite.")

with aggregate.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary[0].keys()),
        delimiter="\t"
    )
    writer.writeheader()
    writer.writerows(summary)

print("Aggregate table: CREATED")

print("\n4. RECONSTRUCT PRIVATE FIGURE")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 5.8))

positions_x = list(range(len(POSITIONS)))
offsets = {"1": -0.10, "2": 0.10}

for group in ("1", "2"):
    mean_key = (
        "group1_mean" if group == "1" else "group2_mean"
    )

    x = [i + offsets[group] for i in positions_x]
    means = [row[mean_key] for row in summary]

    line, = ax.plot(
        x,
        means,
        marker="o",
        linewidth=2,
        label=f"Group {group} mean"
    )

    for index, pos in enumerate(POSITIONS):
        rng = random.Random(
            20261024 + int(pos) * 10 + int(group)
        )

        jittered = [
            x[index] + rng.uniform(-0.06, 0.06)
            for _ in points[(pos, group)]
        ]

        ax.scatter(
            jittered,
            points[(pos, group)],
            color=line.get_color(),
            s=20,
            alpha=0.40,
            linewidths=0
        )

ax.set_xticks(positions_x)
ax.set_xticklabels(POSITIONS)
ax.set_xlabel("IGHV1-2*02 V-relative CDRH2 position")
ax.set_ylabel("Observed lineage-modal-state acquisition fraction")
ax.set_ylim(-0.025, 1.025)
ax.set_title("Private Figure 1A reconstruction from frozen participant data")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.20)

fig.tight_layout()

figure_path = OUT / "figure1a_private_reconstruction.png"

if figure_path.exists():
    raise RuntimeError("Figure already exists. No overwrite.")

fig.savefig(figure_path, dpi=200)
plt.close(fig)

print("Private figure: CREATED")

print("\n5. SAVE REPRODUCIBILITY DOCUMENTATION")
print("-" * 70)

script = Path(__file__).resolve()

manifest = {
    "status": "PRIVATE_REVIEW_ONLY",
    "participants": 15,
    "group1": 7,
    "group2": 8,
    "positions": list(POSITIONS),
    "maximum_centered_shape_error": max_error,
    "absolute_reference_offset": offset,
    "source_sha256": {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (PROFILE, REFERENCE, TRANSITIONS, GENERATOR)
    },
    "script_sha256": sha256(script),
    "aggregate_sha256": sha256(aggregate),
    "figure_sha256": sha256(figure_path),
    "public_release_approved": False,
    "full_public_source_reproduction_verified": False,
    "pixel_identical_to_original": False
}

(OUT / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

readme = """# Figure 1A private reproducibility package

This is a private computational reconstruction,
not the original approved publication figure.

Source table 198 contains the frozen participant
mutation profile for 15 participants across seven
selected CDRH2 positions.

The script reconstructs individual plotted dots
and participant-balanced group means from table 198.

The centered positional contrast shape matches
the frozen table 212 reference.

The absolute reference offset remains documented.
This check does not establish that table 212 uses
identical centering or eligibility definitions.

The original script 177 generated table 198 from
frozen upstream data, including table 193.
That upstream generation is traced but was not
rerun in this step.

The package is not yet a complete end-to-end
reproduction from the original public G002 release.

The plotted observations require disclosure review
before public distribution.

Do not publish this folder or change the original
manuscript figure based on this preview alone.
"""

(OUT / "README.md").write_text(
    readme, encoding="utf-8"
)

print("Manifest: CREATED")
print("README: CREATED")
print("Reusable script: SAVED")

print("\n6. FINAL STATUS")
print("-" * 70)
print("Frozen scientific result: UNCHANGED")
print("Original manuscript: UNCHANGED")
print("Original publication figure: UNCHANGED")
print("GitHub: UNCHANGED")
print("Dengue project: UNCHANGED")
print("Public release: NO")

print("\nOUTPUT FOLDER")
print(OUT)

print("\nSTEP 4AU COMPLETE")
