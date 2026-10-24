from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

BASE = Path("results/bcr/g002_g003")
OUT = BASE / "phaseG_ablang2"

EMBED = OUT / "570_phaseG_ablang2_paired_mean_480.npz"
INDEX = OUT / "571_phaseG_ablang2_embedding_index.tsv"
META = OUT / "572_phaseG_ablang2_embedding_metadata.json"

FEATURES_PATH = (
    BASE / "246_phase4a_group2_week24_frozen_baseline_features.tsv"
)
BASE_PRED_PATH = (
    BASE / "248_phase4a_frozen_baseline_lopo_predictions.tsv"
)
BASE_SUMMARY_PATH = (
    BASE / "250_phase4a_frozen_baseline_lopo_summary.tsv"
)

PRED_SEQUENCE = OUT / (
    "573_phaseG_ablang2_sequence_only_lopo_predictions.tsv"
)
PRED_COMBINED = OUT / (
    "574_phaseG_baseline_plus_ablang2_lopo_predictions.tsv"
)
SUMMARY = OUT / "575_phaseG_ablang2_lopo_summary.tsv"
FOLDS = OUT / "576_phaseG_ablang2_lopo_fold_metrics.tsv"

KEY = ["prospective_lineage", "pubID"]
FEATURES = [
    "log2_week8_cells",
    "week8_heavy_shm_pct",
    "week8_light_shm_pct",
    "cottrell_mean",
]

OUTPUTS = [PRED_SEQUENCE, PRED_COMBINED, SUMMARY, FOLDS]

print("=" * 74, flush=True)
print("PHASE G — STEP 15")
print("FROZEN ABLANG-2 LEAVE-ONE-PARTICIPANT-OUT BENCHMARK")
print("=" * 74, flush=True)

print("\nSCIENTIFIC QUESTION")
print(
    "Does AbLang-2 improve Week-24 observed lineage recovery "
    "prediction beyond the frozen Week-8 biological baseline?"
)

for path in [
    EMBED, INDEX, META,
    FEATURES_PATH, BASE_PRED_PATH, BASE_SUMMARY_PATH,
]:
    if not path.is_file():
        raise SystemExit(f"STOP: Missing {path}")

for path in OUTPUTS:
    if path.exists():
        raise SystemExit(f"STOP: Output already exists: {path}")

index = pd.read_csv(INDEX, sep="\t")
features = pd.read_csv(FEATURES_PATH, sep="\t")
base_pred = pd.read_csv(BASE_PRED_PATH, sep="\t")
base_summary = pd.read_csv(BASE_SUMMARY_PATH, sep="\t")
metadata = json.loads(META.read_text())

with np.load(EMBED, allow_pickle=False) as saved:
    X_embedding = saved["embeddings"].copy()
    saved_lineages = saved["prospective_lineage"].astype(str)
    saved_pubids = saved["pubID"].astype(str)

print("\n1. FROZEN EMBEDDING INTEGRITY")

assert X_embedding.shape == (2739, 480)
assert np.isfinite(X_embedding).all()
assert len(index) == 2739
assert len(features) == 2739
assert len(base_pred) == 2739
assert len(base_summary) == 1

if hashlib.sha256(EMBED.read_bytes()).hexdigest() != (
    metadata["embedding_sha256"]
):
    raise SystemExit("STOP: Embedding file hash mismatch")

for name, table in [
    ("index", index),
    ("features", features),
    ("baseline predictions", base_pred),
]:
    if table[KEY].isna().any().any():
        raise SystemExit(f"STOP: Missing keys in {name}")
    if table.duplicated(KEY).any():
        raise SystemExit(f"STOP: Duplicate keys in {name}")

index = index.sort_values("embedding_row").reset_index(drop=True)

if not np.array_equal(
    index["embedding_row"].to_numpy(),
    np.arange(2739),
):
    raise SystemExit("STOP: Embedding row indices inconsistent")

if not np.array_equal(
    index["prospective_lineage"].astype(str).to_numpy(),
    saved_lineages,
):
    raise SystemExit("STOP: Saved lineage order mismatch")

if not np.array_equal(
    index["pubID"].astype(str).to_numpy(),
    saved_pubids,
):
    raise SystemExit("STOP: Saved participant order mismatch")

df = index[KEY + ["embedding_row", "group"]].merge(
    features[KEY + FEATURES + ["detected"]],
    on=KEY,
    how="left",
    validate="one_to_one",
).merge(
    base_pred[
        KEY + [
            "outcome",
            "predicted_probability",
            "held_out_pubID",
        ]
    ],
    on=KEY,
    how="left",
    validate="one_to_one",
)

if len(df) != 2739:
    raise SystemExit("STOP: Joined cohort size mismatch")

if not (df["group"].astype(str) == "2").all():
    raise SystemExit("STOP: Unexpected vaccination group")

if df[FEATURES + [
    "detected", "outcome", "predicted_probability"
]].isna().any().any():
    raise SystemExit("STOP: Missing model inputs or labels")

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="raise")

X_baseline = df[FEATURES].to_numpy(dtype=np.float64)
X_embedding = X_embedding.astype(np.float64)

if not np.isfinite(X_baseline).all():
    raise SystemExit("STOP: Nonfinite baseline features")

y = df["outcome"].to_numpy(dtype=int)
pubids = df["pubID"].astype(str).to_numpy()

if not np.array_equal(
    y, df["detected"].to_numpy(dtype=int)
):
    raise SystemExit("STOP: Outcome-label mismatch")

if not np.array_equal(
    pubids,
    df["held_out_pubID"].astype(str).to_numpy(),
):
    raise SystemExit("STOP: Original fold-assignment mismatch")

participants = sorted(set(pubids))

if (
    len(participants) != 17
    or int(y.sum()) != 243
    or not set(np.unique(y)).issubset({0, 1})
):
    raise SystemExit("STOP: Frozen prediction cohort changed")

print("Lineages:", len(df))
print("Participants:", len(participants))
print("Detected:", int(y.sum()))
print("Baseline dimensions:", X_baseline.shape[1])
print("AbLang-2 dimensions:", X_embedding.shape[1])
print("Outcome and cohort parity: PASS")

print("\n2. VERIFY HISTORICAL BASELINE METRICS")

participant_counts = pd.Series(pubids).value_counts()

evaluation_weights = np.array(
    [1.0 / participant_counts[p] for p in pubids],
    dtype=np.float64,
)

def metrics(probabilities):
    if (
        not np.isfinite(probabilities).all()
        or not np.all(
            (probabilities >= 0) & (probabilities <= 1)
        )
    ):
        raise ValueError("Invalid prediction probabilities")

    return {
        "pooled_roc_auc":
            roc_auc_score(y, probabilities),
        "pooled_average_precision":
            average_precision_score(y, probabilities),
        "participant_balanced_roc_auc":
            roc_auc_score(
                y,
                probabilities,
                sample_weight=evaluation_weights,
            ),
        "participant_balanced_average_precision":
            average_precision_score(
                y,
                probabilities,
                sample_weight=evaluation_weights,
            ),
    }

baseline_prob = df[
    "predicted_probability"
].to_numpy(dtype=float)

baseline_metrics = metrics(baseline_prob)

for name, value in baseline_metrics.items():
    reference = float(base_summary.iloc[0][name])
    difference = abs(value - reference)

    print(
        f"{name}: recalculated={value:.6f}, "
        f"frozen={reference:.6f}, "
        f"difference={difference:.8f}"
    )

    if difference > 1e-6:
        raise SystemExit(
            "STOP: Historical metric method does not match. "
            "Do not run a noncomparable benchmark."
        )

print("Frozen baseline metric replication: PASS")

print("\n3. TRAIN FROZEN CLASSIFIERS", flush=True)

def run_model(label, X):
    probabilities = np.full(len(y), np.nan)
    fold_rows = []

    for fold_number, held_out in enumerate(participants, 1):
        test_mask = pubids == held_out
        train_mask = ~test_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        train_pubids = pubids[train_mask]

        if np.any(train_pubids == held_out):
            raise SystemExit("STOP: Participant leakage")

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_counts = pd.Series(
            train_pubids
        ).value_counts()

        sample_weights = np.array(
            [
                1.0 / train_counts[p]
                for p in train_pubids
            ],
            dtype=np.float64,
        )

        sample_weights *= (
            len(sample_weights) / sample_weights.sum()
        )

        classifier = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=1,
        )

        classifier.fit(
            X_train_scaled,
            y_train,
            sample_weight=sample_weights,
        )

        if classifier.n_iter_[0] >= 5000:
            raise SystemExit(
                f"STOP: Nonconverged fit: {label} {held_out}"
            )

        fold_prob = classifier.predict_proba(
            X_test_scaled
        )[:, 1]

        probabilities[test_mask] = fold_prob

        two_classes = len(np.unique(y_test)) == 2

        fold_rows.append({
            "model": label,
            "held_out_pubID": held_out,
            "test_lineages": int(test_mask.sum()),
            "test_detected": int(y_test.sum()),
            "test_not_detected": int(
                len(y_test) - y_test.sum()
            ),
            "roc_auc": (
                roc_auc_score(y_test, fold_prob)
                if two_classes else np.nan
            ),
            "average_precision": (
                average_precision_score(y_test, fold_prob)
                if two_classes else np.nan
            ),
        })

        print(
            f"{label}: "
            f"fold {fold_number}/{len(participants)} "
            f"held_out={held_out} "
            f"test={int(test_mask.sum())} "
            f"positive={int(y_test.sum())}",
            flush=True,
        )

    if not np.isfinite(probabilities).all():
        raise SystemExit(
            f"STOP: Missing predictions for {label}"
        )

    return probabilities, fold_rows

sequence_prob, sequence_folds = run_model(
    "ablang2_sequence_only",
    X_embedding,
)

combined_prob, combined_folds = run_model(
    "baseline_plus_ablang2",
    np.column_stack([X_baseline, X_embedding]),
)

print("\n4. COMPARE FROZEN RESULTS")

sequence_metrics = metrics(sequence_prob)
combined_metrics = metrics(combined_prob)

comparison = [
    ("biological_baseline", baseline_metrics),
    ("ablang2_sequence_only", sequence_metrics),
    ("baseline_plus_ablang2", combined_metrics),
]

summary_rows = []

for name, values in comparison:
    row = {
        "model": name,
        "lineages": len(df),
        "participants": len(participants),
        "detected": int(y.sum()),
        **values,
    }

    row["delta_balanced_ap_vs_baseline"] = (
        values["participant_balanced_average_precision"]
        - baseline_metrics[
            "participant_balanced_average_precision"
        ]
    )

    row["delta_pooled_auc_vs_baseline"] = (
        values["pooled_roc_auc"]
        - baseline_metrics["pooled_roc_auc"]
    )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

print(summary.to_string(index=False, float_format="%.6f"))

print("\n5. SAVE RESULTS")

def prediction_table(probabilities):
    return pd.DataFrame({
        "prospective_lineage": df["prospective_lineage"],
        "pubID": df["pubID"],
        "outcome": y,
        "predicted_probability": probabilities,
        "held_out_pubID": pubids,
    })

prediction_table(sequence_prob).to_csv(
    PRED_SEQUENCE, sep="\t", index=False
)

prediction_table(combined_prob).to_csv(
    PRED_COMBINED, sep="\t", index=False
)

summary.to_csv(SUMMARY, sep="\t", index=False)

pd.DataFrame(
    sequence_folds + combined_folds
).to_csv(
    FOLDS, sep="\t", index=False
)

for path in OUTPUTS:
    print("Saved:", path)

print("\nINTERPRETATION LIMITS")
print(
    "This is leave-one-participant-out predictive "
    "evaluation, not external biological validation."
)
print(
    "Observed Week-24 lineage detection is influenced "
    "by sampling, assay ascertainment and matching."
)
print(
    "Performance differences require uncertainty "
    "assessment before claiming improvement."
)
print(
    "No additional model architectures, hyperparameters "
    "or embedding representations were evaluated."
)

print("\nPHASE G STEP 15 COMPLETE")
