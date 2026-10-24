from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


input_path = Path(
    "results/bcr/g002_g003/"
    "246_phase4a_group2_week24_frozen_baseline_features.tsv"
)

outdir = Path(
    "results/bcr/g002_g003"
)


df = pd.read_csv(
    input_path,
    sep="\t",
)


features = [
    "log2_week8_cells",
    "week8_heavy_shm_pct",
    "week8_light_shm_pct",
    "cottrell_mean",
]


print("PHASE 4A FROZEN BASELINE LOPO")


print("\nREQUIRED COLUMN CHECK")

required = [
    "prospective_lineage",
    "pubID",
    "detected",
] + features

for col in required:

    print(
        col,
        "present=",
        col in df.columns,
    )

    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )


def convert_detected(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)

    text = str(value).strip().lower()

    if text in [
        "true",
        "1",
    ]:
        return 1

    if text in [
        "false",
        "0",
    ]:
        return 0

    raise ValueError(
        f"Unexpected detected value: {value}"
    )


df["outcome"] = (
    df["detected"]
    .map(convert_detected)
)


if df["outcome"].isna().any():

    raise ValueError(
        "Missing outcomes found."
    )


for col in features:

    if df[col].isna().any():

        raise ValueError(
            f"Missing feature values in {col}"
        )


print("\nCOHORT")

print(
    "rows:",
    len(df),
)

print(
    "lineages:",
    df[
        "prospective_lineage"
    ].nunique(),
)

print(
    "participants:",
    df[
        "pubID"
    ].nunique(),
)

print(
    "detected:",
    int(
        df["outcome"].sum()
    ),
)

print(
    "not detected:",
    int(
        (
            df["outcome"] == 0
        ).sum()
    ),
)

print(
    "detection fraction:",
    df["outcome"].mean(),
)


participants = sorted(
    df["pubID"].unique()
)


prediction_rows = []
fold_rows = []
coefficient_rows = []


for held_out in participants:

    train = df[
        df["pubID"] != held_out
    ].copy()

    test = df[
        df["pubID"] == held_out
    ].copy()


    X_train = (
        train[features]
        .to_numpy(
            dtype=float
        )
    )

    X_test = (
        test[features]
        .to_numpy(
            dtype=float
        )
    )

    y_train = (
        train["outcome"]
        .to_numpy(
            dtype=int
        )
    )

    y_test = (
        test["outcome"]
        .to_numpy(
            dtype=int
        )
    )


    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )


    participant_counts = (
        train.groupby(
            "pubID"
        )
        .size()
    )


    sample_weight = (
        train["pubID"]
        .map(
            1.0
            / participant_counts
        )
        .to_numpy(
            dtype=float
        )
    )


    sample_weight = (
        sample_weight
        * len(sample_weight)
        / sample_weight.sum()
    )


    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=1,
    )


    model.fit(
        X_train_scaled,
        y_train,
        sample_weight=sample_weight,
    )


    probabilities = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )


    for i, (_, row) in enumerate(
        test.iterrows()
    ):

        prediction_rows.append({
            "prospective_lineage":
            row["prospective_lineage"],
            "pubID":
            row["pubID"],
            "outcome":
            int(row["outcome"]),
            "predicted_probability":
            probabilities[i],
            "held_out_pubID":
            held_out,
        })


    test_detected = int(
        y_test.sum()
    )

    test_not_detected = int(
        (y_test == 0).sum()
    )

    prevalence = (
        y_test.mean()
    )


    if np.unique(
        y_test
    ).size == 2:

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        ap = average_precision_score(
            y_test,
            probabilities,
        )

        ap_lift = (
            ap / prevalence
        )

    else:

        auc = np.nan
        ap = np.nan
        ap_lift = np.nan


    fold_rows.append({
        "held_out_pubID":
        held_out,
        "test_lineages":
        len(test),
        "test_detected":
        test_detected,
        "test_not_detected":
        test_not_detected,
        "test_prevalence":
        prevalence,
        "roc_auc":
        auc,
        "average_precision":
        ap,
        "ap_lift":
        ap_lift,
    })


    for feature, coefficient in zip(
        features,
        model.coef_[0],
    ):

        coefficient_rows.append({
            "held_out_pubID":
            held_out,
            "feature":
            feature,
            "standardized_coefficient":
            coefficient,
        })


predictions = pd.DataFrame(
    prediction_rows
)

fold_metrics = pd.DataFrame(
    fold_rows
)

coefficients = pd.DataFrame(
    coefficient_rows
)


predictions = predictions.merge(
    df[
        [
            "prospective_lineage",
            "log2_week8_cells",
            "week8_heavy_shm_pct",
            "week8_light_shm_pct",
            "cottrell_mean",
        ]
    ],
    on="prospective_lineage",
    how="left",
)


y_all = (
    predictions["outcome"]
    .to_numpy(
        dtype=int
    )
)

p_all = (
    predictions[
        "predicted_probability"
    ]
    .to_numpy(
        dtype=float
    )
)


pooled_prevalence = (
    y_all.mean()
)


pooled_auc = roc_auc_score(
    y_all,
    p_all,
)

pooled_ap = average_precision_score(
    y_all,
    p_all,
)

pooled_ap_lift = (
    pooled_ap
    / pooled_prevalence
)


participant_counts = (
    predictions.groupby(
        "pubID"
    )
    .size()
)


participant_weights = (
    predictions["pubID"]
    .map(
        1.0
        / participant_counts
    )
    .to_numpy(
        dtype=float
    )
)


balanced_auc = roc_auc_score(
    y_all,
    p_all,
    sample_weight=participant_weights,
)


balanced_ap = (
    average_precision_score(
        y_all,
        p_all,
        sample_weight=participant_weights,
    )
)


balanced_prevalence = np.average(
    y_all,
    weights=participant_weights,
)


balanced_ap_lift = (
    balanced_ap
    / balanced_prevalence
)


evaluable = fold_metrics[
    fold_metrics[
        "roc_auc"
    ].notna()
].copy()


print("\nPOOLED OUT-OF-FOLD PERFORMANCE")

print(
    "prevalence:",
    pooled_prevalence,
)

print(
    "ROC AUC:",
    pooled_auc,
)

print(
    "average precision:",
    pooled_ap,
)

print(
    "AP lift:",
    pooled_ap_lift,
)


print(
    "\nPARTICIPANT-BALANCED "
    "OUT-OF-FOLD PERFORMANCE"
)

print(
    "balanced prevalence:",
    balanced_prevalence,
)

print(
    "balanced ROC AUC:",
    balanced_auc,
)

print(
    "balanced average precision:",
    balanced_ap,
)

print(
    "balanced AP lift:",
    balanced_ap_lift,
)


print("\nWITHIN-PARTICIPANT FOLD PERFORMANCE")

print(
    "evaluable folds:",
    len(evaluable),
    "of",
    len(fold_metrics),
)

print(
    "mean ROC AUC:",
    evaluable[
        "roc_auc"
    ].mean(),
)

print(
    "median ROC AUC:",
    evaluable[
        "roc_auc"
    ].median(),
)

print(
    "mean average precision:",
    evaluable[
        "average_precision"
    ].mean(),
)

print(
    "median average precision:",
    evaluable[
        "average_precision"
    ].median(),
)

print(
    "mean AP lift:",
    evaluable[
        "ap_lift"
    ].mean(),
)

print(
    "median AP lift:",
    evaluable[
        "ap_lift"
    ].median(),
)


print("\nFOLD METRICS")

print(
    fold_metrics.to_string(
        index=False
    )
)


coefficient_summary = (
    coefficients.groupby(
        "feature"
    )[
        "standardized_coefficient"
    ]
    .agg(
        [
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .reset_index()
)


print("\nSTANDARDIZED COEFFICIENT SUMMARY")

print(
    coefficient_summary.to_string(
        index=False
    )
)


summary = pd.DataFrame([{
    "lineages":
    len(predictions),
    "participants":
    predictions[
        "pubID"
    ].nunique(),
    "detected":
    int(
        predictions[
            "outcome"
        ].sum()
    ),
    "pooled_prevalence":
    pooled_prevalence,
    "pooled_roc_auc":
    pooled_auc,
    "pooled_average_precision":
    pooled_ap,
    "pooled_ap_lift":
    pooled_ap_lift,
    "participant_balanced_prevalence":
    balanced_prevalence,
    "participant_balanced_roc_auc":
    balanced_auc,
    "participant_balanced_average_precision":
    balanced_ap,
    "participant_balanced_ap_lift":
    balanced_ap_lift,
    "evaluable_folds":
    len(evaluable),
    "mean_fold_roc_auc":
    evaluable[
        "roc_auc"
    ].mean(),
    "median_fold_roc_auc":
    evaluable[
        "roc_auc"
    ].median(),
    "mean_fold_average_precision":
    evaluable[
        "average_precision"
    ].mean(),
    "median_fold_average_precision":
    evaluable[
        "average_precision"
    ].median(),
    "mean_fold_ap_lift":
    evaluable[
        "ap_lift"
    ].mean(),
    "median_fold_ap_lift":
    evaluable[
        "ap_lift"
    ].median(),
}])


predictions.to_csv(
    outdir /
    "248_phase4a_frozen_baseline_lopo_predictions.tsv",
    sep="\t",
    index=False,
)


fold_metrics.to_csv(
    outdir /
    "249_phase4a_frozen_baseline_lopo_fold_metrics.tsv",
    sep="\t",
    index=False,
)


summary.to_csv(
    outdir /
    "250_phase4a_frozen_baseline_lopo_summary.tsv",
    sep="\t",
    index=False,
)


coefficients.to_csv(
    outdir /
    "251_phase4a_frozen_baseline_lopo_coefficients.tsv",
    sep="\t",
    index=False,
)


print("\nSaved:")

for name in [
    "248_phase4a_frozen_baseline_lopo_predictions.tsv",
    "249_phase4a_frozen_baseline_lopo_fold_metrics.tsv",
    "250_phase4a_frozen_baseline_lopo_summary.tsv",
    "251_phase4a_frozen_baseline_lopo_coefficients.tsv",
]:

    print(
        outdir / name
    )
