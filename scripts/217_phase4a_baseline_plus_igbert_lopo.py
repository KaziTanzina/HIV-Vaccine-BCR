from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


outdir = Path(
    "results/bcr/g002_g003"
)


cohort_path = (
    outdir /
    "246_phase4a_group2_week24_frozen_baseline_features.tsv"
)


embedding_path = (
    outdir /
    "257_phase4a_igbert_full_embeddings.npz"
)


baseline_summary_path = (
    outdir /
    "250_phase4a_frozen_baseline_lopo_summary.tsv"
)


sequence_summary_path = (
    outdir /
    "263_phase4a_igbert_sequence_only_lopo_summary.tsv"
)


baseline_features = [
    "log2_week8_cells",
    "week8_heavy_shm_pct",
    "week8_light_shm_pct",
    "cottrell_mean",
]


print(
    "PHASE 4A FROZEN BASELINE + IGBERT LOPO"
)


print(
    "\nFROZEN MODELING PROTOCOL"
)


print(
    "baseline features:",
    baseline_features,
)


print(
    "IgBERT dimensions: 1024"
)


print(
    "total predictor dimensions: 1028"
)


print(
    "PCA: none"
)


print(
    "feature selection: none"
)


print(
    "scaling: StandardScaler, training fold only"
)


print(
    "classifier: L2 logistic regression"
)


print(
    "C: 1.0"
)


print(
    "class_weight: balanced"
)


print(
    "solver: liblinear"
)


print(
    "participant weighting: inverse training-participant lineage count"
)


print(
    "validation: leave-one-participant-out"
)


# ============================================================
# Frozen cohort and baseline features
# ============================================================

df = pd.read_csv(
    cohort_path,
    sep="\t",
)


required = [
    "prospective_lineage",
    "pubID",
    "detected",
] + baseline_features


print(
    "\nCOHORT COLUMN CHECK"
)


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


if df[
    "prospective_lineage"
].duplicated().any():

    raise ValueError(
        "Duplicate prospective lineages found."
    )


for col in baseline_features:

    if df[
        col
    ].isna().any():

        raise ValueError(
            f"Missing baseline values in {col}"
        )


def convert_detected(
    value,
):

    if pd.isna(
        value
    ):

        return np.nan


    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):

        return int(
            value
        )


    text = str(
        value
    ).strip().lower()


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


df[
    "outcome"
] = (
    df[
        "detected"
    ]
    .map(
        convert_detected
    )
)


if df[
    "outcome"
].isna().any():

    raise ValueError(
        "Missing outcome values."
    )


# ============================================================
# Frozen IgBERT embeddings
# ============================================================

embedding_data = np.load(
    embedding_path,
    allow_pickle=False,
)


embedding_ids = (
    embedding_data[
        "prospective_lineage"
    ]
    .astype(
        str
    )
)


X_sequence = (
    embedding_data[
        "embedding"
    ]
    .astype(
        np.float64
    )
)


print(
    "\nEMBEDDING INPUT"
)


print(
    "rows:",
    X_sequence.shape[
        0
    ],
)


print(
    "dimensions:",
    X_sequence.shape[
        1
    ],
)


print(
    "all finite:",
    bool(
        np.isfinite(
            X_sequence
        ).all()
    ),
)


if X_sequence.shape != (
    2739,
    1024,
):

    raise ValueError(
        "Expected IgBERT shape (2739, 1024)."
    )


if not np.isfinite(
    X_sequence
).all():

    raise ValueError(
        "Non-finite IgBERT values found."
    )


# ============================================================
# Exact cohort / embedding alignment
# ============================================================

cohort_ids = set(
    df[
        "prospective_lineage"
    ]
    .astype(
        str
    )
)


embedding_id_set = set(
    embedding_ids.tolist()
)


print(
    "\nCOHORT / EMBEDDING ALIGNMENT"
)


print(
    "cohort IDs:",
    len(
        cohort_ids
    ),
)


print(
    "embedding IDs:",
    len(
        embedding_id_set
    ),
)


print(
    "same ID set:",
    cohort_ids
    == embedding_id_set,
)


if cohort_ids != embedding_id_set:

    raise ValueError(
        "Cohort and embedding IDs differ."
    )


ordered = (
    df.set_index(
        "prospective_lineage"
    )
    .loc[
        embedding_ids
    ]
    .reset_index()
)


if (
    ordered[
        "prospective_lineage"
    ]
    .astype(
        str
    )
    .tolist()
    != embedding_ids.tolist()
):

    raise ValueError(
        "Failed to align cohort to embedding order."
    )


# ============================================================
# Build frozen combined predictor matrix
# ============================================================

X_baseline = (
    ordered[
        baseline_features
    ]
    .to_numpy(
        dtype=np.float64
    )
)


X_all = np.concatenate(
    [
        X_baseline,
        X_sequence,
    ],
    axis=1,
)


y_all = (
    ordered[
        "outcome"
    ]
    .to_numpy(
        dtype=int
    )
)


pubids = (
    ordered[
        "pubID"
    ]
    .astype(
        str
    )
    .to_numpy()
)


print(
    "\nCOMBINED INPUT"
)


print(
    "rows:",
    X_all.shape[
        0
    ],
)


print(
    "baseline dimensions:",
    X_baseline.shape[
        1
    ],
)


print(
    "sequence dimensions:",
    X_sequence.shape[
        1
    ],
)


print(
    "total dimensions:",
    X_all.shape[
        1
    ],
)


print(
    "participants:",
    len(
        np.unique(
            pubids
        )
    ),
)


print(
    "detected:",
    int(
        y_all.sum()
    ),
)


print(
    "not detected:",
    int(
        (
            y_all == 0
        ).sum()
    ),
)


print(
    "prevalence:",
    float(
        y_all.mean()
    ),
)


if X_all.shape != (
    2739,
    1028,
):

    raise ValueError(
        "Expected combined shape (2739, 1028)."
    )


if len(
    np.unique(
        pubids
    )
) != 17:

    raise ValueError(
        "Expected 17 participants."
    )


if int(
    y_all.sum()
) != 243:

    raise ValueError(
        "Expected 243 detected lineages."
    )


# ============================================================
# Leave-one-participant-out
# ============================================================

participants = sorted(
    np.unique(
        pubids
    )
)


prediction_rows = []
fold_rows = []


for held_out in participants:

    test_mask = (
        pubids
        == held_out
    )


    train_mask = (
        ~test_mask
    )


    X_train = (
        X_all[
            train_mask
        ]
    )


    X_test = (
        X_all[
            test_mask
        ]
    )


    y_train = (
        y_all[
            train_mask
        ]
    )


    y_test = (
        y_all[
            test_mask
        ]
    )


    train_pubids = (
        pubids[
            train_mask
        ]
    )


    test_rows = (
        ordered.loc[
            test_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if np.unique(
        y_train
    ).size != 2:

        raise ValueError(
            "Training fold does not contain both classes."
        )


    # --------------------------------------------------------
    # Training-fold-only scaling
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Participant-equalizing training weights
    # --------------------------------------------------------

    participant_counts = pd.Series(
        train_pubids
    ).value_counts()


    sample_weight = np.array(
        [
            1.0
            / participant_counts[
                participant
            ]
            for participant in train_pubids
        ],
        dtype=float,
    )


    sample_weight = (
        sample_weight
        * len(
            sample_weight
        )
        / sample_weight.sum()
    )


    # --------------------------------------------------------
    # Same frozen classifier
    # --------------------------------------------------------

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
        )[
            :,
            1
        ]
    )


    # --------------------------------------------------------
    # OOF predictions
    # --------------------------------------------------------

    for i in range(
        len(
            test_rows
        )
    ):

        prediction_rows.append({
            "prospective_lineage":
                test_rows.loc[
                    i,
                    "prospective_lineage"
                ],

            "pubID":
                test_rows.loc[
                    i,
                    "pubID"
                ],

            "outcome":
                int(
                    y_test[
                        i
                    ]
                ),

            "predicted_probability":
                float(
                    probabilities[
                        i
                    ]
                ),

            "held_out_pubID":
                held_out,
        })


    test_detected = int(
        y_test.sum()
    )


    test_not_detected = int(
        (
            y_test
            == 0
        ).sum()
    )


    prevalence = float(
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
            ap
            / prevalence
        )


    else:

        auc = np.nan
        ap = np.nan
        ap_lift = np.nan


    fold_rows.append({
        "held_out_pubID":
            held_out,

        "train_lineages":
            int(
                train_mask.sum()
            ),

        "train_detected":
            int(
                y_train.sum()
            ),

        "test_lineages":
            int(
                test_mask.sum()
            ),

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

        "model_iterations":
            int(
                model.n_iter_[
                    0
                ]
            ),

        "coefficient_l2_norm":
            float(
                np.linalg.norm(
                    model.coef_[
                        0
                    ]
                )
            ),
    })


predictions = pd.DataFrame(
    prediction_rows
)


fold_metrics = pd.DataFrame(
    fold_rows
)


# ============================================================
# Exact OOF coverage
# ============================================================

print(
    "\nOUT-OF-FOLD COVERAGE"
)


print(
    "prediction rows:",
    len(
        predictions
    ),
)


print(
    "unique lineages:",
    predictions[
        "prospective_lineage"
    ].nunique(),
)


print(
    "participants:",
    predictions[
        "pubID"
    ].nunique(),
)


if len(
    predictions
) != 2739:

    raise ValueError(
        "Expected 2739 OOF predictions."
    )


if predictions[
    "prospective_lineage"
].nunique() != 2739:

    raise ValueError(
        "Expected 2739 unique OOF lineages."
    )


# ============================================================
# Pooled metrics
# ============================================================

y_pred_all = (
    predictions[
        "outcome"
    ]
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


pooled_prevalence = float(
    y_pred_all.mean()
)


pooled_auc = roc_auc_score(
    y_pred_all,
    p_all,
)


pooled_ap = average_precision_score(
    y_pred_all,
    p_all,
)


pooled_ap_lift = (
    pooled_ap
    / pooled_prevalence
)


print(
    "\nPOOLED OUT-OF-FOLD PERFORMANCE"
)


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


# ============================================================
# Participant-balanced pooled metrics
# ============================================================

participant_counts = (
    predictions.groupby(
        "pubID"
    )
    .size()
)


participant_weights = (
    predictions[
        "pubID"
    ]
    .map(
        1.0
        / participant_counts
    )
    .to_numpy(
        dtype=float
    )
)


balanced_prevalence = float(
    np.average(
        y_pred_all,
        weights=participant_weights,
    )
)


balanced_auc = roc_auc_score(
    y_pred_all,
    p_all,
    sample_weight=participant_weights,
)


balanced_ap = average_precision_score(
    y_pred_all,
    p_all,
    sample_weight=participant_weights,
)


balanced_ap_lift = (
    balanced_ap
    / balanced_prevalence
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


# ============================================================
# Within-participant fold summaries
# ============================================================

evaluable = (
    fold_metrics[
        fold_metrics[
            "roc_auc"
        ].notna()
    ]
    .copy()
)


print(
    "\nWITHIN-PARTICIPANT FOLD PERFORMANCE"
)


print(
    "evaluable folds:",
    len(
        evaluable
    ),
    "of",
    len(
        fold_metrics
    ),
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


print(
    "\nFOLD METRICS"
)


print(
    fold_metrics.to_string(
        index=False
    )
)


# ============================================================
# Compare all three frozen Phase-4A models
# ============================================================

baseline_summary = pd.read_csv(
    baseline_summary_path,
    sep="\t",
)


sequence_summary = pd.read_csv(
    sequence_summary_path,
    sep="\t",
)


if len(
    baseline_summary
) != 1:

    raise ValueError(
        "Expected one baseline summary row."
    )


if len(
    sequence_summary
) != 1:

    raise ValueError(
        "Expected one sequence summary row."
    )


baseline = (
    baseline_summary.iloc[
        0
    ]
)


sequence = (
    sequence_summary.iloc[
        0
    ]
)


comparison = pd.DataFrame([
    {
        "model":
            "baseline",

        "pooled_roc_auc":
            float(
                baseline[
                    "pooled_roc_auc"
                ]
            ),

        "pooled_average_precision":
            float(
                baseline[
                    "pooled_average_precision"
                ]
            ),

        "participant_balanced_roc_auc":
            float(
                baseline[
                    "participant_balanced_roc_auc"
                ]
            ),

        "participant_balanced_average_precision":
            float(
                baseline[
                    "participant_balanced_average_precision"
                ]
            ),
    },

    {
        "model":
            "igbert_sequence_only",

        "pooled_roc_auc":
            float(
                sequence[
                    "pooled_roc_auc"
                ]
            ),

        "pooled_average_precision":
            float(
                sequence[
                    "pooled_average_precision"
                ]
            ),

        "participant_balanced_roc_auc":
            float(
                sequence[
                    "participant_balanced_roc_auc"
                ]
            ),

        "participant_balanced_average_precision":
            float(
                sequence[
                    "participant_balanced_average_precision"
                ]
            ),
    },

    {
        "model":
            "baseline_plus_igbert",

        "pooled_roc_auc":
            pooled_auc,

        "pooled_average_precision":
            pooled_ap,

        "participant_balanced_roc_auc":
            balanced_auc,

        "participant_balanced_average_precision":
            balanced_ap,
    },
])


print(
    "\nTHREE-MODEL PHASE 4A COMPARISON"
)


print(
    comparison.to_string(
        index=False
    )
)


baseline_pooled_auc = float(
    baseline[
        "pooled_roc_auc"
    ]
)


baseline_pooled_ap = float(
    baseline[
        "pooled_average_precision"
    ]
)


baseline_balanced_auc = float(
    baseline[
        "participant_balanced_roc_auc"
    ]
)


baseline_balanced_ap = float(
    baseline[
        "participant_balanced_average_precision"
    ]
)


print(
    "\nCOMBINED MODEL DELTA VS FROZEN BASELINE"
)


print(
    "pooled ROC AUC delta:",
    pooled_auc
    - baseline_pooled_auc,
)


print(
    "pooled AP delta:",
    pooled_ap
    - baseline_pooled_ap,
)


print(
    "participant-balanced ROC AUC delta:",
    balanced_auc
    - baseline_balanced_auc,
)


print(
    "participant-balanced AP delta:",
    balanced_ap
    - baseline_balanced_ap,
)


# ============================================================
# Save
# ============================================================

summary = pd.DataFrame([{
    "model":
        "baseline_plus_igbert",

    "baseline_dimensions":
        4,

    "igbert_dimensions":
        1024,

    "total_dimensions":
        1028,

    "pca":
        "none",

    "classifier":
        "l2_logistic_regression",

    "C":
        1.0,

    "lineages":
        len(
            predictions
        ),

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
        len(
            evaluable
        ),

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

    "delta_vs_baseline_pooled_roc_auc":
        pooled_auc
        - baseline_pooled_auc,

    "delta_vs_baseline_pooled_ap":
        pooled_ap
        - baseline_pooled_ap,

    "delta_vs_baseline_balanced_roc_auc":
        balanced_auc
        - baseline_balanced_auc,

    "delta_vs_baseline_balanced_ap":
        balanced_ap
        - baseline_balanced_ap,
}])


prediction_path = (
    outdir /
    "264_phase4a_baseline_plus_igbert_lopo_predictions.tsv"
)


fold_path = (
    outdir /
    "265_phase4a_baseline_plus_igbert_lopo_fold_metrics.tsv"
)


summary_path = (
    outdir /
    "266_phase4a_baseline_plus_igbert_lopo_summary.tsv"
)


comparison_path = (
    outdir /
    "267_phase4a_three_model_comparison.tsv"
)


predictions.to_csv(
    prediction_path,
    sep="\t",
    index=False,
)


fold_metrics.to_csv(
    fold_path,
    sep="\t",
    index=False,
)


summary.to_csv(
    summary_path,
    sep="\t",
    index=False,
)


comparison.to_csv(
    comparison_path,
    sep="\t",
    index=False,
)


print(
    "\nSaved:"
)


for path in [
    prediction_path,
    fold_path,
    summary_path,
    comparison_path,
]:

    print(
        path
    )
