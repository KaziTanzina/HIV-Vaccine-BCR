from pathlib import Path
import csv
import hashlib
import os
import platform
import time

import numpy as np
import torch
import transformers
from transformers import BertModel, BertTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


input_path = Path(
    "results/bcr/g002_g003/"
    "253_phase4a_group2_week24_igbert_clean_sequences.tsv"
)

outdir = Path(
    "results/bcr/g002_g003"
)


model_name = "Exscientia/IgBert"

model_revision = (
    "81a4abb8a4dac5cdcbefc28e5b331a82224c485d"
)


heavy_col = "igbert_heavy_sequence"
light_col = "igbert_light_sequence"


allowed = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


batch_size = 8


print(
    "PHASE 4A FULL FROZEN IGBERT EMBEDDING GENERATION"
)


print(
    "\nSOFTWARE"
)

print(
    "python:",
    platform.python_version(),
)

print(
    "torch:",
    torch.__version__,
)

print(
    "transformers:",
    transformers.__version__,
)

print(
    "numpy:",
    np.__version__,
)

print(
    "device: cpu"
)

print(
    "batch size:",
    batch_size,
)


print(
    "\nMODEL"
)

print(
    "name:",
    model_name,
)

print(
    "revision:",
    model_revision,
)


# ============================================================
# Read canonical cleaned sequence input
# ============================================================

rows = []


with input_path.open() as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    fieldnames = reader.fieldnames


    print(
        "\nINPUT COLUMN CHECK"
    )


    required = [
        "prospective_lineage",
        "pubID",
        "group",
        "representative_cellid",
        heavy_col,
        light_col,
    ]


    for col in required:

        print(
            col,
            "present=",
            col in fieldnames,
        )

        if col not in fieldnames:

            raise ValueError(
                f"Missing required column: {col}"
            )


    for row in reader:

        heavy = (
            row[heavy_col]
            .strip()
            .upper()
        )

        light = (
            row[light_col]
            .strip()
            .upper()
        )


        if "-" in heavy:

            raise ValueError(
                "Unexpected heavy alignment gap after cleanup: "
                f"{row['prospective_lineage']}"
            )


        if "-" in light:

            raise ValueError(
                "Unexpected light alignment gap after cleanup: "
                f"{row['prospective_lineage']}"
            )


        heavy_invalid = (
            set(heavy)
            - allowed
        )

        light_invalid = (
            set(light)
            - allowed
        )


        if heavy_invalid:

            raise ValueError(
                "Invalid heavy characters for "
                f"{row['prospective_lineage']}: "
                f"{sorted(heavy_invalid)}"
            )


        if light_invalid:

            raise ValueError(
                "Invalid light characters for "
                f"{row['prospective_lineage']}: "
                f"{sorted(light_invalid)}"
            )


        sequence_hash = hashlib.sha256(
            (
                heavy
                + "\n"
                + light
            ).encode(
                "utf-8"
            )
        ).hexdigest()


        rows.append({
            "prospective_lineage":
                row["prospective_lineage"],

            "pubID":
                row["pubID"],

            "group":
                row["group"],

            "representative_cellid":
                row["representative_cellid"],

            "heavy":
                heavy,

            "light":
                light,

            "sequence_sha256":
                sequence_hash,
        })


# ============================================================
# Exact frozen-cohort checks
# ============================================================

print(
    "\nINPUT COHORT"
)

print(
    "rows:",
    len(rows),
)


unique_lineages = len(
    {
        row["prospective_lineage"]
        for row in rows
    }
)


print(
    "unique lineages:",
    unique_lineages,
)


participants = len(
    {
        row["pubID"]
        for row in rows
    }
)


print(
    "participants:",
    participants,
)


if len(rows) != 2739:

    raise ValueError(
        "Expected exactly 2739 frozen lineages."
    )


if unique_lineages != 2739:

    raise ValueError(
        "Expected exactly 2739 unique lineages."
    )


if participants != 17:

    raise ValueError(
        "Expected exactly 17 participants."
    )


# Deterministic lineage ordering.
rows = sorted(
    rows,
    key=lambda x:
        x["prospective_lineage"],
)


# ============================================================
# Load tokenizer and model
# ============================================================

print(
    "\nLOADING TOKENIZER"
)


tokenizer = BertTokenizer.from_pretrained(
    model_name,
    revision=model_revision,
    do_lower_case=False,
)


print(
    "tokenizer loaded"
)


print(
    "\nLOADING MODEL"
)


model = BertModel.from_pretrained(
    model_name,
    revision=model_revision,
    add_pooling_layer=False,
)


model.eval()

model.to(
    "cpu"
)


torch.set_num_threads(
    1
)


print(
    "model loaded"
)


print(
    "hidden size:",
    model.config.hidden_size,
)


print(
    "max position embeddings:",
    model.config.max_position_embeddings,
)


if model.config.hidden_size != 1024:

    raise ValueError(
        "Expected IgBERT hidden size 1024."
    )


# ============================================================
# Allocate output arrays
# ============================================================

n_lineages = len(rows)

embedding_dim = (
    model.config.hidden_size
)


embeddings = np.empty(
    (
        n_lineages,
        embedding_dim,
    ),
    dtype=np.float32,
)


token_lengths = np.empty(
    n_lineages,
    dtype=np.int32,
)


sep_counts_all = np.empty(
    n_lineages,
    dtype=np.int32,
)


# ============================================================
# Frozen mean-pooled embedding function
# ============================================================

def embed_batch(
    batch_rows,
):

    paired_sequences = []


    for row in batch_rows:

        paired = (
            " ".join(
                row["heavy"]
            )
            + " [SEP] "
            + " ".join(
                row["light"]
            )
        )

        paired_sequences.append(
            paired
        )


    tokens = tokenizer.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )


    input_ids = (
        tokens[
            "input_ids"
        ]
    )


    attention_mask = (
        tokens[
            "attention_mask"
        ]
    )


    current_token_lengths = (
        attention_mask
        .sum(
            dim=1
        )
        .cpu()
        .numpy()
        .astype(
            np.int32
        )
    )


    current_sep_counts = (
        (
            input_ids
            == tokenizer.sep_token_id
        )
        .sum(
            dim=1
        )
        .cpu()
        .numpy()
        .astype(
            np.int32
        )
    )


    if not np.all(
        current_sep_counts == 2
    ):

        raise ValueError(
            "Expected exactly two [SEP] "
            "tokens per paired sequence."
        )


    if (
        current_token_lengths.max()
        >
        model.config.max_position_embeddings
    ):

        raise ValueError(
            "Sequence exceeds IgBERT "
            "positional capacity."
        )


    with torch.no_grad():

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


        residue_embeddings = (
            output.last_hidden_state
        )


        residue_mask = (
            attention_mask.bool()
        )


        for special_id in (
            tokenizer.all_special_ids
        ):

            residue_mask = (
                residue_mask
                &
                (
                    input_ids
                    != special_id
                )
            )


        mask = (
            residue_mask
            .unsqueeze(
                -1
            )
            .to(
                residue_embeddings.dtype
            )
        )


        counts = (
            mask.sum(
                dim=1
            )
        )


        if (
            counts <= 0
        ).any():

            raise ValueError(
                "A sequence has zero residues "
                "available for pooling."
            )


        summed = (
            residue_embeddings
            * mask
        ).sum(
            dim=1
        )


        pooled = (
            summed
            / counts
        )


    if not torch.isfinite(
        pooled
    ).all():

        raise ValueError(
            "Non-finite embedding encountered."
        )


    return (
        pooled
        .cpu()
        .numpy()
        .astype(
            np.float32
        ),
        current_token_lengths,
        current_sep_counts,
    )


# ============================================================
# Generate embeddings
# ============================================================

print(
    "\nGENERATING EMBEDDINGS"
)


start_time = time.time()


n_batches = (
    n_lineages
    + batch_size
    - 1
) // batch_size


for batch_number, start in enumerate(
    range(
        0,
        n_lineages,
        batch_size,
    ),
    start=1,
):

    end = min(
        start + batch_size,
        n_lineages,
    )


    batch_rows = (
        rows[
            start:end
        ]
    )


    (
        batch_embeddings,
        batch_token_lengths,
        batch_sep_counts,
    ) = embed_batch(
        batch_rows
    )


    embeddings[
        start:end,
        :
    ] = batch_embeddings


    token_lengths[
        start:end
    ] = batch_token_lengths


    sep_counts_all[
        start:end
    ] = batch_sep_counts


    if (
        batch_number == 1
        or batch_number % 25 == 0
        or batch_number == n_batches
    ):

        print(
            "batch",
            batch_number,
            "of",
            n_batches,
            "lineages",
            end,
            "of",
            n_lineages,
        )


elapsed_seconds = (
    time.time()
    - start_time
)


# ============================================================
# Basic generation checks
# ============================================================

print(
    "\nGENERATION SUMMARY"
)


print(
    "embedding shape:",
    embeddings.shape,
)


print(
    "all finite:",
    bool(
        np.isfinite(
            embeddings
        ).all()
    ),
)


print(
    "minimum token length:",
    int(
        token_lengths.min()
    ),
)


print(
    "maximum token length:",
    int(
        token_lengths.max()
    ),
)


print(
    "minimum SEP count:",
    int(
        sep_counts_all.min()
    ),
)


print(
    "maximum SEP count:",
    int(
        sep_counts_all.max()
    ),
)


print(
    "elapsed seconds:",
    elapsed_seconds,
)


if embeddings.shape != (
    2739,
    1024,
):

    raise ValueError(
        "Unexpected final embedding shape."
    )


if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Final embedding matrix contains "
        "non-finite values."
    )


if not np.all(
    sep_counts_all == 2
):

    raise ValueError(
        "Unexpected SEP-token count."
    )


# ============================================================
# Save full embedding matrix
# ============================================================

embedding_path = (
    outdir /
    "257_phase4a_igbert_full_embeddings.npz"
)


np.savez(
    embedding_path,
    prospective_lineage=np.array(
        [
            row[
                "prospective_lineage"
            ]
            for row in rows
        ]
    ),
    embedding=embeddings,
    token_length=token_lengths,
)


# ============================================================
# Save label-blind embedding index
# ============================================================

index_path = (
    outdir /
    "258_phase4a_igbert_full_embedding_index.tsv"
)


with index_path.open(
    "w",
    newline="",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )


    writer.writerow([
        "prospective_lineage",
        "pubID",
        "group",
        "representative_cellid",
        "sequence_sha256",
        "heavy_length",
        "light_length",
        "token_length",
    ])


    for i, row in enumerate(
        rows
    ):

        writer.writerow([
            row[
                "prospective_lineage"
            ],
            row[
                "pubID"
            ],
            row[
                "group"
            ],
            row[
                "representative_cellid"
            ],
            row[
                "sequence_sha256"
            ],
            len(
                row[
                    "heavy"
                ]
            ),
            len(
                row[
                    "light"
                ]
            ),
            int(
                token_lengths[
                    i
                ]
            ),
        ])


# ============================================================
# Save reproducibility metadata
# ============================================================

metadata_path = (
    outdir /
    "259_phase4a_igbert_generation_metadata.tsv"
)


with metadata_path.open(
    "w",
    newline="",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )


    writer.writerow([
        "model_name",
        "model_revision",
        "python_version",
        "torch_version",
        "transformers_version",
        "numpy_version",
        "device",
        "batch_size",
        "lineages",
        "embedding_dimension",
        "minimum_token_length",
        "maximum_token_length",
        "minimum_sep_count",
        "maximum_sep_count",
        "all_finite",
        "elapsed_seconds",
    ])


    writer.writerow([
        model_name,
        model_revision,
        platform.python_version(),
        torch.__version__,
        transformers.__version__,
        np.__version__,
        "cpu",
        batch_size,
        n_lineages,
        embedding_dim,
        int(
            token_lengths.min()
        ),
        int(
            token_lengths.max()
        ),
        int(
            sep_counts_all.min()
        ),
        int(
            sep_counts_all.max()
        ),
        bool(
            np.isfinite(
                embeddings
            ).all()
        ),
        elapsed_seconds,
    ])


print(
    "\nFULL EMBEDDING GENERATION COMPLETE"
)


print(
    "\nSaved:"
)

print(
    embedding_path
)

print(
    index_path
)

print(
    metadata_path
)
