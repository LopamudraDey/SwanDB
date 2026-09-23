# dnabert6_pipeline.py
#
# One-file pipeline for comparing EBF1 WT-specific vs KO-specific ATAC-seq
# peak sequences using DNABERT-6 (original DNABERT, 6-mer tokenization).
#
# Does:
#   1. Load + clean + split data
#   2. Fine-tune DNABERT-6 classifier
#   3. Save BEST model
#   4. Reload BEST model
#   5. Evaluate on held-out TEST set
#   6. Print confusion matrix
#   7. Save confusion matrix as CSV + PNG
#   8. Print accuracy, F1, sensitivity, specificity
#   9. In-silico mutagenesis for interpretability
#  10. Save positional importance
#
# Example:
#
# python dnabert6_pipeline.py \
#     --wt_csv wt_specificfile.csv \
#     --ko_csv Ko_specificfile.csv \
#     --out_dir dnabert6_out \
#     --epochs 5 \
#     --batch_size 16 \
#     --max_length 512 \
#     --n_mutagenesis_per_class 200
#
# Requires:
#   torch
#   transformers
#   scikit-learn
#   pandas
#   numpy
#   matplotlib


import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MODEL_NAME = "zhihan1996/DNA_bert_6"

K = 6

BASES = ["A", "C", "G", "T"]


# --------------------------------------------------------------------------
# 1. Data loading / cleaning / splitting
# --------------------------------------------------------------------------

def load_and_label(path: str, label: int) -> pd.DataFrame:

    # Auto-detect delimiter
    df = pd.read_csv(
        path,
        sep=None,
        engine="python"
    )

    # Clean column names
    df.columns = [
        c.replace("\ufeff", "")
         .strip()
         .strip('"')
         .strip("'")
         .lower()
        for c in df.columns
    ]

    rename_map = {}

    for c in df.columns:

        if c in (
            "chr",
            "chrom",
            "chromosome",
            "seqnames",
            "seqname",
            "#chrom",
            "#chr"
        ):
            rename_map[c] = "chrom"

        elif c in (
            "start",
            "chromstart",
            "peak_start",
            "start_x"
        ):
            rename_map[c] = "start"

        elif c in (
            "end",
            "chromend",
            "stop",
            "peak_end",
            "end_x"
        ):
            rename_map[c] = "end"

        elif c in (
            "seq",
            "sequence",
            "dna_sequence",
            "peak_sequence",
            "fasta_sequence"
        ):
            rename_map[c] = "sequence"

    df = df.rename(columns=rename_map)

    required = {
        "chrom",
        "start",
        "end",
        "sequence"
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Columns actually found in the file: {list(df.columns)}\n"
            f"Fix: either rename these columns in your CSV to "
            f"chrom/start/end/sequence, or add the actual names "
            f"to the rename_map in load_and_label()."
        )

    df = df[
        [
            "chrom",
            "start",
            "end",
            "sequence"
        ]
    ].copy()

    df["label"] = label

    return df


def clean_sequences(
    df: pd.DataFrame,
    min_len: int,
    max_len: int,
    max_n_frac: float = 0.05
) -> pd.DataFrame:

    df = df.copy()

    df["sequence"] = (
        df["sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    valid_chars = re.compile(r"^[ACGTN]+$")

    df = df[
        df["sequence"].apply(
            lambda s: bool(valid_chars.match(s))
        )
    ]

    df["length"] = df["sequence"].str.len()

    df = df[
        (df["length"] >= min_len) &
        (df["length"] <= max_len)
    ]

    n_frac = df["sequence"].apply(
        lambda s: s.count("N") / max(len(s), 1)
    )

    df = df[n_frac <= max_n_frac]

    df = df.drop_duplicates(
        subset="sequence"
    )

    return df.drop(columns=["length"])


def center_trim(
    df: pd.DataFrame,
    window: int
) -> pd.DataFrame:

    df = df.copy()

    def trim(seq: str) -> str:

        L = len(seq)

        if L <= window:
            return seq

        mid = L // 2

        half = window // 2

        start = max(
            0,
            mid - half
        )

        return seq[
            start:start + window
        ]

    df["sequence"] = df["sequence"].apply(trim)

    return df


def seq2kmer(
    seq: str,
    k: int = K
) -> str:

    """
    Convert raw DNA into DNABERT overlapping k-mer format.

    Example:
        ATCGAT -> ATCGAT TCGATC ...
    """

    return " ".join(
        seq[i:i + k]
        for i in range(
            len(seq) - k + 1
        )
    )


def prepare_data(args):

    wt = load_and_label(
        args.wt_csv,
        label=0
    )

    ko = load_and_label(
        args.ko_csv,
        label=1
    )

    print(
        f"Loaded WT: {len(wt)} peaks, "
        f"KO: {len(ko)} peaks"
    )

    wt = clean_sequences(
        wt,
        args.min_len,
        args.max_len
    )

    ko = clean_sequences(
        ko,
        args.min_len,
        args.max_len
    )

    print(
        f"After QC filtering: "
        f"WT: {len(wt)}, "
        f"KO: {len(ko)}"
    )

    if args.center_window > 0:

        wt = center_trim(
            wt,
            args.center_window
        )

        ko = center_trim(
            ko,
            args.center_window
        )

        print(
            f"Centered all sequences to "
            f"{args.center_window}bp window"
        )

    df = pd.concat(
        [wt, ko],
        ignore_index=True
    )

    df = df.sample(
        frac=1.0,
        random_state=args.seed
    ).reset_index(drop=True)

    gc = df["sequence"].apply(
        lambda s:
        (
            s.count("G") +
            s.count("C")
        ) / max(len(s), 1)
    )

    gc_by_label = (
        df.assign(gc=gc)
          .groupby("label")["gc"]
          .mean()
    )

    print(
        "Mean GC content by class "
        "(0=WT, 1=KO):"
    )

    print(gc_by_label)

    if abs(
        gc_by_label.iloc[0] -
        gc_by_label.iloc[1]
    ) > 0.03:

        print(
            "WARNING: GC content differs "
            "by >3pp between classes -- "
            "classifier may partly pick up "
            "composition rather than motif grammar."
        )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    train_val, test = train_test_split(
        df,
        test_size=args.test_frac,
        stratify=df["label"],
        random_state=args.seed
    )

    val_size_adj = (
        args.val_frac /
        (1.0 - args.test_frac)
    )

    train, val = train_test_split(
        train_val,
        test_size=val_size_adj,
        stratify=train_val["label"],
        random_state=args.seed
    )

    for name, split in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:

        counts = (
            split["label"]
            .value_counts()
            .to_dict()
        )

        print(
            f"{name}: {len(split)} sequences, "
            f"label counts (0=WT,1=KO): "
            f"{counts}"
        )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# 2. Dataset + fine-tuning
# --------------------------------------------------------------------------

class KmerDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer,
        max_length,
        k=K
    ):

        self.sequences = (
            df["sequence"].tolist()
        )

        self.labels = (
            df["label"].tolist()
        )

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.k = k

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        kmer_str = seq2kmer(
            self.sequences[idx],
            self.k
        )

        enc = self.tokenizer(
            kmer_str,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            k: v.squeeze(0)
            for k, v in enc.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probs = torch.softmax(
        torch.tensor(logits),
        dim=-1
    )[:, 1].numpy()

    preds = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            preds
        ),

        "f1": f1_score(
            labels,
            preds
        ),

        "auroc": roc_auc_score(
            labels,
            probs
        ),

        "auprc": average_precision_score(
            labels,
            probs
        ),
    }


def train_classifier(
    train_df,
    val_df,
    tokenizer,
    args
):

    print("\nLoading pretrained DNABERT model...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_name,
            num_labels=2
        )
    )

    train_ds = KmerDataset(
        train_df,
        tokenizer,
        args.max_length
    )

    val_ds = KmerDataset(
        val_df,
        tokenizer,
        args.max_length
    )

    training_args = TrainingArguments(

        output_dir=os.path.join(
            args.out_dir,
            "checkpoints"
        ),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=
            args.batch_size,

        per_device_eval_batch_size=
            args.batch_size,

        learning_rate=args.lr,

        weight_decay=0.01,

        warmup_ratio=0.06,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_steps=50,

        load_best_model_at_end=True,

        metric_for_best_model="auroc",

        greater_is_better=True,

        save_total_limit=2,

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=args.seed,
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    print("\nStarting training...")

    trainer.train()

    # ---------------------------------------------------------
    # Validation evaluation
    # ---------------------------------------------------------

    metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print("FINAL VALIDATION METRICS")
    print("=" * 60)

    for k, v in metrics.items():

        print(
            f"{k}: {v}"
        )

    # ---------------------------------------------------------
    # SAVE BEST MODEL
    # ---------------------------------------------------------

    best_dir = os.path.join(
        args.out_dir,
        "best_model"
    )

    trainer.save_model(
        best_dir
    )

    tokenizer.save_pretrained(
        best_dir
    )

    print("\n" + "=" * 60)
    print("BEST MODEL SAVED")
    print("=" * 60)

    print(
        f"Best model directory:\n"
        f"{best_dir}"
    )

    # ---------------------------------------------------------
    # Validation warning
    # ---------------------------------------------------------

    if metrics.get(
        "eval_auroc",
        0.5
    ) < 0.6:

        print(
            "\nNOTE: validation AUROC is "
            "near chance (0.5) -- WT vs KO "
            "may not have a strongly learnable "
            "sequence-level grammar difference here."
        )

        print(
            "Treat mutagenesis results cautiously."
        )

    return trainer.model, metrics


# --------------------------------------------------------------------------
# 3. Prediction
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(
    model,
    tokenizer,
    seqs,
    device,
    max_length,
    k=K,
    batch_size=32
):

    """
    P(class=1, i.e. KO-specific)
    for raw DNA sequences.
    """

    probs = []

    for i in range(
        0,
        len(seqs),
        batch_size
    ):

        batch = [
            seq2kmer(s, k)
            for s in seqs[
                i:i + batch_size
            ]
        ]

        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        logits = model(
            **enc
        ).logits

        p = F.softmax(
            logits,
            dim=-1
        )[:, 1]

        probs.append(
            p.cpu().numpy()
        )

    return np.concatenate(
        probs
    )


# --------------------------------------------------------------------------
# 4. TEST SET EVALUATION + CONFUSION MATRIX
# --------------------------------------------------------------------------

def evaluate_test_set(
    model,
    tokenizer,
    test_df,
    args
):

    print("\n" + "=" * 60)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("=" * 60)

    device = next(
        model.parameters()
    ).device

    # ---------------------------------------------------------
    # Predict test probabilities
    # ---------------------------------------------------------

    probs = predict_proba(
        model,
        tokenizer,
        test_df["sequence"].tolist(),
        device,
        args.max_length
    )

    # ---------------------------------------------------------
    # Convert probability to class
    #
    # label 0 = WT-specific
    # label 1 = KO-specific
    # ---------------------------------------------------------

    predictions = (
        probs >= 0.5
    ).astype(int)

    true_labels = (
        test_df["label"].values
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("TEST SET CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\n                 Predicted"
    )

    print(
        "                 WT       KO"
    )

    print(
        f"True WT       {tn:8d} {fp:8d}"
    )

    print(
        f"True KO       {fn:8d} {tp:8d}"
    )

    print("\nConfusion matrix array:")

    print(cm)

    # ---------------------------------------------------------
    # Individual values
    # ---------------------------------------------------------

    print("\nConfusion matrix components:")

    print(
        f"TN (True WT predicted WT) : {tn}"
    )

    print(
        f"FP (True WT predicted KO) : {fp}"
    )

    print(
        f"FN (True KO predicted WT) : {fn}"
    )

    print(
        f"TP (True KO predicted KO) : {tp}"
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    f1 = f1_score(
        true_labels,
        predictions
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # AUROC
    auroc = roc_auc_score(
        true_labels,
        probs
    )

    # AUPRC
    auprc = average_precision_score(
        true_labels,
        probs
    )

    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"F1 score    : {f1:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"AUROC       : {auroc:.4f}"
    )

    print(
        f"AUPRC       : {auprc:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        true_labels,
        predictions,
        target_names=[
            "WT-specific",
            "KO-specific"
        ],
        digits=4
    )

    print(report)

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    predictions_df = test_df.copy()

    predictions_df[
        "predicted_probability_KO"
    ] = probs

    predictions_df[
        "predicted_label"
    ] = predictions

    predictions_df[
        "correct"
    ] = (
        predictions_df["label"]
        ==
        predictions_df["predicted_label"]
    )

    predictions_path = os.path.join(
        args.out_dir,
        "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"Saved test predictions:\n"
        f"{predictions_path}"
    )

    # ---------------------------------------------------------
    # Save confusion matrix CSV
    # ---------------------------------------------------------

    cm_df = pd.DataFrame(

        cm,

        index=[
            "True_WT",
            "True_KO"
        ],

        columns=[
            "Pred_WT",
            "Pred_KO"
        ]
    )

    cm_csv = os.path.join(
        args.out_dir,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_csv
    )

    print(
        f"Saved confusion matrix CSV:\n"
        f"{cm_csv}"
    )

    # ---------------------------------------------------------
    # Save confusion matrix PNG
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "WT-specific",
            "KO-specific"
        ]
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "DNABERT-6 Test Set Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    plt.tight_layout()

    cm_png = os.path.join(
        args.out_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix PNG:\n"
        f"{cm_png}"
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_df = pd.DataFrame({
        "metric": [
            "accuracy",
            "f1",
            "sensitivity",
            "specificity",
            "auroc",
            "auprc",
            "TN",
            "FP",
            "FN",
            "TP"
        ],

        "value": [
            accuracy,
            f1,
            sensitivity,
            specificity,
            auroc,
            auprc,
            tn,
            fp,
            fn,
            tp
        ]
    })

    metrics_path = os.path.join(
        args.out_dir,
        "test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"Saved test metrics:\n"
        f"{metrics_path}"
    )

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "auprc": auprc,
    }


# --------------------------------------------------------------------------
# 5. In-silico mutagenesis
# --------------------------------------------------------------------------

@torch.no_grad()
def mutagenize_one(
    seq: str,
    model,
    tokenizer,
    device,
    max_length: int
):

    base_p = predict_proba(
        model,
        tokenizer,
        [seq],
        device,
        max_length
    )[0]

    variants = []

    positions = []

    for i, ref in enumerate(seq):

        if ref not in BASES:
            continue

        for alt in BASES:

            if alt == ref:
                continue

            variants.append(
                seq[:i]
                + alt
                + seq[i + 1:]
            )

            positions.append(i)

    if not variants:

        return (
            np.zeros(len(seq)),
            base_p
        )

    mut_probs = predict_proba(
        model,
        tokenizer,
        variants,
        device,
        max_length,
        batch_size=64
    )

    deltas = np.abs(
        mut_probs - base_p
    )

    importance = np.zeros(
        len(seq)
    )

    for pos, d in zip(
        positions,
        deltas
    ):

        importance[pos] = max(
            importance[pos],
            d
        )

    return (
        importance,
        base_p
    )


def run_mutagenesis(
    model,
    tokenizer,
    test_df,
    args
):

    device = next(
        model.parameters()
    ).device

    def crop(s):

        if len(s) <= args.max_seq_bp:
            return s

        mid = len(s) // 2

        half = (
            args.max_seq_bp // 2
        )

        start = max(
            0,
            mid - half
        )

        return s[
            start:
            start + args.max_seq_bp
        ]

    test_df = test_df.copy()

    test_df["seq_cropped"] = (
        test_df["sequence"]
        .apply(crop)
    )

    all_probs = predict_proba(
        model,
        tokenizer,
        test_df["seq_cropped"].tolist(),
        device,
        args.max_length
    )

    test_df["pred_label"] = (
        all_probs >= 0.5
    ).astype(int)

    test_df["correct"] = (
        test_df["pred_label"]
        ==
        test_df["label"]
    )

    records = []

    profiles = {
        0: [],
        1: []
    }

    for label in [0, 1]:

        subset = test_df[
            (test_df["label"] == label)
            &
            (test_df["correct"])
        ]

        n = min(
            args.n_mutagenesis_per_class,
            len(subset)
        )

        subset = subset.sample(
            n=n,
            random_state=args.seed
        )

        print(
            f"Mutagenizing {len(subset)} "
            f"correctly-classified "
            f"label={label} sequences..."
        )

        for _, row in subset.iterrows():

            importance, _ = mutagenize_one(
                row["seq_cropped"],
                model,
                tokenizer,
                device,
                args.max_length
            )

            profiles[label].append(
                importance
            )

            for pos, score in enumerate(
                importance
            ):

                records.append({

                    "chrom":
                        row.get(
                            "chrom",
                            ""
                        ),

                    "start":
                        row.get(
                            "start",
                            ""
                        ),

                    "end":
                        row.get(
                            "end",
                            ""
                        ),

                    "label":
                        label,

                    "position":
                        pos,

                    "importance":
                        score,

                    "base":
                        row[
                            "seq_cropped"
                        ][pos],
                })

    # ---------------------------------------------------------
    # Save per-position importance
    # ---------------------------------------------------------

    out_csv = os.path.join(
        args.out_dir,
        "per_position_importance.csv"
    )

    pd.DataFrame(
        records
    ).to_csv(
        out_csv,
        index=False
    )

    print(
        f"Saved per-position importance "
        f"scores to:\n{out_csv}"
    )

    # ---------------------------------------------------------
    # Plot positional importance
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 4)
    )

    for label, name in [
        (0, "WT-specific"),
        (1, "KO-specific")
    ]:

        arrs = [
            p
            for p in profiles[label]
            if len(p) == args.max_seq_bp
        ]

        if not arrs:
            continue

        mean_profile = np.mean(
            np.stack(arrs),
            axis=0
        )

        plt.plot(
            mean_profile,
            label=name,
            alpha=0.8
        )

    plt.xlabel(
        "Position (bp, center-cropped window)"
    )

    plt.ylabel(
        "Mean mutagenesis importance"
    )

    plt.title(
        "DNABERT-6: average positional importance, "
        "WT-specific vs KO-specific"
    )

    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        args.out_dir,
        "positional_importance_profile.png"
    )

    plt.savefig(
        plot_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved profile plot to:\n"
        f"{plot_path}"
    )


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--wt_csv",
        required=True
    )

    ap.add_argument(
        "--ko_csv",
        required=True
    )

    ap.add_argument(
        "--out_dir",
        default="dnabert6_out"
    )

    ap.add_argument(
        "--model_name",
        default=MODEL_NAME
    )

    ap.add_argument(
        "--min_len",
        type=int,
        default=50
    )

    ap.add_argument(
        "--max_len",
        type=int,
        default=1000
    )

    ap.add_argument(
        "--center_window",
        type=int,
        default=0,
        help=(
            "If >0, trim every sequence "
            "to this fixed width centered "
            "on its midpoint."
        )
    )

    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--test_frac",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    ap.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "Maximum k-mer tokens "
            "(DNABERT max is 512)."
        )
    )

    ap.add_argument(
        "--n_mutagenesis_per_class",
        type=int,
        default=200
    )

    ap.add_argument(
        "--max_seq_bp",
        type=int,
        default=300,
        help=(
            "Cap raw bp actually "
            "mutagenized per sequence "
            "for runtime control."
        )
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--skip_mutagenesis",
        action="store_true"
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    train_df, val_df, test_df = (
        prepare_data(args)
    )

    # Save splits

    for name, split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df)
    ]:

        split.to_csv(
            os.path.join(
                args.out_dir,
                f"{name}.csv"
            ),
            index=False
        )

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------

    print(
        f"\nLoading tokenizer/model: "
        f"{args.model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    model, validation_metrics = (
        train_classifier(
            train_df,
            val_df,
            tokenizer,
            args
        )
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Reload the BEST SAVED model
    # ---------------------------------------------------------

    best_dir = os.path.join(
        args.out_dir,
        "best_model"
    )

    print("\n" + "=" * 60)
    print("RELOADING BEST SAVED MODEL")
    print("=" * 60)

    print(
        f"Loading from:\n{best_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        best_dir
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            best_dir,
            num_labels=2
        )
    )

    model = (
        model
        .to(device)
        .eval()
    )

    print(
        "Best model loaded successfully."
    )

    # ---------------------------------------------------------
    # TEST SET + CONFUSION MATRIX
    # ---------------------------------------------------------

    test_results = evaluate_test_set(
        model,
        tokenizer,
        test_df,
        args
    )

    # ---------------------------------------------------------
    # MUTAGENESIS
    # ---------------------------------------------------------

    if not args.skip_mutagenesis:

        run_mutagenesis(
            model,
            tokenizer,
            test_df,
            args
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model:\n"
        f"  {best_dir}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"  {os.path.join(args.out_dir, 'confusion_matrix.png')}"
    )

    print(
        "\nConfusion matrix CSV:"
    )

    print(
        f"  {os.path.join(args.out_dir, 'confusion_matrix.csv')}"
    )

    print(
        "\nTest predictions:"
    )

    print(
        f"  {os.path.join(args.out_dir, 'test_predictions.csv')}"
    )

    print(
        "\nTest metrics:"
    )

    print(
        f"  {os.path.join(args.out_dir, 'test_metrics.csv')}"
    )

    print("\nDone.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":

    main()