# --------------------- Imports ---------------------
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from collections import defaultdict
import matplotlib.pyplot as plt

# --------------------- Output directory ---------------------
OUT_DIR = "ebf1_bertcnn_out_enhancers"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------- Tokenization ---------------------
def tokenize_dna(sequence, k=6):
    sequence = sequence.upper()
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

# --------------------- Data Loading ---------------------
#wt_df = pd.read_csv("wtspecific_promoters.csv", delimiter=";")
#back_df = pd.read_csv("kospecific_promoters.csv", delimiter=";")
#wt_df = pd.read_csv("wt_specificfile.csv", delimiter=";")
#back_df = pd.read_csv("ebf1ko_specificfile.csv", delimiter=";")
wt_df = pd.read_csv("/home/lopde33/EBF1_Enhancers/WT_NoPromoter512bp.csv", delimiter=";")
back_df = pd.read_csv("/home/lopde33/EBF1_Enhancers/KO_NoPromoter512bp.csv", delimiter=";")
# Labels
wt_df["label"] = 0
back_df["label"] = 1

# Balance classes
min_size = min(len(wt_df), len(back_df))
wt_balanced = wt_df.sample(n=min_size, random_state=42)
back_balanced = back_df.sample(n=min_size, random_state=42)
dfall = pd.concat([wt_balanced, back_balanced], ignore_index=True)

# --------------------- Data Augmentation helper ---------------------
def reverse_complement(seq):
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]

# NOTE: augmentation is intentionally applied ONLY after the chromosome
# split, and ONLY to the training partition (see below). Applying it here,
# before the split, was a bug in the previous version: it duplicated the
# reverse-complement into BOTH train and test sets, and since RC(RC(seq))
# == seq, doing it again later on train_df just re-created rows that were
# already present rather than adding new variety.

dfall["tokenized_sequence"] = dfall["sequence"].apply(lambda x: tokenize_dna(x, k=6))

# --------------------- Train/Test Split by Chromosome ---------------------
dfall["chrom"] = dfall["chrom"].astype(str)
train_chroms = [f"chr{i}" for i in range(1, 20) if i not in [6, 7]]

train_df = dfall[dfall["chrom"].isin(train_chroms)].copy()
test_df = dfall[dfall["chrom"].isin(["chr6", "chr7"])].copy()

# Reverse-complement augmentation applied ONLY to training data, AFTER
# the chromosome split, so the test set (chr6/chr7) stays untouched and
# genuinely independent.
aug_train = train_df.copy()
aug_train["sequence"] = aug_train["sequence"].apply(reverse_complement)
aug_train["tokenized_sequence"] = aug_train["sequence"].apply(lambda x: tokenize_dna(x, k=6))

train_df = pd.concat([train_df, aug_train], ignore_index=True)

df_train_val = train_df[["tokenized_sequence", "label"]]
df_test = test_df[["tokenized_sequence", "label"]]

# test_df, re-indexed 0..n-1, is what seq_idx (built inside
# SequenceDatasetSliding from df_test's row order) lines up with. We keep
# this around so we can pull chrom/start/end/raw sequence back out later
# for confusion-matrix bookkeeping and mutagenesis, since df_test itself
# only carries tokenized_sequence/label.
test_df_meta = test_df.reset_index(drop=True)

# --------------------- Tokenizer ---------------------
tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")

# --------------------- Dataset with Sliding Windows ---------------------
class SequenceDatasetSliding(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512, stride=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.samples = []
        self.seq_indices = []  # track original sequence index for aggregation

        self.starts = []  # k-mer offset of each window, needed to re-align
                           # attention scores back onto the full sequence

        for idx, row in self.data.iterrows():
            seq = row['tokenized_sequence'].split()  # list of k-mers
            label = int(row['label'])

            for start in range(0, len(seq), self.stride):
                end = start + self.max_len
                window = seq[start:end]
                if len(window) == 0:
                    continue
                self.samples.append({
                    'window': " ".join(window),
                    'label': label
                })
                self.seq_indices.append(idx)
                self.starts.append(start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample['window'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'seq_idx': self.seq_indices[idx],
            'start': self.starts[idx]
        }

# --------------------- Model Definition ---------------------
class BertCNN(nn.Module):
    def __init__(self, num_classes=2, num_unfrozen_layers=1):
        """
        num_unfrozen_layers: how many of the LAST transformer layers to keep
        trainable. Everything else (embeddings + earlier layers) is frozen.
        Set to 0 to freeze all of BERT and only train the CNN+FC head.
        Set to 12 to fine-tune the whole model (highest overfitting risk).
        """
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained("zhihan1996/DNA_bert_6", output_attentions=True)

        total_layers = self.bert.config.num_hidden_layers  # 12 for base BERT
        unfreeze_from = total_layers - num_unfrozen_layers  # e.g. 12-2=10 -> unfreeze layers 10,11

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer."):
                layer_num = int(name.split(".")[2])
                param.requires_grad = layer_num >= unfreeze_from
            else:
                # embeddings, pooler, etc. -> keep frozen
                param.requires_grad = False

        # Sanity check / log which layers are trainable
        trainable = sorted({
            name.split(".")[2] for name, p in self.bert.named_parameters()
            if p.requires_grad and name.startswith("encoder.layer.")
        }, key=int)
        n_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.bert.parameters())
        print(f"Trainable BERT layers: {trainable} "
              f"({n_trainable:,} / {n_total:,} params, {100*n_trainable/n_total:.1f}%)")

        self.conv1 = nn.Conv1d(768, 128, kernel_size=11, padding=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        attentions = outputs.attentions[-1].mean(dim=1)
        x = outputs.last_hidden_state.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits, attentions

# --------------------- Train/Validation Split ---------------------
train_len = int(0.9 * len(df_train_val))
val_len = len(df_train_val) - train_len
train_subset, val_subset = random_split(
    df_train_val,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = SequenceDatasetSliding(df_train_val.iloc[train_subset.indices], tokenizer)
val_dataset = SequenceDatasetSliding(df_train_val.iloc[val_subset.indices], tokenizer)
test_dataset = SequenceDatasetSliding(df_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --------------------- Training Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertCNN(num_classes=2, num_unfrozen_layers=1).to(device)

# Discriminative learning rates: BERT gets a much smaller LR than the new head,
# since BERT is pretrained and the head is trained from scratch.
bert_params = [p for n, p in model.named_parameters() if n.startswith("bert.") and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]

optimizer = AdamW([
    {"params": bert_params, "lr": 2e-6},
    {"params": head_params, "lr": 2e-4},
], weight_decay=1e-4)

# Class weights
y_train = train_dataset.data['label'].tolist()
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# Scheduler
num_epochs = 15
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

# --------------------- Training Loop ---------------------
best_val_loss = float('inf')
patience, patience_counter = 3, 0  # lowered from 5 -> val loss rose quickly last time
CHECKPOINT_PATH = "Ebf128wtvskoenhancers_v1.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# --------------------- Testing with Window + Attention Aggregation ---------------------
# Reload best checkpoint before testing (in case training continued past the best epoch)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

sequence_logits = defaultdict(list)
sequence_labels = {}
sequence_preds = {}
sequence_probs = {}  # P(class=1, i.e. KO/Background) per full sequence

# total k-mer length of each full test sequence, keyed by the same
# positional index used as seq_idx in SequenceDatasetSliding
seq_lengths = test_dataset.data["tokenized_sequence"].apply(
    lambda x: len(x.split())
).to_dict()

attn_sum = {}
attn_count = {}

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        seq_idx = batch["seq_idx"].item()
        start = batch["start"].item()

        logits, attentions = model(input_ids, attention_mask)

        sequence_logits[seq_idx].append(logits.cpu().numpy())
        sequence_labels[seq_idx] = labels.item()

        # --- per-position attention, re-aligned to full-sequence coords ---
        # attentions: (1, seq_len, seq_len) -> how much attention each
        # position RECEIVES on average, across all query positions
        token_scores = attentions.mean(dim=1).squeeze(0).cpu().numpy()

        mask = attention_mask.squeeze(0).cpu().numpy()
        real_len = int(mask.sum())              # includes [CLS] + [SEP]
        kmer_scores = token_scores[1:real_len - 1]  # drop [CLS]/[SEP]
        n_kmers = len(kmer_scores)

        total_len = seq_lengths[seq_idx]
        if seq_idx not in attn_sum:
            attn_sum[seq_idx] = np.zeros(total_len)
            attn_count[seq_idx] = np.zeros(total_len)

        end = min(start + n_kmers, total_len)
        usable = end - start
        if usable > 0:
            attn_sum[seq_idx][start:end] += kmer_scores[:usable]
            attn_count[seq_idx][start:end] += 1

# average attention across overlapping windows -> one profile per sequence
attn_profiles = {
    seq_idx: attn_sum[seq_idx] / np.maximum(attn_count[seq_idx], 1)
    for seq_idx in attn_sum
}

# Aggregate predictions per original sequence
final_preds, final_labels = [], []
for seq_idx, logit_list in sequence_logits.items():
    avg_logits = np.mean(logit_list, axis=0)
    pred = np.argmax(avg_logits)
    prob_class1 = F.softmax(torch.tensor(avg_logits), dim=-1).numpy()[0, 1]
    sequence_preds[seq_idx] = pred
    sequence_probs[seq_idx] = float(prob_class1)
    final_preds.append(pred)
    final_labels.append(sequence_labels[seq_idx])

# Evaluate
print("Test Accuracy:", accuracy_score(final_labels, final_preds))
print("\nClassification Report:")
print(classification_report(final_labels, final_preds, target_names=["WT", "Background"]))

# --------------------- Confusion Matrix + Metrics (added) ---------------------
# Same reporting as the DNABERT-6 single-pass pipeline, but built from the
# per-sequence predictions/probabilities aggregated above (i.e. after the
# sliding-window logits have already been averaged per full sequence).

final_probs = [sequence_probs[seq_idx] for seq_idx in sequence_logits.keys()]

cm = confusion_matrix(final_labels, final_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("TEST SET CONFUSION MATRIX")
print("=" * 60)
print("\n                 Predicted")
print("                 WT       Background")
print(f"True WT          {tn:8d} {fp:8d}")
print(f"True Background  {fn:8d} {tp:8d}")
print("\nConfusion matrix array:")
print(cm)

print("\nConfusion matrix components:")
print(f"TN (True WT predicted WT)               : {tn}")
print(f"FP (True WT predicted Background)       : {fp}")
print(f"FN (True Background predicted WT)       : {fn}")
print(f"TP (True Background predicted Background): {tp}")

accuracy = accuracy_score(final_labels, final_preds)
f1 = f1_score(final_labels, final_preds)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
auroc = roc_auc_score(final_labels, final_probs)
auprc = average_precision_score(final_labels, final_probs)

print("\n" + "=" * 60)
print("TEST SET METRICS")
print("=" * 60)
print(f"Accuracy    : {accuracy:.4f}")
print(f"F1 score    : {f1:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"AUROC       : {auroc:.4f}")
print(f"AUPRC       : {auprc:.4f}")

# Save per-sequence predictions (with chrom/start/end pulled back in from
# test_df_meta, since seq_idx lines up with its row order)
pred_rows = []
for seq_idx in sequence_logits.keys():
    meta = test_df_meta.iloc[seq_idx]
    pred_rows.append({
        "chrom": meta.get("chrom", ""),
        "start": meta.get("start", ""),
        "end": meta.get("end", ""),
        "label": sequence_labels[seq_idx],
        "predicted_label": int(sequence_preds[seq_idx]),
        "predicted_probability_class1": sequence_probs[seq_idx],
        "correct": sequence_labels[seq_idx] == sequence_preds[seq_idx],
    })
predictions_df = pd.DataFrame(pred_rows)
predictions_path = os.path.join(OUT_DIR, "test_predictions.csv")
predictions_df.to_csv(predictions_path, index=False)
print(f"\nSaved test predictions:\n{predictions_path}")

# Save confusion matrix CSV
cm_df = pd.DataFrame(
    cm,
    index=["True_WT", "True_Background"],
    columns=["Pred_WT", "Pred_Background"]
)
cm_csv = os.path.join(OUT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv)
print(f"Saved confusion matrix CSV:\n{cm_csv}")

# Save confusion matrix PNG
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["WT", "Background"])
disp.plot(ax=ax, values_format="d")
ax.set_title("BertCNN Test Set Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()
cm_png = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix PNG:\n{cm_png}")

# Save metrics CSV
metrics_df = pd.DataFrame({
    "metric": ["accuracy", "f1", "sensitivity", "specificity", "auroc", "auprc", "TN", "FP", "FN", "TP"],
    "value": [accuracy, f1, sensitivity, specificity, auroc, auprc, tn, fp, fn, tp]
})
metrics_path = os.path.join(OUT_DIR, "test_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved test metrics:\n{metrics_path}")

if auroc < 0.6:
    print("\nNOTE: test AUROC is near chance (0.5) -- WT vs KO may not have a "
          "strongly learnable sequence-level grammar difference here.")
    print("Treat mutagenesis results below cautiously.")

# --------------------- WT vs KO Differential Attention (discriminative motifs) ---------------------
# Raw per-sequence attention (saved above) tells you what each sequence's
# tokens generally attended to -- not necessarily what separates the two
# classes. To find positions that DISCRIMINATE WT vs KO, we average the
# attention profile separately over correctly-classified WT sequences and
# correctly-classified KO sequences, then take the difference. Large
# positive/negative values mark k-mer positions where the two classes'
# internal attention patterns diverge most -- these are your candidate
# discriminative motif locations, not just "high attention" positions.

print("\nSaving per-sequence attention profiles and WT-vs-KO differential motif profile...")

for seq_idx, profile in attn_profiles.items():
    pd.DataFrame({
        "position": np.arange(len(profile)),
        "attention_score": profile
    }).to_csv(os.path.join(OUT_DIR, f"attention_profile_seq{seq_idx}.csv"), index=False)

# Only use CORRECTLY classified sequences for the class-conditional average,
# so mispredicted sequences (which may have misleading attention) don't
# pollute the discriminative signal.
wt_profiles, ko_profiles = [], []
for seq_idx, profile in attn_profiles.items():
    if sequence_preds[seq_idx] != sequence_labels[seq_idx]:
        continue  # skip misclassified sequences
    if sequence_labels[seq_idx] == 0:      # WT
        wt_profiles.append(profile)
    else:                                   # KO / Background
        ko_profiles.append(profile)

if len(wt_profiles) > 0 and len(ko_profiles) > 0:
    # sequences can differ in length; align on the shortest common length
    min_len = min(min(len(p) for p in wt_profiles), min(len(p) for p in ko_profiles))
    wt_avg = np.mean([p[:min_len] for p in wt_profiles], axis=0)
    ko_avg = np.mean([p[:min_len] for p in ko_profiles], axis=0)
    diff_profile = wt_avg - ko_avg  # positive = more WT-associated, negative = more KO-associated

    pd.DataFrame({
        "position": np.arange(min_len),
        "wt_avg_attention": wt_avg,
        "ko_avg_attention": ko_avg,
        "wt_minus_ko_diff": diff_profile
    }).to_csv(os.path.join(OUT_DIR, "wt_vs_ko_differential_attention.csv"), index=False)

    top_wt_positions = np.argsort(diff_profile)[-10:][::-1]
    top_ko_positions = np.argsort(diff_profile)[:10]

    print(f"\nSaved wt_vs_ko_differential_attention.csv "
          f"(based on {len(wt_profiles)} correct WT and {len(ko_profiles)} correct KO sequences)")
    print(f"Top 10 positions with WT-associated attention: {top_wt_positions.tolist()}")
    print(f"Top 10 positions with KO-associated attention: {top_ko_positions.tolist()}")
    print("NOTE: 'position' is a k-mer index; multiply/offset appropriately "
          "to map back to genomic coordinates for motif lookup (e.g. against known EBF1 sites).")
else:
    print("\nSkipped differential attention: not enough correctly-classified "
          "sequences in one or both classes.")

# --------------------- In-silico Mutagenesis (added) ---------------------
# Ported from the DNABERT-6 single-pass pipeline. The key difference here is
# that a BertCNN prediction for one raw sequence is itself an AGGREGATE over
# sliding windows (same logic as the test loop above), so predict_proba_seq()
# below replays that windowing + logit-averaging for one arbitrary sequence
# on demand. Because that's expensive per point mutation (3 * len(seq) calls,
# each an average over ceil(len/stride) windows), sequences are first cropped
# to MAX_SEQ_BP around their midpoint, same as in the DNABERT-6 pipeline, and
# only a capped, correctly-classified sample per class is mutagenized.

K = 6
MAX_SEQ_BP = 300           # raw bp cropped+mutagenized per sequence
N_MUTAGENESIS_PER_CLASS = 50
MUT_MAX_LEN = 512
MUT_STRIDE = 256
BASES = ["A", "C", "G", "T"]


def crop_center(seq, window):
    L = len(seq)
    if L <= window:
        return seq
    mid = L // 2
    half = window // 2
    start = max(0, mid - half)
    return seq[start:start + window]


@torch.no_grad()
def predict_proba_seq(model, tokenizer, raw_seq, device, k=K,
                       max_len=MUT_MAX_LEN, stride=MUT_STRIDE):
    """
    P(class=1, i.e. KO/Background-specific) for one raw DNA sequence,
    replicating the sliding-window + logit-averaging used at test time.
    """
    kmer_str = tokenize_dna(raw_seq, k=k)
    tokens = kmer_str.split()
    if len(tokens) == 0:
        return 0.5

    logits_list = []
    for start in range(0, len(tokens), stride):
        end = start + max_len
        window = tokens[start:end]
        if len(window) == 0:
            continue
        enc = tokenizer(
            " ".join(window),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        logits, _ = model(enc["input_ids"], enc["attention_mask"])
        logits_list.append(logits.cpu().numpy())
        if end >= len(tokens):
            break

    avg_logits = np.mean(logits_list, axis=0)
    probs = F.softmax(torch.tensor(avg_logits), dim=-1).numpy()[0]
    return float(probs[1])


def mutagenize_one(raw_seq, model, tokenizer, device, k=K,
                    max_len=MUT_MAX_LEN, stride=MUT_STRIDE):
    base_p = predict_proba_seq(model, tokenizer, raw_seq, device, k, max_len, stride)
    importance = np.zeros(len(raw_seq))

    for i, ref in enumerate(raw_seq):
        if ref not in BASES:
            continue
        best_delta = 0.0
        for alt in BASES:
            if alt == ref:
                continue
            mut_seq = raw_seq[:i] + alt + raw_seq[i + 1:]
            mut_p = predict_proba_seq(model, tokenizer, mut_seq, device, k, max_len, stride)
            delta = abs(mut_p - base_p)
            if delta > best_delta:
                best_delta = delta
        importance[i] = best_delta

    return importance, base_p


print("\n" + "=" * 60)
print("IN-SILICO MUTAGENESIS")
print("=" * 60)

test_df_meta["seq_cropped"] = test_df_meta["sequence"].apply(lambda s: crop_center(s, MAX_SEQ_BP))

records = []
mut_profiles = {0: [], 1: []}

for label in [0, 1]:
    label_name = "WT" if label == 0 else "Background"

    candidate_idxs = [
        seq_idx for seq_idx in sequence_logits.keys()
        if sequence_labels[seq_idx] == label and sequence_preds[seq_idx] == label
    ]

    n = min(N_MUTAGENESIS_PER_CLASS, len(candidate_idxs))
    if n == 0:
        print(f"No correctly-classified {label_name} sequences available; skipping.")
        continue

    rng = np.random.RandomState(42)
    chosen_idxs = rng.choice(candidate_idxs, size=n, replace=False)

    print(f"Mutagenizing {n} correctly-classified {label_name} sequences "
          f"(cropped to {MAX_SEQ_BP}bp)...")

    for seq_idx in chosen_idxs:
        meta = test_df_meta.iloc[seq_idx]
        cropped_seq = meta["seq_cropped"]

        importance, _ = mutagenize_one(cropped_seq, model, tokenizer, device)
        mut_profiles[label].append(importance)

        for pos, score in enumerate(importance):
            records.append({
                "chrom": meta.get("chrom", ""),
                "start": meta.get("start", ""),
                "end": meta.get("end", ""),
                "label": label,
                "position": pos,
                "importance": score,
                "base": cropped_seq[pos],
            })

importance_csv = os.path.join(OUT_DIR, "per_position_importance.csv")
pd.DataFrame(records).to_csv(importance_csv, index=False)
print(f"Saved per-position importance scores to:\n{importance_csv}")

# --------------------- Positional importance plot ---------------------
plt.figure(figsize=(10, 4))
for label, name in [(0, "WT-specific"), (1, "Background/KO-specific")]:
    arrs = [p for p in mut_profiles[label] if len(p) == MAX_SEQ_BP]
    if not arrs:
        continue
    mean_profile = np.mean(np.stack(arrs), axis=0)
    plt.plot(mean_profile, label=name, alpha=0.8)

plt.xlabel("Position (bp, center-cropped window)")
plt.ylabel("Mean mutagenesis importance")
plt.title("BertCNN: average positional importance, WT-specific vs Background-specific")
plt.legend()
plt.tight_layout()
profile_png = os.path.join(OUT_DIR, "positional_importance_profile.png")
plt.savefig(profile_png, dpi=150)
plt.close()
print(f"Saved positional importance profile plot to:\n{profile_png}")

print("\nDone.")