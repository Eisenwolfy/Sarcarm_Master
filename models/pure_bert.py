"""
Sarcasm detector — PURE DistilBERT baseline, Kaggle Notebook version.

This is the plain-vanilla baseline everything else should be compared
against: comment and parent_comment are concatenated into a single
sequence ("[CLS] comment [SEP] parent_comment [SEP]", the standard way
BERT-family models handle sentence-pair tasks), fed through DistilBERT,
and the [CLS] token's representation goes straight into a linear
classifier. No CNN, no LSTM, no metadata, no dual-branch fusion —
just "fine-tune BERT on the pair and see what you get".

Comparing this number against the CNN+metadata version tells you how
much of your final model's accuracy actually comes from the extra
architecture/features, versus what BERT gets you for free.

HOW TO USE ON KAGGLE: same setup as before (add the dataset, enable
GPU — T4 x2, NOT P100 — and Internet, paste into a cell, run).
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# =====================================================================
# 0. CONFIG
# =====================================================================
DATA_PATH = "/kaggle/input/datasets/sherinclaudia/sarcastic-comments-on-reddit/train-balanced-sarcasm.csv"
OUTPUT_DIR = "/kaggle/working"

SUBSAMPLE_FRAC = 1.0

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 80          # comment + parent combined need more room than either alone
FREEZE_BERT_LAYERS = 2 # keep identical to the other version for a fair comparison

BATCH_SIZE = 32
EPOCHS = 6
BERT_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1
PATIENCE = 2
USE_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================================
# 1. LOAD DATA
# =====================================================================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["comment", "parent_comment"])

if SUBSAMPLE_FRAC < 1.0:
    df = df.sample(frac=SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(df)} rows.")

comments = df["comment"].astype(str).values
parents = df["parent_comment"].astype(str).values
labels = df["label"].astype(int).values

print("Total examples:", len(comments))
print("Class balance:\n", df["label"].value_counts())


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


comments = [clean_text(t) for t in comments]
parents = [clean_text(t) for t in parents]

# =====================================================================
# 2. TRAIN / VAL / TEST SPLIT (same seed/ratios as the other version)
# =====================================================================
indices = np.arange(len(comments))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])


def subset(arr, idx):
    return [arr[i] for i in idx]


c_train, c_val, c_test = subset(comments, train_idx), subset(comments, val_idx), subset(comments, test_idx)
p_train, p_val, p_test = subset(parents, train_idx), subset(parents, val_idx), subset(parents, test_idx)
y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

print(f"Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")

# =====================================================================
# 3. TOKENIZER + DATASET (sentence-pair encoding: comment [SEP] parent)
# =====================================================================
print(f"Loading tokenizer for {BERT_MODEL_NAME}...")
tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
print("Tokenizer ready.")


class SarcasmPairDataset(Dataset):
    def __init__(self, comment_texts, parent_texts, labels_):
        # text_pair encodes both sequences together with a [SEP] between
        # them, which is the standard BERT sentence-pair format — this
        # is what makes this a "plain" baseline rather than a custom
        # dual-branch architecture.
        self.encodings = tokenizer(
            list(comment_texts), list(parent_texts),
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels_, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


print("Tokenizing datasets...")
train_dataset = SarcasmPairDataset(c_train, p_train, y_train)
val_dataset = SarcasmPairDataset(c_val, p_val, y_val)
test_dataset = SarcasmPairDataset(c_test, p_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

# =====================================================================
# 4. MODEL — DistilBERT + linear classifier on [CLS], nothing else
# =====================================================================
class PlainDistilBertClassifier(nn.Module):
    def __init__(self, bert_model_name, freeze_layers):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(bert_dim, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]  # [CLS] token representation
        x = self.dropout(cls_vec)
        logits = self.classifier(x).squeeze(1)
        return logits


model = PlainDistilBertClassifier(BERT_MODEL_NAME, FREEZE_BERT_LAYERS).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# =====================================================================
# 5. OPTIMIZER
# =====================================================================
bert_params = [p for n, p in model.named_parameters() if n.startswith("bert.") and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]

optimizer = torch.optim.AdamW(
    [{"params": bert_params, "lr": BERT_LR}, {"params": head_params, "lr": HEAD_LR}],
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))


def move_batch(batch):
    return {k: v.to(device) for k, v in batch.items()}


# =====================================================================
# 6. TRAINING LOOP
# =====================================================================
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        batch = move_batch(batch)
        y = batch["labels"]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * y.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            batch = move_batch(batch)
            y = batch["labels"]
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, y)
            val_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
    val_loss /= len(val_dataset)
    val_acc = correct / len(val_dataset)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# =====================================================================
# 7. TEST EVALUATION
# =====================================================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        batch_gpu = move_batch(batch)
        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].numpy().tolist())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nTest accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["not sarcasm", "sarcasm"]))
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

# =====================================================================
# 8. SAVE (optional — mainly useful for the comparison table in README)
# =====================================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "sarcasm_plain_distilbert_baseline.pt"))
print("\nSaved baseline model — this one is for the README comparison table, not for the API.")
