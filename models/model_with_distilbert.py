import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from tqdm import tqdm


# CONFIG
DATA_PATH = "train-balanced-sarcasm.csv"

SUBSAMPLE_FRAC = 1.0
MAX_LEN_COMMENT = 40
MAX_LEN_PARENT = 40
LSTM_HIDDEN_DIM = 64
BATCH_SIZE = 32  # smaller than the pure-LSTM version, DistilBERT needs more memory
EPOCHS = 4 # BERT-based models converge in far fewer epochs than from-scratch LSTM
BERT_LR = 2e-5 # small LR for pretrained BERT weights (standard fine-tuning value)
HEAD_LR = 1e-3 # larger LR for the new LSTM/attention/classifier layers
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0 # BERT fine-tuning typically uses a tighter clip than plain LSTM
FREEZE_BERT_LAYERS = 4 # freeze embeddings + first N of 6 DistilBERT transformer layers
WARMUP_RATIO = 0.1  # fraction of total steps used for LR warmup
PATIENCE = 2

USE_AMP = True # mixed precision training (faster + less memory on GPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cpu":
    print("WARNING: no GPU detected. Fine-tuning DistilBERT on CPU will be very slow.")


# LOAD AND LIGHTLY CLEAN DATA
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["comment", "parent_comment"])

if SUBSAMPLE_FRAC < 1.0:
    df = df.sample(frac=SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(df)} rows ({SUBSAMPLE_FRAC*100:.0f}% of the cleaned data).")

comments = df["comment"].astype(str).values
parents = df["parent_comment"].astype(str).values
labels = df["label"].astype(int).values

print("Total examples:", len(comments))
print("Class balance:\n", df["label"].value_counts())


def clean_text(text):
    """
    Only remove URLs and collapse whitespace. Unlike the from-scratch LSTM version, we do NOT strip punctuation or lowercase aggressively here:
    DistilBERT's own tokenizer (WordPiece) is trained on natural text and already handles casing/punctuation well, so over-cleaning would throw
    away signal the pretrained model knows how to use.
    """
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


comments = [clean_text(t) for t in comments]
parents = [clean_text(t) for t in parents]

(
    c_train, c_temp,
    p_train, p_temp,
    y_train, y_temp,
) = train_test_split(
    comments, parents, labels, test_size=0.2, random_state=42, stratify=labels
)
(
    c_val, c_test,
    p_val, p_test,
    y_val, y_test,
) = train_test_split(
    c_temp, p_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")



# DistilBERT
print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
print("Tokenizer ready.")


class SarcasmBertDataset(Dataset):
    """
    Tokenizes comment and parent_comment SEPARATELY (two independent branches, like the two LSTM branches in the previous version).
    Each example returns 4 tensors (comment ids/mask, parent ids/mask) plus the label.
    """

    def __init__(self, comment_texts, parent_texts, labels):
        self.comment_enc = tokenizer(
            list(comment_texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN_COMMENT,
            return_tensors="pt",
        )
        self.parent_enc = tokenizer(
            list(parent_texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN_PARENT,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "comment_input_ids": self.comment_enc["input_ids"][idx],
            "comment_attention_mask": self.comment_enc["attention_mask"][idx],
            "parent_input_ids": self.parent_enc["input_ids"][idx],
            "parent_attention_mask": self.parent_enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }


print("Tokenizing train/val/test sets (this can take a while for large datasets)...")
train_dataset = SarcasmBertDataset(c_train, p_train, y_train)
val_dataset = SarcasmBertDataset(c_val, p_val, y_val)
test_dataset = SarcasmBertDataset(c_test, p_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



# ATTENTION MODULE
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        energy = torch.tanh(self.attn(lstm_out))
        scores = self.context_vector(energy).squeeze(-1)
        '''Use the minimum representable value for the CURRENT dtype instead of a hardcoded -1e9. Under autocast (mixed precision), scores are
        float16, and float16's range tops out around +/-65504 -- -1e9 overflows it and crashes. torch.finfo(dtype).min is always safe.'''
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, mask_value)
        weights = F.softmax(scores, dim=1)
        weighted = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return weighted, weights



# HYBRID MODEL: DistilBERT (embeddings) -> BiLSTM -> Attention -> Fusion -> Classifier
class SarcasmBertLSTM(nn.Module):
    def __init__(self, lstm_hidden_dim, freeze_layers=4):
        super().__init__()

        # Single shared DistilBERT encoder used for BOTH comment and parent.
        # Sharing keeps parameter count reasonable and both branches use the same "understanding" of English.
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_dim = self.bert.config.hidden_size
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.comment_lstm = nn.LSTM(
            bert_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        )
        self.comment_attention = Attention(lstm_hidden_dim * 2)

        self.parent_lstm = nn.LSTM(
            bert_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        )
        self.parent_attention = Attention(lstm_hidden_dim * 2)

        combined_dim = lstm_hidden_dim * 2 * 2
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(combined_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def encode_branch(self, input_ids, attention_mask, lstm, attention):
        # DistilBERT gives one 768-dim contextual vector per token.
        bert_out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        lstm_out, _ = lstm(bert_out)  # (batch, seq_len, lstm_hidden_dim*2)
        attended, attn_weights = attention(lstm_out, attention_mask)
        return attended, attn_weights

    def forward(self, comment_ids, comment_mask, parent_ids, parent_mask):
        comment_vec, comment_attn = self.encode_branch(
            comment_ids, comment_mask, self.comment_lstm, self.comment_attention
        )
        parent_vec, parent_attn = self.encode_branch(
            parent_ids, parent_mask, self.parent_lstm, self.parent_attention
        )

        combined = torch.cat([comment_vec, parent_vec], dim=1)
        combined = self.dropout(combined)
        x = self.relu(self.fc1(combined))
        logits = self.fc2(x).squeeze(1)
        return logits, comment_attn, parent_attn


model = SarcasmBertLSTM(
    lstm_hidden_dim=LSTM_HIDDEN_DIM, freeze_layers=FREEZE_BERT_LAYERS
).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,}")


# OPTIMIZER WITH DIFFERENT LEARNING RATES FOR BERT VS NEW LAYERS

'''BERT weights are pretrained and only need small updates (2e-5 is the standard fine-tuning LR). The new LSTM/attention/classifier layers are
randomly initialized and need a much bigger LR (1e-3) to learn quickly.'''

bert_params = [p for n, p in model.named_parameters() if n.startswith("bert.") and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": bert_params, "lr": BERT_LR},
        {"params": head_params, "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))


# TRAINING
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
    for batch in progress:
        comment_ids = batch["comment_input_ids"].to(device)
        comment_mask = batch["comment_attention_mask"].to(device)
        parent_ids = batch["parent_input_ids"].to(device)
        parent_mask = batch["parent_attention_mask"].to(device)
        y = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits, _, _ = model(comment_ids, comment_mask, parent_ids, parent_mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * y.size(0)
        progress.set_postfix(loss=loss.item())

    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            comment_ids = batch["comment_input_ids"].to(device)
            comment_mask = batch["comment_attention_mask"].to(device)
            parent_ids = batch["parent_input_ids"].to(device)
            parent_mask = batch["parent_attention_mask"].to(device)
            y = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits, _, _ = model(comment_ids, comment_mask, parent_ids, parent_mask)
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


# EVALUATION
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        comment_ids = batch["comment_input_ids"].to(device)
        comment_mask = batch["comment_attention_mask"].to(device)
        parent_ids = batch["parent_input_ids"].to(device)
        parent_mask = batch["parent_attention_mask"].to(device)
        y = batch["labels"]

        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits, _, _ = model(comment_ids, comment_mask, parent_ids, parent_mask)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nTest accuracy: {test_acc:.4f}")

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["not sarcasm", "sarcasm"]))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))


# SAVING MODEL
torch.save(model.state_dict(), "sarcasm_bert_lstm_model.pt")
print("\nModel saved to sarcasm_bert_lstm_model.pt")


# INFERENCE HELPER
def predict_sarcasm(comment_text, parent_text="", threshold=0.5):
    model.eval()
    c_clean = clean_text(comment_text)
    p_clean = clean_text(parent_text) if parent_text else ""

    c_enc = tokenizer(
        c_clean, truncation=True, padding="max_length",
        max_length=MAX_LEN_COMMENT, return_tensors="pt",
    )
    p_enc = tokenizer(
        p_clean, truncation=True, padding="max_length",
        max_length=MAX_LEN_PARENT, return_tensors="pt",
    )

    comment_ids = c_enc["input_ids"].to(device)
    comment_mask = c_enc["attention_mask"].to(device)
    parent_ids = p_enc["input_ids"].to(device)
    parent_mask = p_enc["attention_mask"].to(device)

    with torch.no_grad():
        logits, comment_attn, parent_attn = model(
            comment_ids, comment_mask, parent_ids, parent_mask
        )
        prob = torch.sigmoid(logits).item()

    label = "sarcasm" if prob > threshold else "not sarcasm"
    return label, prob


examples = [
    ("Oh great, another Monday. Just what I needed.", ""),
    ("I love waiting two hours in line, it's my favorite hobby.", ""),
    ("The weather is really nice today.", ""),
    ("Sure, because that always works out so well.", "I think I'll try that plan again."),
]

for comment_text, parent_text in examples:
    label, prob = predict_sarcasm(comment_text, parent_text)
    print(f"{comment_text!r} -> {label} (p={prob:.3f})")
