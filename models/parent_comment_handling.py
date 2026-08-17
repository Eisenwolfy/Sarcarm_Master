import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

 
# CONFIG
DATA_PATH = "train-balanced-sarcasm.csv"

VOCAB_SIZE = 20000
MAX_LEN_COMMENT = 40
MAX_LEN_PARENT = 40
EMBED_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 5.0
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# DATA
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["comment", "parent_comment"])

comments = df["comment"].astype(str).values
parents = df["parent_comment"].astype(str).values
labels = df["label"].astype(int).values

print("Total examples:", len(comments))
print("Class balance:\n", df["label"].value_counts())


def clean_text(text):
    """
    Lowercase and strip URLs/special characters, but KEEP punctuation
    like '!' and '...' since they are often sarcasm markers
    (e.g. 'oh great...', 'wow!!!').
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s'!?.,]", " ", text)
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


# VOCABULARY (shared between comment and parent_comment)
PAD_TOKEN, OOV_TOKEN = "<PAD>", "<OOV>"

counter = Counter()
for t in c_train:
    counter.update(t.split())
for t in p_train:
    counter.update(t.split())

most_common = counter.most_common(VOCAB_SIZE - 2)
vocab = {PAD_TOKEN: 0, OOV_TOKEN: 1}
for word, _ in most_common:
    vocab[word] = len(vocab)

print("Vocabulary size:", len(vocab))


def text_to_ids(text, max_len):
    ids = [vocab.get(w, vocab[OOV_TOKEN]) for w in text.split()]
    ids = ids[:max_len]
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


# DATASET / DATALOADER (returns comment, parent, label)
class SarcasmDataset(Dataset):
    def __init__(self, comments, parents, labels):
        self.comments = comments
        self.parents = parents
        self.labels = labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        c_ids = text_to_ids(self.comments[idx], MAX_LEN_COMMENT)
        p_ids = text_to_ids(self.parents[idx], MAX_LEN_PARENT)
        c = torch.tensor(c_ids, dtype=torch.long)
        p = torch.tensor(p_ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return c, p, y


train_ds = SarcasmDataset(c_train, p_train, y_train)
val_ds = SarcasmDataset(c_val, p_val, y_val)
test_ds = SarcasmDataset(c_test, p_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# ATTENTION MODULE
class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention over LSTM outputs.

    Instead of using only the LAST hidden state of the LSTM as the
    sentence representation, attention computes a weighted sum over ALL
    time steps. The weights are learned end-to-end together with the
    rest of the model, so no external component is needed.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        # lstm_out: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len)
        energy = torch.tanh(self.attn(lstm_out))
        scores = self.context_vector(energy).squeeze(-1)

        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)

        # weighted sum of LSTM outputs - single vector per sequence
        weighted = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return weighted, weights



# MODEL: dual-branch LSTM + attention, embeddings trained from scratch
class SarcasmLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        # Embedding layer trained from scratch, no pretrained vectors.
        # Shared between comment and parent_comment branches since they use the same vocabulary and the same word meanings.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.comment_lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.comment_attention = Attention(hidden_dim * 2)

        self.parent_lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.parent_attention = Attention(hidden_dim * 2)

        # input = attended comment vector + attended parent vector
        combined_dim = hidden_dim * 2 * 2
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(combined_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def encode_branch(self, token_ids, lstm, attention):
        mask = (token_ids != 0).long()
        emb = self.embedding(token_ids)
        lstm_out, _ = lstm(emb)
        attended, attn_weights = attention(lstm_out, mask)
        return attended, attn_weights

    def forward(self, comment_ids, parent_ids):
        comment_vec, comment_attn = self.encode_branch(
            comment_ids, self.comment_lstm, self.comment_attention
        )
        parent_vec, parent_attn = self.encode_branch(
            parent_ids, self.parent_lstm, self.parent_attention
        )

        combined = torch.cat([comment_vec, parent_vec], dim=1)
        combined = self.dropout(combined)
        x = self.relu(self.fc1(combined))
        logits = self.fc2(x).squeeze(1)
        return logits, comment_attn, parent_attn


model = SarcasmLSTMWithAttention(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
).to(device)

print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,   # L2 regularization, helps reduce overfitting
)

# Reduce LR when validation loss plateaus, instead of using a fixed LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.3, patience=1
)


# TRAINING
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for c, p, y in train_loader:
        c, p, y = c.to(device), p.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(c, p)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        train_loss += loss.item() * c.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for c, p, y in val_loader:
            c, p, y = c.to(device), p.to(device), y.to(device)
            logits, _, _ = model(c, p)
            loss = criterion(logits, y)
            val_loss += loss.item() * c.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}"
    )

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
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for c, p, y in test_loader:
        c, p = c.to(device), p.to(device)
        logits, _, _ = model(c, p)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nTest accuracy: {test_acc:.4f}")

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["not sarcasm", "sarcasm"]))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))


# SAVE MODEL AND VOCAB
torch.save(model.state_dict(), "sarcasm_lstm_attention_model.pt")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\nModel and vocabulary saved (sarcasm_lstm_attention_model.pt, vocab.pkl).")


# INFERENCE HELPER
def predict_sarcasm(comment_text, parent_text="", threshold=0.5):
    model.eval()
    c_clean = clean_text(comment_text)
    p_clean = clean_text(parent_text) if parent_text else ""

    c_ids = text_to_ids(c_clean, MAX_LEN_COMMENT)
    p_ids = text_to_ids(p_clean, MAX_LEN_PARENT)

    c_tensor = torch.tensor([c_ids], dtype=torch.long).to(device)
    p_tensor = torch.tensor([p_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, comment_attn, parent_attn = model(c_tensor, p_tensor)
        prob = torch.sigmoid(logits).item()

    label = "sarcasm" if prob > threshold else "not sarcasm"
    return label, prob, comment_attn.cpu().numpy()[0]


examples = [
    ("Oh great, another Monday. Just what I needed.", ""),
    ("I love waiting two hours in line, it's my favorite hobby.", ""),
    ("The weather is really nice today.", ""),
    ("Sure, because that always works out so well.", "I think I'll try that plan again."),
]

for comment_text, parent_text in examples:
    label, prob, attn_weights = predict_sarcasm(comment_text, parent_text)
    print(f"{comment_text!r} -> {label} (p={prob:.3f})")
