import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "train-balanced-sarcasm.csv"

VOCAB_SIZE = 20000
MAX_LEN = 40
EMBED_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
PATIENCE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------
# DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["comment"])

texts = df["comment"].astype(str).values
labels = df["label"].astype(int).values

print("Samples in total:", len(texts))
print("Labels balance:\n", df["label"].value_counts())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


texts = [clean_text(t) for t in texts]

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -----------------------------
# VOCAB
# -----------------------------
PAD_TOKEN, OOV_TOKEN = "<PAD>", "<OOV>"

counter = Counter()
for t in X_train:
    counter.update(t.split())

most_common = counter.most_common(VOCAB_SIZE - 2)
vocab = {PAD_TOKEN: 0, OOV_TOKEN: 1}
for word, _ in most_common:
    vocab[word] = len(vocab)

print("Vocabulary:", len(vocab))


def text_to_ids(text):
    ids = [vocab.get(w, vocab[OOV_TOKEN]) for w in text.split()]
    ids = ids[:MAX_LEN]
    ids = ids + [vocab[PAD_TOKEN]] * (MAX_LEN - len(ids))
    return ids


# -----------------------------
# DATASET
# -----------------------------
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = text_to_ids(self.texts[idx])
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


train_ds = SarcasmDataset(X_train, y_train)
val_ds = SarcasmDataset(X_val, y_val)
test_ds = SarcasmDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL
# -----------------------------
class SarcasmLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm1 = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm1(emb)
        out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_cat = self.dropout2(h_cat)
        x = self.relu(self.fc1(h_cat))
        x = self.fc2(x)
        return x.squeeze(1)


model = SarcasmLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING
# -----------------------------
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# -----------------------------
# VALIDATION
# -----------------------------
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nValidation: {test_acc:.4f}")

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=["normal", "sarcasm"]))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

# -----------------------------
# SAVING MODEL AND VOCAB
# -----------------------------
torch.save(model.state_dict(), "sarcasm_lstm_model.pt")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\nModel and vocabulary are saved (sarcasm_lstm_model.pt, vocab.pkl).")

# -----------------------------
# INFERENCE
# -----------------------------
def predict_sarcasm(text, threshold=0.5):
    model.eval()
    cleaned = clean_text(text)
    ids = text_to_ids(cleaned)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    label = "sarcasm" if prob > threshold else "normal"
    return label, prob


examples = [
    "Oh great, another Monday. Just what I needed.",
    "I love waiting two hours in line, it's my favorite hobby.",
    "The weather is really nice today.",
]

for ex in examples:
    label, prob = predict_sarcasm(ex)
    print(f"{ex!r} -> {label} (p={prob:.3f})")
