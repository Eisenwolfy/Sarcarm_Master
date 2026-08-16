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
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# =====================================================================
# 0. CONFIG
# =====================================================================
DATA_PATH = "train-balanced-sarcasm.csv"

# --- text encoding ---
BERT_MODEL_NAME = "distilbert-base-uncased"  # swap to "bert-base-uncased" for a further quality bump (slower)
MAX_LEN_COMMENT = 40
MAX_LEN_PARENT = 40
FREEZE_BERT_LAYERS = 2        # freeze embeddings + first 2 of 6 layers, fine-tune the last 4 (more than v3's 4/6 frozen)

# --- LSTM branch ---
LSTM_HIDDEN_DIM = 64

# --- CNN branch ---
CNN_KERNEL_SIZES = [3, 4, 5]   # captures 3-gram, 4-gram, 5-gram local patterns
CNN_NUM_FILTERS = 64            # filters per kernel size

# metadata
USE_METADATA = True
SUBREDDIT_VOCAB_SIZE = 500       # keep the top-N most frequent subreddits, rest -> <OOV>
SUBREDDIT_EMBED_DIM = 16
NUMERIC_FEATURE_DIM = 1          # just `score` for now; easy to extend with ups/downs/comment length etc.
METADATA_HIDDEN_DIM = 32

# training
BATCH_SIZE = 32
EPOCHS = 6
BERT_LR = 2e-5
HEAD_LR = 1e-3                  # LSTM / CNN / metadata / classifier layers
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1
PATIENCE = 2
USE_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cpu":
    print("WARNING: no GPU detected. Training will be very slow.")

# =====================================================================
# 1. LOAD DATA (text + metadata columns)
# =====================================================================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["comment", "parent_comment"])

# subreddit/score can have occasional missing values; fill with safe defaults
# instead of dropping rows, since these are auxiliary signals, not the main input.
df["subreddit"] = df["subreddit"].fillna("<unknown>")
df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

comments = df["comment"].astype(str).values
parents = df["parent_comment"].astype(str).values
subreddits = df["subreddit"].astype(str).values
scores = df["score"].astype(float).values
labels = df["label"].astype(int).values

print("Total examples:", len(comments))
print("Class balance:\n", df["label"].value_counts())
print("Unique subreddits:", df["subreddit"].nunique())


def clean_text(text):
    """
    Only strip URLs and collapse whitespace. DistilBERT's own tokenizer
    handles casing/punctuation, so we keep the text close to natural
    language rather than over-cleaning it.
    """
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


comments = [clean_text(t) for t in comments]
parents = [clean_text(t) for t in parents]

# =====================================================================
# 2. TRAIN / VAL / TEST SPLIT (keep all aligned arrays in sync)
# =====================================================================
indices = np.arange(len(comments))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx]
)


def subset(arr, idx):
    return [arr[i] for i in idx] if isinstance(arr, list) else arr[idx]


c_train, c_val, c_test = subset(comments, train_idx), subset(comments, val_idx), subset(comments, test_idx)
p_train, p_val, p_test = subset(parents, train_idx), subset(parents, val_idx), subset(parents, test_idx)
sub_train, sub_val, sub_test = subset(subreddits, train_idx), subset(subreddits, val_idx), subset(subreddits, test_idx)
score_train, score_val, score_test = subset(scores, train_idx), subset(scores, val_idx), subset(scores, test_idx)
y_train, y_val, y_test = subset(labels, train_idx), subset(labels, val_idx), subset(labels, test_idx)

print(f"Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")

# =====================================================================
# 3. METADATA PREPROCESSING (subreddit vocab + score normalization)
# =====================================================================
SUBREDDIT_PAD, SUBREDDIT_OOV = "<PAD>", "<OOV>"

subreddit_counter = Counter(sub_train)
most_common_subs = subreddit_counter.most_common(SUBREDDIT_VOCAB_SIZE - 2)
subreddit_vocab = {SUBREDDIT_PAD: 0, SUBREDDIT_OOV: 1}
for sub, _ in most_common_subs:
    subreddit_vocab[sub] = len(subreddit_vocab)

print("Subreddit vocab size:", len(subreddit_vocab))


def subreddit_to_id(sub):
    return subreddit_vocab.get(sub, subreddit_vocab[SUBREDDIT_OOV])


# Reddit scores can be negative and span a wide range, so we apply a
# signed log transform (compresses large magnitudes) and then standardize
# using TRAIN statistics only (never peek at val/test stats -> avoids leakage).
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


score_train_arr = np.array(score_train, dtype=np.float32)
score_transformed_train = signed_log(score_train_arr)
SCORE_MEAN = score_transformed_train.mean()
SCORE_STD = score_transformed_train.std() + 1e-6
print(f"Score normalization: mean={SCORE_MEAN:.4f}, std={SCORE_STD:.4f}")


def normalize_score(x):
    return (signed_log(np.array(x, dtype=np.float32)) - SCORE_MEAN) / SCORE_STD


# =====================================================================
# 4. TOKENIZER
# =====================================================================
print(f"Loading tokenizer for {BERT_MODEL_NAME}...")
tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
print("Tokenizer ready.")


class SarcasmDataset(Dataset):
    """
    Returns everything the model needs for one example:
    - tokenized comment (ids + attention mask)
    - tokenized parent_comment (ids + attention mask)
    - subreddit id (single integer, embedded inside the model)
    - normalized score (single float)
    - label
    """

    def __init__(self, comment_texts, parent_texts, subs, raw_scores, labels_):
        self.comment_enc = tokenizer(
            list(comment_texts), truncation=True, padding="max_length",
            max_length=MAX_LEN_COMMENT, return_tensors="pt",
        )
        self.parent_enc = tokenizer(
            list(parent_texts), truncation=True, padding="max_length",
            max_length=MAX_LEN_PARENT, return_tensors="pt",
        )
        self.subreddit_ids = torch.tensor(
            [subreddit_to_id(s) for s in subs], dtype=torch.long
        )
        self.score_feat = torch.tensor(
            normalize_score(raw_scores), dtype=torch.float32
        ).unsqueeze(1)  # shape (N, 1), ready to concat with other numeric feats later
        self.labels = torch.tensor(labels_, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "comment_input_ids": self.comment_enc["input_ids"][idx],
            "comment_attention_mask": self.comment_enc["attention_mask"][idx],
            "parent_input_ids": self.parent_enc["input_ids"][idx],
            "parent_attention_mask": self.parent_enc["attention_mask"][idx],
            "subreddit_id": self.subreddit_ids[idx],
            "score_feat": self.score_feat[idx],
            "labels": self.labels[idx],
        }


print("Tokenizing datasets...")
train_dataset = SarcasmDataset(c_train, p_train, sub_train, score_train, y_train)
val_dataset = SarcasmDataset(c_val, p_val, sub_val, score_val, y_val)
test_dataset = SarcasmDataset(c_test, p_test, sub_test, score_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =====================================================================
# 5. ATTENTION MODULE (for the LSTM branch)
# =====================================================================
class Attention(nn.Module):
    """Additive attention: learns to weight each token's LSTM output."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        energy = torch.tanh(self.attn(lstm_out))
        scores = self.context_vector(energy).squeeze(-1)
        # dtype-safe masking value, works under float16 autocast too
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, mask_value)
        weights = F.softmax(scores, dim=1)
        weighted = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return weighted, weights


# =====================================================================
# 6. CNN BRANCH
# =====================================================================
class CNNTextBranch(nn.Module):
    """
    Runs several 1D convolutions with different kernel sizes over the
    BERT token embeddings, each followed by ReLU + max-over-time pooling.
    This captures local n-gram-like patterns (fixed phrases), which is
    complementary to what the LSTM captures (long-range sequential flow).
    """

    def __init__(self, input_dim, num_filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

    def forward(self, token_embeddings, mask):
        # token_embeddings: (batch, seq_len, input_dim)
        # zero out padded positions before convolving so padding doesn't
        # contribute to the learned n-gram patterns.
        masked = token_embeddings * mask.unsqueeze(-1).float()
        x = masked.transpose(1, 2)  # -> (batch, input_dim, seq_len) for Conv1d

        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))                     # (batch, num_filters, seq_len - k + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            pooled_outputs.append(pooled)

        return torch.cat(pooled_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))


# =====================================================================
# 7. FULL MODEL: shared DistilBERT -> {CNN branch, BiLSTM+Attention branch}
#    per input (comment/parent) -> concat with metadata -> classifier
# =====================================================================
class SarcasmHybridModel(nn.Module):
    def __init__(
        self,
        bert_model_name,
        lstm_hidden_dim,
        cnn_num_filters,
        cnn_kernel_sizes,
        freeze_layers,
        subreddit_vocab_size,
        subreddit_embed_dim,
        numeric_feature_dim,
        metadata_hidden_dim,
        use_metadata=True,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        # --- shared BERT encoder (used for both comment and parent) ---
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # --- per-branch text encoders (separate weights for comment vs parent) ---
        self.comment_lstm = nn.LSTM(bert_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.comment_attention = Attention(lstm_hidden_dim * 2)
        self.comment_cnn = CNNTextBranch(bert_dim, cnn_num_filters, cnn_kernel_sizes)

        self.parent_lstm = nn.LSTM(bert_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.parent_attention = Attention(lstm_hidden_dim * 2)
        self.parent_cnn = CNNTextBranch(bert_dim, cnn_num_filters, cnn_kernel_sizes)

        lstm_out_dim = lstm_hidden_dim * 2
        cnn_out_dim = cnn_num_filters * len(cnn_kernel_sizes)
        branch_dim = lstm_out_dim + cnn_out_dim   # per input (comment or parent)
        text_dim = branch_dim * 2                  # comment branch + parent branch

        # --- metadata branch ---
        if use_metadata:
            self.subreddit_embedding = nn.Embedding(subreddit_vocab_size, subreddit_embed_dim, padding_idx=0)
            metadata_input_dim = subreddit_embed_dim + numeric_feature_dim
            self.metadata_mlp = nn.Sequential(
                nn.Linear(metadata_input_dim, metadata_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            fusion_dim = text_dim + metadata_hidden_dim
        else:
            fusion_dim = text_dim

        # --- classifier head ---
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def encode_branch(self, input_ids, attention_mask, lstm, attention, cnn):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        lstm_out, _ = lstm(bert_out)
        lstm_vec, attn_weights = attention(lstm_out, attention_mask)

        cnn_vec = cnn(bert_out, attention_mask)

        branch_vec = torch.cat([lstm_vec, cnn_vec], dim=1)
        return branch_vec, attn_weights

    def forward(self, comment_ids, comment_mask, parent_ids, parent_mask,
                subreddit_id=None, score_feat=None):
        comment_vec, comment_attn = self.encode_branch(
            comment_ids, comment_mask, self.comment_lstm, self.comment_attention, self.comment_cnn
        )
        parent_vec, parent_attn = self.encode_branch(
            parent_ids, parent_mask, self.parent_lstm, self.parent_attention, self.parent_cnn
        )

        combined = torch.cat([comment_vec, parent_vec], dim=1)

        if self.use_metadata:
            sub_emb = self.subreddit_embedding(subreddit_id)         # (batch, subreddit_embed_dim)
            meta_input = torch.cat([sub_emb, score_feat], dim=1)      # (batch, subreddit_embed_dim + numeric_dim)
            meta_vec = self.metadata_mlp(meta_input)                  # (batch, metadata_hidden_dim)
            combined = torch.cat([combined, meta_vec], dim=1)

        combined = self.dropout(combined)
        x = self.relu(self.fc1(combined))
        logits = self.fc2(x).squeeze(1)
        return logits, comment_attn, parent_attn


model = SarcasmHybridModel(
    bert_model_name=BERT_MODEL_NAME,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    cnn_num_filters=CNN_NUM_FILTERS,
    cnn_kernel_sizes=CNN_KERNEL_SIZES,
    freeze_layers=FREEZE_BERT_LAYERS,
    subreddit_vocab_size=len(subreddit_vocab),
    subreddit_embed_dim=SUBREDDIT_EMBED_DIM,
    numeric_feature_dim=NUMERIC_FEATURE_DIM,
    metadata_hidden_dim=METADATA_HIDDEN_DIM,
    use_metadata=USE_METADATA,
).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# =====================================================================
# 8. OPTIMIZER (separate LR for pretrained BERT vs everything else)
# =====================================================================
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
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))


def move_batch_to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


# =====================================================================
# 9. TRAINING LOOP
# =====================================================================
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
    for batch in progress:
        batch = move_batch_to_device(batch)
        y = batch["labels"]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits, _, _ = model(
                batch["comment_input_ids"], batch["comment_attention_mask"],
                batch["parent_input_ids"], batch["parent_attention_mask"],
                batch["subreddit_id"], batch["score_feat"],
            )
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
            batch = move_batch_to_device(batch)
            y = batch["labels"]
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits, _, _ = model(
                    batch["comment_input_ids"], batch["comment_attention_mask"],
                    batch["parent_input_ids"], batch["parent_attention_mask"],
                    batch["subreddit_id"], batch["score_feat"],
                )
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
# 10. TEST EVALUATION
# =====================================================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        batch_gpu = move_batch_to_device(batch)
        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
            logits, _, _ = model(
                batch_gpu["comment_input_ids"], batch_gpu["comment_attention_mask"],
                batch_gpu["parent_input_ids"], batch_gpu["parent_attention_mask"],
                batch_gpu["subreddit_id"], batch_gpu["score_feat"],
            )
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

# ---------------------------------------------------------------------
# 11. SAVE MODEL + PREPROCESSING ARTIFACTS
#     (the API service will need all of these to reproduce inference)
# ---------------------------------------------------------------------
torch.save(model.state_dict(), "sarcasm_hybrid_model.pt")

with open("preprocessing.pkl", "wb") as f:
    pickle.dump({
        "subreddit_vocab": subreddit_vocab,
        "score_mean": float(SCORE_MEAN),
        "score_std": float(SCORE_STD),
        "max_len_comment": MAX_LEN_COMMENT,
        "max_len_parent": MAX_LEN_PARENT,
        "bert_model_name": BERT_MODEL_NAME,
    }, f)

print("\nSaved sarcasm_hybrid_model.pt and preprocessing.pkl")
print("These two files are what the API service will load for inference.")
