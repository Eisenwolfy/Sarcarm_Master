import os
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


# CONFIG
DATA_PATH = "train-balanced-sarcasm.csv"  # download from Kaggle, place next to main.py

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN_COMMENT = 40
MAX_LEN_PARENT = 40
FREEZE_BERT_LAYERS = 2
CNN_KERNEL_SIZES = [3, 4, 5]
CNN_NUM_FILTERS = 64
USE_METADATA = True
SUBREDDIT_VOCAB_SIZE = 500
SUBREDDIT_EMBED_DIM = 16
NUMERIC_FEATURE_DIM = 1
METADATA_HIDDEN_DIM = 32
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.pt")
PREPROCESSING_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
SUBREDDIT_PAD, SUBREDDIT_OOV = "<PAD>", "<OOV>"
BATCH_SIZE = 32
EPOCHS = 6
BERT_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1
PATIENCE = 2
USE_AMP = True
def artifacts_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSING_PATH)



# PREPROCESSING
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def normalize_score(raw_score, score_mean, score_std):
    transformed = signed_log(np.array(raw_score, dtype="float32"))
    return (transformed - score_mean) / score_std


def subreddit_to_id(subreddit_name, subreddit_vocab):
    return subreddit_vocab.get(subreddit_name, subreddit_vocab[SUBREDDIT_OOV])



# ARCHITECTURE
class CNNTextBranch(nn.Module):
    """Parallel 1D convs over BERT token embeddings, several kernel sizes, max-pooled."""

    def __init__(self, input_dim, num_filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

    def forward(self, token_embeddings, mask):
        masked = token_embeddings * mask.unsqueeze(-1).float()
        x = masked.transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)
        return torch.cat(pooled_outputs, dim=1)


class SarcasmModel(nn.Module):
    """DistilBERT (shared) -> CNN branch per input (comment/parent) -> metadata fusion -> classifier."""

    def __init__(self, bert_model_name, cnn_num_filters, cnn_kernel_sizes,
                 freeze_layers, subreddit_vocab_size, subreddit_embed_dim,
                 numeric_feature_dim, metadata_hidden_dim, use_metadata=True):
        super().__init__()
        self.use_metadata = use_metadata

        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.transformer.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.comment_cnn = CNNTextBranch(bert_dim, cnn_num_filters, cnn_kernel_sizes)
        self.parent_cnn = CNNTextBranch(bert_dim, cnn_num_filters, cnn_kernel_sizes)

        cnn_out_dim = cnn_num_filters * len(cnn_kernel_sizes)
        text_dim = cnn_out_dim * 2

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

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def encode_branch(self, input_ids, attention_mask, cnn):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return cnn(bert_out, attention_mask)

    def forward(self, comment_ids, comment_mask, parent_ids, parent_mask, subreddit_id=None, score_feat=None):
        comment_vec = self.encode_branch(comment_ids, comment_mask, self.comment_cnn)
        parent_vec = self.encode_branch(parent_ids, parent_mask, self.parent_cnn)
        combined = torch.cat([comment_vec, parent_vec], dim=1)

        if self.use_metadata:
            sub_emb = self.subreddit_embedding(subreddit_id)
            meta_input = torch.cat([sub_emb, score_feat], dim=1)
            meta_vec = self.metadata_mlp(meta_input)
            combined = torch.cat([combined, meta_vec], dim=1)

        combined = self.dropout(combined)
        x = self.relu(self.fc1(combined))
        logits = self.fc2(x).squeeze(1)
        return logits


def build_model(subreddit_vocab_size, device):
    return SarcasmModel(
        bert_model_name=BERT_MODEL_NAME,
        cnn_num_filters=CNN_NUM_FILTERS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        freeze_layers=FREEZE_BERT_LAYERS,
        subreddit_vocab_size=subreddit_vocab_size,
        subreddit_embed_dim=SUBREDDIT_EMBED_DIM,
        numeric_feature_dim=NUMERIC_FEATURE_DIM,
        metadata_hidden_dim=METADATA_HIDDEN_DIM,
        use_metadata=USE_METADATA,
    ).to(device)



# TRAINING (runs automatically the first time — see artifacts_exist())
def train_model(status_callback=None):
    def report(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    report(f"No trained model found — training from {DATA_PATH} (this takes a while)...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Download it from "
            f"https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit "
            f"and place it next to main.py."
        )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report(f"Device: {device}")
    if device.type == "cpu":
        report("WARNING: no GPU detected — this will be very slow (hours, not minutes).")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["comment", "parent_comment"])
    df["subreddit"] = df["subreddit"].fillna("<unknown>") if "subreddit" in df.columns else "<unknown>"
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0) if "score" in df.columns else 0.0

    comments = df["comment"].astype(str).values
    parents = df["parent_comment"].astype(str).values
    subreddits = df["subreddit"].astype(str).values
    scores = df["score"].astype(float).values
    labels = df["label"].astype(int).values

    report(f"Total examples: {len(comments)}")

    comments = [clean_text(t) for t in comments]
    parents = [clean_text(t) for t in parents]

    indices = np.arange(len(comments))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

    def subset(arr, idx):
        return [arr[i] for i in idx] if isinstance(arr, list) else arr[idx]

    c_train, c_val, c_test = subset(comments, train_idx), subset(comments, val_idx), subset(comments, test_idx)
    p_train, p_val, p_test = subset(parents, train_idx), subset(parents, val_idx), subset(parents, test_idx)
    sub_train, sub_val, sub_test = subset(subreddits, train_idx), subset(subreddits, val_idx), subset(subreddits, test_idx)
    score_train, score_val, score_test = subset(scores, train_idx), subset(scores, val_idx), subset(scores, test_idx)
    y_train, y_val, y_test = subset(labels, train_idx), subset(labels, val_idx), subset(labels, test_idx)

    report(f"Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")

    subreddit_counter = Counter(sub_train)
    most_common_subs = subreddit_counter.most_common(SUBREDDIT_VOCAB_SIZE - 2)
    subreddit_vocab = {SUBREDDIT_PAD: 0, SUBREDDIT_OOV: 1}
    for sub, _ in most_common_subs:
        subreddit_vocab[sub] = len(subreddit_vocab)

    score_train_arr = np.array(score_train, dtype=np.float32)
    score_transformed_train = signed_log(score_train_arr)
    score_mean = float(score_transformed_train.mean())
    score_std = float(score_transformed_train.std() + 1e-6)

    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    class TrainDataset(Dataset):
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
                [subreddit_to_id(s, subreddit_vocab) for s in subs], dtype=torch.long
            )
            self.score_feat = torch.tensor(
                normalize_score(raw_scores, score_mean, score_std), dtype=torch.float32
            ).unsqueeze(1)
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

    report("Tokenizing datasets...")
    train_dataset = TrainDataset(c_train, p_train, sub_train, score_train, y_train)
    val_dataset = TrainDataset(c_val, p_val, sub_val, score_val, y_val)
    test_dataset = TrainDataset(c_test, p_test, sub_test, score_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = build_model(subreddit_vocab_size=len(subreddit_vocab), device=device)

    bert_params = [p for n, p in net.named_parameters() if n.startswith("bert.") and p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("bert.") and p.requires_grad]

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

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        report(f"Epoch {epoch}/{EPOCHS} — training...")
        net.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            batch = move_batch(batch)
            y = batch["labels"]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits = net(
                    batch["comment_input_ids"], batch["comment_attention_mask"],
                    batch["parent_input_ids"], batch["parent_attention_mask"],
                    batch["subreddit_id"], batch["score_feat"],
                )
                loss = criterion(logits, y)

            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_dataset)

        net.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                batch = move_batch(batch)
                y = batch["labels"]
                with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                    logits = net(
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
        report(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                report("Early stopping triggered.")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_gpu = move_batch(batch)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits = net(
                    batch_gpu["comment_input_ids"], batch_gpu["comment_attention_mask"],
                    batch_gpu["parent_input_ids"], batch_gpu["parent_attention_mask"],
                    batch_gpu["subreddit_id"], batch_gpu["score_feat"],
                )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    report(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["not sarcasm", "sarcasm"]))
    print(confusion_matrix(all_labels, all_preds))

    torch.save(net.state_dict(), MODEL_PATH)
    with open(PREPROCESSING_PATH, "wb") as f:
        pickle.dump({"subreddit_vocab": subreddit_vocab, "score_mean": score_mean, "score_std": score_std}, f)

    report(f"Saved model to {MODEL_PATH}")



# INFERENCE
state = {
    "model": None,
    "tokenizer": None,
    "subreddit_vocab": None,
    "score_mean": None,
    "score_std": None,
    "device": None,
}


def load_artifacts():
    if state["model"] is not None:
        return

    if not artifacts_exist():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Call train_model() first "
            f"(main.py / the GUI does this automatically)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PREPROCESSING_PATH, "rb") as f:
        preprocessing = pickle.load(f)

    subreddit_vocab = preprocessing["subreddit_vocab"]

    model = build_model(subreddit_vocab_size=len(subreddit_vocab), device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["subreddit_vocab"] = subreddit_vocab
    state["score_mean"] = preprocessing["score_mean"]
    state["score_std"] = preprocessing["score_std"]
    state["device"] = device

    print(f"[final_model] Loaded on {device}.")


def ensure_ready(status_callback=None):
    if not artifacts_exist():
        train_model(status_callback=status_callback)
    load_artifacts()


def predict(comment: str, parent_comment: str = "", subreddit: str | None = None, score: float = 0.0):
    if state["model"] is None:
        load_artifacts()

    model = state["model"]
    tokenizer = state["tokenizer"]
    device = state["device"]

    c_clean = clean_text(comment)
    p_clean = clean_text(parent_comment) if parent_comment else ""

    c_enc = tokenizer(c_clean, truncation=True, padding="max_length",
                       max_length=MAX_LEN_COMMENT, return_tensors="pt")
    p_enc = tokenizer(p_clean, truncation=True, padding="max_length",
                       max_length=MAX_LEN_PARENT, return_tensors="pt")

    sub_id = subreddit_to_id(subreddit or "<unknown>", state["subreddit_vocab"])
    sub_id_tensor = torch.tensor([sub_id], dtype=torch.long).to(device)

    score_val = normalize_score(score, state["score_mean"], state["score_std"])
    score_tensor = torch.tensor([[score_val]], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(
            c_enc["input_ids"].to(device), c_enc["attention_mask"].to(device),
            p_enc["input_ids"].to(device), p_enc["attention_mask"].to(device),
            sub_id_tensor, score_tensor,
        )
        prob = torch.sigmoid(logits).item()

    label = "sarcasm" if prob > 0.5 else "not sarcasm"
    return label, prob


if __name__ == "__main__":
    ensure_ready()
