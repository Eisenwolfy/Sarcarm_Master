import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

import model

DATA_PATH = "train-balanced-sarcasm.csv"

BATCH_SIZE = 32
EPOCHS = 6
BERT_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1
PATIENCE = 2
USE_AMP = True


# =====================================================================
# TRAINING (only runs if artifacts/ doesn't already have a trained model)
# =====================================================================
def train_model():
    print("No trained model found — training a new one from", DATA_PATH)
    print("(This takes a while. A GPU is strongly recommended — see the")
    print(" Kaggle training script if you'd rather not use your own machine.)\n")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Download it from "
            f"https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit "
            f"and place it next to main.py."
        )

    os.makedirs(model.ARTIFACTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: no GPU detected. This will be very slow.")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["comment", "parent_comment"])
    df["subreddit"] = df["subreddit"].fillna("<unknown>") if "subreddit" in df.columns else "<unknown>"
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0) if "score" in df.columns else 0.0

    comments = df["comment"].astype(str).values
    parents = df["parent_comment"].astype(str).values
    subreddits = df["subreddit"].astype(str).values
    scores = df["score"].astype(float).values
    labels = df["label"].astype(int).values

    print("Total examples:", len(comments))

    comments = [model.clean_text(t) for t in comments]
    parents = [model.clean_text(t) for t in parents]

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

    print(f"Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")

    subreddit_counter = Counter(sub_train)
    most_common_subs = subreddit_counter.most_common(model.SUBREDDIT_VOCAB_SIZE - 2)
    subreddit_vocab = {model.SUBREDDIT_PAD: 0, model.SUBREDDIT_OOV: 1}
    for sub, _ in most_common_subs:
        subreddit_vocab[sub] = len(subreddit_vocab)

    score_train_arr = np.array(score_train, dtype=np.float32)
    score_transformed_train = model.signed_log(score_train_arr)
    score_mean = float(score_transformed_train.mean())
    score_std = float(score_transformed_train.std() + 1e-6)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model.BERT_MODEL_NAME)

    class TrainDataset(Dataset):
        def __init__(self, comment_texts, parent_texts, subs, raw_scores, labels_):
            self.comment_enc = tokenizer(
                list(comment_texts), truncation=True, padding="max_length",
                max_length=model.MAX_LEN_COMMENT, return_tensors="pt",
            )
            self.parent_enc = tokenizer(
                list(parent_texts), truncation=True, padding="max_length",
                max_length=model.MAX_LEN_PARENT, return_tensors="pt",
            )
            self.subreddit_ids = torch.tensor(
                [model.subreddit_to_id(s, subreddit_vocab) for s in subs], dtype=torch.long
            )
            self.score_feat = torch.tensor(
                model.normalize_score(raw_scores, score_mean, score_std), dtype=torch.float32
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

    print("Tokenizing datasets...")
    train_dataset = TrainDataset(c_train, p_train, sub_train, score_train, y_train)
    val_dataset = TrainDataset(c_val, p_val, sub_val, score_val, y_val)
    test_dataset = TrainDataset(c_test, p_test, sub_test, score_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = model.build_model(subreddit_vocab_size=len(subreddit_vocab), device=device)

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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
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
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
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
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["not sarcasm", "sarcasm"]))
    print(confusion_matrix(all_labels, all_preds))

    torch.save(net.state_dict(), model.MODEL_PATH)
    with open(model.PREPROCESSING_PATH, "wb") as f:
        pickle.dump({"subreddit_vocab": subreddit_vocab, "score_mean": score_mean, "score_std": score_std}, f)

    print(f"\nSaved model to {model.MODEL_PATH}")


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    if not model.artifacts_exist():
        train_model()
    else:
        print("Found existing trained model in artifacts/ — skipping training.")

    print("\nLaunching desktop app...")
    import interface
    interface.main()


if __name__ == "__main__":
    main()
