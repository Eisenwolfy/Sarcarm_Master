# Sarcasm Master
An LSTM-based deep learning model for sarcasm detection in text.


---

## Motivation



---

## Models

1) basic.py

2) parent_commnet_handling.py

  No external pretrained models are used (no GloVe, no BERT, etc.).
  Everything, including word embeddings, is learned from scratch using
  only this dataset.

  Improvements over the very first basic version:

  1. DUAL INPUT (comment + parent_comment)
     Sarcasm on Reddit is often only obvious when you know what the comment
     is replying to. We encode both the comment and its parent with separate
     LSTM branches and combine them before classification.

  2. ATTENTION LAYER
     Instead of using only the final LSTM hidden state, attention lets the
     model learn to focus on the most informative words in the sequence
     (e.g. "love", "favorite", "great" combined with a negative situation),
     which is exactly the kind of contrast that signals sarcasm.

  3. Better regularization/training setup (dropout, weight decay,
     gradient clipping, LR scheduler, early stopping) to reduce overfitting
     and make training more stable, all without any external data or models.
  
3) model_with_distilbert.py
  Instead of training word embeddings from scratch (previous version), we use
  DistilBERT as a contextual embedding extractor: for every token, DistilBERT
  outputs a 768-dim vector that already "understands" grammar, word sense and
  some world knowledge from pretraining. We then feed those per-token vectors
  into a BiLSTM + attention (same idea as before) to build a sentence vector,
  separately for comment and parent_comment, and combine them for the final
  classifier.

  This is a genuine hybrid: DistilBERT replaces only the embedding layer,
  LSTM+attention still does the sequence modeling and the two-input fusion.

  IMPORTANT — resource requirements
  ----------------------------------
  Fine-tuning DistilBERT on ~1M examples is much heavier than the pure-LSTM
  version:
  - Needs a GPU (CPU will be extremely slow).
  - Use a smaller batch size (16-32) since DistilBERT eats a lot of GPU memory.
  - Consider SUBSAMPLE_FRAC below to work with less data while iterating,
    then scale up once the pipeline works.
  - Mixed precision (torch.cuda.amp) is used to reduce memory usage and
    speed up training on GPU.
