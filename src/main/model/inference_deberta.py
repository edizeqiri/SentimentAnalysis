import pandas as pd
import numpy as np
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification
from datasets import Dataset

# 1) load your saved model + tokenizer
model_path = "my_model_dir"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2) rebuild a Trainer for easy batched pred
trainer = Trainer(model=model, tokenizer=tokenizer)
# 1) Load test set (preserving original IDs)
test_df = pd.read_csv("../resources/data/test.csv")

# 2) Build a 🤗 Dataset
test_ds = Dataset.from_pandas(test_df[["sentence"]])

# 3) Tokenize
def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

test_ds = test_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.remove_columns(["sentence"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4) Run inference
pred_out = trainer.predict(test_ds)
# 5) Unpack & squeeze safely
raw_preds = pred_out.predictions
if isinstance(raw_preds, tuple):
    raw_preds = raw_preds[0]

raw_preds = np.squeeze(raw_preds, axis=-1) 

# 5) Map via thresholds
thr_low = 0.45
thr_high = 1.55
def apply_thresholds(x):
    if x <= thr_low:
        return 0
    elif x >= thr_high:
        return 2
    else:
        return 1

pred_ids = np.array([apply_thresholds(x) for x in raw_preds], dtype=int)

# 6) Map back to labels
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
labels = [ID2LABEL[i] for i in pred_ids]

# 7) Build & save submission
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": labels
})
submission_df.to_csv("../resources/data/submission_deberta_ordinal_thr.csv", index=False)
print(f" Wrote submission_deberta_ordinal_thr.csv ({len(submission_df)} rows)")