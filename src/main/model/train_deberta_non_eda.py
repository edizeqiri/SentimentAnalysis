import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoConfig,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import random
import os
from pathlib import Path
import warnings
from torch import nn
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
print("device:", device)

os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_path = Path("../resources/data/training_split.csv")
val_path   = Path("../resources/data/validation_split.csv")

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

train_df["label"] = train_df["label"].map(LABEL2ID).astype("float32")
val_df["label"]   = val_df["label"].map(LABEL2ID).astype("float32")

assert train_df["label"].isna().sum() == 0
assert val_df["label"].isna().sum()   == 0

model_name = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
train_ds = Dataset.from_pandas(train_df[["sentence", "label"]])
val_ds   = Dataset.from_pandas(val_df[["sentence", "label"]])

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128 
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["sentence"])
val_ds = val_ds.remove_columns(["sentence"])

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
)

def compute_metrics(eval_pred: EvalPrediction):
    
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.squeeze(preds)
    mae = mean_absolute_error(labels, preds)
    ints = np.clip(np.round(preds), 0, 2).astype(int)
    labels_int = labels.astype(int)
    acc = accuracy_score(labels_int, ints)
    f1  = f1_score(labels_int, ints, average="macro")
    return {"eval_mae": mae, "accuracy": acc, "f1": f1}



#hyper parameters:
lr = 2.2e-5
batch_size = 8
num_epochs = 4
warmup_ratio = 0.2
weight_decay = 0.04

collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

steps_per_epoch = len(train_ds) // batch_size
half_epoch = max(1, steps_per_epoch // 2)

arguments = dict(
    output_dir="./deberta_non_eda",
    eval_strategy="steps",
    eval_steps=half_epoch,
    save_strategy="steps",
    save_steps=half_epoch,
    logging_strategy="steps",
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_mae",
    greater_is_better=False,
    do_eval=True,
    logging_dir="./deberta_non_eda/logs",
    report_to="none",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    gradient_accumulation_steps=1,
    seed=seed,
    fp16=True,
    save_total_limit=2,
)

training_args = TrainingArguments(**arguments)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print('training...')
trainer.train()

trainer.save_model("./deberta_non_eda/best")

print('evaluating...')
trainer.evaluate()
