from pathlib import Path
import numpy as np
import pandas as pd
import torch, random, os
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer,
    EarlyStoppingCallback
)

# ------------------------------------------------------------------
# 1.  Minimal, model-compatible text cleaning
# ------------------------------------------------------------------
import re


def preprocess_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"\\'", "'", text).replace("`", "'")  # fix escaped/back-tick quotes
    text = text.lower()
    text = re.sub(r'@\w+', '@user', text)  # user placeholder
    text = re.sub(r'http\S+|www\S+', 'http', text)  # URL placeholder
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ------------------------------------------------------------------
# 2.  Main training function
# ------------------------------------------------------------------
def train_roberta_sentiment(
        df: pd.DataFrame,
        text_col: str = "sentence",
        label_col: str = "label",
        output_dir: str = "roberta_sentiment_finetuned",
        num_epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 32,
        seed: int = 42
):
    """
    Fine-tune twitter-roberta-base-sentiment-latest on a labelled DataFrame.
    Returns (trainer, metrics_dict).
    """
    # ---------- 2.1  Reproducibility ----------
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---------- 2.2  Prepare dataset ----------
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].map(preprocess_tweet)

    # encode string labels → ids
    label2id = {lbl: i for i, lbl in enumerate(sorted(df[label_col].unique()))}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label_id"] = df[label_col].map(label2id)

    # keep only the numeric label and the text, then rename ► unique column names
    dataset_df = df[[text_col, "label_id"]].rename(
        columns={text_col: "text", "label_id": "label"}
    )

    # Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(dataset_df)
    hf_dataset = hf_dataset.class_encode_column("label")

    # train / validation split (90-10 stratified)
    hf_dataset = hf_dataset.train_test_split(
        test_size=0.1, stratify_by_column="label", seed=seed
    )
    train_ds, val_ds = hf_dataset["train"], hf_dataset["test"]

    # ---------- 2.3  Tokeniser & model ----------
    model_ckpt = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(label2id), hidden_dropout_prob=0.2,  # Add dropout to hidden layers
        attention_probs_dropout_prob=0.2,  # Add dropout to attention
        id2label=id2label, label2id=label2id
    )

    if os.name == "nt":  # only needed on Windows
        model.floating_point_ops = lambda *a, **k: 0

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,  # plenty for tweets
        )

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        batch_size=1000,  # bigger batch → fewer Python calls
        num_proc=8,  # avoid Windows/mp hang; raise on Linux if you like
        remove_columns=["text"],
        desc="Tokenising train",
        load_from_cache_file=False,
    )

    val_ds = val_ds.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=["text"],
        desc="Tokenising val",
        load_from_cache_file=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # ---------- 2.4  Metrics ----------

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    # ---------- 2.5  TrainingArguments ----------
    args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir=output_dir,
        do_eval=True,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.03,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        report_to="none",
        seed=seed,
    )

    # ---------- 2.6  Trainer ----------
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---------- 2.7  Train & evaluate ----------
    trainer.train()
    eval_results = trainer.evaluate()

    # ---------- 2.8  Save ----------
    trainer.save_model(Path(output_dir) / "best")

    return trainer, eval_results
