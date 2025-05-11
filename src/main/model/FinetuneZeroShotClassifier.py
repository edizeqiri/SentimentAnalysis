from __future__ import annotations
"""finetune_zero_shot_classifier.py
-------------------------------------------------
A reusable helper class to fine‑tune a zero‑shot Natural Language Inference (NLI) model for
semantic (sentiment) classification with small datasets.

🔄 **New in v1.1**
• Optional *k‑fold cross‑validation* (stratified) with macro‑F1 aggregation
• Quick‑train toggle (`fast_mode=True`) that swaps in a lighter backbone (DeBERTa‑v3‑*base*‑MNLI) and reduces training epochs / batch size for rapid iteration

Usage
-----
```python
import pandas as pd
from finetune_zero_shot_classifier import FinetuneZeroShotClassifier

df = pd.read_csv("sentences.csv")
clf = FinetuneZeroShotClassifier(k_folds=5, fast_mode=False)
clf.load_dataframe(df)
results = clf.train()            # returns metrics dict or list of dicts (CV)
print(results)
```
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_metric
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

warnings.filterwarnings("ignore")

def _clean_text(text: str) -> str:
    """Basic normalisation: strip & collapse whitespace (keeps case/punctuation)."""
    return " ".join(text.strip().split())

@dataclass
class FinetuneZeroShotClassifier:
    # Model / training hyper‑parameters
    model_name: str = "microsoft/deberta-v3-large-mnli"
    fast_model_name: str = "microsoft/deberta-v3-base-mnli"
    fast_mode: bool = False  # if True override model_name & shrink training budget

    # Data column names
    id_col: str = "id"
    text_col: str = "sentence"
    label_col: str = "label"

    # General training settings
    seed: int = 42
    test_size: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 5
    patience: int = 2  # early stopping
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    freeze_base_model: bool = False  # optional encoder freezing
    k_folds: int = 1  # >1 triggers stratified CV

    # Internals (populated post‑init)
    tokenizer: Optional[AutoTokenizer] = field(init=False, default=None)
    model: Optional[AutoModelForSequenceClassification] = field(init=False, default=None)
    label2id: Dict[str, int] = field(init=False, default_factory=dict)
    id2label: Dict[int, str] = field(init=False, default_factory=dict)
    data_collator: Optional[DataCollatorWithPadding] = field(init=False, default=None)

    metric_acc: Any = field(init=False, default=None)
    metric_f1: Any = field(init=False, default=None)

    def __post_init__(self):
        set_seed(self.seed)
        self.metric_acc = load_metric("accuracy")
        self.metric_f1 = load_metric("f1")
        if self.fast_mode:
            # Down‑shift to lighter backbone & training budget
            self.model_name = self.fast_model_name
            self.batch_size = max(4, self.batch_size // 2)
            self.num_epochs = max(2, self.num_epochs // 2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    # ------------------------------------------------------------------
    # Data loading / preprocessing
    # ------------------------------------------------------------------
    def load_dataframe(self, df: pd.DataFrame):
        """Convert a pandas DataFrame to a HuggingFace Dataset (cleaned)."""
        assert self.text_col in df.columns and self.label_col in df.columns, "Required columns missing."
        df = df.copy()
        df[self.text_col] = df[self.text_col].astype(str).apply(_clean_text)
        self.label2id = {label: idx for idx, label in enumerate(sorted(df[self.label_col].unique()))}
        self.id2label = {v: k for k, v in self.label2id.items()}
        df["label_id"] = df[self.label_col].map(self.label2id)
        self.dataset = Dataset.from_pandas(df[[self.text_col, "label_id"]])
        self.dataset = self.dataset.class_encode_column("label_id")
        return self.dataset

    def _tokenise(self, batch):
        return self.tokenizer(batch[self.text_col], truncation=True)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = self.metric_acc.compute(predictions=preds, references=labels)
        f1 = self.metric_f1.compute(predictions=preds, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

    def _init_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        if self.freeze_base_model:
            for p in self.model.base_model.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Training (single split or k‑fold CV)
    # ------------------------------------------------------------------
    def train(self, output_dir: str = "./finetuned_zs_model") -> Union[Dict[str, float], List[Dict[str, float]]]:
        assert hasattr(self, "dataset"), "Call load_dataframe() first."
        self.data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

        if self.k_folds and self.k_folds > 1:
            return self._train_cross_validation(output_dir)
        else:
            return self._train_single(output_dir)

    # ---- single split helper
    def _train_single(self, output_dir: str):
        train_ds, val_ds = train_test_split(
            self.dataset,
            test_size=self.test_size,
            stratify=self.dataset["label_id"],
            random_state=self.seed,
        )
        metrics = self._fit(train_ds, val_ds, output_dir)
        return metrics

    # ---- k‑fold helper
    def _train_cross_validation(self, output_dir: str):
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        labels = np.array(self.dataset["label_id"])
        all_metrics: List[Dict[str, float]] = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            fold_dir = os.path.join(output_dir, f"fold_{fold}")
            train_ds = self.dataset.select(train_idx)
            val_ds = self.dataset.select(val_idx)
            metrics = self._fit(train_ds, val_ds, fold_dir, fold)
            all_metrics.append(metrics)
        # Aggregate macro‑F1 & accuracy across folds
        avg_metrics = {
            k: round(float(np.mean([m[k] for m in all_metrics])), 4)
            for k in all_metrics[0].keys()
            if k.startswith("eval_")
        }
        return [*all_metrics, {"cv_mean": avg_metrics}]

    # ---- core training routine shared by single / CV
    def _fit(self, train_ds, val_ds, output_dir: str, fold: Optional[int] = None):
        self._init_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Class imbalance weighting
        label_counts = np.bincount(train_ds["label_id"], minlength=len(self.label2id))
        class_weights = (1.0 / label_counts) * (len(train_ds) / len(label_counts))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # TrainingArgs (unique folder per fold)
        run_name = f"run_fold{fold}" if fold is not None else "run_single"
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            save_total_limit=2,
            seed=self.seed,
            dataloader_num_workers=4,
            logging_strategy="epoch",
        )

        # Custom loss to inject class weights
        def compute_loss(model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
            compute_loss=compute_loss,
        )
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(output_dir)
        # Keep the last trainer instance for potential inference
        self.trainer = trainer
        return {k: round(float(v), 4) for k, v in metrics.items() if k.startswith("eval_")}

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def predict(self, texts: List[str]) -> List[str]:
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        cleaned = [_clean_text(t) for t in texts]
        toks = self.tokenizer(cleaned, truncation=True, padding=True, return_tensors="pt")
        device = self.model.device if hasattr(self, "model") else torch.device("cpu")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            logits = self.model(**toks).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        return [self.id2label[int(i)] for i in preds]

    def evaluate(self) -> Dict[str, float]:
        if self.trainer is None:
            raise RuntimeError("Call train() first.")
        metrics = self.trainer.evaluate()
        return {k: round(float(v), 4) for k, v in metrics.items() if k.startswith("eval_")}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        obj = cls(**kwargs)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.id2label = obj.model.config.id2label
        obj.label2id = obj.model.config.label2id
        return obj
