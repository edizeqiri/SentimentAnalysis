import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RobertaModel(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def tokenize_batch(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def training_step(self, batch, batch_idx):
        inputs = self.tokenize_batch(batch["text"])
        labels = batch["label"]
        outputs = self(**inputs, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.tokenize_batch(batch["text"])
        labels = batch["label"]
        outputs = self(**inputs, labels=labels)
        self.log("val_loss", outputs.loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
