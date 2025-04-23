# pip install transformers vaderSentiment torch scikit-learn datasets --quiet
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import pipeline
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset

test = pd.read_csv("src/resources/data/test.csv")
train = pd.read_csv("src/resources/data/training.csv")
CANDIDATES = ["positive", "negative", "neutral"]
ds = train.copy(10000)

def convert_labels_to_ids(labels):
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    return [label_map[label.lower()] for label in labels]

def convert_ids_to_labels(ids):
    id_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return [id_to_label[id] for id in ids]


def fine_tune_roberta(train_data):
    print("Fine-tuning RoBERTa model...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    train_texts = train_data['sentence'].tolist()
    train_labels = convert_labels_to_ids(train_data['label'].tolist())

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Fine-tuning complete!")
    return model, tokenizer


def get_fine_tuned_predictions(sentences, model, tokenizer):

    fine_tuned_clf = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=8,
        padding=True,
        max_length=512,
        truncation=True
    )

    predictions = fine_tuned_clf(sentences, truncation=True, batch_size=64)

    return [p["label"].lower() for p in predictions]


def get_roberta_predictions(sentences):
    roberta_clf = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=8,
        padding=True,
        max_length=512,
        truncation=True
    )
    roberta_preds = roberta_clf(sentences, truncation=True)
    return [p["label"].lower() for p in roberta_preds]


def perform_cross_validation(ds, k=5, model_type="roberta", fine_tune=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []
    all_true = []
    all_pred = []

    results_df = pd.DataFrame()

    fold_num = 1
    for train_idx, val_idx in kf.split(ds):
        print(f"Processing fold {fold_num}/{k}...")

        train_data = ds.iloc[train_idx]
        val_data = ds.iloc[val_idx]

        print(
            f"  Using {len(train_data)} examples for training fold {fold_num}")

        if fine_tune:
            fine_tuned_model, fine_tuned_tokenizer = fine_tune_roberta(
                train_data)

            val_preds = get_fine_tuned_predictions(
                val_data['sentence'].tolist(),
                fine_tuned_model,
                fine_tuned_tokenizer
            )
        else:
            val_preds = get_roberta_predictions(val_data['sentence'].tolist())

        val_truth = val_data['label'].tolist()
        fold_accuracy = accuracy_score(val_truth, val_preds)
        fold_accuracies.append(fold_accuracy)

        all_true.extend(val_truth)
        all_pred.extend(val_preds)

        fold_results = pd.DataFrame({
            'fold': fold_num,
            'sentence': val_data['sentence'],
            'true_label': val_truth,
            'predicted_label': val_preds,
            'correct': [t == p for t, p in zip(val_truth, val_preds)]
        })
        results_df = pd.concat([results_df, fold_results])

        fold_num += 1

    model_name = f"{model_type.upper()}{' (Fine-tuned)' if fine_tune else ''}"
    print(f"\n{model_name} Model Cross-Validation Results:")
    print(
        f"Average Accuracy across {k} folds: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred))

    cm = confusion_matrix(all_true, all_pred, labels=CANDIDATES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CANDIDATES, yticklabels=CANDIDATES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Model Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return results_df, np.mean(fold_accuracies), all_true, all_pred


def analyze_errors(results_df, model_name):
    """Analyze the errors made by the model"""
    misclassified = results_df[~results_df['correct']]
    print(f"\n{model_name} Error Analysis:")
    print(f"Total misclassified examples: {len(misclassified)}")

    error_patterns = misclassified.groupby(
        ['true_label', 'predicted_label']).size().reset_index(name='count')
    error_patterns = error_patterns.sort_values('count', ascending=False)
    print("\nMost common misclassification patterns:")
    print(error_patterns.head(5))

    print("\nExample misclassifications:")
    for _, row in error_patterns.head(3).iterrows():
        true_label = row['true_label']
        pred_label = row['predicted_label']
        examples = misclassified[(misclassified['true_label'] == true_label) &
                                 (misclassified['predicted_label'] == pred_label)].head(2)
        print(f"\nTrue: {true_label}, Predicted: {pred_label}")
        for _, example in examples.iterrows():
            print(f"  • {example['sentence'][:100]}...")


# print("Starting cross-validation for pre-trained RoBERTa model...")
# roberta_results, roberta_accuracy, roberta_true, roberta_pred = perform_cross_validation(
#     ds, k=5, model_type="roberta", fine_tune=False
# )

print("\nStarting cross-validation for fine-tuned RoBERTa model...")
fine_tuned_results, fine_tuned_accuracy, fine_tuned_true, fine_tuned_pred = perform_cross_validation(
    ds, k=2, model_type="roberta", fine_tune=True
)

# analyze_errors(roberta_results, "Pre-trained RoBERTa")

analyze_errors(fine_tuned_results, "Fine-tuned RoBERTa")

# roberta_results.to_csv(
#     "roberta_pretrained_cross_validation_results.csv", index=False)
fine_tuned_results.to_csv(
    "roberta_finetuned_cross_validation_results.csv", index=False)

print("\nPerformance Comparison:")
# print(f"Pre-trained RoBERTa accuracy: {roberta_accuracy:.4f}")
print(f"Fine-tuned RoBERTa accuracy: {fine_tuned_accuracy:.4f}")
# print(f"Improvement: {fine_tuned_accuracy - roberta_accuracy:.4f} ({((fine_tuned_accuracy - roberta_accuracy) / roberta_accuracy) * 100:.2f}%)")

print("\nCross-validation complete!")
