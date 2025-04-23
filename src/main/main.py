# pip install transformers vaderSentiment torch scikit-learn --quiet
from transformers import pipeline
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv("src/resources/data/test.csv")
train = pd.read_csv("src/resources/data/training.csv")
CANDIDATES = ["positive", "negative", "neutral"]
ds = train.copy() 

def get_roberta_predictions(sentences):
    roberta_clf = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0,
        batch_size=64,
        padding=True,
        max_length=512,                    
        truncation=True
    )
    roberta_preds = roberta_clf(sentences, truncation=True)
    return [p["label"].lower() for p in roberta_preds]


def perform_cross_validation(ds, k=5, model_type="roberta"):
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
        
  
        print(f"  Using {len(train_data)} examples for training fold {fold_num}")

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
    
    print(f"\n{model_type.upper()} Model Cross-Validation Results:")
    print(f"Average Accuracy across {k} folds: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred))
    
    cm = confusion_matrix(all_true, all_pred, labels=CANDIDATES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CANDIDATES, yticklabels=CANDIDATES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.upper()} Model Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return results_df, np.mean(fold_accuracies), all_true, all_pred


print("Starting cross-validation for RoBERTa model...")
roberta_results, roberta_accuracy, roberta_true, roberta_pred = perform_cross_validation(ds, k=5, model_type="roberta")


def analyze_errors(results_df, model_name):
    misclassified = results_df[~results_df['correct']]
    print(f"\n{model_name} Error Analysis:")
    print(f"Total misclassified examples: {len(misclassified)}")
    
    error_patterns = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
    error_patterns = error_patterns.sort_values('count', ascending=False)
    print("\nMost common misclassification patterns:")
    print(error_patterns.head(5))
    
    # Sample misclassified examples for each pattern
    print("\nExample misclassifications:")
    for _, row in error_patterns.head(3).iterrows():
        true_label = row['true_label']
        pred_label = row['predicted_label']
        examples = misclassified[(misclassified['true_label'] == true_label) & 
                                (misclassified['predicted_label'] == pred_label)].head(2)
        print(f"\nTrue: {true_label}, Predicted: {pred_label}")
        for _, example in examples.iterrows():
            print(f"  • {example['sentence'][:100]}...")

# Analyze errors for RoBERTa
analyze_errors(roberta_results, "RoBERTa")

# Optionally, save the results to CSV
roberta_results.to_csv("roberta_cross_validation_results.csv", index=False)

print("\nCross-validation complete!")