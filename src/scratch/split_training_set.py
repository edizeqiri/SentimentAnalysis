import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("resources/data/training.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("resources/data/training_split.csv", index=False)
val_df.to_csv("resources/data/validation_split.csv", index=False)
print("Train:", len(train_df), "  Val:", len(val_df))