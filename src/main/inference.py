from transformers import pipeline
import pandas as pd
import re

def preprocess_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"\\'", "'", text).replace("`", "'")  # fix escaped/back-tick quotes
    text = text.lower()
    text = re.sub(r'@\w+', '@user', text)  # user placeholder
    text = re.sub(r'http\S+|www\S+', 'http', text)  # URL placeholder
    text = re.sub(r'\s+', ' ', text).strip()
    return text
model_dir = "roberta_sentiment_finetuned/best"   # same as in training
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model_dir,
    tokenizer=model_dir,
    device=0          # -1 for CPU
)

test_df = pd.read_csv("../resources/data/test.csv")
test_df["clean"] = test_df["sentence"].map(preprocess_tweet)

outputs = sentiment_pipe(list(test_df["clean"]), batch_size=64)

test_df["label"] = [o["label"]  for o in outputs]
test_df["conf"]       = [o["score"]  for o in outputs]

test_df[['id','label']].to_csv("test_with_predictions.csv", index=False)
