import pandas as pd
from model.FinetuneZeroShotClassifier import FinetuneZeroShotClassifier

training_data = pd.read_csv("src/resources/data/training.csv")
test_data = pd.read_csv("src/resources/data/test.csv")
clf = FinetuneZeroShotClassifier()
clf.load_dataframe(training_data)
clf.train()
clf.evaluate()
preds = clf.predict(test_data['sentence'])

print(preds)