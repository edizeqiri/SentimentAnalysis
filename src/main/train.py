# run_roberta.py  ───────────────────────────────────────────
from multiprocessing import freeze_support
import multiprocessing as mp
import pandas as pd

from src.main.model.roberta import train_roberta_sentiment


def main():
    train = pd.read_csv("../resources/data/training.csv")
    # test  = pd.read_csv("../resources/data/test.csv")  # if you need it

    trainer, metrics = train_roberta_sentiment(train)
    print(trainer)
    print(metrics)


if __name__ == "__main__":
    freeze_support()          # safe on every OS, needed on Windows
    main()
