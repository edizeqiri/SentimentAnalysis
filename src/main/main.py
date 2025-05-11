import pandas as pd
from model.Logistic import Logistic
from icecream import ic
ic.configureOutput(includeContext=True)


def main():

    data = pd.read_csv('src/resources/data/training.csv')
    test = pd.read_csv('src/resources/data/test.csv')
    model = Logistic(learning_rate=0.01, num_iterations=1000)
    result = model.train(data[['sentence']], data['label'])
    
    print("Training Results:")
    print(result)
    
    predictions = ic(model.predict(ic(model.vectorizer.transform(ic(model.pre_process(test)['clean'])))))
    print("Predictions:", predictions)
    print("Accuracy:", model.calculate_sentiment_score(test['label'], predictions))
    
    # save to csv
    predictions[['id', 'label']].to_csv('src/resources/data/test_predictions.csv', index=False)

if __name__ == "__main__":
    main()