from src.dataloader import load_dataset
from src.model import EmotionClassifier

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model = EmotionClassifier()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)

if __name__ == '__main__':
    main()

