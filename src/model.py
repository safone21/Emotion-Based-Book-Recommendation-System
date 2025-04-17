from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class EmotionClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))