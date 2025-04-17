import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text

def load_dataset(path='data/emotions.csv'):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['emotion'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test