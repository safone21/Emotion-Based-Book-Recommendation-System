import streamlit as st
import pandas as pd
import joblib
from preprocessing import clean_text


exclude_keywords = [
    'technology & engineering', 'women\'s health', 'military science',
    'health & fitness', 'diet & nutrition', 'cooking', 'reference',
    'cookbook', 'recipe', 'cooking', 'manual', 'how to', 'business',
    'marketing', 'guide', 'biography', 'memoir', 'family & relationships',
    'life stages', 'teenagers', 'religion', 'general', 'christian life'
]


def is_relevant(category):
    category = str(category).lower()
    return not any(keyword in category for keyword in exclude_keywords)


books = pd.read_csv('../data/books.csv')
books = books[['Title', 'Authors', 'Description','Category']].dropna()
books = books[books['Description'].str.strip() != ""]
books = books[books['Category'].apply(is_relevant)]

model = joblib.load('../outputs/emotion_model.pkl')


emotion_dict = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}


def detect_emotion(text):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    return emotion_dict[int(pred)]


def recommend_books(emotion_label, books_df, top_n=5):
    recommendation_map = {
        'Sadness': 'Joy',
        'Joy': 'Joy',
        'Love': 'Love',
        'Anger': 'Calm',
        'Fear': 'Inspiration',
        'Surprise': 'Adventure'
    }
    # Get the target vibe for the emotion based on the recommendation map.
    # If no mapping is found for the given emotion, it will return the emotion itself as the default.
    target_vibe = recommendation_map.get(emotion_label, emotion_label)
    filtered = books_df[books_df['Description'].str.contains(target_vibe, case=False, na=False)]

    return filtered.sample(top_n) if not filtered.empty else books_df.sample(top_n)

st.title(" Emotion-Based Book Recommender (Beta)")
st.write("Tell us how you're feeling, and we'll suggest a book that matches your mood.")

user_input = st.text_area(" How are you feeling today?", height=150)

if st.button(" Recommend Me a Book"):
    if user_input.strip():
        emotion = detect_emotion(user_input)
        st.success(f"**Detected Emotion:** {emotion}")

        recommendations = recommend_books(emotion, books)
        st.write("### Recommended Books:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['Title']}** by *{row['Authors']}*")
    else:
        st.warning("Please enter how you're feeling first!")
