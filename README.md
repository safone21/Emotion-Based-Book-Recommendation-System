BY SAFOUANE LASMAR


#  Emotion-Based Book Recommendation System

This project is an intelligent book recommendation engine that understands your emotions and suggests books accordingly. It combines a custom-trained emotion detection model with a curated book dataset to enhance the reading experience.

##  Features

-  Emotion detection using your own trained NLP model
-  Book recommendations based on user mood
-  Data preprocessing, visualization, and word clouds
-  Modular code with clean folder structure
-  Jupyter Notebook interface for testing and experimentation

##  Emotions Covered

The emotion detection model identifies six key emotions:
- Sadness 
- Joy 
- Love 
- Anger 
- Fear 
- Surprise 

##  Install dependencies

pip install -r requirements.txt

##  Prepare data

Place your emotions.csv and books.csv files in the data/ directory.

##  Train the emotion model

notebooks/emotion_detection.ipynb

##  Recommend books

notebooks/book_recommender.ipynb

##  Run the Streamlit App

cd src
streamlit run app.py
