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

##  Dataset Setup

To run this project, you need to manually download the dataset and place it in the correct folder:
Create a folder named data/ at the root of your project.


Download the dataset from Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
After downloading, extract the dataset (usually a .csv file).
Rename the file to emotions.csv and move it into the data/ folder.



In the same data/ folder, download the books dataset from:
https://www.kaggle.com/datasets/elvinrustam/books-dataset
After downloading and extracting, locate the file named something like booksdata.csv.
Rename the file to books.csv and place it in the same data/ directory.



##  Train the emotion model

notebooks/emotion_detection.ipynb

##  Recommend books

notebooks/book_recommender.ipynb

##  Run the Streamlit App

cd src
streamlit run app.py
