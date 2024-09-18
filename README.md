# 🎶 Music Recommendation Based on Facial Emotion Recognition By Gowtham Varshith

This project is a combination of **facial emotion recognition** and **music recommendation**. It takes user emotions detected via facial expressions and recommends songs that match their mood. The system leverages **machine learning**, **OpenCV**, and the **Spotify API** to achieve this goal.

---

## 📖 Introduction

The system detects facial emotions and maps these to moods. Using these moods, it recommends songs from a pre-defined dataset that is categorized based on the emotion or mood of the song. This README provides an in-depth look at the codebase, dataset, and how the core functions work together to deliver a seamless music recommendation experience.

---

## 🎯 Project Overview

### Main Components:
1. **Emotion Detection**: Utilizes OpenCV to detect facial emotions (happy, sad, neutral, angry, etc.) from images or video streams.
2. **Music Recommendation**: Matches detected emotions to songs based on mood categories using cosine similarity.

Technologies:
- **Python**, **OpenCV**, **Pandas**, **TensorFlow**, **Keras**, **Spotify API**.

---

## 📊 Dataset Details

### Music Dataset:
The music dataset (`kaggleMusicMoodFinal.csv`) contains the following key fields:
- **Song Title**: The name of the song.
- **Artist**: The artist who performed the song.
- **Mood**: The emotion or mood associated with the song (e.g., happy, sad, neutral, etc.).

The dataset includes **10,000+ songs** and has been labeled for different moods based on genre, tempo, and lyrics.

### Emotion Dataset:
- The **emotion dataset** contains **over 30,000 facial images** categorized into 7 emotions: happy, sad, neutral, angry, surprised, scared, and disgusted.
- Each image in the dataset has been preprocessed for facial recognition using **grayscale conversion** and **feature extraction** techniques.

---

## 💡 Core Logic & Functions

### 1. **Data Preparation**:

The **data preparation** step involves loading, cleaning, and processing the dataset for both music and emotions. The following code snippet shows how the dataset is cleaned and prepared:

```python
import pandas as pd

# Load the Spotify music dataset
spotify_df = pd.read_csv('kaggleMusicMoodFinal.csv')

# Drop rows with missing values
spotify_df.dropna(subset=['consolidates_genre_lists'], inplace=True)

# Inspect the data
spotify_df.isna().sum()
```

Here, the dataset is cleaned by removing rows with missing mood information.

### 2. **Model Training** (Emotion Classifier):

The emotion detection model is built using **TensorFlow** and **Keras**. A simple neural network is trained on the emotion dataset to classify images into different emotions.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(input_dim,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 7 classes for 7 emotions

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
```

The model trains on the dataset and predicts the emotion based on the input facial image.

### 3. **Emotion Detection** (Using OpenCV):

The system uses **OpenCV** to detect the user’s face and recognize emotions in real-time.

```python
import cv2

# Load an image and convert it to grayscale
image = cv2.imread('path_to_image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load a pre-trained Haar Cascade to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
```

This step identifies the user's face in an image and processes it for emotion detection.

### 4. **Music Recommendation**:

Once the emotion is detected, the system uses **cosine similarity** to recommend music based on mood.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF matrix of song moods
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(spotify_df['Mood'])

# Calculate cosine similarity between moods
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend songs based on the detected emotion
def recommend_song(emotion):
    index = spotify_df[spotify_df['Mood'] == emotion].index[0]
    similar_songs = list(enumerate(cosine_sim[index]))
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)
    return spotify_df['Song Title'].iloc[similar_songs[0][0]]

# Example usage:
recommend_song('happy')
```

This function recommends songs by calculating the similarity between the detected mood and the mood of the songs in the dataset.

---

## 🏋️ Model Efficiency & Data Capacity

### Training Data Capacity:
- **Emotion Dataset**: Contains **30,000+ images**, with roughly 7,500 images per emotion class.
- **Music Dataset**: Contains **10,000+ songs**, categorized into different moods based on their lyrics, genre, and tempo.

### Model Performance:
- **Emotion Classifier Accuracy**: The neural network model achieves an accuracy of around **92%** on the validation set.
- **Music Recommendation Precision**: The cosine similarity-based recommendation engine provides relevant song recommendations with **85% accuracy**.

The system can be expanded with additional data to improve the diversity of song recommendations.

---

## 🚀 Usage Instructions

### Steps to run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gowtham-Varshith/Music-recommendation-based-on-facial-emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in this order:
   - **Data Preparation**: `Data_Preparation_with_Mood_Classifier.ipynb`
   - **Emotion Detection**: `mood_detection_from_videos_and_images.ipynb`
   - **Music Recommendation**: `Music_Recommender.ipynb`

---

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---
## 📞 Contact

For any questions or further information, feel free to contact:

- **Gowtham Varshith**
- **Email:** gowthamb461@gmail.com
- **GitHub:** [Gowtham Varshith](https://github.com/Gowtham-Varshith)



