# Tamizhan-skills-Project-1-8-Solved
#Project 1
# Project 1- Email Spam Detection using Naive Bayes

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load Dataset
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Data preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)

# Features and Labels
X = df['message']
y = df['label']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Building - Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
![image](https://github.com/user-attachments/assets/8dc3379d-f2a5-42b3-b203-1765fbbba99a)

#Project-2 Handwritten Digit Recognition (MNIST)




# Handwritten Digit Recognition using CNN (MNIST dataset)

# Import libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset directly from Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data (28x28x1 for CNN input)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Normalize pixel values (0-1 range)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes (0-9)
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\nTest Accuracy:", test_acc*100)
Ouput: 
![image](https://github.com/user-attachments/assets/fb13f3c6-e51d-48ca-acd6-6286203c0c03)
