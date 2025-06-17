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

Project -3  Loan Eligibility Predictor

# Loan Eligibility Predictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
data = pd.read_csv(url)

# Step 2: Drop unnecessary columns
data.drop('Loan_ID', axis=1, inplace=True)

# Step 3: Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Step 4: Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Step 5: Split dataset into features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train Model (Random Forest or Logistic Regression)
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = LogisticRegression()   # You can switch here if you want Logistic Regression

model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC

# Step 10: Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

Output:

![image](https://github.com/user-attachments/assets/b4f4f866-4a8e-4108-8355-dafb272e286b)
![image](https://github.com/user-attachments/assets/88522ee3-928e-4cc8-8187-5d031e5792c2)

Project-4:Fake News Detection (Using Scikit-learn Built-in Dataset)

# Project 4: Fake News Detection (Using Scikit-learn Built-in Dataset)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Step 1: Load built-in dataset (2 categories only to simulate fake vs real)
categories = ['sci.space', 'talk.politics.misc']  # Space (Real) vs Politics (Fake for simulation)
news = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X = news.data  # Text data
y = news.target  # Labels (0 or 1)

# Step 2: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 4: Model Training (Passive Aggressive Classifier)
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(X_train_tfidf, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


Output:

![image](https://github.com/user-attachments/assets/b8025e83-0967-4283-a552-31f4b9ce7db0)

Project-5 Movie recommendation System
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Sample MovieLens-like dataset manually created for demonstration (since direct URLs fail)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'movie': ['Avengers', 'Titanic', 'Gladiator', 'Avengers', 'Gladiator',
              'Titanic', 'Gladiator', 'Avengers', 'Titanic', 'Gladiator'],
    'rating': [5, 4, 3, 5, 4, 5, 4, 3, 5, 4]
}

df = pd.DataFrame(data)
print("Sample Data:\n", df)

# Pivot to create User-Item matrix
user_movie_matrix = df.pivot_table(index='user_id', columns='movie', values='rating').fillna(0)
print("\nUser-Movie Matrix:\n", user_movie_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
print("\nUser Similarity Matrix:\n", user_similarity)

# Recommend movies for User 1
user_id = 1
similar_users = list(enumerate(user_similarity[user_id - 1]))
similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

print("\nTop similar users to User", user_id, ":", similar_users[1:])

# Find movies rated by the most similar user but not rated by User 1
top_user = similar_users[1][0] + 1
user_movies = df[df.user_id == user_id]['movie'].tolist()
top_user_movies = df[df.user_id == top_user]

# Recommend unseen movies
recommend_movies = top_user_movies[~top_user_movies['movie'].isin(user_movies)]
recommend_movies = recommend_movies.sort_values(by='rating', ascending=False)

print("\nRecommended Movies for User", user_id, ":\n", recommend_movies[['movie', 'rating']])
Output:

![image](https://github.com/user-attachments/assets/1bebd512-08dc-4127-8dca-dc1f4e749beb)
![image](https://github.com/user-attachments/assets/f91a2cb6-a855-44be-a7c3-d0136659e360)

Project 6: Stock Price Prediction using LSTM 
# Stock Price Prediction using LSTM 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Step 1: Load Stock Price Data (Example: Apple 'AAPL')
df = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

print("Stock Data Sample:\n", df.head())

# Step 2: Use 'Close' price for prediction
data = df[['Close']].values

# Step 3: Normalize data (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 4: Prepare training data (60 timesteps)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 5: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Step 6: Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# Step 7: Predict and inverse transform to original scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 8: Plot Actual vs Predicted Prices
plt.figure(figsize=(10,6))
plt.plot(y_test_scaled, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

Output:
![image](https://github.com/user-attachments/assets/e5ce4946-8f35-455c-8186-f62562ce666f)
![image](https://github.com/user-attachments/assets/a0bea1c4-6e95-42b3-bb84-d87b0d921f76)


#PROJECT -7 Emotion Detection from Text
import nltk
from nltk.corpus import twitter_samples
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK Twitter Dataset
nltk.download('twitter_samples')
nltk.download('punkt')

# Step 1: Load positive and negative tweets as sample emotional data
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

# Step 2: Create DataFrame
tweets = pd.DataFrame({'text': pos_tweets + neg_tweets, 
                       'emotion': ['positive']*len(pos_tweets) + ['negative']*len(neg_tweets)})

print("Sample Tweets:\n", tweets.head())

# Step 3: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(tweets['text'])
y = tweets['emotion']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#Output

![image](https://github.com/user-attachments/assets/f51f7b80-2916-4347-aa5e-16d0eb6cb7a0)
