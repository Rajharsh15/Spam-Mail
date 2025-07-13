import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_mail_data():
    data = pd.read_csv(r"C:\Users\Acer\Pictures\Screenshots\coding\SPAM MAIL\mail_data.csv")
    data = data.where((pd.notnull(data)), '')
    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1
    return data

def train_spam_model(data):
    X = data['Message']
    y = data['Category'].astype('int')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # Text feature extraction
    feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extractor.fit_transform(X_train)
    X_test_features = feature_extractor.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_features, y_train)

    # Accuracy on training data
    train_prediction = model.predict(X_train_features)
    train_accuracy = accuracy_score(y_train, train_prediction)
    print(f"✅ Training Accuracy: {train_accuracy:.4f}")

    # Accuracy on testing data
    test_prediction = model.predict(X_test_features)
    test_accuracy = accuracy_score(y_test, test_prediction)
    print(f"✅ Testing Accuracy: {test_accuracy:.4f}")

    return model, feature_extractor, train_accuracy, test_accuracy

def predict_spam(model, feature_extractor, input_text):
    input_features = feature_extractor.transform([input_text])
    prediction = model.predict(input_features)[0]
    return "✅ Ham Mail" if prediction == 1 else "🚫 Spam Mail"
