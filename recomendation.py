# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Loading Data
def load_data("C:\Users\91761\OneDrive\Documents\Desktop\python_projects\.vscode\tracks.csv"):
    try:
        return pd.read_csv("C:\Users\91761\OneDrive\Documents\Desktop\python_projects\.vscode\tracks.csv")
    except FileNotFoundError:
        print(f"Error: The file '"C:\Users\91761\OneDrive\Documents\Desktop\python_projects\.vscode\SpotifyAudioFeaturesApril2019.csv.zip
              ",' is not found.")
        exit()

# Data Preprocessing
def preprocess_data(df_tracks, df_features):
    try:
        df = pd.merge(df_tracks, df_features, on='id')
        df = df.dropna()
        return df
    except ValueError:
        print("Error: The 'id' column is not present in both datasets.")
        exit()
    except Exception as e:
        print("Error: An error occurred while cleaning the data.")
        print(str(e))
        exit()

# Feature Engineering
def feature_engineering(df):
    features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 
                'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
    try:
        X = df[features]
        y = (df['popularity'] > df['popularity'].median()).astype(int)
        return X, y
    except KeyError:
        print("Error: Some features are not present in the dataset.")
        exit()

# Handling Categorical Variables
def handle_categorical_variables(X):
    le = LabelEncoder()
    try:
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = le.fit_transform(X[column])
        return X
    except Exception as e:
        print("Error: An error occurred while handling categorical variables.")
        print(str(e))
        exit()

# Splitting Data
def split_data(X, y):
    try:
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print("Error: An error occurred while splitting the data.")
        print(str(e))
        exit()

# Training Model
def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print("Error: An error occurred while training the model.")
        print(str(e))
        exit()

# Making Predictions
def make_predictions(model, X_test):
    try:
        return model.predict(X_test)
    except Exception as e:
        print("Error: An error occurred while making predictions.")
        print(str(e))
        exit()

# Evaluating Model
def evaluate_model(y_test, y_pred):
    try:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
    except Exception as e:
        print("Error: An error occurred while evaluating the model.")
        print(str(e))
        exit()

# Main Function
def main():
    df_tracks = load_data(r'C:\Users\91761\OneDrive\Documents\Data\tracks.csv')
    df_features = load_data(r'C:\Users\91761\OneDrive\Documents\Data\audio_features.csv')
    df = preprocess_data(df_tracks, df_features)
    X, y = feature_engineering(df)
    X = handle_categorical_variables(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = make_predictions(model, X_test)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()