# Import necessary libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data_url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv"
data = pd.read_csv(data_url)

# Split features and target
X = data.iloc[:, :-1].values  # Features
y = data['species'].values    # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the model
joblib.dump(clf, 'iris_random_forest_model.pkl')
print("Model saved as 'iris_random_forest_model.pkl'")