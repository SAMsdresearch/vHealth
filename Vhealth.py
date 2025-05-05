import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Load the dataset
data = pd.read_csv('Heart Prediction Quantum Dataset.csv')
# Preprocessing
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})  # Example encoding
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
print(classification_report(y_test, y_pred))