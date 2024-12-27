import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("cardio_data.csv")  # Replace with your actual dataset path

# Preprocessing
X = data[["age", "blood_pressure", "cholesterol", "heart_rate", "bmi", "smoking", "exercise"]]  # Features
 # Features
y = data["cardio_disease"]  # Target (1 = Disease, 0 = No Disease)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("cardio_model.pkl", "wb") as file:
    pickle.dump(model, file)
