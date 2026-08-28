import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("data\student_placement.csv")

# Features and target
X = data[
    ["CGPA", "Attendance", "CodingScore", "Projects", "Internship"]
]

y = data["Placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Test Accuracy:", accuracy)

# Save model
joblib.dump(model, "student_placement_model.pkl")

print("Model saved as student_placement_model.pkl")