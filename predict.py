import joblib


# Load trained model
model = joblib.load("student_placement_model.pkl")


# Student details
student = [[
    8.5,   # CGPA
    92,    # Attendance
    85,    # Coding Score
    3,     # Projects
    1      # Internship
]]

# Make prediction
prediction = model.predict(student)

if prediction[0] == 1:
    result = "PLACED"
else:
    result = "NOT PLACED"

print("Predicted Placement:", result)