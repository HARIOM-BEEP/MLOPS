import joblib


def test_model_file_exists():
    model = joblib.load("student_placement_model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("student_placement_model.pkl")

    student = [[
        8.5,
        92,
        85,
        3,
        1
    ]]

    prediction = model.predict(student)

    assert prediction[0] == 1