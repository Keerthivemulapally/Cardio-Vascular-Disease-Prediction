from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("cardio_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        blood_pressure = float(request.form["blood_pressure"])
        cholesterol = float(request.form["cholesterol"])
        heart_rate = float(request.form["heart_rate"])
        bmi = float(request.form["bmi"])
        smoking = int(request.form["smoking"])
        exercise = float(request.form["exercise"])

        # Predict
        features = np.array([[age, blood_pressure, cholesterol, heart_rate, bmi, smoking, exercise]])
        prediction = model.predict(features)[0]
        risk = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("index.html", prediction_text=f"Prediction: {risk}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
