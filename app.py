from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load ML Model
model = joblib.load("waiting_time_model.pkl")
service_encoder = joblib.load("service_encoder.pkl")
day_encoder = joblib.load("day_encoder.pkl")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    service = data["service"]
    queue = data["queue_length"]
    day = data["day"]
    hour = data["hour"]

    service = service_encoder.transform([service])[0]
    day = day_encoder.transform([day])[0]

    sample = pd.DataFrame({
        "Service_Type": [service],
        "Queue_Length": [queue],
        "Day": [day],
        "Hour": [hour]
    })

    prediction = model.predict(sample)[0]

    return jsonify({
        "predicted_wait_time": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)