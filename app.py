from flask import Flask, render_template, request
import pandas as pd
import joblib
import os  # to read environment variables

app = Flask(__name__)

# Load model
model = joblib.load("model/upi_fraud_detector_model_weighted_smote.pkl")

def explain_prediction(data, prediction):
    if prediction == 1:
        if data['user_app_status'] == "Success" and data['bank_actual_status'] == "Not Credited":
            return "⚠️ This transaction was marked as Success by the app but failed at the bank. Possible reasons: technical glitch, fraud attempt, or bad payment gateway. 😱🥵"
        elif data['merchant_category'].lower() in ["spam site", "unknown"]:
            return "⚠️ Suspicious merchant category. Could be a spam or phishing attempt. 👻"
        else:
            return "⚠️ This transaction was flagged as fraud based on unusual patterns. 😩😨"
    else:
        if data['user_app_status'] != data['bank_actual_status']:
            return "ℹ️ The transaction is Not fraud, and it is Successful 😎🤩😍."
        else:
            return "✅ This transaction looks Not fraud. No suspicious indicators detected.😙😘🥰"

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/predict", methods=["GET", "POST"])
def index():
    result = None
    explanation = None

    if request.method == "POST":
        if request.form.get("mode") == "single":
            data = {
                "sender_upi_id": request.form["sender_upi_id"],
                "receiver_upi_id": request.form["receiver_upi_id"],
                "amount": float(request.form["amount"]),
                "user_app_status": request.form["user_app_status"],
                "bank_actual_status": request.form["bank_actual_status"],
                "timestamp_date": request.form["timestamp_date"],
                "timestamp_time": request.form["timestamp_time"],
                "location": request.form["location"],
                "device_id": request.form["device_id"],
                "merchant_category": request.form["merchant_category"]
            }

            input_df = pd.DataFrame([data])
            prediction = model.predict(input_df)[0]
            label = "Fraud Transaction" if prediction == 1 else "Not fraud Transaction"
            explanation = explain_prediction(data, prediction)
            result = label

    return render_template("index.html", result=result, explanation=explanation)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
