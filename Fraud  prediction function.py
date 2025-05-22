import joblib
import pandas as pd

# Load saved model
model = joblib.load("upi_fraud_detector_model_weighted_smote.pkl")

def predict_transactions(input_list, model):
    input_df = pd.DataFrame(input_list)
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]
    results = []
    for pred, prob in zip(preds, probs):
        label = "Fraudulent Transaction" if pred == 1 else "Legitimate Transaction"
        results.append({"prediction": label, "fraud_probability": prob})
    return results

# Sample test cases
test_samples = [
    {
        "sender_upi_id": "user123@upi",
        "receiver_upi_id": "merchant789@upi",
        "amount": 2500,
        "user_app_status": "Success",
        "bank_actual_status": "Credited",
        "timestamp_date": "15/05/2025",
        "timestamp_time": "14:30:00",
        "location": "Hyderabad",
        "device_id": "device001",
        "merchant_category": "Online Shopping"
    },
    {
        "sender_upi_id": "user456@upi",
        "receiver_upi_id": "spam111@upi",
        "amount": 15000,
        "user_app_status": "Success",
        "bank_actual_status": "Not Credited",
        "timestamp_date": "16/05/2025",
        "timestamp_time": "09:15:00",
        "location": "Mumbai",
        "device_id": "device002",
        "merchant_category": "Spam Site"
    }
]

results = predict_transactions(test_samples, model)
for i, res in enumerate(results, 1):
    print(f"Transaction {i}: {res['prediction']} with fraud probability {res['fraud_probability']:.2f}")
