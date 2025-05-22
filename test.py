import pandas as pd
import joblib

# Load saved model
model = joblib.load("upi_fraud_detector_model_weighted_smote.pkl")

# Prepare 5 test transactions as a list of dicts
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
    },
    {
        "sender_upi_id": "user789@upi",
        "receiver_upi_id": "bank555@upi",
        "amount": 500,
        "user_app_status": "Success",
        "bank_actual_status": "Credited",
        "timestamp_date": "20/05/2025",
        "timestamp_time": "18:45:00",
        "location": "Delhi",
        "device_id": "device003",
        "merchant_category": "Bank"
    },
    {
        "sender_upi_id": "user222@upi",
        "receiver_upi_id": "merchant123@upi",
        "amount": 3000,
        "user_app_status": "Failed",
        "bank_actual_status": "Not Credited",
        "timestamp_date": "21/05/2025",
        "timestamp_time": "12:00:00",
        "location": "Bangalore",
        "device_id": "device004",
        "merchant_category": "Online Shopping"
    },
    {
        "sender_upi_id": "user333@upi",
        "receiver_upi_id": "merchant456@upi",
        "amount": 7500,
        "user_app_status": "Success",
        "bank_actual_status": "Credited",
        "timestamp_date": "22/05/2025",
        "timestamp_time": "20:10:00",
        "location": "Chennai",
        "device_id": "device005",
        "merchant_category": "Online Shopping"
    }
]

# Convert to DataFrame
test_df = pd.DataFrame(test_samples)

# Predict
predictions = model.predict(test_df)

# Output results
for i, pred in enumerate(predictions, 1):
    label = "Fraud" if pred == 1 else "Not Fraud"
    print(f"Transaction {i}: {label}")
