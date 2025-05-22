import pandas as pd
import joblib

def batch_predict(input_csv, output_csv, model_path):
    # Load the batch transactions CSV file
    df = pd.read_csv(input_csv)
    
    # Load your trained model pipeline
    model = joblib.load(model_path)
    
    # Predict fraud labels and probabilities
    df['prediction'] = model.predict(df)
    df['fraud_probability'] = model.predict_proba(df)[:, 1]
    
    # Map numeric prediction to human-readable labels
    df['prediction_label'] = df['prediction'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    # Save the dataframe with predictions to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Batch prediction completed. Results saved to '{output_csv}'.")

if __name__ == "__main__":
    # File paths
    input_csv = "batch_transactions.csv"  # your batch input file
    output_csv = "batch_predictions.csv" # output file with predictions
    model_path = "upi_fraud_detector_model_weighted_smote.pkl"  # your saved model

    batch_predict(input_csv, output_csv, model_path)
