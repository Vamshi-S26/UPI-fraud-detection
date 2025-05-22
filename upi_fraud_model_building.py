import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("realistic_upi_fraud_dataset.csv")

# Features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Categorical and numerical features
categorical_features = [
    'sender_upi_id',
    'receiver_upi_id',
    'user_app_status',
    'bank_actual_status',
    'timestamp_date',
    'timestamp_time',
    'location',
    'device_id',
    'merchant_category'
]

numerical_features = ['amount']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Train-test split (60-40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# SMOTE for oversampling minority class (fraud)
smote = SMOTE(random_state=42)

# Pipeline with preprocessor, SMOTE, and weighted Random Forest
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "upi_fraud_detector_model_weighted_smote.pkl")
print("\nModel saved as 'upi_fraud_detector_model_weighted_smote.pkl'")


