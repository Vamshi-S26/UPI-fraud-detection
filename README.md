
# 🛡️ **UPI Fraud Detection System**

---

## 🚀 **Overview**

This project is a realistic and intelligent fraud detection system designed to identify potentially fraudulent transactions in UPI (Unified Payments Interface) payments.  
It leverages a custom dataset that mimics real-world transactions and uses advanced machine learning techniques such as SMOTE and Random Forest to accurately detect anomalies.

The system is deployed as a **Flask web application** featuring a stylish, user-friendly interface for easy interaction.

---

## ✅ **Key Features**

### 📊 Realistic Dataset  
Includes sender/receiver UPI IDs, transaction amount, date & time, merchant category, user-visible payment status, bank confirmation status, and location.

### 🧠 Machine Learning Pipeline  
- Data preprocessing with `OneHotEncoder` and `StandardScaler`  
- Handling imbalanced classes with `SMOTE`  
- Classification model: `RandomForestClassifier` trained and saved as a `.pkl` file

### 🧪 Prediction Modes  
- **Single transaction prediction** via a web form  
- **Batch prediction** supported via a separate script

### 🌐 Flask Web Application  
- Responsive and stylish dashboard with welcome page  
- Animated fraud explanation messages  
- Image-enhanced frontend for better UX

---

## 🔄 **Live Deployment**

The app is hosted and running on **Render.com**.

### 🌍 **Live Demo:**  
🔗 Visit the live web app here:https://upi-fraud-detection-mvgu.onrender.com

## 🔒 **Security Note**

This project is intended **only for educational and demonstration purposes**.  
Real-world fraud detection systems must comply with strict financial regulations and security protocols before deployment.

---

## 🙌 **Acknowledgments**

This project was developed with guidance and code assistance from **ChatGPT by OpenAI**.
