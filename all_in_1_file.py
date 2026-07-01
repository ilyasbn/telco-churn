"""Quick script to run the Telco churn pipeline.

The reusable code lives in src/. This file only gives a short, readable way to
train the model and try one prediction.
"""

from src.predict import predict_one_customer
from src.train import train_model


def main() -> None:
    result = train_model()
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Churn recall: {result['recall']:.3f}")
    print(f"ROC AUC: {result['roc_auc']:.3f}")

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65,
    }
    print(predict_one_customer(sample_customer))


if __name__ == "__main__":
    main()
