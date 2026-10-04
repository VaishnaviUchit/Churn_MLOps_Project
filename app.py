from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API - Day 3")

# Load model
model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {
        "message": "Churn Model API is LIVE!",
        "accuracy": "79.84%",
        "status": "Day 3 Complete"
    }

@app.get("/health")
def health():
    return {"status": "Model is loaded and ready"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0].max()
        return {
            "churn_prediction": int(prediction),
            "confidence": round(float(proba), 2),
            "result": "Will Churn" if prediction == 1 else "Will Not Churn"
        }
    except Exception as e:
        return {"error": str(e)}