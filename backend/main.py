import sqlite3
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = "../models"
DB_PATH = "predictions.db"

metadata = joblib.load(f"{MODEL_DIR}/metadata.pkl")
WINNER_MODEL_PATH = f"{MODEL_DIR}/{'xgb_pipeline.pkl' if metadata['winner']=='xgboost' else 'logreg_pipeline.pkl'}"
pipeline = joblib.load(WINNER_MODEL_PATH)

preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_json TEXT NOT NULL,
            churn_probability REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerInput(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(examples=[0])
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[3])
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["No"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["No"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[85.5])
    TotalCharges: float = Field(examples=[256.5])


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["winner"]}

@app.post("/predict")
def predict(customer: CustomerInput):
    row = pd.DataFrame([customer.model_dump()])
    row = row[metadata["feature_columns"]]

    proba = float(pipeline.predict_proba(row)[0, 1])
    risk_level = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"

    top_reasons = []
    try:
        transformed = preprocessor.transform(row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model) if metadata["winner"] == "xgboost" else shap.LinearExplainer(model, transformed)
        shap_values = explainer.shap_values(transformed)
        shap_row = shap_values[0] if hasattr(shap_values, "ndim") and shap_values.ndim > 1 else shap_values

        contributions = list(zip(feature_names, shap_row))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_reasons = [
            {"feature": name.split("__")[-1], "impact": round(float(val), 3)}
            for name, val in contributions[:3]
        ]
    except Exception:
        top_reasons = [{"feature": "unavailable", "impact": 0.0}]

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, input_json, churn_probability, risk_level) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), json.dumps(customer.model_dump()), proba, risk_level),
    )
    conn.commit()
    conn.close()

    return {
        "churn_probability": round(proba, 4),
        "risk_level": risk_level,
        "top_reasons": top_reasons,
        "model_used": metadata["winner"],
    }

