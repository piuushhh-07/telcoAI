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