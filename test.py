import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
logreg_pipeline.fit(X_train, y_train)
logreg_preds = logreg_pipeline.predict(X_test)
logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression (baseline) ===")
print(classification_report(y_test, logreg_preds, target_names=["Stay", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, logreg_proba):.3f}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )),
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_test)
xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

print("\n=== XGBoost (production candidate) ===")
print(classification_report(y_test, xgb_preds, target_names=["Stay", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.3f}")

logreg_auc = roc_auc_score(y_test, logreg_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)

winner_name = "xgboost" if xgb_auc > logreg_auc else "logistic_regression"
winner_pipeline = xgb_pipeline if xgb_auc > logreg_auc else logreg_pipeline

print(f"\n>>> Winner: {winner_name} (AUC {max(xgb_auc, logreg_auc):.3f} vs {min(xgb_auc, logreg_auc):.3f})")

joblib.dump(logreg_pipeline, "models/logreg_pipeline.pkl")
joblib.dump(xgb_pipeline, "models/xgb_pipeline.pkl")

metadata = {
    "winner": winner_name,
    "logreg_auc": float(logreg_auc),
    "xgb_auc": float(xgb_auc),
    "feature_columns": list(X.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}
joblib.dump(metadata, "models/metadata.pkl")

print("\nSaved: logreg_pipeline.pkl, xgb_pipeline.pkl, metadata.pkl")