# Churn Risk Predictor

A live tool that scores whether a telecom customer is likely to cancel — 
built end-to-end from raw data to a deployed product.

**Live demo:** https://telco-ai-azure.vercel.app/
**API docs:** https://telcoai-1.onrender.com/docs

## What it does
Given a customer's tenure, contract type, services, and billing details, 
it returns a churn probability and the top factors driving that score 
(via SHAP), so a retention team could act on it before a renewal call — 
not just a number with no explanation.

## Approach
- Trained and compared Logistic Regression (baseline) vs XGBoost on the 
  IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn rate)
- Used ROC-AUC for model selection instead of accuracy, since accuracy 
  is misleading on imbalanced data — a model that always predicts "no 
  churn" scores ~73% accuracy while being useless
- XGBoost won narrowly (AUC 0.843 vs 0.842) — handled with 
  `scale_pos_weight` for the class imbalance
- FastAPI backend serves predictions, logs every request to SQLite, and 
  exposes a `/stats` endpoint for monitoring — not just a notebook 
  wrapped in an API

## Stack
Python, scikit-learn, XGBoost, SHAP, FastAPI, SQLite, vanilla HTML/JS 
frontend, deployed on Render + Vercel.

## What I'd improve with more time
- Move logging from SQLite to a hosted database for production durability
- A/B test the "would intervening on high-risk customers actually 
  reduce churn" question, not just predict the risk
