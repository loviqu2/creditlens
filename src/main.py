from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append('.')
from src.inference import load_pipeline, explain_applicant, build_query_from_shap, retrieve_policy_context, generate_explanation, validate_explanation
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="CreditLens API")

# Load everything ONCE when the server starts, not on every request
xgb_model, explainer, embedder, collection = load_pipeline()


class ApplicantFeatures(BaseModel):
    features: dict


@app.get("/")
def read_root():
    return {"message": "CreditLens API is running"}


@app.post("/predict")
def predict(applicant: ApplicantFeatures):
    X_input = pd.DataFrame([applicant.features])
    shap_values = explainer.shap_values(X_input)
    explanation_data = explain_applicant(0, X_input, shap_values, xgb_model)
    query = build_query_from_shap(explanation_data)
    policy_chunks = retrieve_policy_context(embedder, collection, query)
    final_explanation = generate_explanation(explanation_data, policy_chunks)
    checks = validate_explanation(final_explanation, explanation_data)
    
    return {
        "default_probability": explanation_data["default_probability"],
        "risk_tier": explanation_data["risk_tier"],
        "explanation": final_explanation,
        "validation_checks": checks
    }