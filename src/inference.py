import joblib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import os 


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load everything once ---
def load_pipeline():
    xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
    explainer = joblib.load(os.path.join(BASE_DIR, "models", "shap_explainer.pkl"))
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "data", "chroma_db"))
    collection = client.get_collection(name="credit_policy")
    return xgb_model, explainer, embedder, collection

def get_risk_tier(probability):
    if probability < 0.15:
        return "Low Risk"
    elif probability < 0.35:
        return "Moderate Risk"
    elif probability < 0.60:
        return "High Risk"
    else:
        return "Very High Risk"

def explain_applicant(index, X_data, shap_vals, model):
    applicant_shap = shap_vals[index]
    pred_proba = model.predict_proba(X_data.iloc[[index]])[0][1]
    shap_series = pd.Series(applicant_shap, index=X_data.columns)
    return {
        "default_probability": round(float(pred_proba), 4),
        "risk_tier": get_risk_tier(pred_proba),
        "top_risk_factors": shap_series.sort_values(ascending=False).head(5).to_dict(),
        "top_protective_factors": shap_series.sort_values().head(5).to_dict()
    }

def validate_explanation(explanation_text, explanation_data): #validate that the explanation text contains the predicted probability and risk tier"
    expected_pct = f"{explanation_data['default_probability']:.1%}"
    checks = {
        "probability_mentioned": expected_pct in explanation_text or expected_pct.rstrip('%') in explanation_text,
        "risk_tier_mentioned": explanation_data['risk_tier'].lower() in explanation_text.lower()
    }
    return checks

def build_query_from_shap(explanation_dict):
    top_risk = list(explanation_dict["top_risk_factors"].keys())
    return f"applicant denied primarily due to {', '.join(top_risk[:3])}"


def retrieve_policy_context(embedder, collection, query_text, n_results=3):
    query_embedding = embedder.encode([query_text])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    return results["documents"][0]


def generate_explanation(explanation_data, policy_chunks):
    prompt = f"""You are writing an adverse action explanation for a loan applicant, compliant with ECOA/Regulation B.

    IMPORTANT: The following number is the model's PREDICTED DEFAULT RISK PROBABILITY 
    for this applicant. It is NOT a credit utilization rate, interest rate, or any 
    other percentage. Do not reinterpret or relabel this number.

Applicant's predicted default probability: {explanation_data['default_probability']:.1%}

Top factors that increased risk:
{explanation_data['top_risk_factors']}

Top factors that decreased risk:
{explanation_data['top_protective_factors']}

Predicted default risk probability: {explanation_data['default_probability']:.1%}
Risk tier (already determined by our risk model, do not recalculate): {explanation_data['risk_tier']}

Relevant regulatory guidance:
{chr(10).join(policy_chunks)}



Write a clear, specific, plain-English explanation (3-5 sentences) of why this
applicant received this risk assessment. Do not use vague language like
"did not meet internal criteria."

Output ONLY the explanation text itself. Do not include any notes, meta-commentary, 
disclaimers about your own process, or explanations of how you approached the task.

"""
    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']