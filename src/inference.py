import joblib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# --- Load everything once ---
def load_pipeline():
    xgb_model = joblib.load("../models/xgboost_model.pkl")
    explainer = joblib.load("../models/shap_explainer.pkl")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection(name="credit_policy")
    return xgb_model, explainer, embedder, collection


def explain_applicant(index, X_data, shap_vals, model):
    applicant_shap = shap_vals[index]
    pred_proba = model.predict_proba(X_data.iloc[[index]])[0][1]
    shap_series = pd.Series(applicant_shap, index=X_data.columns)
    return {
        "default_probability": round(float(pred_proba), 4),
        "top_risk_factors": shap_series.sort_values(ascending=False).head(5).to_dict(),
        "top_protective_factors": shap_series.sort_values().head(5).to_dict()
    }


def build_query_from_shap(explanation_dict):
    top_risk = list(explanation_dict["top_risk_factors"].keys())
    return f"applicant denied primarily due to {', '.join(top_risk[:3])}"


def retrieve_policy_context(embedder, collection, query_text, n_results=3):
    query_embedding = embedder.encode([query_text])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    return results["documents"][0]


def generate_explanation(explanation_data, policy_chunks):
    prompt = f"""You are writing an adverse action explanation for a loan applicant, compliant with ECOA/Regulation B.

Applicant's predicted default probability: {explanation_data['default_probability']:.1%}

Top factors that increased risk:
{explanation_data['top_risk_factors']}

Top factors that decreased risk:
{explanation_data['top_protective_factors']}

Relevant regulatory guidance:
{chr(10).join(policy_chunks)}

Write a clear, specific, plain-English explanation (3-5 sentences) of why this
applicant received this risk assessment. Do not use vague language like
"did not meet internal criteria."
"""
    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']