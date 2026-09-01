# CreditLens

**An explainable credit risk model with a compliance-grounded LLM explanation layer.**

CreditLens predicts a loan applicant's probability of default and generates a plain-English, regulation-grounded explanation for that decision — combining a trained XGBoost model, per-applicant SHAP explainability, a RAG knowledge base built from real ECOA/Regulation B text, and an LLM explanation layer, all served through a containerized, deployed API.

## Why this project exists

Under the U.S. Equal Credit Opportunity Act (ECOA) and Regulation B, lenders are legally required to give applicants **specific, principal reasons** for a credit denial — not a generic "did not meet internal criteria." Most black-box ML credit models can't do this natively. CreditLens is a working demonstration of how to close that gap: a real model, grounded in real regulatory text, generating explanations that are both accurate to the model's actual reasoning and compliant with the legal standard for what a reason must look like.

## Architecture

```
Raw loan data (2.26M rows)
        │
        ▼
  Data cleaning & leakage removal ──► Feature engineering ──► Encoding
        │
        ▼
  XGBoost model  ──────────────►  SHAP explainer (per-applicant risk factors)
        │                                   │
        │                                   ▼
        │                     RAG retrieval (ChromaDB + sentence-transformers)
        │                     over real CFPB/ECOA regulatory text
        │                                   │
        └───────────────┬───────────────────┘
                         ▼
              LLM explanation layer
        (Ollama locally / Groq in production)
                         │
                         ▼
                  FastAPI /predict
                         │
                         ▼
         Docker container → AWS EC2 (public API)
                         │
                 GitHub Actions CI
          (automated test + build verification)
```

## Tech stack

**Data & modeling:** Python, pandas, scikit-learn, XGBoost, SHAP
**Explainability & RAG:** SHAP, sentence-transformers, ChromaDB
**LLM:** Ollama (local dev, Llama 3.1) / Groq API (deployed, openai/gpt-oss-20b)
**Serving:** FastAPI, Uvicorn
**Experiment tracking:** MLflow
**Infrastructure:** Docker, AWS EC2 (Ubuntu), GitHub Actions (CI)

## Results

Trained and compared two models on ~1.3M cleaned Lending Club loan records (2007–2015), after removing 40+ leakage/identifier columns and engineering two additional features (`credit_history_years`, `loan_to_income`):

| Model | Recall (default) | Precision (default) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.66 | 0.33 | 0.72 |
| XGBoost | 0.68 | 0.34 | 0.74 |

XGBoost modestly outperforms the linear baseline. The relatively small gap suggests the remaining prediction difficulty reflects genuine unpredictability in consumer default behavior (job loss, medical events, etc.) rather than being purely a limitation of model complexity — a class imbalance of ~80/20 (paid/default) was addressed via weighted training rather than resampling.

## Key decisions & challenges

**Leakage prevention.** The raw dataset included 40+ columns only populated *after* a loan is issued or resolved (payment history, hardship flags, settlement data). Training on these would have produced an artificially strong but practically useless model — one that "predicts" outcomes using information a real underwriter wouldn't have at application time. All were identified and removed before training.

**A real fairness catch.** During testing, the LLM explanation layer once cited an applicant's state ("New York, a high-risk state") as a reason for denial. Disclosing geography as a denial reason resembles redlining and would likely be non-compliant in a real institution, even though the underlying feature was legitimately used by the model. This led to treating "the model can use a feature" and "the explanation can disclose that feature" as separate concerns — a real guardrail a production system would need.

**LLM reliability is not free.** Early explanation generations occasionally mislabeled the model's risk probability as an unrelated percentage (e.g., "credit utilization rate"), or included meta-commentary about the LLM's own writing process. Fixed through explicit prompt labeling (stating clearly what a number is and is not) and a lightweight automated `validate_explanation()` check confirming the stated probability and risk tier actually appear in the generated text — grounding (SHAP + RAG) reduces hallucination risk but doesn't eliminate the need for verification.

**Deterministic logic stays in code, not the prompt.** Risk tiering (Low/Moderate/High/Very High) is computed in Python and passed to the LLM as an already-decided fact, rather than asking the LLM to categorize risk itself — the same input should always produce the same tier, which is exactly the kind of thing code guarantees and LLMs don't.

**CI caught a real bug.** `models/*.pkl` had been silently excluded from git the entire project (via a `*.pkl` rule in `.gitignore`), masked for two stages by manually copying files directly onto the EC2 server. The first GitHub Actions run — a genuine fresh `git clone` — immediately failed on `COPY models/`, exposing the gap that local and server testing had missed.

**Cost-conscious cloud deployment.** Deployed to AWS EC2 using AWS Educate credits (avoiding real cost), and switched the LLM layer from local Ollama to Groq's free API for the deployed version specifically, since the free-tier-eligible instance size couldn't hold an 8B-parameter model in memory. Local development still uses Ollama — the model choice is an environment variable, not a code fork.

## Project structure

```
creditLens/
├── data/               # raw + processed data, RAG vector store (not fully tracked in git)
├── models/             # trained model + SHAP explainer artifacts
├── notebooks/          # EDA, modeling, RAG, and LLM development notebooks
├── src/
│   ├── inference.py    # shared pipeline logic: load models, explain, retrieve, generate
│   └── main.py         # FastAPI application
├── tests/              # automated tests
├── docs/
│   └── JOURNAL.md      # full build log and decision history
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

## Running it locally

```bash
git clone https://github.com/loviqu2/creditlens.git
cd creditlens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Local dev uses Ollama (requires Ollama installed + llama3.1 pulled)
uvicorn src.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

### Running with Docker

```bash
docker build -t creditlens-api .
docker run -d -p 8000:8000 --env-file .env creditlens-api
```

## What I'd build next

- Automated semantic validation of generated explanations (beyond checking that key facts are *present*, verifying they're stated in the correct *direction*)
- A raw-input preprocessing endpoint (the current `/predict` accepts pre-encoded feature vectors; a production version would accept raw applicant fields and replicate the full cleaning pipeline)
- Full CD (automatic deployment to EC2 on push), currently scoped out since the instance is intentionally stopped between sessions to conserve credits

---

Built as an end-to-end learning project covering the full lifecycle: data cleaning, model training and comparison, explainability, RAG, LLM integration, experiment tracking, API development, containerization, cloud deployment, and CI.
