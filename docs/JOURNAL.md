

dti means income to debt ratio 


## Leakage columns identified
Dropped payment/hardship/settlement columns since they're only populated 
after a loan is funded and resolved — including them would leak the 
outcome into the model's inputs. Kept application-time features: loan_amnt, 
int_rate, term, grade/sub_grade, annual_inc, dti, emp_length, home_ownership, 
purpose, delinq_2yrs, inq_last_6mths, open_acc, pub_rec, revol_bal, 
revol_util, total_acc, mort_acc, verification_status.


## Data cleaning stage
- Filtered to resolved loans only (~1.3M rows)
- Built binary target (default vs paid), ~80/20 split
- Dropped leakage/ID columns and sparse joint-application/late-collection columns
- Imputed remaining missing numeric columns with median, emp_length with "Unknown"
- Dropped 25 rows missing earliest_cr_line
- Final shape: [fill in final rows/cols]


## Feature engineering
- credit_history_years: issue_d minus earliest_cr_line, converted to years
- loan_to_income: loan_amnt / annual_inc, dropped 123 rows where ratio > 2 (unrealistic)


## Encoding complete
- Ordinal encoded: grade, sub_grade, emp_length, term (preserve natural order)
- One-hot encoded: home_ownership, verification_status, purpose, addr_state,
  initial_list_status, application_type, disbursement_method, pymnt_plan
- Final shape: (1,305,165, 139), 0 missing values

## Baseline model — Logistic Regression
- Scaled features with StandardScaler (fit on train only)
- class_weight='balanced' to address 80/20 class imbalance
- Results: precision (default)=0.33, recall (default)=0.66, ROC-AUC=0.72
- Trade-off: balanced weighting favors catching more defaulters at cost of more false positives
- Next: try XGBoost to see if a more complex model improves both precision and recall together

## Model comparison
- Logistic Regression: recall=0.66, precision=0.33, AUC=0.72
- XGBoost: recall=0.68, precision=0.34, AUC=0.74
- XGBoost modestly outperforms; remaining gap likely reflects genuine 
  unpredictability in default risk, not model limitations. Chose XGBoost 
  as production model.

## SHAP explainability
- Used TreeExplainer on XGBoost model
- Confirmed per-applicant explanations make sense (e.g. sub_grade, dti driving risk up;
  loan_to_income, shorter term driving risk down)
- Global summary plot confirms sub_grade is the single strongest predictor overall,
  consistent with it being LendingClub's own internal risk assessment
- Built explain_applicant() function returning: probability + top risk factors + 
  top protective factors, ready to be reused inside the API later


## RAG KNOWLEDGE BASE
- https://www.consumerfinance.gov/rules-policy/regulations/1002/9/
- https://www.consumerfinance.gov/rules-policy/regulations/1002/c/
- https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/
- Sourced real regulatory text from CFPB (ECOA/Regulation B, 12 CFR 1002.9)
- 3 policy documents, chunked by markdown headers into 9 chunks
- Embedded with sentence-transformers (all-MiniLM-L6-v2), stored in ChromaDB (persistent, local)
- Test query on a debt-to-income/credit-scoring scenario correctly retrieved all 3 
  most relevant chunks, confirming semantic search works (matched "credit scoring 
  model" to "Credit Scoring Systems" despite different wording)

  ## LLM explanation layer (Stage 6)
- Used Ollama (local, free) running llama3.1 for generation — no API key/cost required
- Refactored reusable logic (model/explainer/embedder loading, SHAP explanation,
  RAG retrieval, LLM generation) into src/inference.py for use across notebooks
  and the future API, instead of duplicating code per notebook
- Built full pipeline: applicant -> SHAP risk factors -> query built from top
  factors -> RAG retrieval of relevant policy chunks -> LLM generates final
  plain-English, ECOA/Reg B-grounded explanation
- Tested end-to-end on a real applicant: generated explanation correctly cited
  interest rate, dti, and account balance history as risk factors

### Issue found: geographic bias risk
- LLM-generated explanation cited "New York, a high-risk state" as a denial
  reason. Using location as a disclosed reason resembles redlining and would
  likely be non-compliant in a real institution, even though addr_state was
  a legitimate model feature.
- Follow-up: exclude location-based SHAP factors from what's passed to the
  LLM prompt, to prevent geography from appearing in generated explanations.

### Key learnings
- Notebooks don't share memory across files — each has its own kernel.
  Solved by persisting cleaned data (parquet) and trained artifacts (joblib/
  ChromaDB) to disk, and moving reusable logic into src/ modules.
- A technically accurate model-derived reason isn't automatically a
  compliant explanation — real deployment needs guardrails on what the LLM
  is allowed to surface, not just accurate SHAP values.