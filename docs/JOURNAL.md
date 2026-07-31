

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