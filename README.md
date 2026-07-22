# CreditLens
A credit risk prediction model with LLM-generated, policy-grounded explanations.


Dataset: Lending Club loan data (2007–2015), ~2.26M records, 145 features per loan, sourced from Kaggle.

Target construction: Loans with unresolved status (Current, Late, In Grace Period) were excluded, since their final outcome is unknown. Remaining loans were labeled binary: 1 (default) for Charged Off / Default, 0 (paid) for Fully Paid. This yields ~1.3M usable records with roughly an 80/20 paid/default split.

Class imbalance: Defaults represent ~20% of resolved loans, addressed during modeling via class weighting and evaluation metrics beyond accuracy (precision, recall, ROC-AUC).