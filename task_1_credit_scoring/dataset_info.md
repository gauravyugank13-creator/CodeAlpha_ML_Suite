# Dataset Profile: German Credit Dataset

This file documents the characteristics and schema details of the selected German Credit Dataset used for Task 1: Credit Scoring.

## Overview
- **Dataset Name:** Statlog German Credit Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) (CSV format sourced via ziadasal's mirror)
- **Direct Link:** `https://raw.githubusercontent.com/ziadasal/Credit-Risk-Assessment/main/german_credit_data.csv`
- **Target Column:** `Risk` (binary: `good`, `bad`)
- **Number of Samples:** 1000
- **Number of Features:** 9 (excluding row index and target column)

---

## Feature Descriptions

| Feature Column | Datatype | Type | Description |
|----------------|----------|------|-------------|
| **Age** | Integer | Numerical | Age of the applicant in years (ranges from 19 to 75). |
| **Sex** | Object (String) | Categorical | Gender of the applicant (`male`, `female`). |
| **Job** | Integer | Categorical | Professional skill level of the applicant:<br>0: Unskilled and non-resident<br>1: Unskilled and resident<br>2: Skilled<br>3: Highly skilled |
| **Housing** | Object (String) | Categorical | Type of housing the applicant resides in (`own`, `rent`, `free`). |
| **Saving accounts** | Object (String) | Categorical | Saving account balance class (`little`, `moderate`, `quite rich`, `rich`). Contains missing values. |
| **Checking account** | Object (String) | Categorical | Checking account balance class (`little`, `moderate`, `rich`). Contains missing values. |
| **Credit amount** | Integer | Numerical | The credit amount requested in DM (Deutsche Mark) (ranges from 250 to 18,424). |
| **Duration** | Integer | Numerical | The repayment period duration of the loan in months (ranges from 4 to 72). |
| **Purpose** | Object (String) | Categorical | Purpose of the credit request (e.g., `car`, `furniture/equipment`, `radio/TV`, `education`, `business`, `repairs`, `vacation/others`, `domestic appliances`). |

---

## Target Column: `Risk`
- Represents creditworthiness of the loan applicant.
- **Value Class Distribution:**
  - `good` (Low risk): 700 samples (70%)
  - `bad` (High risk / likely to default): 300 samples (30%)

---

## Expected Preprocessing Requirements
1. **Index Handling:** Drop the first column (`Unnamed: 0`) representing raw row IDs.
2. **Missing Values Imputation:**
   - `Saving accounts` is missing 183 entries. We can impute using a placeholder class like `'unknown'` or the mode value.
   - `Checking account` is missing 394 entries. We can impute using a placeholder class like `'unknown'` or the mode value.
3. **Categorical Encoding:**
   - Use `LabelEncoder` or binary mapping for the target variable: `good -> 0` (No Risk), `bad -> 1` (Risk / Default).
   - Use `OneHotEncoder` or ordinal encoding for categorical features (`Sex`, `Housing`, `Saving accounts`, `Checking account`, `Purpose`).
4. **Numerical Scaling:** Scale continuous features (`Age`, `Credit amount`, `Duration`) using standard Z-score scaling (`StandardScaler`) to prevent scaling discrepancies in distance-based algorithms.
5. **Class Imbalance Mitigation:** Since there is a 70/30 class imbalance, we may need to monitor class weights during model training.
