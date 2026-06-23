"""
Feature engineering module for Task 1: Credit Scoring Model.
Implements a reusable scikit-learn transformer to construct new mathematical features 
from numerical applicant attributes.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to build features for tabular credit scoring data.
    Ensures feature engineering steps are cleanly integrated in scikit-learn pipelines.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        """
        No-op fit method. Returns self since no parameters are learned.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs engineered columns and returns the enriched DataFrame.
        """
        # Ensure deep copy to avoid modifying original datasets
        X_out = X.copy()
        
        # 1. credit_per_month: Repayment rate proxy (Credit amount requested / loan repayment duration)
        # Adding a small constant in denominator to prevent DivisionByZero warnings
        X_out["credit_per_month"] = X_out["Credit amount"] / (X_out["Duration"] + 1e-5)
        
        # 2. age_group: Binning Age into demographic categories
        def get_age_group(age: int) -> str:
            if age <= 25:
                return "young"
            elif age <= 40:
                return "adult"
            elif age <= 60:
                return "middle_age"
            else:
                return "senior"
                
        X_out["age_group"] = X_out["Age"].apply(get_age_group)
        
        # 3. credit_to_age_ratio: Debt profile to maturity ratio (Credit amount requested / age of applicant)
        X_out["credit_to_age_ratio"] = X_out["Credit amount"] / (X_out["Age"] + 1e-5)
        
        return X_out
