"""
Data loading and validation module for Task 1: Credit Scoring Model.
Handles raw dataset ingestion (caching locally), schema verification, 
and computing statistics.
"""
import os
import requests
import pandas as pd
from src.config import Config

class DataLoader:
    """
    Ingests and validates the Credit Scoring tabular dataset.
    """
    def __init__(self, raw_data_dir: str = Config.RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir

    def load_dataset(self, filename: str = "german_credit_data.csv", fallback_url: str = Config.DATASET_URL) -> pd.DataFrame:
        """
        Loads the tabular credit scoring dataset. If the dataset does not exist 
        locally, it is downloaded from the fallback URL and cached locally.
        """
        local_path = os.path.join(self.raw_data_dir, filename)
        
        # Download data if missing locally
        if not os.path.exists(local_path):
            print(f"[DataLoader] Local file missing. Downloading from: {fallback_url} ...")
            os.makedirs(self.raw_data_dir, exist_ok=True)
            try:
                response = requests.get(fallback_url)
                response.raise_for_status()
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"[DataLoader] Successfully downloaded and saved to: {local_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download credit scoring dataset. Check your connection: {e}")
        
        # Ingest CSV and drop the index column (which is Unnamed: 0 in Kaggle export)
        df = pd.read_csv(local_path, index_col=0)
        return df

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Performs assertion checks to validate dataset integrity, column schema, 
        and values.
        """
        # 1. Row count validation
        assert len(df) == 1000, f"Validation Error: Expected 1000 rows, got {len(df)}"
        
        # 2. Schema check (numerical features)
        for col in Config.NUMERICAL_FEATURES:
            assert col in df.columns, f"Validation Error: Missing numerical feature '{col}'"
            
        # 3. Schema check (categorical features)
        for col in Config.CATEGORICAL_FEATURES:
            assert col in df.columns, f"Validation Error: Missing categorical feature '{col}'"
            
        # 4. Target column check
        assert Config.TARGET_COLUMN in df.columns, f"Validation Error: Target column '{Config.TARGET_COLUMN}' is missing"
        
        # 5. Target value verification
        unique_targets = set(df[Config.TARGET_COLUMN].unique())
        expected_targets = {"good", "bad"}
        assert unique_targets.issubset(expected_targets), f"Validation Error: Unexpected targets found: {unique_targets}"
        
        print("[DataLoader] [PASS] Dataset schema and integrity verification passed.")
        return True

    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        """
        Computes descriptive statistics, missing counts, category counts, 
        and duplicate flags.
        """
        num_cols = list(Config.NUMERICAL_FEATURES)
        cat_cols = list(Config.CATEGORICAL_FEATURES)
        
        summary = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum()),
            "class_distribution": df[Config.TARGET_COLUMN].value_counts(dropna=False).to_dict(),
            "datatype_map": {col: str(df[col].dtype) for col in df.columns}
        }
        return summary
