"""
Prediction and inference module for Task 1: Credit Scoring Model.
Loads serialized models and preprocessors to perform inference on applicant records.
"""
import joblib
from src.config import Config

class CreditPredictor:
    """
    Loads saved pipelines and outputs risk predictions.
    """
    def __init__(self, model_path: str = None):
        import os
        if model_path is None:
            self.model_path = os.path.join(Config.MODEL_DIR, "final_credit_scoring_model.joblib")
        else:
            self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.is_loaded = False

    def load_pipeline(self) -> None:
        """
        Loads the trained classification model and preprocessing pipeline.
        """
        checkpoint = joblib.load(self.model_path)
        self.model = checkpoint["model"]
        self.preprocessor = checkpoint["preprocessor"]
        self.is_loaded = True

    def predict_risk_score(self, X_input) -> float:
        """
        Predict the risk probability score for default status (class 1: bad).
        """
        if not self.is_loaded:
            self.load_pipeline()
        
        import pandas as pd
        if isinstance(X_input, dict):
            X_input = pd.DataFrame([X_input])
            
        X_trans = self.preprocessor.transform(X_input)
        probs = self.model.predict_proba(X_trans)
        return float(probs[0, 1])

    def predict_decision(self, X_input) -> int:
        """
        Predict the binary decision outcome (0 = Low Risk / Good, 1 = High Risk / Bad).
        """
        if not self.is_loaded:
            self.load_pipeline()
            
        import pandas as pd
        if isinstance(X_input, dict):
            X_input = pd.DataFrame([X_input])
            
        X_trans = self.preprocessor.transform(X_input)
        preds = self.model.predict(X_trans)
        return int(preds[0])

