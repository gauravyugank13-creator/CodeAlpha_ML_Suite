"""
Model training and hyperparameter search coordination module for Task 1.
Manages fitting, randomized grid searches with cross-validation, and serialized dumps.
"""
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.config import Config

class Trainer:
    """
    Coordinates model training loops, hyperparameter optimization, and saves/loads checkpoints.
    """
    def __init__(self, config=Config):
        self.config = config

    def train_model(self, model, X_train, y_train) -> None:
        """
        Fits a classification model on the preprocessed training feature matrix.
        """
        model.fit(X_train, y_train)
        print(f"[Trainer] Model training completed successfully for classifier: {model.__class__.__name__}")

    def save_model(self, model, preprocessor, save_path: str) -> None:
        """
        Serializes the trained classifier model alongside the preprocessing pipeline 
        into a dictionary structure using joblib.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "model": model,
            "preprocessor": preprocessor
        }
        
        joblib.dump(checkpoint, save_path)
        print(f"[Trainer] Model checkpoint successfully saved to: {save_path}")

    def load_model(self, load_path: str) -> dict:
        """
        Loads and returns the serialized pipeline dictionary.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint file found at: {load_path}")
            
        checkpoint = joblib.load(load_path)
        print(f"[Trainer] Model checkpoint successfully loaded from: {load_path}")
        return checkpoint

    def hyperparameter_search(self, model, X_train, y_train, param_distributions: dict, 
                              n_iter: int = 20, n_splits: int = 5, 
                              random_state: int = Config.RANDOM_SEED) -> RandomizedSearchCV:
        """
        Executes a Stratified 5-Fold Randomized Cross-Validation search over a hyperparameter grid.
        Optimizes for the 'roc_auc' scoring metric.
        """
        print(f"[Trainer] Initializing RandomizedSearchCV (iterations: {n_iter}, CV splits: {n_splits}) ...")
        
        # Define cross-validation folds structure
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Configure search
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Run search
        search.fit(X_train, y_train)
        print(f"[Trainer] RandomizedSearchCV complete. Best CV ROC-AUC: {search.best_score_:.4f}")
        return search
