"""
Model builder and pipeline factory module for Task 2: Emotion Recognition.
Provides methods to assemble standard classification pipelines (SVM, Random Forest, MLP).
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.config import Config

class ModelBuilder:
    """
    Factory to construct standard Scikit-Learn classification pipelines.
    """
    
    @staticmethod
    def build_svm(C: float = 1.0, kernel: str = "rbf", 
                  random_state: int = Config.RANDOM_SEED) -> Pipeline:
        """
        Assembles a Support Vector Machine pipeline (StandardScaler + SVC).
        """
        return Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel=kernel, probability=True, class_weight="balanced", random_state=random_state))
        ])

    @staticmethod
    def build_random_forest(n_estimators: int = 200, max_depth: int = None, 
                            random_state: int = Config.RANDOM_SEED) -> Pipeline:
        """
        Assembles a Random Forest pipeline (StandardScaler + RandomForestClassifier).
        """
        return Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                class_weight="balanced", 
                random_state=random_state
            ))
        ])

    @staticmethod
    def build_mlp(hidden_layer_sizes: tuple = (256, 128), max_iter: int = 500, 
                  random_state: int = Config.RANDOM_SEED) -> Pipeline:
        """
        Assembles a Multi-Layer Perceptron neural network pipeline (StandardScaler + MLPClassifier).
        """
        return Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes, 
                max_iter=max_iter, 
                random_state=random_state
            ))
        ])
