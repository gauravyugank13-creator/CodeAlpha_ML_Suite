"""
Model building and assembly module for Task 1: Credit Scoring Model.
Instantiates scikit-learn classification models configured for credit scoring.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import Config

class ModelBuilder:
    """
    Utility factory class to construct classification algorithm wrappers.
    """
    @staticmethod
    def get_logistic_regression(random_state: int = Config.RANDOM_SEED, max_iter: int = 5000, **kwargs) -> LogisticRegression:
        """
        Instantiate a Logistic Regression classifier wrapper.
        """
        return LogisticRegression(random_state=random_state, max_iter=max_iter, **kwargs)

    @staticmethod
    def get_decision_tree(random_state: int = Config.RANDOM_SEED, max_depth: int = None, **kwargs) -> DecisionTreeClassifier:
        """
        Instantiate a Decision Tree classifier wrapper.
        """
        return DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, **kwargs)

    @staticmethod
    def get_random_forest(random_state: int = Config.RANDOM_SEED, n_estimators: int = 200, **kwargs) -> RandomForestClassifier:
        """
        Instantiate a Random Forest ensemble classifier wrapper.
        """
        return RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, **kwargs)

    @staticmethod
    def build_logistic_regression(random_state: int = Config.RANDOM_SEED, max_iter: int = 5000, class_weight: str = "balanced", **kwargs) -> LogisticRegression:
        """
        Builds a Logistic Regression model configured for the Credit Scoring baseline.
        """
        return LogisticRegression(random_state=random_state, max_iter=max_iter, class_weight=class_weight, **kwargs)

    @staticmethod
    def build_decision_tree(random_state: int = Config.RANDOM_SEED, class_weight: str = "balanced", **kwargs) -> DecisionTreeClassifier:
        """
        Builds a Decision Tree model configured for the Credit Scoring baseline.
        """
        return DecisionTreeClassifier(random_state=random_state, class_weight=class_weight, **kwargs)

    @staticmethod
    def build_random_forest(n_estimators: int = 200, random_state: int = Config.RANDOM_SEED, class_weight: str = "balanced", **kwargs) -> RandomForestClassifier:
        """
        Builds a Random Forest model configured for the Credit Scoring baseline.
        """
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight, **kwargs)
