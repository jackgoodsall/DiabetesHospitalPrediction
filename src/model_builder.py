from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

def build_estimator(name: str, **model_configs):
    """
    Helper function for a model builder factory, builds a classifer based on the name paramater.
    Config of the model defined in model_configs.
    """
    if name == "logreg":
        return LogisticRegression(model_configs)
    if name == "random_forest":
        return RandomForestClassifier(model_configs)
    if name == "xgboost":
        return XGBClassifier(model_configs)
    return ValueError(f"Unknown model: {name}")

