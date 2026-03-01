from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


__MODELS__ = {

    "logistic_regression" : LogisticRegression,
    "random_forest" : RandomForestClassifier,
    "xgboost"  : XGBClassifier
 
}

def build_estimator(name: str, **model_configs):
    """
    Helper function for a model builder factory, builds a classifer based on the name paramater.
    Config of the model defined in model_configs.
    """
    model_type = __MODELS__.get(name.strip().lower(), False)
    if model_type == False:
        raise ValueError(f"Unknown model: {name}")
    return model_type(**model_configs)
