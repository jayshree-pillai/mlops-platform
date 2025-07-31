import shap

def run_shap(model, features):
    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    return shap_values
