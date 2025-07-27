from models.evaluate import log_and_report

def train_model(model, model_name, X_train, y_train, X_val, y_val, params=None):
    model.fit(X_train, y_train)
    log_and_report(model, model_name, X_val, y_val, params)
