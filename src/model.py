import joblib

def save_model(model, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)