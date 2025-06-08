from pydantic import BaseModel
import joblib
import numpy as np

try:
    model = joblib.load("trained_model.pkl")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def predict_productivity(idle_men, style_changes, num_workers, month):
    if model is None:
        raise ValueError("Model belum dimuat.")
    
    X = np.array([[idle_men, style_changes, num_workers, month]])
    return model.predict(X)[0]

class DermaInput(BaseModel):
    date: str
    quarter: int
    department: str
    day: str
    team: int
    targeted_productivity: float
    smv: float
    wip: int
    over_time: int
    incentive: float
    idle_time: float
    idle_men: int
    no_of_style_change: int
    no_of_workers: int

class PredictionResult(BaseModel):
    predicted_actual_productivity: float
