# app/models/schemas.py
from pydantic import BaseModel

class InputFeatures(BaseModel):
    feature1: float
    feature2: float
    # ... other features

class PredictionOutput(BaseModel):
    prediction: float # Or appropriate type