# app/routers/predictions.py
from fastapi import APIRouter
import joblib 
from app.models.schemas import InputFeatures, PredictionOutput 

# Assume model path is configured or known
MODEL_PATH = "model.joblib" 
model = joblib.load(MODEL_PATH)

router = APIRouter(
    prefix="/predict", # All routes in this router will start with /predict
    tags=["predictions"] # Group endpoints in API docs
)

@router.post("/", response_model=PredictionOutput) # Path is now relative to prefix
async def make_prediction(input_data: InputFeatures):
    """
    Accepts input features and returns a prediction.
    (Logic remains the same as before)
    """
    features = [[input_data.feature1, input_data.feature2]]
    prediction_result = model.predict(features)
    return PredictionOutput(prediction=prediction_result[0])

# You could add other prediction-related endpoints here later,
# e.g., @router.post("/batch", ...) 
