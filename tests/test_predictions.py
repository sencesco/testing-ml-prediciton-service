# tests/test_predictions.py
from fastapi.testclient import TestClient
from app.main import app # Import the FastAPI app instance
from app.models.schemas import InputFeatures # Import for type hints if needed

# Create a TestClient instance using our FastAPI app
client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Prediction service is running"}

def test_make_prediction_success():
    """Test the prediction endpoint with valid input."""
    # Define valid input data matching InputFeatures schema
    valid_input = {"feature1": 5.1, "feature2": 3.5} 

    # Make a POST request to the /predict/ endpoint
    response = client.post("/predict/", json=valid_input) 

    # Assert the request was successful (HTTP 200 OK)
    assert response.status_code == 200

    # Assert the response body structure matches PredictionOutput
    response_data = response.json()
    assert "prediction" in response_data

    # Optionally, assert the type of the prediction
    assert isinstance(response_data["prediction"], float) 

    # Note: Asserting the exact prediction value depends on your model 

def test_make_prediction_invalid_input_type():
    """Test the prediction endpoint with incorrect input data type."""
    # Send data where a feature is a string instead of a float
    invalid_input = {"feature1": "wrong_type", "feature2": 3.5}

    response = client.post("/predict/", json=invalid_input)

    # FastAPI/Pydantic automatically handles validation errors
    # Expect HTTP 422 Unprocessable Entity
    assert response.status_code == 422 

    # Check if the response body contains validation error details
    response_data = response.json()
    assert "detail" in response_data
    # You can add more specific checks on the error message if needed
    # e.g., assert "feature1" in str(response_data["detail"])

def test_make_prediction_missing_input_feature():
    """Test the prediction endpoint with missing input data."""
    # Send data missing 'feature2'
    missing_input = {"feature1": 5.1} 

    response = client.post("/predict/", json=missing_input)

    # Expect HTTP 422 Unprocessable Entity
    assert response.status_code == 422

    response_data = response.json()
    assert "detail" in response_data
    
    
    # e.g., assert "feature2" in str(response_data["detail"]) 
    # e.g., assert "field required" in str(response_data["detail"])
    # If we test now we will get an error because we have not added 
    # the validation for missing fields in the endpoint, 
    # we will add it in later in the real project.