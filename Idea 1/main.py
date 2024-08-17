from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your model
model = joblib.load('tuned_model.pkl')  # 現在のディレクトリにある場合


# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add other features as needed

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Prepare the input data for prediction
    input_data = [[data.feature1, data.feature2, data.feature3]]  # Add other features accordingly
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return {"prediction": prediction[0]}

# To run the app, use the following command in the terminal:
# uvicorn main:app --reload

