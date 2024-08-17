from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.responses import RedirectResponse

# Load your model
with open(r'C:\Users\takuma nishimoto\Desktop\Snow\tuned_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add other features as per your model

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.dict()])
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

