# Integrating ML model with FAST API
import os
import joblib
import pandas as pd
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException


import logging
# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for the FastAPI application."""
    # Load ML model
    try:
        logger.info("Loading ML model...")
        model_path = os.path.join(current_dir, 'model.pkl')
        app.state.model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        yield

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")

app = FastAPI(lifespan=lifespan)

class InputData(BaseModel):
    # Define the input data structure
    Weight : float
    ProductVisibility : float
    MRP : float
    EstablishmentYear : int
    ProductID : str
    FatContent : str
    ProductType : str
    OutletID : str
    OutletSize : str
    LocationType : str
    OutletType : str

@app.get("/")
async def root():
    return {"Welcome to our ML application."}

@app.post('/predict')
async def predict(data: InputData):
    """ Predict the OutletSales based on the input data """
    try:
        input_df = pd.DataFrame([data.model_dump()])
        logger.info(f"Input features DataFrame: {input_df}")
        model = app.state.model
        prediction = model.predict(input_df)
        logger.info(f"Prediction: {prediction}")

        return {"predicted OutletSales": float(prediction[0])}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please check the input data.")

if __name__ == '__main__':
    uvicorn.run(app, port=8001)