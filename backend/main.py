# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
import joblib

app = FastAPI()

# Enable CORS for frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model, scaler, and feature names
MODEL_DIR = Path(__file__).parent / "model"

try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    print("✓ Scaler loaded successfully")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")
    scaler = None

try:
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    print(f"✓ Feature names loaded successfully: {len(feature_names)} features")
except Exception as e:
    print(f"✗ Error loading feature names: {e}")
    feature_names = None


# Define input model
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # example: "NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"


def preprocess_input(data: HouseFeatures) -> pd.DataFrame:
    """
    Preprocess input data to match the training pipeline:
    1. Apply log transformations
    2. One-hot encode ocean_proximity
    3. Create engineered features (bedroom_ratio, household_rooms)
    4. Ensure all features match training data
    """
    # Create DataFrame from input
    df = pd.DataFrame([{
        'longitude': data.longitude,
        'latitude': data.latitude,
        'housing_median_age': data.housing_median_age,
        'total_rooms': data.total_rooms,
        'total_bedrooms': data.total_bedrooms,
        'population': data.population,
        'households': data.households,
        'median_income': data.median_income,
        'ocean_proximity': data.ocean_proximity
    }])
    
    # Apply log transformations (same as training)
    df['total_rooms'] = np.log1p(df['total_rooms'] + 1)
    df['total_bedrooms'] = np.log1p(df['total_bedrooms'] + 1)
    df['population'] = np.log1p(df['population'] + 1)
    df['households'] = np.log1p(df['households'] + 1)
    
    # One-hot encode ocean_proximity
    df = pd.get_dummies(df, columns=['ocean_proximity'])
    
    # Convert boolean columns to int (if any)
    bool_cols = df.select_dtypes('bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Create engineered features
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    
    # Ensure all training features are present (add missing columns with 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    return df


@app.get("/")
def read_root():
    return {
        "message": "House Price Prediction API",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": feature_names is not None
    }


@app.post("/predict")
def predict_price(data: HouseFeatures):
    """
    Predict house price based on input features
    """
    # Check if model components are loaded
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model components not loaded properly. Please check server logs."
        )
    
    try:
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Scale the features
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        return {
            "predicted_price": round(float(prediction), 2),
            "input_features": data.dict(),
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/model-info")
def get_model_info():
    """
    Get information about the loaded model
    """
    if model is None or feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "ocean_proximity_categories": [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"
        ]
    }


# Example usage endpoint
@app.get("/example")
def get_example():
    """
    Get an example input for testing
    """
    return {
        "example_input": {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
    }