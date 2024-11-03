from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from random import randrange
import pickle
import pandas as pd

App = FastAPI()

# Load the model and scalers
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

class Post(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@App.post("/predict/")
def predict(data: Post):
    try:
        # Convert data to dataframe
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])

        # Encode categorical features
        for column, label_encoder in label_encoders.items():
            if column in df.columns:
                df[column] = label_encoder.transform(df[column])

        # Scale numerical features
        df = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df)
        prediction = prediction[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


@App.get("/")
async def root():
    return {"message": "Hello World"}
