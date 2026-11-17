from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# Load the trained scikit-learn pipeline (preprocessing + RandomForest)
model = joblib.load(BASE_DIR / "Mental_health_model.pkl")

app = FastAPI(title="Mental Health Prediction API")

# Lets the UI work even if you open index.html from a different origin/port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Countries the model was trained on; everything else is grouped as "Other"
top_country = ['Other', 'India', 'USA', 'Canada', 'Australia',
               'UK', 'Germany', 'Mexico', 'Turkey', 'France']


class StudentData(BaseModel):
    Age: int = Field(..., ge=10, le=95)
    Gender: Literal['Male', 'Female']
    Country: str
    Academic_Level: Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
        'Twitter', 'YouTube', 'TikTok', 'LINE',
        'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ]
    Purpose_Of_Use: Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24)
    Daily_Unlocks: int = Field(..., ge=0)
    Study_Hours: float = Field(..., ge=0, le=24)
    Physical_Activity_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)
    # FIX: the model's OrdinalEncoder expects these 4 labels, not a 1-10 number
    Stress_Level: Literal['Low', 'Medium', 'High', 'Very High']


class PredictionResponse(BaseModel):
    # FIX: was `= float` (assignment); it must be a type annotation `: float`.
    # Also fixed the spelling "predicated" -> "predicted".
    predicted_mental_health_score: float


@app.get("/api")
def greet():
    return {"Welcome": "Mental Health Prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def home():
    # Serves the UI at http://127.0.0.1:8000/
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    country_group = data.Country if data.Country in top_country else "Other"

    # Columns must match what the pipeline was trained on.
    # FIX: removed the raw 'Country' column (the model only uses 'Group_Country').
    input_data = pd.DataFrame([{
        'Age': data.Age,
        'Gender': data.Gender,
        'Academic_Level': data.Academic_Level,
        'Most_Used_Platform': data.Most_Used_Platform,
        'Purpose_Of_Use': data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks': data.Daily_Unlocks,
        'Study_Hours': data.Study_Hours,
        'Physical_Activity_Hours': data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
        'Stress_Level': data.Stress_Level,
        'Group_Country': country_group,
    }])

    prediction = model.predict(input_data)[0]

    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )