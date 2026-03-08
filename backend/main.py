from dotenv import load_dotenv
load_dotenv()
# Load environment variables from .env file (for local development)

from fastapi import FastAPI, HTTPException
# FastAPI is a web framework for building APIs in Python.
# HTTPException is used to handle errors in API requests.
from fastapi.middleware.cors import CORSMiddleware
# CORS lets your frontend talk to your backend when they are on different domains.
from pydantic import BaseModel
# Pydantic is used for data validation. It ensures incoming JSON looks like {"text": "some string"}.
from transformers import pipeline
# Transformers is a library for natural language processing. We use it to load our ML model.
import os
# os is used to read environment variables

app = FastAPI()  # Create an instance of the FastAPI application

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from Hugging Face Hub
# This downloads the model when the server starts
MODEL = "YunaTakele/distilbert-fakenews"
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# Define the shape of incoming requests
class PredictRequest(BaseModel):
    text: str

# Root endpoint — just confirms the API is running
@app.get("/")
def root():
    return {"message": "API is running"}

# Main prediction endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = clf(req.text)[0]
        raw_label = result['label']

        # Map LABEL_0 / LABEL_1 to human-readable values
        if raw_label == "LABEL_0":
            label = "FAKE"
        else:
            label = "REAL"
        score = float(result["score"])

        return {
            "label": label,
            "confidence": score,
            "explanation": f"The model predicts this text is {label} with confidence {score:.3f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))