from fastapi.middleware.cors import CORSMiddleware
# CORS lets your frontend talk to your backend when they are on different domains.
from fastapi import FastAPI, HTTPException
# FastAPI is a web framework for building APIs in Python.
# HTTPException is used to handle errors in API requests.
from pydantic import BaseModel
# Pydantic is used for data validation. It ensures incoming JSON looks like {"text": "some string"}.
from transformers import pipeline
# Transformers is a library for natural language processing. We use it to load our ML model.
from mangum import Mangum 
# Mangum is an adapter that allows FastAPI to run on AWS Lambda, making it easy to deploy our API in a serverless environment.

app = FastAPI()  # Create an instance of the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the ML model once at startup
MODEL = "YunaTakele/distilbert-fakenews"  # Path to the saved model
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# Define the data model for the prediction request
class PredictRequest(BaseModel):
    text: str

@app.get("/")  # Root endpoint
def root():
    return {"message": "API is running"}  # Must return a dictionary

@app.post("/predict")  # Endpoint for making predictions
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


handler = Mangum(app)  # Wrap the FastAPI app with Mangum for AWS Lambda compatibility