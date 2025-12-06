# Author: Xmind404 Franciszek Karbowniczek

# Include necessary imports
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils import predict_emotion
import os

# Define API router
router = APIRouter()


# Endpoint definition
@router.post('/')
async def root():
    return {"message": "Hello Face Emotion Recognition API!"}

@router.post('/predict_emotion/')
async def predict_emotion_api(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())

    # Predict emotion using the utility function
    emotion = predict_emotion(temp_file)

    return JSONResponse(content={"predicted_emotion": emotion})
