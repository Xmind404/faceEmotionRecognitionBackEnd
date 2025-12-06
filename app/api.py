# Author: Xmind404 Franciszek Karbowniczek

# Include necessary imports
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils import predict_emotion
import os

# Define API router
router = APIRouter()


# Endpoint definition
@router.get("/")
async def api_home():
    return {"message": "Hello Face Emotion Recognition API!"}


@router.post("/predict_emotion/")
async def predict(file: UploadFile = File(...)):
    # Create temp directory if not exists
    os.makedirs("temp", exist_ok=True)

    # Save uploaded file to temp folder
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    # Predict emotion using the utility function
    emotion = predict_emotion(temp_path)

    # Remove the temporary file
    os.remove(temp_path)

    return JSONResponse(content={"predicted_emotion": emotion})
