# Face Emotion Recognition Backend

**Author:** Xmind404 Franciszek Karbowniczek

A simple backend for predicting facial emotions using PyTorch and FastAPI.  
The API allows uploading a grayscale 48x48 image and returns the predicted emotion.

---

## **Project Structure**

```
faceEmotionRecognitionBackEnd/
│── main.py                 # Entry point for FastAPI application
│── requirements.txt        # Python dependencies
│── train.py                # Training script for the model
│── teach.py                # Optional training/utility script
│
├── app/
│   │── __init__.py
│   │── api.py               # API endpoints
│   │── utils.py             # Model loading and prediction utilities
│
├── models/
│   └── emotion_cnn.pth     # Pre-trained PyTorch model
├── data/
│   └── (folders with emotions and 48x48 images)
├── temp/                    # Temporary folder for uploaded files
```

---

## **Requirements**

- Python 3.9+
- torch
- torchvision
- fastapi
- uvicorn
- pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Run the Backend**

Run FastAPI server:

```bash
python main.py
```

The server will start on:

```
http://127.0.0.1:8000
```

---

## **API Endpoints**

### 1. Root Endpoint

**URL:** `/`  
**Method:** GET

Returns a simple status JSON:

```json
{
  "status": "API running"
}
```

---

### 2. Predict Emotion

**URL:** `/api/predict_emotion/`  
**Method:** POST  
**Content-Type:** multipart/form-data  
**Parameter:** `file` (image file, 48x48 grayscale)

**Example using curl:**

```bash
curl -X POST "http://127.0.0.1:8000/api/predict_emotion/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@smiling-face.jpg;type=image/jpeg"
```

**Response:**

```json
{
  "predicted_emotion": "happy"
}
```

---

## **Model Training**

- Place your dataset in the `data/` folder. Each emotion should have its own folder with 48x48 images.
- Run `train.py` or `teach.py` to train your CNN model. The trained model will be saved to `models/emotion_cnn.pth`.

---

## **Notes**

- The backend is designed for learning purposes and simple projects.
- Uses GPU if available (PyTorch handles it automatically).
- Temporary files from uploads are saved in the `temp/` folder.
- You can explore the API using Swagger UI:

```
http://127.0.0.1:8000/docs
```

