# Author: Xmind404 Franciszek Karbowniczek

# Include necessary imports
from fastapi import FastAPI
from app.api import router

# Initialize FastAPI application
app = FastAPI(title="Face Emotion Recognition API")
app.include_router(router)


if __name__ == "__main__":

    # Run the application using Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)