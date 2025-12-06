# Author: Xmind404 Franciszek Karbowniczek

# Include necessary imports
from fastapi import FastAPI
from app.api import router

# Initialize FastAPI application
app = FastAPI(title="Face Emotion Recognition API")

# Include API router with prefix
app.include_router(router, prefix="/api")

# Root endpoint
@app.get("/")
def root():
    return {"status": "API running"}

if __name__ == "__main__":

    # Run the application using Uvicorn
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
