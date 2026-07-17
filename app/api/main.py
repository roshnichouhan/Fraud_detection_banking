from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model_loader import load_artifacts
from .predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML artifacts...")
    load_artifacts()
    print("API is ready.")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="Bank Fraud Detection API",
    description="API for predicting fraudulent banking transactions using a Machine Learning model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Home"])
def home():
    return {
        "message": "Welcome to the Bank Fraud Detection API",
        "status": "Running",
        "version": "1.0.0",
    }


app.include_router(
    predict_router,
    prefix="/api",
)