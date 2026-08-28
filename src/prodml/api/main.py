# src/prodml/api/main.py


from fastapi import FastAPI
from pydantic import BaseModel

from prodml.config import settings
from prodml.predict import DurationPredictor


app = FastAPI(
    title="Trip Duration Prediction API",
    version="0.1.0",
)


class TripFeatures(BaseModel):
    PU_DO: str
    trip_distance: float


class BatchPredictionRequest(BaseModel):
    trips: list[TripFeatures]


predictor = DurationPredictor.load(settings.model_path)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Trip Duration Prediction API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    trip: TripFeatures,
) -> dict[str, float]:
    prediction = predictor.predict_one(trip.model_dump())

    return {"predicted_duration_minutes": prediction}


@app.post("/predict-batch")
async def predict_batch(
    request: BatchPredictionRequest,
) -> dict[str, list[float]]:
    predictions = predictor.predict_batch([trip.model_dump() for trip in request.trips])

    return {"predictions": predictions}
