import uuid
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import logging
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.logging_conf import (
    configure_logging,
    correlation_id_context,
)

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trip Duration Prediction API",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:

    logger.error(
        "Validation rejection: %s",
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
) -> Response:

    correlation_id = str(uuid.uuid4())

    token = correlation_id_context.set(correlation_id)

    try:
        logger.info(
            "Request received: %s %s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            "Request completed: status=%s",
            response.status_code,
        )

        return response

    finally:
        correlation_id_context.reset(token)


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
