from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    PU_DO: str = Field(
        ...,
        description="Pickup-dropoff location pair.",
        examples=["1_2"],
    )

    trip_distance: float = Field(
        ...,
        gt=0,
        lt=200,
        description="Trip distance in miles.",
        examples=[5.0],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "PU_DO": "1_2",
                "trip_distance": 5.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    trips: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        description="List of trips to predict.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "trips": [
                    {
                        "PU_DO": "1_2",
                        "trip_distance": 5.0,
                    },
                    {
                        "PU_DO": "2_3",
                        "trip_distance": 10.0,
                    },
                ]
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float
