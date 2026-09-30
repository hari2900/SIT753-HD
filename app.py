import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

from predictor import build_features, load_artifacts, predict_price

model, meta = load_artifacts()

app = FastAPI(title="Sydney Housing Price API", version="1.0.0")

# ---------------- Prometheus metrics ----------------
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter("predictions_total", "Total successful predictions")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed predictions")


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    path = request.url.path
    REQUEST_LATENCY.labels(endpoint=path).observe(time.time() - start)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=path, status=response.status_code
    ).inc()
    return response


# ---------------- Request schema (ranges match your Streamlit app) ----------------
class PropertyInput(BaseModel):
    suburb: str
    property_type: str
    bedrooms: int = Field(ge=0, le=10)
    bathrooms: int = Field(ge=0, le=10)
    car_spaces: int = Field(ge=0, le=10)
    building_size_sqm: float = Field(ge=10, le=1500)
    year_built: int = Field(ge=1850)

    @field_validator("year_built")
    @classmethod
    def year_not_in_future(cls, v):
        if v > datetime.now().year:
            raise ValueError("year_built cannot be in the future")
        return v


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/options")
def options():
    return {"suburbs": list(meta["suburbs"]), "property_types": list(meta["property_types"])}


@app.post("/predict")
def predict(data: PropertyInput):
    if data.suburb not in meta["suburbs"]:
        PREDICTION_ERRORS.inc()
        raise HTTPException(422, f"suburb must be one of {list(meta['suburbs'])}")
    if data.property_type not in meta["property_types"]:
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            422, f"property_type must be one of {list(meta['property_types'])}"
        )
    try:
        features = build_features(**data.model_dump())
        price = predict_price(model, features)
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(500, f"Prediction failed: {exc}")
    PREDICTION_COUNT.inc()
    return {"predicted_price": round(price, 2), "currency": "AUD"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
