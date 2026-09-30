from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent


def load_artifacts():
    """Load the trained model pipeline and its metadata (suburbs, property types)."""
    model = joblib.load(BASE_DIR / "sydney_housing_model.pkl")
    meta = joblib.load(BASE_DIR / "model_metadata.pkl")
    return model, meta


def build_features(
    suburb,
    property_type,
    bedrooms,
    bathrooms,
    car_spaces,
    building_size_sqm,
    year_built,
    current_year=None,
):
    """Turn raw inputs into the exact single-row DataFrame the model expects."""
    year = current_year or datetime.now().year
    return pd.DataFrame(
        [
            {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "car_spaces": car_spaces,
                "building_size_sqm": building_size_sqm,
                "property_age": year - year_built,
                "total_rooms": bedrooms + bathrooms,
                "suburb": suburb,
                "property_type": property_type,
            }
        ]
    )


def predict_price(model, features):
    return float(model.predict(features)[0])
