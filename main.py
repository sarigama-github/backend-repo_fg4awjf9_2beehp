import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from database import create_document
from schemas import CarListing, PredictionResult
import json
import math

MODEL_PATH = "car_price_model.pkl"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API is running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

class TrainResponse(BaseModel):
    r2: float
    rmse: float
    n_samples: int
    model_path: str
    backend_model: str

# Simple synthetic dataset + heuristic model (no heavy deps)

def _brands():
    return ["Maruti", "Hyundai", "Honda", "Mahindra", "Tata", "Toyota", "Renault"]

def _models():
    return ["Swift", "i20", "City", "Bolero", "Nexon", "Innova", "Kwid"]

def _fuels():
    return ["Petrol", "Diesel", "CNG"]

def _sellers():
    return ["Individual", "Dealer", "Trustmark Dealer"]

def _transmissions():
    return ["Manual", "Automatic"]

def _owners():
    return ["First Owner", "Second Owner", "Third Owner"]


def heuristic_price(sample: dict) -> float:
    brand = sample["brand"]
    model = sample["model"]
    year = int(sample["year"])
    km = int(sample["km_driven"])
    fuel = sample["fuel"]
    seller = sample["seller_type"]
    transmission = sample["transmission"]
    owner = sample["owner"]
    mileage = float(sample["mileage"])
    engine = int(sample["engine"])
    power = float(sample["max_power"])
    seats = int(sample["seats"])

    brands = _brands()
    models = _models()
    owners_list = _owners()

    base = 2.5
    brand_bonus = max(0, brands.index(brand)) * 0.2 if brand in brands else 0.1
    model_bonus = max(0, models.index(model)) * 0.1 if model in models else 0.05
    age_penalty = max(0, (2024 - year)) * 0.15
    km_penalty = max(0, km) / 100000 * 0.8
    fuel_bonus = 0.3 if fuel == "Diesel" else (0.2 if fuel == "CNG" else 0)
    transmission_bonus = 0.4 if transmission == "Automatic" else 0
    owner_penalty = (owners_list.index(owner) if owner in owners_list else 1) * 0.3
    engine_bonus = max(0, (engine - 800)) / 1000 * 0.6
    power_bonus = max(0, (power - 60)) / 40 * 0.5
    seat_bonus = 0.2 if seats >= 6 else 0
    eff_bonus = max(0, (mileage - 15)) * 0.03

    price = base + brand_bonus + model_bonus - age_penalty - km_penalty + fuel_bonus + transmission_bonus - owner_penalty + engine_bonus + power_bonus + seat_bonus + eff_bonus
    return round(max(0.5, price), 2)


def build_synthetic(n: int = 400) -> List[dict]:
    # Deterministic pseudo-random without numpy
    import random
    random.seed(42)
    data = []
    for _ in range(n):
        s = {
            "brand": random.choice(_brands()),
            "model": random.choice(_models()),
            "year": random.randint(2008, 2023),
            "km_driven": random.randint(1000, 200000),
            "fuel": random.choice(_fuels()),
            "seller_type": random.choice(_sellers()),
            "transmission": random.choice(_transmissions()),
            "owner": random.choice(_owners()),
            "mileage": round(random.uniform(14, 24), 1),
            "engine": random.randint(800, 2500),
            "max_power": round(random.uniform(60, 120), 1),
            "seats": random.choice([4, 5, 7])
        }
        s["selling_price"] = heuristic_price(s) + round(random.uniform(-0.3, 0.3), 2)
        data.append(s)
    return data


def save_model(metadata: dict):
    # Save simple metadata to simulate a trained model
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_model() -> dict:
    if not os.path.exists(MODEL_PATH):
        return {}
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.post("/api/train", response_model=TrainResponse)
def api_train():
    data = build_synthetic(800)
    # Evaluate heuristic vs synthetic noisy labels
    preds = [heuristic_price({k: v for k, v in row.items() if k != "selling_price"}) for row in data]
    y = [row["selling_price"] for row in data]

    # Compute simple metrics
    n = len(y)
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
    rmse = math.sqrt(ss_res / n)

    save_model({"type": "heuristic", "trained_on": n})

    return TrainResponse(r2=float(round(r2, 4)), rmse=float(round(rmse, 4)), n_samples=n, model_path=MODEL_PATH, backend_model="heuristic")


@app.post("/api/predict", response_model=PredictionResult)
def predict_price(car: CarListing):
    # store input to DB if available
    try:
        create_document("carlisting", car)
    except Exception:
        pass

    # if we had a heavier ML model, we'd attempt to load it here.
    _ = load_model()
    pred = heuristic_price(car.model_dump())
    return PredictionResult(predicted_price=pred)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
