"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional

class CarListing(BaseModel):
    """
    Car listings submitted by users for price prediction
    Collection name: "carlisting"
    """
    brand: str = Field(..., description="Car brand (e.g., Maruti, Hyundai, Honda)")
    model: str = Field(..., description="Car model (e.g., Swift, i20, City)")
    year: int = Field(..., ge=1990, le=2100, description="Year of manufacture")
    km_driven: int = Field(..., ge=0, description="Total kilometers driven")
    fuel: str = Field(..., description="Fuel type (Petrol, Diesel, CNG, LPG, Electric)")
    seller_type: str = Field(..., description="Seller type (Individual, Dealer, Trustmark Dealer)")
    transmission: str = Field(..., description="Transmission (Manual, Automatic)")
    owner: str = Field(..., description="Owner type (First Owner, Second Owner, etc.)")
    mileage: float = Field(..., ge=0, description="Mileage in kmpl or km/kg")
    engine: int = Field(..., ge=50, description="Engine capacity in CC")
    max_power: float = Field(..., ge=10, description="Maximum power in bhp")
    seats: int = Field(..., ge=2, le=10, description="Number of seats")

class PredictionResult(BaseModel):
    predicted_price: float = Field(..., description="Predicted selling price in INR (lakhs)")

# Keep example schemas for reference (not used directly)
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = None
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str
    in_stock: bool = True
