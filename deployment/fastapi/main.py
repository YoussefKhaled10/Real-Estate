from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from src.pipelines.inference_pipeline import predict_price

app = FastAPI(
    title="Real Estate Price Prediction API",
    version="1.0"
)

# =========================
# Input Schema
# =========================

class PropertyInput(BaseModel):
    building_construction_year: int
    building_total_floors: int
    apartment_floor: int
    apartment_rooms: int
    apartment_bedrooms: int
    apartment_bathrooms: int
    apartment_total_area: float
    apartment_living_area: float
    country: str
    location: str

    @field_validator("building_construction_year")
    @classmethod
    def validate_year(cls, v):
        if v < 1800:
            raise ValueError("Construction year is unrealistically old")
        if v > 2025:
            raise ValueError("Construction year must be 2025 or earlier")
        return v

    @field_validator("building_total_floors")
    @classmethod
    def validate_total_floors(cls, v):
        if v < 1:
            raise ValueError("Building must have at least 1 floor")
        if v > 60:
            raise ValueError("Building with more than 60 floors is not realistic")
        return v

    @field_validator("apartment_floor")
    @classmethod
    def validate_apartment_floor(cls, v, info):
        total = info.data.get("building_total_floors")
        if v < 0:
            raise ValueError("Apartment floor cannot be negative")
        if total is not None and v > total:
            raise ValueError("Apartment floor cannot exceed building total floors")
        return v

    @field_validator("apartment_rooms")
    @classmethod
    def validate_rooms(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Number of rooms must be between 0 and 10")
        return v

    @field_validator("apartment_bedrooms")
    @classmethod
    def validate_bedrooms(cls, v, info):
        rooms = info.data.get("apartment_rooms")
        if v < 0:
            raise ValueError("Bedrooms cannot be negative")
        if rooms is not None and v > rooms:
            raise ValueError("Bedrooms cannot exceed total rooms")
        return v

    @field_validator("apartment_bathrooms")
    @classmethod
    def validate_bathrooms(cls, v, info):
        bedrooms = info.data.get("apartment_bedrooms")
        if v < 0:
            raise ValueError("Bathrooms cannot be negative")
        if bedrooms is not None and v > bedrooms + 1:
            raise ValueError("Bathrooms exceed realistic number")
        return v

    @field_validator("apartment_total_area")
    @classmethod
    def validate_total_area(cls, v):
        if v < 20 or v > 1000:
            raise ValueError("Total area must be between 20 and 1000 m²")
        return v

    @field_validator("apartment_living_area")
    @classmethod
    def validate_living_area(cls, v, info):
        total = info.data.get("apartment_total_area")
        if v < 15:
            raise ValueError("Living area is too small")
        if total is not None and v > total:
            raise ValueError("Living area cannot exceed total area")
        return v

    @field_validator("country", "location")
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# =========================
# GLOBAL ERROR HANDLERS
# =========================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0]
    field = first_error["loc"][-1].replace("_", " ").title()
    message = first_error["msg"].replace("Value error, ", "")
    return JSONResponse(
        status_code=422,
        content={"error": f"{field}: {message}"}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": str(exc)}
    )


# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {"status": "API is running ✅"}

@app.post("/predict")
def predict(data: PropertyInput):
    price = predict_price(data.dict())
    return {"predicted_price_usd": price}