from unittest import result
from urllib import response

import gradio as gr
import requests


API_URL = "http://127.0.0.1:5000/predict"


def predict_price(
    building_construction_year,
    building_total_floors,
    apartment_floor,
    apartment_rooms,
    apartment_bedrooms,
    apartment_bathrooms,
    apartment_total_area,
    apartment_living_area,
    country,
    location
):
    payload = {
        "building_construction_year": building_construction_year,
        "building_total_floors": building_total_floors,
        "apartment_floor": apartment_floor,
        "apartment_rooms": apartment_rooms,
        "apartment_bedrooms": apartment_bedrooms,
        "apartment_bathrooms": apartment_bathrooms,
        "apartment_total_area": apartment_total_area,
        "apartment_living_area": apartment_living_area,
        "country": country,
        "location": location
    }

    response = requests.post(API_URL, json=payload)
    result = response.json()

    if response.status_code != 200:
        message = (
            result.get("error")
            or result.get("detail")
            or "Invalid input"
        )
        return f"❌ {message}"

    return f"${result['predicted_price_usd']:,.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Building Year", value=2018),
        gr.Number(label="Total Floors", value=10),
        gr.Number(label="Apartment Floor", value=3),
        gr.Number(label="Rooms", value=3),
        gr.Number(label="Bedrooms", value=2),
        gr.Number(label="Bathrooms", value=2),
        gr.Number(label="Total Area (m²)", value=120),
        gr.Number(label="Living Area (m²)", value=85),
        gr.Text(label="Country", value="Turkey"),
        gr.Text(
            label="Location (exact format)",
            value="Mahmutlar, Mediterranean Region, Alanya, Turkey"
        )
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="Real Estate Price Prediction"
)

if __name__ == "__main__":
    demo.launch()