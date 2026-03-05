from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd

from src.pipeline.prediction import SalesData, SalesPredictor
from src.logging import logging

app = FastAPI()

# Templates setup
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sales(
    request: Request,
    date: str = Form(...),
    store_nbr: int = Form(...),
    family: str = Form(...),
    onpromotion: int = Form(...),
    dcoilwtico: float = Form(...)
):
    try:
        # 1. User input ko SalesData object mein pack karna
        sales_data = SalesData(
            date=date,
            store_nbr=store_nbr,
            family=family,
            onpromotion=onpromotion,
            dcoilwtico=dcoilwtico,
            city="Quito",  # Default values for demo
            state="Pichincha",
            store_type="A",
            cluster=1
        )
        
        # 2. DataFrame banani
        input_df = sales_data.get_sales_input_dataframe()
        
        # 3. Predictor trigger karna
        predictor = SalesPredictor()
        prediction = predictor.predict(input_df)
        
        result = round(float(prediction), 2)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction_result": f"Predicted Sales: ${result}",
            "context": {"date": date, "store": store_nbr, "family": family}
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"Error: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)