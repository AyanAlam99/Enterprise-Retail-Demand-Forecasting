from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.pipeline.prediction import SalesData, SalesPredictor
from src.data_access.data_export import ExportData
from src.logging import logging

app = FastAPI()

templates = Jinja2Templates(directory="templates")

try :
    stores_df = ExportData().export_collection_as_dataframe(collection_name="Stores")
    STORE_MAPPING = stores_df.set_index('store_nbr').to_dict('index')
    logging.info("Store mapping loaded successfully into memory from MongoDB.")
except Exception as e :
    logging.error(f"Could not load store mapping from MongoDB: {e}")
    STORE_MAPPING ={}

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
        
        store_info = STORE_MAPPING.get(store_nbr , {"city" : 'Quito',"state":"Pichincha","type":"A","cluster":1})

        sales_data = SalesData(
            date=date,
            store_nbr=store_nbr,
            family=family,
            onpromotion=onpromotion,
            dcoilwtico=dcoilwtico,
            city = store_info.get("city"),
            state = store_info.get("state"),
            store_type =store_info.get("type"),
            cluster = store_info.get("cluster"),

            holiday_type = "None",
            holiday_description = "None"
        )
        
       
        input_df = sales_data.get_sales_input_dataframe()
        
    
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