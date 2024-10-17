import dill
from datetime import datetime
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('model/event_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: datetime
    visit_time: datetime
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    hit_date: str
    hit_time: float
    hit_number: int
    hit_type: str
    hit_referer: str
    hit_page_path: str
    event_category: str
    event_action: str
    event_label: str


class Prediction(BaseModel):
    id: str
    event_action: str
    pred_value: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.get('/info')
def info():
    return (
        "Эта модель предсказывает совершение клиентом целевого действия из списка:[sub_car_claim_click, "
        "sub_car_claim_submit_click, sub_open_dialog_click, sub_custom_question_submit_click, sub_call_number_click, "
        "sub_callback_submit_click, sub_submit_success,sub_car_request_submit_click]")


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    form_dict = vars(form)
    df = pd.DataFrame([form_dict])
    y = model['model'].predict(df)

    return {
        'id': form.client_id,
        'event_action': form.event_action,
        'pred_value': y[0]
    }
