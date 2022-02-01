from os import stat
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from typing import List
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:2000'])

class SalesHistory(BaseModel):
    sale_date: date
    location_code: int
    sku_code: int
    sold_quantity: float

class StatisticalForecast(BaseModel):
    forecast_date: date
    location_code: int
    sku_code: int
    statistical_forecast_quantity: float

class StatisticalForecastList(BaseModel):
    __root__: List[StatisticalForecast]

@app.post('/get_forecast', response_model=List[StatisticalForecast])
def forecast(sales_history:List[SalesHistory]):
    prediction_length = 365
    sales_history_df = pd.DataFrame([s.dict() for s in sales_history])
    sales_history_df = sales_history_df.astype({'sale_date':'datetime64','location_code':'int','sku_code':'int','sold_quantity':'float'})
    
    unique_location_sku = sales_history_df.groupby(['location_code','sku_code']).size().reset_index()
    forecast_df = pd.DataFrame({'forecast_date':'datetime64',
                                'location_code':'int',
                                'sku_code':'int',
                                'statistical_forecast_quantity':'float'
                                }, index=[])

    for index, row in unique_location_sku.iterrows():        
        time_series = TimeSeries.from_dataframe (
            sales_history_df [
                (sales_history_df['location_code'] == row['location_code']) &
                (sales_history_df['sku_code'] == row['sku_code'])
            ], time_col ='sale_date', value_cols='sold_quantity', fill_missing_dates=True, freq='D',fillna_value=0
        )
        model = ExponentialSmoothing()
        model.fit(time_series)
        prediction = model.predict(prediction_length, num_samples=1)
        df = prediction.pd_dataframe().reset_index()
        df.rename(columns={'time':'forecast_date','sold_quantity':'statistical_forecast_quantity'},inplace=True)
        df['location_code'] = row['location_code']
        df['sku_code'] = row['sku_code']
        forecast_df = forecast_df.append(df,ignore_index=True)
    statistical_forecast = StatisticalForecastList.parse_obj(forecast_df.to_dict('records'))
    return statistical_forecast.__root__