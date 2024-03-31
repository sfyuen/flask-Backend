from datetime import datetime, timedelta
from retry_requests import retry
from flask_cors import CORS
import numpy as np
import pickle
import holidays
import openmeteo_requests
import requests_cache
import requests
import json

from flask import Flask
app = Flask(__name__)
CORS(app)

def fetchelec():
    r = requests.get('https://api.electricitymap.org/v3/power-breakdown/history?zone=ES')
    actual_load = []
    actual_ts = []
    for x in r.json()['history']:
        actual_load.append(x['powerConsumptionTotal'])
        actual_ts.append(datetime.fromisoformat(x['datetime'][:-1]))
    return actual_ts, actual_load,

def fetchweather():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40,
        "longitude": -4,
        "hourly": ["temperature_2m", "dew_point_2m"],
        "timezone": "Asia/Singapore",
        "past_days": 2,
        "forecast_days": 3
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()

    start_time = datetime.fromtimestamp(hourly.Time())
    end_time = datetime.fromtimestamp(hourly.TimeEnd())

    return hourly_temperature_2m, hourly_dew_point_2m, start_time, end_time

def gen_input(hourly_temperature_2m, hourly_dew_point_2m, start_time, end_time):
    ES_holidays = holidays.country_holidays('ES')
    i = 0
    X_test = []
    predicted_ts = []
    while start_time < end_time:
        predicted_ts.append(start_time)
        t = start_time
        year = t.year

        month = t.month
        week = t.isocalendar().week
        weekday = t.weekday() + 1
        hour = t.hour + 1
        day = t.day

        month_sin = np.sin(month / 12 * 2 * np.pi)
        month_cos = np.cos(month / 12 * 2 * np.pi)

        week_sin = np.sin(week / 53 * 2 * np.pi)
        week_cos = np.cos(week / 53 * 2 * np.pi)

        weekday_sin = np.sin(weekday / 7 * 2 * np.pi)
        weekday_cos = np.cos(weekday / 7 * 2 * np.pi)

        hour_sin = np.sin(hour / 24 * 2 * np.pi)
        hour_cos = np.cos(hour / 24 * 2 * np.pi)

        day_sin = np.sin(day / 30 * 2 * np.pi)
        day_cos = np.cos(day / 30 * 2 * np.pi)

        X_test.append([
            hourly_temperature_2m[i],
            hourly_dew_point_2m[i],
            year,
            month_sin,
            month_cos,
            weekday_sin,
            weekday_cos,
            hour_sin,
            hour_cos,
            week_sin,
            week_cos,
            day_sin,
            day_cos,
            t in ES_holidays or weekday == 7])
        start_time += timedelta(hours=1)
        i += 1
    return X_test, predicted_ts

@app.route('/')
def end_point():
    pickled_model = pickle.load(open("model.pkl", 'rb'))
    actual_ts, actual_load = fetchelec()
    hourly_temperature_2m, hourly_dew_point_2m, start_time, end_time = fetchweather()
    X_test, predicted_ts = gen_input(hourly_temperature_2m, hourly_dew_point_2m, start_time, end_time)
    predicted_load = pickled_model.predict(X_test)

    shift = max(int((actual_ts[0] - start_time).total_seconds() // 3600), 0)

    data = [
        {
            'x': actual_ts,
            'y': actual_load,
            'type': 'scatter',
            'name': 'Actual Load'
        },
        {
            'x': predicted_ts[shift:],
            'y': list(predicted_load)[shift:],
            'type': 'scatter',
            'name': 'Predicted Load'
        }
    ]

    return json.dumps(data, default=str).encode(encoding='utf-8')
