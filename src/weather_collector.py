"""
weather_collector.py
Collect weather data from OpenWeatherMap API
Phase 2 - Data Collection (CRISP-ML(Q))
Group 05 - ISB AMPBA
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENWEATHER_API_KEY, CITIES, PRIMARY_CITY

def get_current_weather(city_name):
    """Get current weather for a city"""
    city = CITIES[city_name]
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat':   city['lat'],
        'lon':   city['lon'],
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200:
            return {
                'city':        city_name,
                'datetime':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': data['main']['temp'],
                'humidity':    data['main']['humidity'],
                'wind_speed':  data['wind']['speed'],
                'cloud_cover': data['clouds']['all'],
                'pressure':    data['main']['pressure'],
                'weather':     data['weather'][0]['main']
            }
        else:
            print(f"  API error for {city_name}: {data.get('message','unknown')}")
            return None
    except Exception as e:
        print(f"  Connection error for {city_name}: {e}")
        return None

def collect_weather_all_cities():
    """Collect current weather for all 8 cities"""
    print(f"\n🌤️  Collecting weather for {len(CITIES)} cities...")
    records = []
    for city_name in CITIES:
        record = get_current_weather(city_name)
        if record:
            records.append(record)
            print(f"  ✅ {city_name}: {record['temperature']}°C, "
                  f"Wind: {record['wind_speed']}m/s, "
                  f"Humidity: {record['humidity']}%")
        time.sleep(0.5)   # respect rate limit
    return pd.DataFrame(records)

def generate_synthetic_weather_features(n_samples, start_date='2021-01-01'):
    """
    Generate realistic synthetic weather features aligned to price dataset.
    Used when historical API data is not available (free tier limitation).
    Based on Delhi climate patterns.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    # Season mapping: 1=Winter(Dec-Feb), 2=Spring(Mar-May),
    #                 3=Summer(Jun-Sep), 4=Autumn(Oct-Nov)
    month = dates.month
    season = np.where(month.isin([12,1,2]), 1,
              np.where(month.isin([3,4,5]),  2,
              np.where(month.isin([6,7,8,9]),3, 4)))

    # Temperature based on season (Delhi climate)
    base_temp = np.where(season==1, 15, np.where(season==2, 28,
                np.where(season==3, 35, 24)))
    temperature = base_temp + np.random.normal(0, 3, n_samples)

    # Hour-based adjustment (cooler at night)
    hour_adj = -5 * np.cos(2 * np.pi * dates.hour / 24)
    temperature = temperature + hour_adj

    # Cooling degree (max(temp - 25, 0)) — committed feature
    cooling_degree = np.maximum(temperature - 25, 0)

    # Wind speed (higher in monsoon/spring)
    base_wind = np.where(season==3, 4.5, np.where(season==2, 3.5, 2.0))
    wind_speed = base_wind + np.abs(np.random.normal(0, 1, n_samples))

    # Low wind flag — committed feature
    low_wind_flag = (wind_speed < 2.0).astype(int)

    # Solar irradiance proxy (based on hour and cloud cover)
    cloud_cover = np.random.uniform(10, 80, n_samples)
    solar_hour  = np.maximum(0, np.sin(np.pi * (dates.hour - 6) / 12))
    solar_irradiance = solar_hour * (1 - cloud_cover/100) * 1000

    humidity = np.where(season==3, 75, np.where(season==1, 55, 45))
    humidity = humidity + np.random.normal(0, 8, n_samples)
    humidity = np.clip(humidity, 20, 95)

    df = pd.DataFrame({
        'temperature':      temperature,
        'cooling_degree':   cooling_degree,
        'wind_speed':       wind_speed,
        'low_wind_flag':    low_wind_flag,
        'humidity':         humidity,
        'cloud_cover':      cloud_cover,
        'solar_irradiance': solar_irradiance,
    }, index=dates)

    return df

if __name__ == "__main__":
    # Test live weather collection
    print("Testing OpenWeatherMap API connection...")
    df = collect_weather_all_cities()
    if len(df) > 0:
        print(f"\n✅ Successfully collected weather for {len(df)} cities")
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/current_weather.csv', index=False)
        print("Saved to data/current_weather.csv")
    else:
        print("❌ No data collected — check API key")
