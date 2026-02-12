import requests
from io import BytesIO

BASE_URL = "http://e2m3.iptime.org:63002/model/25061"

def get_utm_wind_data(yyyy, mm, dd, hh):
    
    vec_filename = f"r_{yyyy}_M{mm}_D{dd}_{hh}00(UTC+0900)_L00_1HR.vec"
    vec_url = f"{BASE_URL}/{yyyy}/{mm}/{dd}/{hh}/00/{vec_filename}"
    
    print(f"📡 Fetching vec: {vec_url}")

    response = requests.get(vec_url, timeout=10)
    
    if response.status_code != 200:
        raise Exception("vec file not found")

    lines = response.text.split('\n')[1:]
    
    wind_list = []
    
    min_lat = float("inf")
    max_lat = float("-inf")
    min_lon = float("inf")
    max_lon = float("-inf")
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        lon = float(parts[0]) * 1000
        lat = float(parts[1]) * 1000
        wd  = abs(float(parts[3]))
        ws  = float(parts[4])
        
        wind_list.append({
            "lon": lon,
            "lat": lat,
            "wd": wd,
            "ws": ws,
        })
        
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        
    extent = [min_lon, min_lat, max_lon, max_lat]
    
    return {
        "windData": wind_list,
        "extent": extent
    }
    
def get_utm_img_data(img_type, yyyy, mm, dd, hh):
    if img_type not in ['conc', 'wind']:
        raise ValueError("Invalid type. Must be 'conc' or 'wind'")
    
    prefix_map = {
        "conc": "rConc",
        "wind": "rWind"
    }
    
    prefix = prefix_map[img_type]
    
    file_name = f"10001_H1.5_{prefix}_{yyyy}{mm}{dd}{hh}00.Main.Trans.PNG"
    full_url = f"{BASE_URL}/{yyyy}/{mm}/{dd}/{hh}/00/{file_name}"
    
    response = requests.get(full_url)
    response.raise_for_status()
    
    return BytesIO(response.content)