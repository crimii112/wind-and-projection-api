import os
import numpy as np
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "data", "2024_M11_D17_0000(UTC+0900)_L00_1HR.vec")
CONC_PATH = os.path.join(BASE_DIR, "data", "2024_M11_D17_0000(UTC+0900)_L00_POLLUTANT01_10001_1HR_CONC_GRID.DAT")

def get_projection_test_utm_data():

    
    # 바람 화살표 데이터
    arrow_data = []
    with open(VEC_PATH, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            parts = line.strip().split()
            arrow_data.append({
                "lat": float(parts[1]) * 1000,
                "lon": float(parts[0]) * 1000,
                "wd": abs(float(parts[3])),
                "ws": float(parts[4]),
            })
            
    ncols = 42
    nrows = 34


    # 폴리곤 데이터
    polygon_data = []
    with open(CONC_PATH, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            parts = line.strip().split()
            polygon_data.append({
                "lat": float(parts[1]) * 1000,
                "lon": float(parts[0]) * 1000,
                "value": float(parts[2])
            })
    
        
    result = {
        "arrowData": arrow_data,
        "polygonData": polygon_data
    }
    
    return result
        
# get_projection_test_utm_data()

def get_projection_test_utm_ol_wind():
    transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)

    lats, lons, wds, wss = [], [], [], []
    with open(VEC_PATH, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            parts = line.strip().split()
            
            x = float(parts[0]) * 1000
            y = float(parts[1]) * 1000
            
            lon, lat = transformer.transform(x, y)
            
            lats.append(lat)
            lons.append(lon)
            wds.append(abs(float(parts[3])))
            wss.append(float(parts[4]))

    lats = np.array(lats)
    lons = np.array(lons)
    wd = np.array(wds)
    ws = np.array(wss)
    
    rad = np.radians(wd)
    u = -ws * np.sin(rad)
    v = -ws * np.cos(rad)
    
    nx, ny = 42, 34
    
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))

    u = np.flipud(u)
    v = np.flipud(v)
    u = u.flatten()
    v = v.flatten()

    lo1 = float(np.min(lons))      
    lo2 = float(np.max(lons))
    la1 = float(np.min(lats))
    la2 = float(np.max(lats)) 
    dx = (lo2 - lo1) / (nx - 1)
    dy = (la2 - la1) / (ny - 1)
    refTime = '2024-11-17_00:00:00'

    wind_data = [
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 2,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": refTime
            },
            "data": u.tolist()
        },
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 3,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": refTime
            },
            "data": v.tolist()
        },
    ]
    
    result = {
        "windData": wind_data
    }
    
    # print(wind_data)
    return result

get_projection_test_utm_ol_wind()