import os
import pygrib
import numpy as np
import json
from scipy.interpolate import griddata

def read_grib2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grib_filename = os.path.join(script_dir, 'data', 'l015v070erlopresh000.2025060500.gb2')
    grib_filename = os.path.join(script_dir, 'data', 'l015v070erlopresh001.2025060500.gb2')
    
    grbs = pygrib.open(grib_filename)
    
    ## 파일 파악
    for grb in grbs:
        print(grb.name)
        # print(grb.keys())
        # print(grb.latlons())
        # break
        
    u_grb = grbs.select(name='U component of wind')[0]  #25:U component of wind:m s**-1 (instant):lambert:isobaricInhPa:level 100000 Pa:fcst time 0 hrs:from 202506050000
    v_grb = grbs.select(name='V component of wind')[0]  #26:V component of wind:m s**-1 (instant):lambert:isobaricInhPa:level 100000 Pa:fcst time 0 hrs:from 202506050000
    tmp_grb = grbs.select(name='Temperature')[0]        #97:Temperature:K (instant):lambert:isobaricInhPa:level 100000 Pa:fcst time 0 hrs:from 202506050000
    lats, lons = u_grb.latlons()
    
    # # 새로운 lat/lon grid 정의 (예시: 0.25도 간격)
    # lon_new = np.linspace(lons.min(), lons.max(), lons.shape[1])
    # lat_new = np.linspace(lats.min(), lats.max(), lats.shape[0])
    # lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    u = np.array(u_grb.values)
    v = np.array(v_grb.values)
    tmp = np.array(tmp_grb.values)
    
    print(lats, lons)
    # print(u_grb)
    # print(tmp_grb.values)
    
    # 2차원 격자를 1차원 벡터로 변환
    # points = np.stack([lats.flatten(), lons.flatten()], axis=-1)
    # u_interp = griddata(points, u.flatten(), (lat_grid, lon_grid), method='linear')
    # v_interp = griddata(points, v.flatten(), (lat_grid, lon_grid), method='linear')
    
    lo1 = float(np.min(lons))      
    la1 = float(np.min(lats))
    lo2 = float(np.max(lons))
    la2 = float(np.max(lats)) 
    nx, ny = u.shape
    dxs = np.diff(lons, axis = 1).mean(0)
    dx = float(dxs.mean())
    # dx = (lo2 - lo1) / (nx - 1)
    dys = np.diff(lats, axis = 0).mean(1)
    dy = float(dys.mean())
    # dy = (la2 - la1) / (ny - 1)
    
    windData = [
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 2,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": "2025-06-05T00:00:00Z",
                "surface1Type": 100,
                "surface1Value": 100000.0,
                "forecastTime": 1,
                "scanMode": 0
            },
            "data": u.flatten().tolist()
        },
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 3,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": "2025-06-05T00:00:00Z",
                "surface1Type": 100,
                "surface1Value": 100000.0,
                "forecastTime": 1,
                "scanMode": 0
            },
            "data": v.flatten().tolist()
        },
    ]
    
    tempData = [
        {
            "header": {
                "parameterCategory": 0,
                "parameterNumber": 0,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": "2025-06-05T00:00:00Z",
                "surface1Type": 1,
                "surface1Value": 2,
                "forecastTime": 1,
                "scanMode": 0
            },
            "data": tmp.flatten().tolist()
        }
    ]
    
    with open('temp-20250605-001-gb2.json', 'w') as f :
        json.dump(tempData, f, indent=4)
    
    with open('wind-20250605-001-gb2.json', 'w') as f :
        json.dump(windData, f, indent=4)

    
read_grib2()