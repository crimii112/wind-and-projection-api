import netCDF4 as nc
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from pyproj import CRS, Transformer
from nc_cache import get_nc_dataset, nc_lock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
# GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
# METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')
ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.09KM.2025063012.nc')
GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_09KM.2025063012.nc')
METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_09KM.2025063012.nc')

GRID_CONFIG = {
    9: {
        "ACONC": "ACONC.09KM.2025063012.nc",
        "GRIDCRO": "GRIDCRO2D_09KM.2025063012.nc",
        "METCRO": "METCRO2D_09KM.2025063012.nc",
        "nrows": 82,
        "ncols": 67,
        "half_cell": 4500
    },
    27: {
        "ACONC": "ACONC.27KM.2025063012.nc",
        "GRIDCRO": "GRIDCRO2D_27KM.2025063012.nc",
        "METCRO": "METCRO2D_27KM.2025063012.nc",
        "nrows": 128,
        "ncols": 174,
        "half_cell": 13500
    },
}

def convert_flatten_array(ds, el, tstep, layer):
    list = [float(v) for v in ds.variables[el][tstep][layer].flatten()]
    return np.array(list)

def wdws_to_uv(wd_deg, ws):
    """
    wd_deg: wind direction in degrees (meteorological convention: direction wind is COMING FROM)
    ws: wind speed
    returns: (u, v) where u is eastward, v is northward
    """
    wd_rad = np.deg2rad(wd_deg)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v

def get_earth_data(grid_km, tstep, layer):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError(f"Unsupported grid_km: {grid_km}")
        
        cfg = GRID_CONFIG[grid_km]
        gridcro_path = os.path.join(SCRIPT_DIR, "data", cfg["GRIDCRO"])
        metcro_path  = os.path.join(SCRIPT_DIR, "data", cfg["METCRO"])
        
        with nc_lock():
            ds_gridcro = get_nc_dataset(os.path.join(SCRIPT_DIR, "data", cfg["GRIDCRO"]))
            ds_metcro  = get_nc_dataset(os.path.join(SCRIPT_DIR, "data", cfg["METCRO"]))
            
            # CRS
            # lcc = CRS.from_proj4(
            #     "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 "
            #     "+a=6370000 +b=6370000 +units=m +no_defs"
            # )
            # wgs84 = CRS.from_epsg(4326)
            # tf = Transformer.from_crs(lcc, wgs84, always_xy=True)

            # grid
            XORIG = ds_gridcro.getncattr("XORIG")
            YORIG = ds_gridcro.getncattr("YORIG")
            XCELL = ds_gridcro.getncattr("XCELL")
            YCELL = ds_gridcro.getncattr("YCELL")

            nrows = cfg["nrows"]
            ncols = cfg["ncols"]    
            half = cfg["half_cell"]

            x = XORIG + np.arange(ncols) * XCELL + half
            y = YORIG + np.arange(nrows) * YCELL + half
            xx, yy = np.meshgrid(x, y)

            # lon, lat = tf.transform(xx, yy)
            lo1 = XORIG + half
            la1 = YORIG + (nrows - 1) * YCELL + half

            # wind
            wdir = ds_metcro["WDIR10"][tstep][layer]
            wspd = ds_metcro["WSPD10"][tstep][layer]

            u = np.zeros_like(wdir)
            v = np.zeros_like(wdir)

            for i in range(nrows):
                for j in range(ncols):
                    u[i, j], v[i, j] = wdws_to_uv(wdir[i, j], wspd[i, j])

            # # temp
            # temp = ds_metcro["TEMP2"][tstep, layer]
            
            # print("top-left lat:", lat[0,0])
            # print("bottom-left lat:", lat[-1,0])
            
            # if lat[0,0] < lat[-1,0]:   # 첫 행이 남쪽이면
            #     u   = np.flipud(u)
            #     v   = np.flipud(v)
            #     lat = np.flipud(lat)
            #     lon = np.flipud(lon)
            #     # temp = np.flipud(temp)
                
            u = np.flipud(u)
            v = np.flipud(v)
            
            header = {
                "nx": ncols,
                "ny": nrows,
                "lo1": float(lo1),
                "la1": float(la1),
                "dx": float(XCELL),
                "dy": -float(YCELL),   # 북 → 남
                "parameterCategory": 2
            }

            now = datetime.now(timezone.utc).isoformat()

            wind = [
                {
                    "header": {**header, "parameterNumber": 2},
                    "data": u.flatten().tolist(),
                    "meta": {"date": now}
                },
                {
                    "header": {**header, "parameterNumber": 3},
                    "data": v.flatten().tolist(),
                    "meta": {"date": now}
                }
            ]
            
            # temp = [
            #     {
            #         "header": {**header, "parameterNumber": 0, "parameterCategory": 0},
            #         "data": temp.flatten().tolist(),
            #         "meta": {"date": now}
            #     },
            # ]

            # with open('earth_wind.json', "w") as f:
            #     json.dump(wind, f)
                
            # with open('earth_temp.json', "w") as f:
            #     json.dump(temp, f)

            # print("✅ earth JSON saved")

            return {"earthData" : wind}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise