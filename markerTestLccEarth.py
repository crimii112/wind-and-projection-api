import numpy as np
import os
from datetime import datetime, timezone
from nc_cache import get_nc_dataset, nc_lock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PM25_ELEMENTS = [
    'A25I', 'A25J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ACLI', 'ACLJ', 'AECI', 'AECJ', 'AISO1J', 
    'AISO2J', 'AISO3J', 'ANAI', 'ANAJ', 'ANH4I', 'ANH4J', 'ANO3I', 'ANO3J', 'AOLGAJ', 'AOLGBJ', 
    'AORGCJ', 'AORGPAI', 'AORGPAJ', 'ASO4I', 'ASO4J', 'ASQTJ', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ATRP1J',
    'ATRP2J', 'AXYL1J', 'AXYL2J', 'AXYL3J'
]
PM10_ELEMENTS = [
    'A25I', 'A25J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ACLI', 'ACLJ', 'ACLK', 'ACORS', 'AECI', 
    'AECJ', 'AISO1J', 'AISO2J', 'AISO3J', 'ANAI', 'ANAJ', 'ANAK', 'ANH4I', 'ANH4J', 'ANH4K',
    'ANO3I', 'ANO3J', 'ANO3K', 'AOLGAJ', 'AOLGBJ', 'AORGCJ', 'AORGPAI', 'AORGPAJ', 'ASO4I', 'ASO4J', 
    'ASO4K', 'ASOIL', 'ASQTJ', 'ATOL1J', 'ATOL2J', 'ATOL3J', 'ATRP1J', 'ATRP2J', 'AXYL1J', 'AXYL2J', 
    'AXYL3J'
]

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

def sum_elements_2d(ds, elements, tstep, layer):
    # ds[var][t][z] 는 보통 (ny, nx) 2D
    acc = None
    for el in elements:
        arr = np.array(ds.variables[el][tstep][layer], dtype=np.float32)  # 2D
        acc = arr if acc is None else acc + arr
    return acc

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

def get_earth_data(grid_km, tstep, layer, bg_poll='TEMP'):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError(f"Unsupported grid_km: {grid_km}")
        
        cfg = GRID_CONFIG[grid_km]
        
        with nc_lock():
            ds_gridcro = get_nc_dataset(os.path.join(SCRIPT_DIR, "data", cfg["GRIDCRO"]))
            ds_metcro  = get_nc_dataset(os.path.join(SCRIPT_DIR, "data", cfg["METCRO"]))
            ds_aconc   = get_nc_dataset(os.path.join(SCRIPT_DIR, "data", cfg["ACONC"]))

            # grid
            XORIG = ds_gridcro.getncattr("XORIG")
            YORIG = ds_gridcro.getncattr("YORIG")
            XCELL = ds_gridcro.getncattr("XCELL")
            YCELL = ds_gridcro.getncattr("YCELL")

            nrows = cfg["nrows"]
            ncols = cfg["ncols"]    
            half = cfg["half_cell"]

            lo1 = XORIG + half
            la1 = YORIG + (nrows - 1) * YCELL + half

            header = {
                "nx": ncols,
                "ny": nrows,
                "lo1": float(lo1),
                "la1": float(la1),
                "dx": float(XCELL),
                "dy": -float(YCELL),   # 북 → 남
            }

            # ===== wind =====
            wdir = ds_metcro["WDIR10"][tstep][layer]
            wspd = ds_metcro["WSPD10"][tstep][layer]

            u = np.zeros_like(wdir)
            v = np.zeros_like(wdir)

            for i in range(nrows):
                for j in range(ncols):
                    u[i, j], v[i, j] = wdws_to_uv(wdir[i, j], wspd[i, j])
                
            u = np.flipud(u)
            v = np.flipud(v)

            # ===== scalar (bg_poll) =====
            if bg_poll == "TEMP":
                scalar = ds_metcro["TEMP2"][tstep][layer] - 273.15
            elif bg_poll == "O3":
                scalar = ds_aconc["O3"][tstep][layer]
            elif bg_poll == "PM10":
                scalar = sum_elements_2d(ds_aconc, PM10_ELEMENTS, tstep, layer)
            elif bg_poll == "PM2.5":
                scalar = sum_elements_2d(ds_aconc, PM25_ELEMENTS, tstep, layer)
            else:
                raise ValueError(f"Unsupported bg_poll: {bg_poll}")
            
            scalar = np.flipud(scalar)
            
            now = datetime.now(timezone.utc).isoformat()

            earth = [
                {
                    "header": {**header, "parameterCategory": 2, "parameterNumber": 2},
                    "data": u.flatten().tolist(),
                    "meta": {"date": now}
                },
                {
                    "header": {**header, "parameterCategory": 2, "parameterNumber": 3},
                    "data": v.flatten().tolist(),
                    "meta": {"date": now}
                },
                {
                    "header": {**header, "parameterCategory": 0, "parameterNumber": 0},
                    "data": scalar.flatten().tolist(),
                    "meta": {"date": now, "bg_poll": bg_poll}
                }
            ]
            

            return {"earthData" : earth}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise