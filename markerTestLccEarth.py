import numpy as np
import os
from datetime import datetime, timezone
from nc_cache import get_nc_dataset, nc_lock
import json

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

# ===== CAI helpers (Seoul CAI breakpoints) =====
CAI_I_LO = np.array([0, 51, 101, 251], dtype=np.float32)
CAI_I_HI = np.array([50, 100, 250, 500], dtype=np.float32)

CAI_BPS = {
    "PM2.5": np.array([[0, 15], [16, 35], [36, 75], [76, 500]], dtype=np.float32),
    "PM10":  np.array([[0, 30], [31, 80], [81, 150], [151, 600]], dtype=np.float32),
    "O3":    np.array([[0.0, 0.03], [0.0301, 0.09], [0.0901, 0.15], [0.1501, 0.6]], dtype=np.float32),
    "NO2":   np.array([[0.0, 0.03], [0.0301, 0.06], [0.0601, 0.2], [0.2001, 2.0]], dtype=np.float32),
    "CO":    np.array([[0.0, 2.0], [2.01, 9.0], [9.01, 15.0], [15.01, 50.0]], dtype=np.float32),
    "SO2":   np.array([[0.0, 0.02], [0.0201, 0.05], [0.0501, 0.15], [0.1501, 1.0]], dtype=np.float32),
}

def cai_subindex_array(cp: np.ndarray, bps: np.ndarray) -> np.ndarray:
    """
    Vectorized CAI sub-index (Ip) using piecewise linear interpolation:
    Ip = (I_HI - I_LO)/(BP_HI - BP_LO) * (Cp - BP_LO) + I_LO
    If Cp exceeds defined BP_HI, clamp to last BP_HI. (Seoul rule) :contentReference[oaicite:1]{index=1}
    """
    cp = np.asarray(cp, dtype=np.float32)
    cp = np.maximum(cp, 0.0)

    # clamp Cp to last BP_HI (very-unhealthy BP_HI)
    cp = np.minimum(cp, bps[-1, 1])

    out = np.zeros_like(cp, dtype=np.float32)

    for k in range(4):
        bp_lo, bp_hi = float(bps[k, 0]), float(bps[k, 1])
        i_lo, i_hi = float(CAI_I_LO[k]), float(CAI_I_HI[k])

        mask = (cp >= bp_lo) & (cp <= bp_hi)
        denom = (bp_hi - bp_lo) if (bp_hi - bp_lo) != 0 else 1.0
        out[mask] = ((i_hi - i_lo) / denom) * (cp[mask] - bp_lo) + i_lo

    return np.rint(out).astype(np.int16)  # 정수 지수

def cai_from_arrays(o3, so2, no2, co, pm10, pm25) -> np.ndarray:
    sub_o3  = cai_subindex_array(o3,  CAI_BPS["O3"])
    sub_so2 = cai_subindex_array(so2, CAI_BPS["SO2"])
    sub_no2 = cai_subindex_array(no2, CAI_BPS["NO2"])
    sub_co  = cai_subindex_array(co,  CAI_BPS["CO"])
    sub_pm10= cai_subindex_array(pm10, CAI_BPS["PM10"])
    sub_pm25= cai_subindex_array(pm25, CAI_BPS["PM2.5"])

    subs = np.stack([sub_o3, sub_so2, sub_no2, sub_co, sub_pm10, sub_pm25], axis=0)

    base = subs.max(axis=0)
    n_unhealthy = (subs >= 101).sum(axis=0)  # '나쁨 이상' 개수 :contentReference[oaicite:2]{index=2}
    addon = np.where(n_unhealthy == 2, 50, np.where(n_unhealthy >= 3, 75, 0))  # :contentReference[oaicite:3]{index=3}

    cai = base + addon
    cai = np.minimum(cai, 500).astype(np.int16)
    return cai

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
                # test용(전부 20)
                # scalar = np.full((nrows, ncols), 20.0, dtype=np.float32)
            elif bg_poll == "O3":
                scalar = ds_aconc["O3"][tstep][layer]
            elif bg_poll == "SO2":
                scalar = ds_aconc["SO2"][tstep][layer]
            elif bg_poll == "NO2":
                scalar = ds_aconc["NO2"][tstep][layer]
            elif bg_poll == "CO":
                scalar = ds_aconc["CO"][tstep][layer]
            elif bg_poll == "PM10":
                scalar = sum_elements_2d(ds_aconc, PM10_ELEMENTS, tstep, layer)
            elif bg_poll == "PM2.5":
                scalar = sum_elements_2d(ds_aconc, PM25_ELEMENTS, tstep, layer)
            elif bg_poll == 'WIND':
                scalar = ds_metcro['WSPD10'][tstep][layer]
            elif bg_poll == 'CAI':
                o3 = np.array(ds_aconc["O3"][tstep][layer], dtype=np.float32)
                so2 = np.array(ds_aconc["SO2"][tstep][layer], dtype=np.float32)
                no2 = np.array(ds_aconc["NO2"][tstep][layer], dtype=np.float32)
                co = np.array(ds_aconc["CO"][tstep][layer], dtype=np.float32)
                pm10 = sum_elements_2d(ds_aconc, PM10_ELEMENTS, tstep, layer)
                pm25 = sum_elements_2d(ds_aconc, PM25_ELEMENTS, tstep, layer)
                
                scalar = cai_from_arrays(o3=o3, so2=so2, no2=no2, co=co, pm10=pm10, pm25=pm25)
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