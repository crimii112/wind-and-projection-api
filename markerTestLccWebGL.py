import netCDF4 as nc
import numpy as np
import os
import json
from io import BytesIO
from nc_cache import get_nc_dataset, nc_lock
from pyproj import CRS, Transformer
from PIL import Image

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

def wdws_to_uv(wd_deg, ws):
    wd_rad = np.deg2rad(wd_deg)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v

def get_lcc_to_wgs84_transformer():
    lcc_crs = CRS.from_proj4(
        "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 "
        "+x_0=0 +y_0=0 "
        "+a=6370000 +b=6370000 +units=m +no_defs"
    )

    wgs84 = CRS.from_epsg(4326)
    return Transformer.from_crs(lcc_crs, wgs84, always_xy=True)

def get_policy_min_max(poll):
    RGBA_RANGES = {
        "O3": [
            { "min": 0.0, "max": 0.01 },
            { "min": 0.01, "max": 0.02 },
            { "min": 0.02, "max": 0.03 },
            { "min": 0.03, "max": 0.04 },
            { "min": 0.04, "max": 0.05 },
            { "min": 0.05, "max": 0.06 },
            { "min": 0.06, "max": 0.07 },
            { "min": 0.07, "max": 0.08 },
            { "min": 0.08, "max": 0.09 },
            { "min": 0.09, "max": 0.1 },
            { "min": 0.1, "max": 0.11 },
            { "min": 0.11, "max": 0.12 },
            { "min": 0.12, "max": 0.13 },
            { "min": 0.13, "max": 0.14 },
            { "min": 0.14, "max": 0.15 },
            { "min": 0.15, "max": 0.16 },
            { "min": 0.16, "max": 0.17 },
            { "min": 0.17, "max": 0.18 },
            { "min": 0.18, "max": 0.19 },
            { "min": 0.19, "max": float("inf") },
        ],
        "PM10": [
            { "min": 6, "max": 18 },
            { "min": 0, "max": 6 },
            { "min": 18, "max": 31 },
            { "min": 31, "max": 40 },
            { "min": 40, "max": 48 },
            { "min": 48, "max": 56 },
            { "min": 56, "max": 64 },
            { "min": 64, "max": 72 },
            { "min": 72, "max": 81 },
            { "min": 81, "max": 93 },
            { "min": 93, "max": 105 },
            { "min": 105, "max": 117 },
            { "min": 117, "max": 130 },
            { "min": 130, "max": 142 },
            { "min": 142, "max": 151 },
            { "min": 151, "max": 191 },
            { "min": 191, "max": 231 },
            { "min": 231, "max": 271 },
            { "min": 271, "max": 320 },
            { "min": 320, "max": float("inf") },
        ],
        "PM2.5": [
            { "min": 0, "max": 5 },
            { "min": 5, "max": 10 },
            { "min": 10, "max": 16 },
            { "min": 16, "max": 19 },
            { "min": 19, "max": 22 },
            { "min": 22, "max": 26 },
            { "min": 26, "max": 30 },
            { "min": 30, "max": 33 },
            { "min": 33, "max": 36 },
            { "min": 36, "max": 42 },
            { "min": 42, "max": 48 },
            { "min": 48, "max": 55 },
            { "min": 55, "max": 62 },
            { "min": 62, "max": 69 },
            { "min": 69, "max": 76 },
            { "min": 76, "max": 107 },
            { "min": 107, "max": 138 },
            { "min": 138, "max": 169 },
            { "min": 169, "max": 200 },
            { "min": 200, "max": float("inf") },
        ],
        "TEMP": [
            { "min": -80.15, "max": -67.15 },
            { "min": -67.15, "max": -54.15 },
            { "min": -54.15, "max": -40.0 },
            { "min": -40.0, "max": -17.78 },
            { "min": -17.78, "max": 0.0 },
            { "min": 0.0, "max": 2.0 },
            { "min": 2.0, "max": 17.85 },
            { "min": 17.85, "max": 24.85 },
            { "min": 24.85, "max": 37.85 },
            { "min": 37.85, "max": 54.85 },
            { "min": 54.85, "max": float("inf") },
        ],
    }
    
    ranges = RGBA_RANGES[poll]
    finite = [r for r in ranges if np.isfinite(r["min"]) and np.isfinite(r["max"])]
    
    policy_min = finite[0]["min"]
    policy_max = finite[-1]["max"]

    return policy_min, policy_max

def get_webgl_wind_png(grid_km, layer, tstep, poll):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError("Unsupported grid")
    
        cfg = GRID_CONFIG[grid_km]
        
        aconc_path = os.path.join(SCRIPT_DIR, 'data', cfg["ACONC"])
        gridcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["GRIDCRO"])
        metcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["METCRO"])
        
        with nc_lock():
            ds_aconc = get_nc_dataset(aconc_path)
            ds_gridcro = get_nc_dataset(gridcro_path)
            ds_metcro = get_nc_dataset(metcro_path)
            
            print(f"✅ NetCDF {grid_km}km files opened successfully.")
            
            ## 격자(좌표계)
            XORIG = ds_gridcro.getncattr('XORIG')   # -180000.0(9km) / -2349000.0(27km)
            YORIG = ds_gridcro.getncattr('YORIG')   # -585000.0(9km) / -1728000.0(27km)
            XCELL = ds_gridcro.getncattr('XCELL')   # 9000.0(9km) / 27000.0(27km)
            YCELL = ds_gridcro.getncattr('YCELL')   # 9000.0(9km) / 27000.0(27km)

            nrows = cfg["nrows"]
            ncols = cfg["ncols"]    
            
            # LCC extent (grid 전체 영역)
            extent_lcc = [
                XORIG,
                YORIG,
                XORIG + ncols * XCELL,
                YORIG + nrows * YCELL,
            ]

            # ==========================================================
            # 바람(u/v) → wind PNG
            # ==========================================================
            # 풍향, 풍속            
            wds = ds_metcro.variables['WDIR10'][tstep][layer]
            wss = ds_metcro.variables['WSPD10'][tstep][layer]
                    
            u = np.zeros_like(wds, dtype=np.float32)
            v = np.zeros_like(wds, dtype=np.float32)

            for i in range(nrows):
                for j in range(ncols):
                    u[i, j], v[i, j] = wdws_to_uv(wds[i, j], wss[i, j])

            # 남→북 뒤집기
            u = np.flipud(u)
            v = np.flipud(v)
            
            # u/v 최소, 최대, GPU는 float 배열을 직접 못 쓰므로 0~255 정수로 압축
            u_min, u_max = float(u.min()), float(u.max())
            v_min, v_max = float(v.min()), float(v.max())
            
            # normalize → 0~255
            u_img = np.clip((u - u_min) / (u_max - u_min) * 255, 0, 255).astype(np.uint8)
            v_img = np.clip((v - v_min) / (v_max - v_min) * 255, 0, 255).astype(np.uint8)
            
            rgba = np.zeros((nrows, ncols, 4), dtype=np.uint8)
            rgba[..., 0] = u_img    # R(u 성분)
            rgba[..., 1] = v_img    # G(v 성분)
            rgba[..., 2] = 0        # B(사용 안 함)
            rgba[..., 3] = 255      # A(불투명)
            
            # 파일 저장 x -> 메모리 png
            wind_png = BytesIO()
            Image.fromarray(rgba, "RGBA").save(wind_png, format="PNG")
            wind_png.seek(0)
            
            # ==========================================================
            # poll이 WIND면 농도 PNG 없이 반환
            # ==========================================================
            if poll == "WIND":
                webgl_meta = {
                    "width": ncols,
                    "height": nrows,
                    "extentLCC": extent_lcc,
                    "gridKm": grid_km,

                    "uMin": u_min,
                    "uMax": u_max,
                    "vMin": v_min,
                    "vMax": v_max,

                    "pollutant": poll
                }
                
                return wind_png, None, webgl_meta
            
            # ==========================================================
            # 농도 데이터 → conc PNG (poll에 따라)
            # ==========================================================
            elif poll == "O3":
                conc = ds_aconc.variables['O3'][tstep][layer]

            elif poll == "PM10":
                arrays = [
                    ds_aconc.variables[el][tstep][layer]
                    for el in PM10_ELEMENTS
                ]
                conc = np.sum(arrays, axis=0)

            elif poll == "PM2.5":
                arrays = [
                    ds_aconc.variables[el][tstep][layer]
                    for el in PM25_ELEMENTS
                ]
                conc = np.sum(arrays, axis=0)

            elif poll == "TEMP":
                conc = ds_metcro.variables["TEMP2"][tstep][layer] - 273.15

            else:
                raise ValueError(f"Unsupported pollutant: {poll}")

            conc = np.flipud(conc)

            # c_min = float(np.nanmin(conc))
            # c_max = float(np.nanmax(conc))

            # conc_img = np.clip(
            #     (conc - c_min) / (c_max - c_min) * 255,
            #     0, 255
            # ).astype(np.uint8)
            
            policy_min, policy_max = get_policy_min_max(poll)
            
            conc_norm = (conc - policy_min) / (policy_max - policy_min)
            conc_norm = np.clip(conc_norm, 0.0, 1.0)

            conc_img = (conc_norm * 255).astype(np.uint8)

            conc_rgba = np.zeros((nrows, ncols, 4), dtype=np.uint8)
            conc_rgba[..., 0] = conc_img
            conc_rgba[..., 1] = 0
            conc_rgba[..., 2] = 0
            conc_rgba[..., 3] = 255

            conc_png = BytesIO()
            Image.fromarray(conc_rgba, "RGBA").save(conc_png, format="PNG")
            conc_png.seek(0)

            # ==========================================================
            # 메타데이터
            # ==========================================================
            webgl_meta = {
                "width": ncols,
                "height": nrows,
                "extentLCC": extent_lcc,
                "gridKm": grid_km,

                "uMin": u_min,
                "uMax": u_max,
                "vMin": v_min,
                "vMax": v_max,

                "cMin": policy_min,
                "cMax": policy_max,
                "pollutant": poll
            }
            
            # with open(f"wind/{out_name}.json", "w") as f:
            #     json.dump(webgl_data, f, indent=2)
            
            # print(f"✅ wind/{out_name}.png & json 생성 완료")

            return wind_png, conc_png, webgl_meta
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

