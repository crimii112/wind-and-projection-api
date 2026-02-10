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
        "SO2": [
            { "min": 0.0, "max": 0.006 },
            { "min": 0.006, "max": 0.013 },
            { "min": 0.013, "max": 0.020 },
            { "min": 0.020, "max": 0.025 },
            { "min": 0.025, "max": 0.030 },
            { "min": 0.030, "max": 0.035 },
            { "min": 0.035, "max": 0.040 },
            { "min": 0.040, "max": 0.045 },
            { "min": 0.045, "max": 0.050 },
            { "min": 0.050, "max": 0.067 },
            { "min": 0.067, "max": 0.084 },
            { "min": 0.084, "max": 0.101 },
            { "min": 0.101, "max": 0.118 },
            { "min": 0.118, "max": 0.134 },
            { "min": 0.134, "max": 0.150 },
            { "min": 0.150, "max": 0.160 },
            { "min": 0.160, "max": 0.170 },
            { "min": 0.170, "max": 0.180 },
            { "min": 0.180, "max": 0.190 },
            { "min": 0.190, "max": float("inf") },
        ],
        "NO2": [
            { "min": 0.0, "max": 0.01 },
            { "min": 0.01, "max": 0.02 },
            { "min": 0.02, "max": 0.03 },
            { "min": 0.03, "max": 0.035 },
            { "min": 0.035, "max": 0.04 },
            { "min": 0.04, "max": 0.045 },
            { "min": 0.045, "max": 0.05 },
            { "min": 0.05, "max": 0.055 },
            { "min": 0.055, "max": 0.06 },
            { "min": 0.06, "max": 0.08 },
            { "min": 0.08, "max": 0.10 },
            { "min": 0.10, "max": 0.12 },
            { "min": 0.12, "max": 0.14 },
            { "min": 0.14, "max": 0.17 },
            { "min": 0.17, "max": 0.20 },
            { "min": 0.20, "max": 0.21 },
            { "min": 0.21, "max": 0.22 },
            { "min": 0.22, "max": 0.23 },
            { "min": 0.23, "max": 0.24 },
            { "min": 0.24, "max": float("inf") },
        ],
        "CO": [
            { "min": 0.0, "max": 0.6 },
            { "min": 0.6, "max": 1.3 },
            { "min": 1.3, "max": 2 },
            { "min": 2, "max": 3.1 },
            { "min": 3.1, "max": 4.2 },
            { "min": 4.2, "max": 5.4 },
            { "min": 5.4, "max": 6.6 },
            { "min": 6.6, "max": 7.8 },
            { "min": 7.8, "max": 9 },
            { "min": 9, "max": 10 },
            { "min": 10, "max": 11 },
            { "min": 11, "max": 12 },
            { "min": 12, "max": 13 },
            { "min": 13, "max": 14 },
            { "min": 14, "max": 15 },
            { "min": 15, "max": 16 },
            { "min": 16, "max": 17 },
            { "min": 17, "max": 18 },
            { "min": 18, "max": 19 },
            { "min": 19, "max": float("inf") },
        ],
        "PM10": [
            { "min": 0, "max": 6 },
            { "min": 6, "max": 18 },
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
        # "TEMP": [
        #     { "min": -80.15, "max": -67.15 },
        #     { "min": -67.15, "max": -54.15 },
        #     { "min": -54.15, "max": -40.0 },
        #     { "min": -40.0, "max": -17.78 },
        #     { "min": -17.78, "max": 0.0 },
        #     { "min": 0.0, "max": 2.0 },
        #     { "min": 2.0, "max": 17.85 },
        #     { "min": 17.85, "max": 24.85 },
        #     { "min": 24.85, "max": 37.85 },
        #     { "min": 37.85, "max": 54.85 },
        #     { "min": 54.85, "max": float("inf") },
        # ],
        "TEMP": [
            { "min": 0,  "max": 4  },
            { "min": 4,  "max": 8  },
            { "min": 8,  "max": 12 },
            { "min": 12, "max": 16 },
            { "min": 16, "max": 20 },
            { "min": 20, "max": 24 },
            { "min": 24, "max": 28 },
            { "min": 28, "max": 32 },
            { "min": 32, "max": 36 },
            { "min": 36, "max": 40 },
            # { "min": -10, "max": -6  },
            # { "min": -6,  "max": -2  },
            # { "min": -2,  "max": 2   },
            # { "min": 2,   "max": 6   },
            # { "min": 6,   "max": 10  },
            # { "min": 10,  "max": 14  },
            # { "min": 14,  "max": 18  },
            # { "min": 18,  "max": 22  },
            # { "min": 22,  "max": 26  },
            # { "min": 26,  "max": 30  }
        ],
        "WIND": [
            { "min": 0, "max": 1 },
            { "min": 1, "max": 2 },
            { "min": 2, "max": 3 },
            { "min": 3, "max": 4 },
            { "min": 4, "max": 5 },
            { "min": 5, "max": 6 },
            { "min": 6, "max": 7 },
            { "min": 7, "max": 8 },
            { "min": 8, "max": 9 },
            { "min": 9, "max": 10 },
        ],
        "CAI": [
            { "min": 0, "max": 50 },
            { "min": 51, "max": 100 },
            { "min": 101, "max": 250 },
            { "min": 251, "max": 500 },
        ]
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
                
            elif poll == "SO2":
                conc = ds_aconc.variables['SO2'][tstep][layer]
                
            elif poll == "NO2":
                conc = ds_aconc.variables['NO2'][tstep][layer]
                
            elif poll == "CO":
                conc = ds_aconc.variables['CO'][tstep][layer]

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
                # test용(전부 20)
                # conc = np.full((nrows, ncols), 20.0, dtype=np.float32)

            elif poll == 'CAI':
                o3  = np.array(ds_aconc.variables["O3"][tstep][layer],  dtype=np.float32)
                so2 = np.array(ds_aconc.variables["SO2"][tstep][layer], dtype=np.float32)
                no2 = np.array(ds_aconc.variables["NO2"][tstep][layer], dtype=np.float32)
                co  = np.array(ds_aconc.variables["CO"][tstep][layer],  dtype=np.float32)

                pm10_arrays = [ds_aconc.variables[el][tstep][layer] for el in PM10_ELEMENTS]
                pm10 = np.sum(pm10_arrays, axis=0).astype(np.float32)

                pm25_arrays = [ds_aconc.variables[el][tstep][layer] for el in PM25_ELEMENTS]
                pm25 = np.sum(pm25_arrays, axis=0).astype(np.float32)
                
                conc = cai_from_arrays(o3=o3, so2=so2, no2=no2, co=co, pm10=pm10, pm25=pm25).astype(np.float32)
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

