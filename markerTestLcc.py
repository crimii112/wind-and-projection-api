import netCDF4 as nc
import numpy as np
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from nc_cache import get_nc_dataset, nc_lock
from pyproj import CRS, Transformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
# GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
# METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')
# ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.09KM.2025063012.nc')
# GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_09KM.2025063012.nc')
# METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_09KM.2025063012.nc')

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

CAI_POLL_META = [
    ("O3",   "ppm"),
    ("SO2",  "ppm"),
    ("NO2",  "ppm"),
    ("CO",   "ppm"),
    ("PM10", "µg/m³"),
    ("PM2.5","µg/m³"),
]

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
    main_idx = subs.argmax(axis=0)  # 대표 물질 index
    
    n_unhealthy = (subs >= 101).sum(axis=0)  # '나쁨 이상' 개수 :contentReference[oaicite:2]{index=2}
    addon = np.where(n_unhealthy == 2, 50, np.where(n_unhealthy >= 3, 75, 0))  # :contentReference[oaicite:3]{index=3}

    cai = base + addon
    cai = np.minimum(cai, 500).astype(np.int16)
    
    return cai, main_idx

KST = timezone(timedelta(hours=9))
def get_datetime_from_tflag(ds, tstep):
    yyyyddd = int(ds.variables['TFLAG'][tstep, 0, 0])
    hhmmss  = int(ds.variables['TFLAG'][tstep, 0, 1])

    year = yyyyddd // 1000
    day_of_year = yyyyddd % 1000

    hour = hhmmss // 10000
    minute = (hhmmss % 10000) // 100
    second = hhmmss % 100

    # UTC datetime(파일 시간이 UTC)
    dt_utc = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_of_year - 1,
        hours=hour,
        minutes=minute,
        seconds=second
    )
    
    # UTC → KST 변환
    dt_kst = dt_utc.astimezone(KST)

    return dt_kst

def convert_flatten_array(ds, el, tstep, layer):
    list = [float(v) for v in ds.variables[el][tstep][layer].flatten()]
    return np.array(list)

def wdws_to_uv(wd_deg, ws):
    wd_rad = np.deg2rad(wd_deg)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v

def get_lcc_to_wgs84_transformer():
    lcc_crs = CRS.from_proj4(
        "+proj=lcc "
        "+lat_1=30 "
        "+lat_2=60 "
        "+lat_0=38 "
        "+lon_0=126 "
        "+x_0=0 "
        "+y_0=0 "
        "+a=6370000 "
        "+b=6370000 "
        "+units=m "
        "+no_defs"
    )

    wgs84 = CRS.from_epsg(4326)

    return Transformer.from_crs(lcc_crs, wgs84, always_xy=True)

def get_marker_test_lcc_data(grid_km, layer, tstep, bg_poll, arrow_gap):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError(f"Unsupported gridKm: {grid_km}")
    
        cfg = GRID_CONFIG[grid_km]
        
        aconc_path = os.path.join(SCRIPT_DIR, 'data', cfg["ACONC"])
        gridcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["GRIDCRO"])
        metcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["METCRO"])
        
        with nc_lock():
            ds_aconc = get_nc_dataset(aconc_path)
            ds_gridcro = get_nc_dataset(gridcro_path)
            ds_metcro = get_nc_dataset(metcro_path)
            
            transformer = get_lcc_to_wgs84_transformer()
            
            print(f"✅ NetCDF {grid_km}km files opened successfully.")
            
            ## datetime test(aconc는 tstep=239, metcro는 tstep=241)
            # print(get_datetime_from_tflag(ds_aconc, tstep))
            # print(get_datetime_from_tflag(ds_metcro, tstep))
            # print(get_datetime_from_tflag(ds_metcro, tstep))
            # print(get_datetime_from_tflag(ds_metcro, tstep))
            
            ## 격자(좌표계)
            XORIG = ds_gridcro.getncattr('XORIG')   # -180000.0(9km) / -2349000.0(27km)
            YORIG = ds_gridcro.getncattr('YORIG')   # -585000.0(9km) / -1728000.0(27km)
            XCELL = ds_gridcro.getncattr('XCELL')   # 9000.0(9km) / 27000.0(27km)
            YCELL = ds_gridcro.getncattr('YCELL')   # 9000.0(9km) / 27000.0(27km)

            # nrows, ncols = 82, 67 # 9km
            # nrows, ncols = 128, 174 # 27km
            nrows = cfg["nrows"]
            ncols = cfg["ncols"]    
            half = cfg["half_cell"]    
            
            lon = [[0 for j in range(ncols)] for i in range(nrows)]
            lat = [[0 for j in range(ncols)] for i in range(nrows)]
            for i in range(nrows):
                for j in range(ncols):
                    lon[i][j] = XORIG + (j * XCELL) + half # 4500(9km) / 13500(27km)
                    lat[i][j] = YORIG + (i * YCELL) + half # 4500(9km) / 13500(27km)

            # lon = np.zeros((nrows, ncols))
            # lat = np.zeros((nrows, ncols))

            # for i in range(nrows):
            #     for j in range(ncols):
            #         x = XORIG + (j * XCELL) + half
            #         y = YORIG + (i * YCELL) + half
            #         lon[i][j], lat[i][j] = transformer.transform(x, y)
            
            lon = np.array(lon)
            lat = np.array(lat)
            
            ########## 격자 폴리곤 데이터 ##########
            if bg_poll == 'O3':
                o3_arr = convert_flatten_array(ds_aconc, 'O3', tstep, layer)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(o3), "overlay": f"O3: {o3:.3f}(ppm)"}
                    for lat, lon, o3 in zip(lat.flatten(), lon.flatten(), o3_arr)
                ]
            elif bg_poll == 'SO2':
                so2_arr = convert_flatten_array(ds_aconc, 'SO2', tstep, layer)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(so2), "overlay": f"SO2: {so2:.3f}(ppm)"}
                    for lat, lon, so2 in zip(lat.flatten(), lon.flatten(), so2_arr)
                ]
            elif bg_poll == 'NO2':
                no2_arr = convert_flatten_array(ds_aconc, 'NO2', tstep, layer)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(no2), "overlay": f"NO2: {no2:.3f}(ppm)"}
                    for lat, lon, no2 in zip(lat.flatten(), lon.flatten(), no2_arr)
                ]
            elif bg_poll == 'CO':
                co_arr = convert_flatten_array(ds_aconc, 'CO', tstep, layer)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(co), "overlay": f"CO: {co:.3f}(ppm)"}
                    for lat, lon, co in zip(lat.flatten(), lon.flatten(), co_arr)
                ]
            elif bg_poll == 'PM10':
                # PM10
                pm10_all_arrays = []
                for el in PM10_ELEMENTS:
                    el_arr = convert_flatten_array(ds_aconc, el, tstep, layer)
                    pm10_all_arrays.append(el_arr)
                
                pm10_arr = np.sum(pm10_all_arrays, axis=0)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(pm10), "overlay": f"PM10: {int(round(pm10))}(µg/m³)"}
                    for lat, lon, pm10 in zip(lat.flatten(), lon.flatten(), pm10_arr)
                ]
            elif bg_poll == 'PM2.5':
                # PM2.5
                pm25_all_arrays = []
                for el in PM25_ELEMENTS:
                    el_array = convert_flatten_array(ds_aconc, el, tstep, layer)
                    pm25_all_arrays.append(el_array)
                
                pm25_arr = np.sum(pm25_all_arrays, axis=0)
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(pm25), "overlay": f"PM2.5: {int(round(pm25))}(µg/m³)"}
                    for lat, lon, pm25 in zip(lat.flatten(), lon.flatten(), pm25_arr)
                ]
            elif bg_poll == 'TEMP':
                # TEMP (K → ℃)
                temp_k = ds_metcro["TEMP2"][tstep][layer]
                temp_c = temp_k - 273.15
                
                # test용(전부 20)
                # temp_c = np.full((nrows,ncols), 20.0, dtype=np.float32)

                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(t), "overlay": f"TEMP: {t:.1f}(°C)"}
                    for lat, lon, t in zip(
                        lat.flatten(),
                        lon.flatten(),
                        temp_c.flatten()
                    )
                ]
            elif bg_poll == 'WIND':
                ws = ds_metcro["WSPD10"][tstep][layer]
                
                polygon_data = [
                    {'lat': float(lat), 'lon': float(lon), 'value': float(ws), "overlay": f"풍속: {ws:.3f}(m/s)"}
                    for lat, lon, ws in zip(
                        lat.flatten(),
                        lon.flatten(),
                        ws.flatten()
                    )
                ]
            elif bg_poll == 'CAI':
                o3_arr = convert_flatten_array(ds_aconc, 'O3', tstep, layer)
                so2_arr = convert_flatten_array(ds_aconc, 'SO2', tstep, layer)
                no2_arr = convert_flatten_array(ds_aconc, 'NO2', tstep, layer)
                co_arr = convert_flatten_array(ds_aconc, 'CO', tstep, layer)
                
                pm10_all = [convert_flatten_array(ds_aconc, el, tstep, layer) for el in PM10_ELEMENTS]
                pm10_arr = np.sum(pm10_all, axis=0)

                pm25_all = [convert_flatten_array(ds_aconc, el, tstep, layer) for el in PM25_ELEMENTS]
                pm25_arr = np.sum(pm25_all, axis=0)
                
                cai_arr, main_idx_arr = cai_from_arrays(
                    o3=o3_arr, so2=so2_arr, no2=no2_arr, co=co_arr,
                    pm10=pm10_arr, pm25=pm25_arr
                )
                
                all_values = [o3_arr, so2_arr, no2_arr, co_arr, pm10_arr, pm25_arr]
                
                polygon_data = []
                
                for lat_v, lon_v, cai, midx, vals in zip(
                    lat.flatten(),
                    lon.flatten(),
                    cai_arr,
                    main_idx_arr,
                    zip(*all_values)
                ):
                    poll, unit = CAI_POLL_META[midx]
                    conc = vals[midx]

                    if poll in ("PM10", "PM2.5"):
                        conc_txt = f"{int(round(conc))}"
                    else:
                        conc_txt = f"{conc:.3f}"
                        
                    overlay = f"CAI: {int(cai)}\n{poll} ({ conc_txt} {unit})"

                    polygon_data.append({
                        "lat": float(lat_v),
                        "lon": float(lon_v),
                        "value": int(cai),
                        "overlay": overlay
                    })
                
                # polygon_data = [
                #     {'lat': float(lat), 'lon': float(lon), 'value': int(cai)}
                #     for lat, lon, cai in zip(lat.flatten(), lon.flatten(), cai_arr)
                # ]
                
            ########## 바람 화살표 데이터 ##########
            # 풍향, 풍속            
            wds = ds_metcro.variables['WDIR10'][tstep][layer]
            wss = ds_metcro.variables['WSPD10'][tstep][layer]
            
            arrow_data = []
            for i in range(0, nrows, arrow_gap):
                for j in range(0, ncols, arrow_gap):
                    # 슬라이스 범위 (경계 체크)
                    i_end = min(i + arrow_gap, nrows)
                    j_end = min(j + arrow_gap, ncols)

                    if(i_end % arrow_gap != 0 or j_end % arrow_gap != 0):
                        break
                    
                    # 해당 블록 추출
                    lon_block = lon[i:i_end, j:j_end]
                    lat_block = lat[i:i_end, j:j_end]
                    wd_block = wds[i:i_end, j:j_end]
                    ws_block = wss[i:i_end, j:j_end]

                    # 각 블록의 중심
                    avg_lon = np.mean(lon_block)
                    avg_lat = np.mean(lat_block)
                    avg_wd = np.mean(wd_block)
                    avg_ws = np.mean(ws_block)

                    arrow_data.append({
                        'lat': float(avg_lat),
                        'lon': float(avg_lon),
                        'wd': float(avg_wd),
                        'ws': float(avg_ws)
                    })
                    
            result = {
                "polygonData": polygon_data,
                "arrowData": arrow_data,
                "datetime": get_datetime_from_tflag(ds_aconc, tstep).strftime("%Y-%m-%d %H:%M:%S")
            }

            return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def get_sido_shp():
    load_dotenv()
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWD')
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password
    )
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
    SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
        'type', 'Feature',
        'id', gid,
        'geometry', ST_AsGeoJSON(geom)::json,
        'properties', json_build_object(
            'gid', gid,
            'ctprvn_cd', ctprvn_cd,
            'ctp_eng_nm', ctp_eng_nm,
            'ctp_kor_nm', ctp_kor_nm,
            'sido_name', sido_name
        )
        )
    )
    ) AS geojson
    FROM public.ctprvn4326;
    """)
    
    result = cur.fetchone()
    geojson = result["geojson"]
    
    cur.close()
    conn.close()

    return {"sidoshp": geojson}
    
    