import netCDF4 as nc
import numpy as np
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from flask import jsonify

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
    print(ds.variables[el][tstep][0])
    list = [float(v) for v in ds.variables[el][tstep][layer].flatten()]
    return np.array(list)

def get_marker_test_lcc_data(grid_km, layer, tstep, bg_poll, arrow_gap):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError(f"Unsupported gridKm: {grid_km}")
    
        cfg = GRID_CONFIG[grid_km]
        
        aconc_path = os.path.join(SCRIPT_DIR, 'data', cfg["ACONC"])
        gridcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["GRIDCRO"])
        metcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["METCRO"])
        
        ds_aconc = nc.Dataset(aconc_path)
        ds_gridcro = nc.Dataset(gridcro_path)
        ds_metcro = nc.Dataset(metcro_path)
        
        print(f"✅ NetCDF {grid_km}km files opened successfully.")
        
        ## datetime test(aconc는 tstep=239, metcro는 tstep=241)
        print(get_datetime_from_tflag(ds_aconc, tstep))
        print(get_datetime_from_tflag(ds_metcro, tstep))
        print(get_datetime_from_tflag(ds_metcro, tstep))
        print(get_datetime_from_tflag(ds_metcro, tstep))
        
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

        lon = np.array(lon)
        lat = np.array(lat)
        
        ########## 격자 폴리곤 데이터 ##########
        if bg_poll == 'O3':
            o3_arr = convert_flatten_array(ds_aconc, 'O3', tstep, layer)
            polygon_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(o3)}
                for lat, lon, o3 in zip(lat.flatten(), lon.flatten(), o3_arr)
            ]
        elif bg_poll == 'PM10':
            # PM10
            pm10_all_arrays = []
            for el in PM10_ELEMENTS:
                el_arr = convert_flatten_array(ds_aconc, el, tstep, layer)
                pm10_all_arrays.append(el_arr)
            
            pm10_arr = np.sum(pm10_all_arrays, axis=0)
            polygon_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(pm10)}
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
                {'lat': float(lat), 'lon': float(lon), 'value': float(pm25)}
                for lat, lon, pm25 in zip(lat.flatten(), lon.flatten(), pm25_arr)
            ]
        
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
        exit(1)
        

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
    
    