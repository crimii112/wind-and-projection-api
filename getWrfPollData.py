import netCDF4 as nc
import numpy as np
import os
import json
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
# GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
# METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')
ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.09KM.2025063012.nc')
GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_09KM.2025063012.nc')
METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_09KM.2025063012.nc')

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

def convert_flatten_array(ds, el, tstep):
    list = [float(v) for v in ds.variables[el][tstep][0].flatten()]
    return np.array(list)

def filter_data_with_gap(arr, gap):
    return arr[::gap, ::gap]

def get_wind_data(wind_gap, tstep):
    ds_gridcro = nc.Dataset(GRIDCRO_PATH)
    ds_metcro = nc.Dataset(METCRO_PATH)

    # 격자
    lats = ds_gridcro.variables['LAT'][0][0]
    lons = ds_gridcro.variables['LON'][0][0]
    filtered_lats = filter_data_with_gap(lats, wind_gap)
    filtered_lons = filter_data_with_gap(lons, wind_gap)
    
    # 풍향, 풍속            
    wds = ds_metcro.variables['WDIR10'][tstep][0]
    wss = ds_metcro.variables['WSPD10'][tstep][0]
    filtered_wd = filter_data_with_gap(wds, wind_gap)
    filtered_ws = filter_data_with_gap(wss, wind_gap)

    wd = np.array([float(v) for v in filtered_wd.flatten()])
    ws = np.array([float(v) for v in filtered_ws.flatten()])
    
    
    # 풍향, 풍속 => U, V 변환
    rad = np.radians(wd)
    # rad = np.deg2rad(wd)
    u = -ws * np.sin(rad)
    v = -ws * np.cos(rad)
    # u = -ws * np.cos(rad)
    # v = -ws * np.sin(rad)

    # ol-wind에 보내야하는 데이터
    lo1 = float(np.min(filtered_lons))      
    la1 = float(np.min(filtered_lats))
    lo2 = float(np.max(filtered_lons))
    la2 = float(np.max(filtered_lats)) 
    nx, ny = filtered_wd.shape
    dx = (lo2 - lo1) / (nx - 1)
    dy = (la2 - la1) / (ny - 1)
    
    # 시간 
    SDATE = ds_gridcro.getncattr('SDATE')
    STIME = ds_gridcro.getncattr('STIME')
    TSTEP = ds_gridcro.getncattr('TSTEP')

    SDATE = int(SDATE)
    STIME = int(STIME)
    
    year = SDATE // 1000
    day_of_year = SDATE % 1000
    date_part = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    hour = STIME // 10000

    start_dt = datetime(
        date_part.year,
        date_part.month,
        date_part.day,
        hour,
        00,
        00
    )   
    refTime = start_dt.strftime('%Y-%m-%d_%H:%M:%S') 
    
    final_dt = start_dt + timedelta(hours=tstep+9)
    time = final_dt.strftime('%Y-%m-%d %H:%M:%S KST')
    print(time)
    
    # ol-wind WindLayer에 사용할 데이터
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
    
    return wind_data, time




def convert_poll_nc_to_json(poll, wind_gap, tstep):
    try:
        ds_aconc = nc.Dataset(ACONC_PATH)
        ds_gridcro = nc.Dataset(GRIDCRO_PATH)
        ds_metcro = nc.Dataset(METCRO_PATH)
        print("✅ NetCDF file opened successfully.")

        # 격자
        lat = ds_gridcro.variables['LAT'][0][0]
        lon = ds_gridcro.variables['LON'][0][0]
        
        ### 물질 ###
        # TMP
        tmp_arr = convert_flatten_array(ds_metcro, 'TEMP2', tstep-1)
        
        # O3
        o3_arr = convert_flatten_array(ds_aconc, 'O3', tstep-1)
        
        # PM10
        pm10_all_arrays = []
        for idx, el in enumerate(PM10_ELEMENTS):
            el_array = convert_flatten_array(ds_aconc, el, tstep-1)
            pm10_all_arrays.append(el_array)
        
        pm10_arr = np.sum(pm10_all_arrays, axis=0)
        
        # PM25
        pm25_all_arrays = []
        for idx, el in enumerate(PM25_ELEMENTS):
            el_array = convert_flatten_array(ds_aconc, el, tstep-1)
            pm25_all_arrays.append(el_array)
        
        pm25_arr = np.sum(pm25_all_arrays, axis=0)
        
        # 물질별 heatmap_data
        if poll == 'tmp':
            heatmap_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(tmp)-273.15}
                for lat, lon, tmp in zip(lat.flatten(), lon.flatten(), tmp_arr)
            ]
        elif poll == 'o3':
            heatmap_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(o3)}
                for lat, lon, o3 in zip(lat.flatten(), lon.flatten(), o3_arr)
            ]
        elif poll == 'pm10':
            heatmap_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(pm10)}
                for lat, lon, pm10 in zip(lat.flatten(), lon.flatten(), pm10_arr)
            ]
        elif poll == 'pm2.5':
            heatmap_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(pm25)}
                for lat, lon, pm25 in zip(lat.flatten(), lon.flatten(), pm25_arr)
            ]

        wind_data, time = get_wind_data(wind_gap, tstep-1)
        
        result = {
            "windData": wind_data,    # 바람은 공통
            "heatmapData": heatmap_data,
            "metaData": {'time': time}
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)