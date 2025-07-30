import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
# GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
# METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')
ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.09KM.2025063012.nc')
GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_09KM.2025063012.nc')
METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_09KM.2025063012.nc')

def convert_flatten_array(ds, el, tstep):
    list = [float(v) for v in ds.variables[el][tstep][0].flatten()]
    return np.array(list)

def filter_data_with_gap(arr, gap):
    return arr[::gap, ::gap]

## 바람 화살표 데이터
def get_arrow_data(arrow_gap):
    ds_gridcro = nc.Dataset(GRIDCRO_PATH)
    ds_metcro = nc.Dataset(METCRO_PATH)
    
    ## 격자(좌표계)
    XORIG = ds_gridcro.getncattr('XORIG')   # -180000.0
    YORIG = ds_gridcro.getncattr('YORIG')   # -585000.0
    XCELL = ds_gridcro.getncattr('XCELL')   # 9000.0
    YCELL = ds_gridcro.getncattr('YCELL')   # 9000.0

    lon = [[0 for j in range(67)] for i in range(82)]
    lat = [[0 for j in range(67)] for i in range(82)]
    for i in range(82):
        for j in range(67):
            lon[i][j] = XORIG + (j * XCELL) + 4500
            lat[i][j] = YORIG + (i * YCELL) + 4500

    lon = np.array(lon)
    lat = np.array(lat)
    
    # 풍향, 풍속            
    wds = ds_metcro.variables['WDIR10'][0][0]
    wss = ds_metcro.variables['WSPD10'][0][0]
    
    nrows, ncols = 82, 67
    arrow_data = []
    for i in range(0, nrows, arrow_gap):
        for j in range(0, ncols, arrow_gap):
            # 슬라이스 범위 (경계 체크)
            i_end = min(i + arrow_gap, nrows)
            j_end = min(j + arrow_gap, ncols)

            # 해당 블록 추출
            lon_block = lon[i:i_end, j:j_end]
            lat_block = lat[i:i_end, j:j_end]
            wd_block = wds[i:i_end, j:j_end]
            ws_block = wss[i:i_end, j:j_end]

            # 각 블록의 평균(혹은 중심)
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
    
    # lonForGap = lon[::arrow_gap, ::arrow_gap]
    # latForGap = lat[::arrow_gap, ::arrow_gap]
    
    # wdForGap = np.array(wds)[::arrow_gap, ::arrow_gap]
    # wsForGap = np.array(wss)[::arrow_gap, ::arrow_gap]
    
    # arrow_data = [
    #     {'lat': float(lat), 'lon': float(lon), 'wd': float(wd), 'ws': float(ws)}
    #     for lat, lon, wd, ws in zip(latForGap.flatten(), lonForGap.flatten(), wdForGap.flatten(), wsForGap.flatten())
    # ]
    
    return arrow_data

def get_projection_test_data(arrow_gap):
    try:
        ds_aconc = nc.Dataset(ACONC_PATH)
        ds_gridcro = nc.Dataset(GRIDCRO_PATH)
        ds_metcro = nc.Dataset(METCRO_PATH)
        print("✅ NetCDF file opened successfully.")

        # 격자(위경도)
        lat = ds_gridcro.variables['LAT'][0][0]
        lon = ds_gridcro.variables['LON'][0][0]
        
        ## 격자(좌표계)
        XORIG = ds_gridcro.getncattr('XORIG')   # -180000.0
        YORIG = ds_gridcro.getncattr('YORIG')   # -585000.0
        XCELL = ds_gridcro.getncattr('XCELL')   # 9000.0
        YCELL = ds_gridcro.getncattr('YCELL')   # 9000.0

        lon = [[0 for j in range(67)] for i in range(82)]
        lat = [[0 for j in range(67)] for i in range(82)]
        for i in range(82):
            for j in range(67):
                lon[i][j] = XORIG + (j * XCELL) + 4500
                lat[i][j] = YORIG + (i * YCELL) + 4500

        lon = np.array(lon)
        lat = np.array(lat)
        
        # 풍향, 풍속            
        wds = ds_metcro.variables['WDIR10'][0][0]
        wss = ds_metcro.variables['WSPD10'][0][0]
        
        print(wds.shape)
        wd = np.array([float(v) for v in wds.flatten()])
        ws = np.array([float(v) for v in wss.flatten()])
        
        # 풍향, 풍속 => U, V 변환
        rad = np.radians(wd)
        # rad = np.deg2rad(wd)
        u = -ws * np.sin(rad)
        v = -ws * np.cos(rad)
        # u = -ws * np.cos(rad)
        # v = -ws * np.sin(rad)

        # ol-wind에 보내야하는 데이터
        lo1 = float(np.min(lon))      
        la1 = float(np.min(lat))
        lo2 = float(np.max(lon))
        la2 = float(np.max(lat)) 
        nx, ny = wds.shape
        dxs = np.diff(lon, axis = 1).mean(0)
        dx = float(dxs.mean())
        # dx = (lo2 - lo1) / (nx - 1)
        dys = np.diff(lat, axis = 0).mean(1)
        dy = float(dys.mean())
        # dy = (la2 - la1) / (ny - 1)
        
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
        refTime = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ') 
        
        final_dt = start_dt + timedelta(hours=9)
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
                    # "dx": 21912, "dy": 27810,     #0.25
                    # "dx": 87649, "dy": 111239,      #1.0
                    "refTime": refTime,
                    "surface1Type": 100,
                    "surface1Value": 100000.0,
                    "forecastTime": 0,
                    "scanMode": 0
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
                    # "dx": 21912, "dy": 27810,     #0.25
                    # "dx": 87649, "dy": 111239,      #1.0
                    "refTime": refTime,
                    "surface1Type": 100,
                    "surface1Value": 100000.0,
                    "forecastTime": 0,
                    "scanMode": 0
                },
                "data": v.tolist()
            },
        ]

        ### 물질 ###
        # TMP
        tmp_arr = convert_flatten_array(ds_metcro, 'TEMP2', 0)
        
        # 물질별 heatmap_data
        heatmap_data = [
            {'lat': float(lat), 'lon': float(lon), 'value': float(tmp)-273.15}
            for lat, lon, tmp in zip(lat.flatten(), lon.flatten(), tmp_arr)
        ]
        
        # temp_data = [
        #     {
        #         "header":{
        #             "parameterCategory": 0,
        #             "parameterNumber": 0,
        #             "nx": nx, "ny": ny,
        #             "lo1": lo1, "la1": la1,
        #             "lo2": lo2, "la2": la2,
        #             "dx": dx, "dy": dy,
        #             "refTime": refTime,
        #             "surface1Type": 1,
        #             "surface1Value": 2,
        #             "forecastTime": 0,
        #             "scanMode": 0
        #         },
        #         "data": tmp_arr.tolist()
        #     }
        # ]

        
        # 바람 화살표 데이터
        
        
        result = {
            "windData": wind_data,    # 바람은 공통
            "heatmapData": heatmap_data,
            "arrowData": get_arrow_data(arrow_gap),
            "metaData": {'time': time}
        }
        
        # with open('wind.json', 'w') as f :
        #     json.dump(wind_data, f, indent=4)
        
        # with open('temp.json', 'w') as f :
        #     json.dump(temp_data, f, indent=4)

        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
        
