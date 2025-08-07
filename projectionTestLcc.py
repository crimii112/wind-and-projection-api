import netCDF4 as nc
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')
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

def convert_flatten_array(ds, el, tstep):
    print(ds.variables[el][tstep][0])
    list = [float(v) for v in ds.variables[el][tstep][0].flatten()]
    return np.array(list)

def get_projection_test_lcc_data(bg_poll, arrow_gap):
    try:
        ds_aconc = nc.Dataset(ACONC_PATH)
        ds_gridcro = nc.Dataset(GRIDCRO_PATH)
        ds_metcro = nc.Dataset(METCRO_PATH)
        print("✅ NetCDF file opened successfully.")
        
        ## 격자(좌표계)
        XORIG = ds_gridcro.getncattr('XORIG')   # -180000.0(9km) / -2349000.0(27km)
        YORIG = ds_gridcro.getncattr('YORIG')   # -585000.0(9km) / -1728000.0(27km)
        XCELL = ds_gridcro.getncattr('XCELL')   # 9000.0(9km) / 27000.0(27km)
        YCELL = ds_gridcro.getncattr('YCELL')   # 9000.0(9km) / 27000.0(27km)

        # nrows, ncols = 82, 67 # 9km
        nrows, ncols = 128, 174 # 27km    
        
        lon = [[0 for j in range(ncols)] for i in range(nrows)]
        lat = [[0 for j in range(ncols)] for i in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                lon[i][j] = XORIG + (j * XCELL) + (XCELL / 2) # 4500(9km) / 13500(27km)
                lat[i][j] = YORIG + (i * YCELL) + (XCELL / 2) # 4500(9km) / 13500(27km)

        lon = np.array(lon)
        lat = np.array(lat)
        
        ########## 격자 폴리곤 데이터 ##########
        if bg_poll == 'O3':
            o3_arr = convert_flatten_array(ds_aconc, 'O3', 0)
            polygon_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(o3)}
                for lat, lon, o3 in zip(lat.flatten(), lon.flatten(), o3_arr)
            ]
        elif bg_poll == 'PM10':
            # PM10
            pm10_all_arrays = []
            for el in PM10_ELEMENTS:
                el_arr = convert_flatten_array(ds_aconc, el, 0)
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
                el_array = convert_flatten_array(ds_aconc, el, 0)
                pm25_all_arrays.append(el_array)
            
            pm25_arr = np.sum(pm25_all_arrays, axis=0)
            polygon_data = [
                {'lat': float(lat), 'lon': float(lon), 'value': float(pm25)}
                for lat, lon, pm25 in zip(lat.flatten(), lon.flatten(), pm25_arr)
            ]
        
        ##### TMP #####
        # tmp_arr = convert_flatten_array(ds_metcro, 'TEMP2', 0)
        # polygon_data = [
        #     {'lat': float(lat), 'lon': float(lon), 'value': float(tmp)-273.15}
        #     for lat, lon, tmp in zip(lat.flatten(), lon.flatten(), tmp_arr)
        # ]
        
        
        ########## 바람 화살표 데이터 ##########
        # 풍향, 풍속            
        wds = ds_metcro.variables['WDIR10'][0][0]
        wss = ds_metcro.variables['WSPD10'][0][0]
        
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
        
        result = {
            "polygonData": polygon_data,
            "arrowData": arrow_data,
        }
        
        # with open('wind.json', 'w') as f :
        #     json.dump(wind_data, f, indent=4)
        
        # with open('temp.json', 'w') as f :
        #     json.dump(temp_data, f, indent=4)

        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
        
