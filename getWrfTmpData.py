import netCDF4 as nc
import numpy as np
import os

###
# / 페이지에서 사용
# 'wrfout_d02_2025-06-29_12_00_00.nc' 파일 데이터 사용
# netCDF => json
# 바람/히트맵(온도) 데이터 전송
###

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, 'data', 'wrfout_d02_2025-06-29_12_00_00.nc')

def convert_tmp_nc_to_json():
    try:
        ds = nc.Dataset(FILE_PATH)
        print("✅ NetCDF file opened successfully.")

        lats = ds.variables['XLAT'][0]
        lons = ds.variables['XLONG'][0]
        
        u10 = ds.variables['U10'][0]
        v10 = ds.variables['V10'][0]
        t2 = ds.variables['T2'][0]

        ny, nx = u10.shape

        lo1 = float(np.min(lons))      
        la1 = float(np.min(lats))
        lo2 = float(np.max(lons))
        la2 = float(np.max(lats))  
        
        dx = (lo2 - lo1) / (nx - 1)
        dy = (la2 - la1) / (ny - 1)

        refTime = str(ds.variables['Times'][0].tobytes().decode('utf-8')).strip()
        
        def make_data(arr):
            return [float(v) for v in arr.flatten()]
        
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
                "data": make_data(u10)
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
                "data": make_data(v10)
            },
        ]
        
        heatmap_data = [
            {'lat': float(lat), 'lon': float(lon), 'value': float(tmp) - 273.15}
            for lat, lon, tmp in zip(lats.flatten(), lons.flatten(), t2.flatten())
        ]
        
        result = {
            "windData": wind_data,
            "heatmapData": heatmap_data
        }

        return result

    except Exception as e:
        print(f"❌ Error opening NetCDF file: {e}")
        exit(1)
        