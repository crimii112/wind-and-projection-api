import os
import numpy as np
import netCDF4 as nc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACONC_PATH = os.path.join(SCRIPT_DIR, 'data', 'ACONC.27KM.2025063012.nc')
GRIDCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'GRIDCRO2D_27KM.2025063012.nc')
METCRO_PATH = os.path.join(SCRIPT_DIR, 'data', 'METCRO2D_27KM.2025063012.nc')

def set_grid():
    ds_gridcro = nc.Dataset(GRIDCRO_PATH)
    ds_metcro = nc.Dataset(METCRO_PATH)
    
    # 격자(위경도)
    lat = ds_gridcro.variables['LAT'][0][0]
    lon = ds_gridcro.variables['LON'][0][0]
    
    lo1 = float(np.min(lon))      
    la1 = float(np.min(lat))
    lo2 = float(np.max(lon))
    la2 = float(np.max(lat)) 
        
    print(lat, lon)
    print(lat.shape)
    print(lo1, lo2) # 91.72314453125, 160.27685546875
    print(la1, la2) # 19.91510009765625, 53.917694091796875
    
set_grid()