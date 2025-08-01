import os
import numpy as np
import netCDF4 as nc
import json
import math

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
    
    lon_min = float(np.min(lon))      
    lat_min = float(np.min(lat))
    lon_max = float(np.max(lon))
    lat_max = float(np.max(lat)) 
        
    # print(lat, lon)
    # print(lat.shape)  # (128, 174)
    # print(lon_min, lon_max) # 91.72314453125, 160.27685546875
    # print(lat_min, lat_max) # 19.91510009765625, 53.917694091796875
    
    # 풍향, 풍속            
    wds = ds_metcro.variables['WDIR10'][0][0]
    wss = ds_metcro.variables['WSPD10'][0][0]
    wd = np.array([float(v) for v in wds.flatten()])
    ws = np.array([float(v) for v in wss.flatten()])
    
    # 풍향, 풍속 => U, V 변환
    rad = np.radians(wd)
    u = -ws * np.sin(rad)
    v = -ws * np.cos(rad)
    
    original_data = [
        {'lon': float(lon), 'lat': float(lat), 'u': float(u), 'v': float(v)}
        for lon, lat, u, v in zip(lon.flatten(), lat.flatten(), u, v)
    ]
    print(np.array(original_data).shape)    # (22272,) => 128 * 174
    print(np.array(original_data))    # (22272,) => 128 * 174
    
    # numpy 배열로도 만들기
    orig_lon = np.array([d['lon'] for d in original_data])
    orig_lat = np.array([d['lat'] for d in original_data])
    orig_u = np.array([d['u'] for d in original_data])
    orig_v = np.array([d['v'] for d in original_data])
    
    grid = []   
    for lat in np.arange(19.5, 54.1, 0.1):
        row = []
        for lon in np.arange(91.5, 160.6, 0.1):
            lon_min = lon - 0.05
            lon_max = lon + 0.05
            lat_min = lat - 0.05
            lat_max = lat + 0.05
            mask = (
                (orig_lon >= lon_min) & (orig_lon < lon_max) &
                (orig_lat >= lat_min) & (orig_lat < lat_max)
            )
            
            if np.any(mask):
                u_mean = np.mean(orig_u[mask])
                v_mean = np.mean(orig_v[mask])
            else:
                u_mean = np.nan
                v_mean = np.nan
                
            row.append({'lon': float(round(lon, 1)), 'lat': float(round(lat, 1)), 'u': float(u_mean), 'v': float(v_mean)})
        grid.append(row)
    
    # filtered_grid = [
    #     cell for row in grid for cell in row
    #     if not (math.isnan(cell['u']) or math.isnan(cell['v']))
    # ]
    # print(np.array(filtered_grid).shape)
    
    # print(np.array(grid).shape) # (346, 691)
    # print(np.array(grid))
    
    ## nan처리 => 가까운 값들 평균내서 넣기
    def fill_nan_with_neighbors(grid):
        rows = len(grid)
        cols = len(grid[0])
        filled_grid = [[cell.copy() for cell in row] for row in grid]

        changed = True
        while changed:
            changed = False
            for i in range(rows):
                for j in range(cols):
                    cell = filled_grid[i][j]
                    if math.isnan(cell['u']) or math.isnan(cell['v']):
                        neighbors = []
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < rows and 0 <= nj < cols:
                                n_cell = filled_grid[ni][nj]
                                if not (math.isnan(n_cell['u']) or math.isnan(n_cell['v'])):
                                    neighbors.append(n_cell)
                                    
                        if neighbors:
                            cell['u'] = float(np.mean([n['u'] for n in neighbors]))
                            cell['v'] = float(np.mean([n['v'] for n in neighbors]))
                            changed = True
        return filled_grid
    
    grid = fill_nan_with_neighbors(grid)
    
    # print(np.array(grid).shape) # (346, 691)
    # print(np.array(grid))
    
    # with open('gridFinal.json', 'w') as f :
    #     json.dump(grid, f, indent=4)
    
    ## earth 데이터에 맞춰서 만들기
    lo1 = 91.5
    la1 = 19.5
    lo2 = 160.5
    la2 = 54.0
    nx = np.array(grid).shape[1]
    ny = np.array(grid).shape[0]
    dx = 0.1
    dy = 0.1
    
    grid_u = np.array([d['u'] for d in np.array(grid).flatten()])
    grid_v = np.array([d['v'] for d in np.array(grid).flatten()])
    
    windData = [
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 2,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": "2025-06-30T12:00:00Z",
                "surface1Type": 100,
                "surface1Value": 100000.0,
                "forecastTime": 0,
                "scanMode": 0
            },
            "data": grid_u.tolist()
        },
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 3,
                "nx": nx, "ny": ny,
                "lo1": lo1, "la1": la1,
                "lo2": lo2, "la2": la2,
                "dx": dx, "dy": dy,
                "refTime": "2025-06-30T12:00:00Z",
                "surface1Type": 100,
                "surface1Value": 100000.0,
                "forecastTime": 0,
                "scanMode": 0
            },
            "data": grid_v.tolist()
        },
    ]
    
    # with open('set-grid-wind-0.1.json', 'w') as f :
    #     json.dump(windData, f, indent=4)
    
set_grid()