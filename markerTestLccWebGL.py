import netCDF4 as nc
import numpy as np
import os
import json
from io import BytesIO
from nc_cache import get_nc_dataset, nc_lock
from pyproj import CRS, Transformer
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def get_webgl_wind_png(grid_km, layer, tstep):
    try:
        if grid_km not in GRID_CONFIG:
            raise ValueError("Unsupported grid")
    
        cfg = GRID_CONFIG[grid_km]
        
        gridcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["GRIDCRO"])
        metcro_path = os.path.join(SCRIPT_DIR, 'data', cfg["METCRO"])
        
        with nc_lock():
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
            half = cfg["half_cell"]    
            
            # LCC extent (grid 전체 영역)
            extent_lcc = [
                XORIG,
                YORIG,
                XORIG + ncols * XCELL,
                YORIG + nrows * YCELL,
            ]

            ########## 바람 화살표 데이터 ##########
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
            
            # PNG 저장
            # Image.fromarray(rgba, "RGBA").save(f"wind/{out_name}.png")
            
            # 파일 저장 x -> 메모리 png
            png_buffer = BytesIO()
            Image.fromarray(rgba, "RGBA").save(png_buffer, format="PNG")
            png_buffer.seek(0)

            webgl_meta = {
                "width": ncols,
                "height": nrows,
                "uMin": u_min,
                "uMax": u_max,
                "vMin": v_min,
                "vMax": v_max,
                "gridKm": grid_km,
                "extentLCC": extent_lcc,
            }
            
            # with open(f"wind/{out_name}.json", "w") as f:
            #     json.dump(webgl_data, f, indent=2)
            
            # print(f"✅ wind/{out_name}.png & json 생성 완료")

            return png_buffer, webgl_meta
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

