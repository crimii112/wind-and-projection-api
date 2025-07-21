import os
import pygrib
import numpy as np

###
# /test 페이지에서 사용
# 사용자가 선택한 date, time 받아옴
# 'https://nomads.ncep.noaa.gov'에서 데이터 호출
# grib2 => json
# 바람/히트맵(온도) 데이터 전송
###

def download_and_convert(date: str, time: str):
    # base_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/'
    # date_part = f"gfs.{date}/{time}/atmos/"
    # file_part = f"gfs.t{time}z.pgrb2.0p25.f000"
    
    # gfs_url = base_url + date_part + file_part
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grib_filename = os.path.join(script_dir, 'data', 'wind-korea.grib2')

    # print(f"📥 Downloading GFS data: {gfs_url}")
    # res = requests.get(gfs_url, stream=True)
    # if res.status_code != 200:
    #     raise Exception(f"GFS file not found or failed: {gfs_url}")

    # with open(grib_filename, 'wb') as f:
    #     for chunk in res.iter_content(8192):
    #         f.write(chunk)
            
    # print("✅ GRIB downloaded. Converting...")
    
    grbs = pygrib.open(grib_filename)
    
    # 10m 높이 바람
    u_grb = grbs.select(name='U component of wind', level=10)[0]
    v_grb = grbs.select(name='V component of wind', level=10)[0]
    t_grb = grbs.select(name='Temperature', level=2)[0]  # 2m 온도 (선택적)

    # 위치 좌표 및 값 가져오기
    lats, lons = u_grb.latlons()
    u_vals = u_grb.values
    v_vals = v_grb.values
    t_vals = t_grb.values
    
    # 필터링: 120~135E, 30~40N
    mask = (lons >= 120) & (lons <= 135) & (lats >= 30) & (lats <= 40)

    # 마스킹된 데이터만 사용
    u_masked = np.where(mask, u_vals, np.nan)
    v_masked = np.where(mask, v_vals, np.nan)
    t_masked = np.where(mask, t_vals, np.nan)

    # NaN이 포함되지 않는 행/열만 추출
    valid_rows = ~np.all(np.isnan(u_masked), axis=1)
    valid_cols = ~np.all(np.isnan(u_masked), axis=0)
    
    u_crop = u_masked[valid_rows][:, valid_cols]
    v_crop = v_masked[valid_rows][:, valid_cols]
    t_crop = t_masked[valid_rows][:, valid_cols]
    lats_crop = lats[valid_rows][:, valid_cols]
    lons_crop = lons[valid_rows][:, valid_cols]

    ny, nx = u_crop.shape
    lo1 = float(np.min(lons_crop))
    la1 = float(np.max(lats_crop))  # 위쪽 위도
    lo2 = float(np.max(lons_crop))
    la2 = float(np.min(lats_crop))  # 아래쪽 위도
    dx = (lo2 - lo1) / (nx - 1)
    dy = (la1 - la2) / (ny - 1)
    
    def make_data(arr):
        return [float(arr[i][j]) if not np.isnan(arr[i][j]) else 0.0
                for i in range(ny) for j in range(nx)]

    # heatmap 데아터
    lats_flat = lats_crop.ravel()
    lons_flat = lons_crop.ravel()
    tmps_flat = t_crop.ravel()
    
    heatmap_data = [
        {'lat': lat, 'lon': lon, 'tmp': tmp}
        for lat, lon, tmp in zip(lats_flat, lons_flat, tmps_flat)
    ]
    
    result = {
        "windData": [
            {
                "header": {
                    "parameterCategory": 2,
                    "parameterNumber": 2,
                    "nx": nx, "ny": ny,
                    "lo1": lo1, "la1": la1,
                    "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy,
                    "refTime": f"{date[:4]}-{date[4:6]}-{date[6:]}T{time}:00:00Z"
                },
                "data": make_data(u_crop)
            },
            {
                "header": {
                    "parameterCategory": 2,
                    "parameterNumber": 3,
                    "nx": nx, "ny": ny,
                    "lo1": lo1, "la1": la1,
                    "lo2": lo2, "la2": la2,
                    "dx": dx, "dy": dy,
                    "refTime": f"{date[:4]}-{date[4:6]}-{date[6:]}T{time}:00:00Z"
                },
                "data": make_data(v_crop)
            },
        ],
        "heatmapData": heatmap_data
    }

    return result
