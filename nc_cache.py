import netCDF4 as nc
import os
import threading

_NC_CACHE = {}
_NC_LOCK = threading.RLock()


def get_nc_dataset(path: str):
    path = os.path.abspath(path)
    with _NC_LOCK:
        ds = _NC_CACHE.get(path)
    
        if ds is None:
            print(f"📂 Opening netCDF file: {os.path.basename(path)}")
            ds = nc.Dataset(path, mode="r")
            _NC_CACHE[path] = ds
            
        return ds

def nc_lock():
    return _NC_LOCK