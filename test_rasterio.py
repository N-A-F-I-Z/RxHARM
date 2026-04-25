import time
import rasterio
from rasterio.windows import from_bounds
import urllib.request

url = 'https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/2025/BGD/v1/100m/constrained/bgd_t_00_2025_CN_100m_R2025A_v1.tif'

# Test if server supports range requests
req = urllib.request.Request(url, method='HEAD')
with urllib.request.urlopen(req) as resp:
    print(f"Accept-Ranges: {resp.headers.get('Accept-Ranges')}")

t0 = time.time()
try:
    with rasterio.Env(GDAL_HTTP_TIMEOUT=10):
        with rasterio.open(url) as src:
            print("Opened directly! Profile:", src.profile)
            
            # Tiny bounding box in Bangladesh (Dhaka roughly)
            bounds = (90.35, 23.75, 90.45, 23.85)
            win = from_bounds(*bounds, src.transform)
            
            data = src.read(1, window=win)
            print(f"Read shape: {data.shape}")
            print(f"Sum: {data.sum()}")
            
    print(f"Direct read took {time.time() - t0:.2f} seconds")
except Exception as e:
    print("Error:", e)
