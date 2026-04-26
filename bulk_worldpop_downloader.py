import os
import requests
import rasterio
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress rasterio NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ==========================================
# CONFIGURATION
# ==========================================
# List of countries to process (ISO3 codes)
COUNTRIES = [
    "BGD", "IND", "PAK", "AUS", "NGA", 
    "ETH", "PHL", "IDN", "BRA", "MEX",
    # Add more up to 100...
]

START_YEAR = 2000
END_YEAR = 2030

# Age bands definition
CHILD_BANDS = [0, 1]                                      # Children (< 5)
ELDERLY_BANDS = [60, 65, 70, 75, 80, 85, 90]              # Elderly (60+)
IN_BETWEEN_BANDS = list(range(5, 60, 5))                  # In-between (5 - 59)
ALL_BANDS = CHILD_BANDS + IN_BETWEEN_BANDS + ELDERLY_BANDS

# Output directory
OUTPUT_DIR = Path("worldpop_bulk_export")
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR = OUTPUT_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)


# ==========================================
# FUNCTIONS
# ==========================================
def get_worldpop_url(iso3: str, year: int, age: int, sex: str = "t") -> str:
    """Generate the WorldPop URL based on the year (2000-2020 vs 2015-2030 formats)."""
    iso_lower = iso3.lower()
    
    # WorldPop Global 2000-2020 dataset
    if year < 2015:
        filename = f"{iso_lower}_{sex}_{age}_{year}.tif"
        return f"https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/{year}/{iso3}/{filename}"
    
    # WorldPop Global 2015-2030 (R2025A) 100m dataset
    else:
        filename = f"{iso_lower}_{sex}_{age:02d}_{year}_CN_100m_R2025A_v1.tif"
        return f"https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/{year}/{iso3}/v1/100m/constrained/{filename}"


def download_band(iso3: str, year: int, age: int) -> str:
    """Downloads a single band and returns the local file path."""
    url = get_worldpop_url(iso3, year, age)
    local_path = TMP_DIR / f"{iso3}_{year}_t_{age}.tif"
    
    if local_path.exists():
        return str(local_path)
    
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            # Older dataset (pre-2015) uses separate 'f' and 'm' instead of 't' for total
            if year < 2015:
                return download_and_merge_fm(iso3, year, age, local_path)
            return None
            
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(local_path)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def download_and_merge_fm(iso3: str, year: int, age: int, out_path: Path) -> str:
    """Fallback for pre-2015 data: downloads male and female bands, sums them."""
    f_url = get_worldpop_url(iso3, year, age, "f")
    m_url = get_worldpop_url(iso3, year, age, "m")
    
    f_path = TMP_DIR / f"{iso3}_{year}_f_{age}.tif"
    m_path = TMP_DIR / f"{iso3}_{year}_m_{age}.tif"
    
    for url, path in [(f_url, f_path), (m_url, m_path)]:
        if not path.exists():
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code != 200:
                return None
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    # Read, sum, and save to out_path
    with rasterio.open(f_path) as src_f, rasterio.open(m_path) as src_m:
        meta = src_f.meta
        f_arr = src_f.read(1).astype(float)
        m_arr = src_m.read(1).astype(float)
        
        nodata = src_f.nodata
        f_arr[f_arr == nodata] = 0
        m_arr[m_arr == nodata] = 0
        
        total = f_arr + m_arr
        total[f_arr == nodata] = nodata # Restore nodata where both were nodata (rough approximation)
        
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(total.astype(meta['dtype']), 1)
        
    f_path.unlink()
    m_path.unlink()
    
    return str(out_path)


def process_country_year(iso3: str, year: int):
    """Downloads all bands for a country/year, aggregates into 3 bands, and exports a GeoTIFF."""
    out_file = OUTPUT_DIR / f"WorldPop_100m_{iso3}_{year}_3band.tif"
    if out_file.exists():
        print(f"Skipping {iso3} {year} - already processed.")
        return

    print(f"[{iso3} {year}] Downloading {len(ALL_BANDS)} bands...")
    
    downloaded_files = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_age = {executor.submit(download_band, iso3, year, age): age for age in ALL_BANDS}
        for future in as_completed(future_to_age):
            age = future_to_age[future]
            path = future.result()
            if path:
                downloaded_files[age] = path
            else:
                print(f"[{iso3} {year}] Missing data for age {age}")
                
    if len(downloaded_files) < len(ALL_BANDS):
        print(f"[{iso3} {year}] FAILED: Missing bands. Skipping.")
        return

    print(f"[{iso3} {year}] Aggregating bands...")
    
    # Open the first file to get metadata
    with rasterio.open(downloaded_files[0]) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)
        nodata = src.nodata

    def _sum_bands(ages):
        total = np.zeros(shape, dtype=float)
        for a in ages:
            with rasterio.open(downloaded_files[a]) as src:
                arr = src.read(1).astype(float)
                if nodata is not None:
                    arr[arr == nodata] = 0
                total += arr
        return total

    children = _sum_bands(CHILD_BANDS)
    elderly = _sum_bands(ELDERLY_BANDS)
    in_between = _sum_bands(IN_BETWEEN_BANDS)

    # Update metadata for 3 bands
    meta.update({
        'count': 3,
        'dtype': 'float32',
        'nodata': -9999.0
    })

    print(f"[{iso3} {year}] Writing {out_file.name}...")
    with rasterio.open(out_file, 'w', **meta) as dst:
        dst.write(children.astype('float32'), 1)
        dst.write(elderly.astype('float32'), 2)
        dst.write(in_between.astype('float32'), 3)
        dst.set_band_description(1, 'Children (0-4)')
        dst.set_band_description(2, 'Elderly (60+)')
        dst.set_band_description(3, 'In_Between (5-59)')

    # Cleanup tmp files
    for path in downloaded_files.values():
        Path(path).unlink(missing_ok=True)
        
    print(f"[{iso3} {year}] DONE!")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    for country in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            process_country_year(country, year)
