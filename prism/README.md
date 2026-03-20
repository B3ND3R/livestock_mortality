# PRISM Kenya/Marsabit — Local Development Setup

## Architecture

```
Windows machine
├── Docker Desktop
│   └── TiTiler container (serves COGs as map tiles on port 8000)
├── prism-app/
│   └── frontend/ (React app on port 3000)
└── C:\Users\abhas\prism-project\cogs\
    ├── ndvi/   (217 COGs, ~500m MODIS monthly 2008-2026)
    ├── evi/
    ├── fpar/
    └── lai/
```

## Prerequisites

- Docker Desktop for Windows
- Node.js v18+
- Yarn package manager
- Git

## Step 1: NetCDF → COG Conversion (Google Colab)

```python
import xarray as xr
import rioxarray
from pathlib import Path

ds = xr.open_dataset('your_file.nc')
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
ds = ds.rio.write_crs("EPSG:4326")

output_dir = Path('cogs')
for var in ['ndvi', 'evi', 'fpar', 'lai']:
    (output_dir / var).mkdir(parents=True, exist_ok=True)
    for t in range(len(ds.time)):
        date_str = str(ds.time[t].values)[:10]
        da = ds[var].isel(time=t)
        da.rio.to_raster(
            output_dir / var / f"{var}_{date_str}.tif",
            driver="COG",
            compress="deflate"
        )
        print(f"{var} {date_str} done")
```

Then zip and download:
```python
!zip -r cogs.zip cogs/
```

Unzip to: `C:\Users\abhas\prism-project\cogs\`

## Step 2: Start TiTiler

```powershell
docker run -d --name titiler -p 8000:80 -v C:\Users\abhas\prism-project\cogs:/data ghcr.io/developmentseed/titiler:latest
```

**Note:** TiTiler listens on port 80 internally, so map 8000:80 (not 8000:8000).

Test: http://localhost:8000/cog/info?url=file:///data/ndvi/ndvi_2020-01-01.tif

Preview: http://localhost:8000/cog/preview?url=file:///data/ndvi/ndvi_2020-01-01.tif&rescale=0,1&colormap_name=ylgn

## Step 3: Clone and Configure PRISM

```powershell
git clone https://github.com/WFP-VAM/prism-app.git
cd prism-app/frontend
```

Copy the config files from this directory into:
```
frontend/src/config/kenya/
├── prism.json
├── layers.json
├── index.ts
└── admin_boundaries.json  (Marsabit county boundary GeoJSON)
```

## Step 4: Get Admin Boundaries

Download Marsabit county boundary from HDX or GADM. The GeoJSON needs admin code properties that PRISM expects.

## Step 5: Run PRISM

```powershell
cd frontend
yarn clean
yarn install
yarn setup:common
$env:REACT_APP_COUNTRY="kenya"; yarn start
```

Opens at http://localhost:3000

## Data Specs

- **Source:** 1.3GB NetCDF, MODIS vegetation indices
- **Variables:** NDVI, EVI, FPAR, LAI
- **Resolution:** ~500m (0.0045° grid, 709×679 pixels)
- **Temporal:** Monthly, Jan 2008 – Jan 2026 (217 timesteps)
- **CRS:** EPSG:4326 (WGS84)
- **Extent:** Marsabit County (lat 1.3°N–4.5°N, lon 36°E–39°E)
- **COG size:** ~2MB per file, ~900MB total for 868 files

## Layer Config

All 4 layers use `static_raster` type pointing at TiTiler:
```
http://localhost:8000/cog/tiles/WebMercatorQuad/{z}/{x}/{y}@1x?url=file:///data/{var}/{var}_{YYYY}-{MM}-{DD}.tif&rescale=...&colormap_name=...
```

PRISM's date picker substitutes `{YYYY}-{MM}-{DD}` → `{YYYY_MM_DD}` format.

## TiTiler CORS (if needed)

If you get CORS errors, restart TiTiler with:
```powershell
docker rm -f titiler
docker run -d --name titiler -p 8000:80 -e CORS_ORIGINS="*" -v C:\Users\abhas\prism-project\cogs:/data ghcr.io/developmentseed/titiler:latest
```
