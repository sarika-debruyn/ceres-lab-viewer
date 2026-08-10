"""
Fast path. Grabs four single months instead of averaging the whole record,
so it finishes in seconds rather than minutes. Enough to prove the pipeline
and get a map on screen.

    python quick_test.py

Once a map renders, go back to convert.py for the real climatology.
"""

import shutil
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr

BUCKET = "leap-pangeo-pipeline"
PREFIX = "CERES_EBAF/store.icechunk"
ENDPOINT = "https://nyu1.osn.mghpcc.org"

GROUP = None                      # set if inspect_store.py reported groups
VARIABLE = "toa_sw_all_mon"       # just one, for speed
YEAR = 2010                       # any year in the record
MONTHS = [1, 3, 7, 9]             # the four the labs ask for

OUT = Path("ceres.zarr")


def main():
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT,
        region="us-east-1", anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    store = repo.readonly_session(branch="main").store

    kwargs = {"chunks": {}}
    if GROUP:
        kwargs["group"] = GROUP
    ds = xr.open_zarr(store, **kwargs)

    if VARIABLE not in ds.data_vars:
        print(f"! '{VARIABLE}' not found. Available: {list(ds.data_vars)}")
        return

    ds = ds[[VARIABLE]]

    # Four timesteps only. This is the whole reason it's fast.
    if "time" in ds.dims:
        sel = ds.sel(time=ds.time.dt.year == YEAR)
        sel = sel.sel(time=sel.time.dt.month.isin(MONTHS))
        if sel.time.size == 0:
            print(f"! no data for {YEAR}. Range: "
                  f"{ds.time.min().values} to {ds.time.max().values}")
            return
        ds = sel.assign_coords(
            month=("time", [f"{m:02d}" for m in sel.time.dt.month.values])
        ).swap_dims({"time": "month"}).drop_vars("time")
        print(f"  selected {ds.month.size} months: {list(ds.month.values)}")

    # -180/180 for the map
    lon = next((n for n in ds.coords if n.lower() in ("lon", "longitude")), None)
    if lon and float(ds[lon].max()) > 180:
        print(f"  rolling {lon} to -180/180")
        ds = ds.assign_coords({lon: (((ds[lon] + 180) % 360) - 180)}).sortby(lon)

    spatial = [d for d in ds.dims
               if d.lower() in ("lat", "latitude", "lon", "longitude")]
    ds = ds.chunk({d: -1 for d in spatial})

    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype("float32")

    if OUT.exists():
        shutil.rmtree(OUT)

    print(f"writing {OUT}…")
    try:
        ds.load().to_zarr(OUT, mode="w", zarr_format=3, consolidated=True)
        version = 3
    except TypeError:
        ds.load().to_zarr(OUT, mode="w", consolidated=True)
        version = 2

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    v = ds[VARIABLE].isel(month=0)

    print(f"\ndone — {size:.1f} MB, Zarr v{version}")
    print("\nPaste into zarr-test.html:")
    print(f"  source        http://localhost:8000/{OUT.name}")
    print(f"  variable      {VARIABLE}")
    print(f"  zarr version  {version}")
    print(f"  selector      {{\"month\": \"{ds.month.values[0]}\"}}")
    print(f"  range         {float(v.min()):.0f} to {float(v.max()):.0f}")
    print("\nThen:  python3 -m http.server 8000")


if __name__ == "__main__":
    main()
