"""
Step 2 of 2. Reads the Icechunk store on OSN and writes a small plain-Zarr
copy into this folder, so the test page can render it with no CORS involved.

Edit the CONFIG block below using what inspect_store.py printed, then:

    python convert.py
"""

import shutil
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr

# ----------------------------- CONFIG ---------------------------------

BUCKET = "leap-pangeo-pipeline"
PREFIX = "CERES_EBAF/store.icechunk"
ENDPOINT = "https://nyu1.osn.mghpcc.org"

# If inspect_store.py reported groups, put the group path here.
# Leave as None for a flat store.
GROUP = None

# Variable names exactly as inspect_store.py printed them. Keep this short:
# every variable you add is more to write and more to load.
VARIABLES = [
    "toa_sw_all_mon",
    "toa_lw_all_mon",
    "solar_mon",
]

# The labs only ever ask for Jan / Mar / Jul / Sep, so a 12-month climatology
# covers both of them. Set False to keep the full monthly record (~30x larger).
CLIMATOLOGY_ONLY = True

OUT = Path("ceres.zarr")

# ----------------------------------------------------------------------


def open_source():
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint_url=ENDPOINT,
        region="us-east-1",  # stops the EC2 metadata probe
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    store = repo.readonly_session(branch="main").store
    kwargs = {"chunks": {}}
    if GROUP:
        kwargs["group"] = GROUP
    return xr.open_zarr(store, **kwargs)


def normalise_longitude(ds):
    """Web maps expect -180/180. CERES ships 0-360."""
    name = next((n for n in ds.coords if n.lower() in ("lon", "longitude")), None)
    if name is None:
        print("! no longitude coord found; skipping roll")
        return ds
    if float(ds[name].max()) <= 180:
        print(f"  longitude already -180/180 ({name})")
        return ds
    print(f"  rolling {name} from 0-360 to -180/180")
    ds = ds.assign_coords({name: (((ds[name] + 180) % 360) - 180)})
    return ds.sortby(name)


def main():
    print("opening source store…")
    ds = open_source()

    missing = [v for v in VARIABLES if v not in ds.data_vars]
    if missing:
        print(f"\n! not in this dataset: {missing}")
        print(f"  available: {list(ds.data_vars)}")
        if GROUP is None:
            print("  (if the store is grouped, set GROUP at the top)")
        return

    ds = ds[VARIABLES]
    print(f"  {len(VARIABLES)} variables, dims {dict(ds.sizes)}")

    ds = normalise_longitude(ds)

    if CLIMATOLOGY_ONLY and "time" in ds.dims:
        print("  averaging to a 12-month climatology…")
        ds = ds.groupby("time.month").mean("time", keep_attrs=True)
        # Give the month coord readable labels. zarr-layer matches selector
        # values against the coordinate array, so these become the API:
        # setSelector({month: 'Mar'}) instead of {month: 3}.
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ds = ds.assign_coords(
            month_name=("month", [names[m - 1] for m in ds.month.values])
        )

    # One map per chunk: a single view is then a single HTTP request.
    spatial = [d for d in ds.dims if d.lower() in ("lat", "latitude", "lon", "longitude")]
    ds = ds.chunk({d: -1 for d in spatial})

    # float32 halves the size and is well beyond the precision of the data.
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype("float32")

    if OUT.exists():
        print(f"  removing existing {OUT}")
        shutil.rmtree(OUT)

    print(f"writing {OUT}…")
    try:
        ds.to_zarr(OUT, mode="w", zarr_format=3, consolidated=True)
        version = 3
    except TypeError:
        # Older zarr-python has no zarr_format argument.
        ds.to_zarr(OUT, mode="w", consolidated=True)
        version = 2
        print("  (wrote Zarr v2 — set zarrVersion: 2 in the test page)")

    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\ndone. {OUT} is {size_mb:.1f} MB, Zarr v{version}")

    print("\nUse these in zarr-test.html:")
    print(f"  source       http://localhost:8000/{OUT.name}")
    print(f"  variable     {VARIABLES[0]}")
    print(f"  zarr version {version}")
    print(f"  selector     {{\"month\": {list(ds.month.values)[:1][0] if 'month' in ds.coords else 0}}}")
    for name in ds.data_vars:
        v = ds[name]
        s = v.isel(month=0) if "month" in v.dims else v
        print(f"  clim for {name}: [{float(s.min()):.0f}, {float(s.max()):.0f}]")

    print("\nThen:  python3 -m http.server 8000")


if __name__ == "__main__":
    main()
