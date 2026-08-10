"""
Builds a small plain-Zarr store from just the arrays the labs need.

The source store is flat with ~170 arrays, so xr.open_zarr is slow: it makes
one network round-trip per array. This reads the ten arrays we actually want
using zarr directly, which skips that entirely.

    python build_labs.py
"""

import shutil
import time
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr
import zarr

BUCKET = "leap-pangeo-pipeline"
PREFIX = "CERES_EBAF/store.icechunk"
ENDPOINT = "https://nyu1.osn.mghpcc.org"

OUT = Path("ceres.zarr")

# Which averaging period. Options in this store: monthly_clim, seasonal_clim,
# annual_mean. monthly_clim gives 12 maps, which is what both labs use.
EPOCH = "monthly_clim"
DIM = "month"          # matching coordinate: month / season / year

# Source array -> friendly name. Named for what students look at, not for
# how CERES names its files.
WANTED = {
    "solar_mon":         "incoming_sw",       # Lab 1 Task 3
    "toa_sw_all_mon":    "reflected_sw",      # Lab 2 Task 1.5
    "toa_sw_clr_c_mon":  "reflected_sw_clear",  # Lab 1 Task 2
    "toa_lw_all_mon":    "outgoing_lw",       # Lab 2 Task 3, Figure 4
    "toa_lw_clr_c_mon":  "outgoing_lw_clear", # Lab 1 Task 4
    "toa_cre_net_mon":   "cloud_forcing_net", # Lab 2 Task 4
    "sfc_lw_up_all_mon": "surface_lw_up",     # Lab 2 Task 6
}


def main():
    t0 = time.time()
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT,
        region="us-east-1", anonymous=True,
    )
    store = icechunk.Repository.open(storage).readonly_session(branch="main").store
    root = zarr.open_group(store, mode="r")
    print(f"root open                {time.time() - t0:5.1f}s")

    # Coordinates first.
    t = time.time()
    lat = root["lat"][:]
    lon = root["lon"][:]
    axis = root[DIM][:]
    print(f"coords                   {time.time() - t:5.1f}s")
    print(f"  lat {lat.shape}  lon {lon.shape}  {DIM} {axis.shape} = {axis}")

    data = {}
    for src, friendly in WANTED.items():
        key = f"{src}_{EPOCH}"
        if key not in root:
            print(f"  ! missing {key}")
            continue
        t = time.time()
        arr = root[key][:]
        print(f"{friendly:<24} {time.time() - t:5.1f}s  {arr.shape}")
        data[friendly] = arr

    if not data:
        print("\nNothing read. Check EPOCH and the array names.")
        return

    # Match axes by length so we don't assume dimension order.
    sample = next(iter(data.values()))
    lookup = {len(axis): DIM, len(lat): "lat", len(lon): "lon"}
    dims = tuple(lookup[n] for n in sample.shape)
    print(f"\ninferred dims {dims}")

    ds = xr.Dataset(
        {name: (dims, arr.astype("float32")) for name, arr in data.items()},
        coords={DIM: axis, "lat": lat, "lon": lon},
    )

    # Derived albedo. Guard the polar-night divide-by-zero: where the sun
    # never rises, incoming is ~0 and the ratio is meaningless.
    if "reflected_sw" in ds and "incoming_sw" in ds:
        lit = ds.incoming_sw > 10          # W/m2
        ds["albedo"] = (ds.reflected_sw / ds.incoming_sw).where(lit) * 100
        ds["albedo"].attrs = {"units": "%", "long_name": "Planetary albedo"}
        print("derived albedo")
    if "reflected_sw_clear" in ds and "incoming_sw" in ds:
        lit = ds.incoming_sw > 10
        ds["albedo_clear"] = (ds.reflected_sw_clear / ds.incoming_sw).where(lit) * 100
        ds["albedo_clear"].attrs = {"units": "%", "long_name": "Clear-sky albedo"}
        print("derived albedo_clear")

    # Greenhouse effect: surface emission minus what escapes to space.
    # This is Lab 2 Task 6 without needing the JONES dataset.
    if "surface_lw_up" in ds and "outgoing_lw" in ds:
        ds["greenhouse_lw"] = ds.surface_lw_up - ds.outgoing_lw
        ds["greenhouse_lw"].attrs = {
            "units": "W m-2",
            "long_name": "Surface emission minus TOA emission",
        }
        print("derived greenhouse_lw")

    for name in ds.data_vars:
        ds[name].attrs.setdefault("units", "W m-2")

    # Web maps want -180/180.
    if float(ds.lon.max()) > 180:
        print("rolling longitude to -180/180")
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")

    # One map per chunk, so a single view is a single request.
    ds = ds.chunk({"lat": -1, "lon": -1, DIM: 1})

    ds.attrs = {
        "title": "CERES EBAF for EESC UN2100 labs",
        "source": f"{BUCKET}/{PREFIX}",
        "epoch": EPOCH,
        "note": "Derived fields computed locally; verify before teaching use.",
    }

    if OUT.exists():
        shutil.rmtree(OUT)

    print(f"\nwriting {OUT}…")
    try:
        ds.to_zarr(OUT, mode="w", zarr_format=3, consolidated=True)
        version = 3
    except TypeError:
        ds.to_zarr(OUT, mode="w", consolidated=True)
        version = 2

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"done — {size:.1f} MB, Zarr v{version}, {time.time() - t0:.0f}s total")

    print("\n" + "=" * 58)
    print("Paste into zarr-test.html")
    print("=" * 58)
    print(f"  source        http://localhost:8000/{OUT.name}")
    print(f"  zarr version  {version}")
    print(f"  lat dim       lat")
    print(f"  lon dim       lon")
    print(f"  selector      {{\"{DIM}\": {repr(axis[0].item() if hasattr(axis[0], 'item') else axis[0])}}}")
    print("\n  variable / suggested range:")
    for name in ds.data_vars:
        v = ds[name].isel({DIM: 0})
        lo, hi = float(v.min()), float(v.max())
        print(f"    {name:<20} [{lo:7.0f}, {hi:7.0f}]")

    print("\nThen:  python3 -m http.server 8000")


if __name__ == "__main__":
    main()
