"""
Step 1 of 2. Reads the CERES Icechunk store on OSN and prints what's inside.
Nothing is written or downloaded. Run this first, then use what it prints to
fill in convert.py.

    pip install icechunk "xarray>=2024.10" "zarr>=3" dask
    python inspect_store.py
"""

import icechunk
import xarray as xr

BUCKET = "leap-pangeo-pipeline"
PREFIX = "CERES_EBAF/store.icechunk"
ENDPOINT = "https://nyu1.osn.mghpcc.org"


def open_store():
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint_url=ENDPOINT,
        # Without an explicit region the AWS SDK probes the EC2 metadata
        # endpoint, which isn't reachable from a laptop. That's the warning
        # Robert saw. OSN ignores the value; it just stops the probe.
        region="us-east-1",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    return repo.readonly_session(branch="main").store


def rule(label):
    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")


def describe(ds, label):
    rule(label)
    print(f"dims        {dict(ds.sizes)}")
    print(f"coords      {list(ds.coords)}")
    print(f"variables   {list(ds.data_vars)}")

    for name in ds.coords:
        c = ds[name]
        if c.ndim != 1 or c.size == 0:
            continue
        vals = c.values
        preview = vals[:4] if c.size > 6 else vals
        print(f"\n  {name}  (n={c.size})")
        print(f"    first: {preview}")
        if c.size > 6:
            print(f"    last:  {vals[-2:]}")

    print("\n  --- variable detail ---")
    for name in ds.data_vars:
        v = ds[name]
        units = v.attrs.get("units", "?")
        long_name = v.attrs.get("long_name", "")
        print(f"\n  {name}")
        print(f"    dims  {v.dims}")
        print(f"    units {units}")
        if long_name:
            print(f"    desc  {long_name}")
        try:
            # One timestep only, so this stays a small read.
            sample = v.isel({v.dims[0]: 0}) if v.ndim == 3 else v
            lo, hi = float(sample.min()), float(sample.max())
            print(f"    range {lo:.1f} to {hi:.1f}   <- use for clim")
        except Exception as err:
            print(f"    range unavailable ({type(err).__name__})")


def main():
    store = open_store()

    # Raphael said he'd split the store into groups, so try a tree first.
    tree = None
    try:
        tree = xr.open_datatree(store, engine="zarr", chunks={})
    except Exception:
        pass

    if tree is not None and len(tree.children) > 0:
        rule("GROUPED STORE (datatree)")
        print("Groups found. Use group=... in open_zarr to read one.\n")
        for path, node in tree.subtree_with_keys:
            label = path or "/"
            nvars = len(node.dataset.data_vars)
            print(f"  {label:<28} {nvars} variables")
        for path, node in tree.subtree_with_keys:
            if len(node.dataset.data_vars):
                describe(node.dataset, f"GROUP: {path or '/'}")
    else:
        ds = xr.open_zarr(store, chunks={})
        describe(ds, "FLAT STORE (no groups)")

    # Longitude convention decides whether the map draws in the right place.
    rule("CHECKS")
    ds = tree.dataset if tree is not None and len(tree.dataset.data_vars) else None
    if ds is None:
        try:
            ds = xr.open_zarr(store, chunks={})
        except Exception:
            ds = None

    if ds is not None:
        loname = next((n for n in ds.coords if n.lower() in ("lon", "longitude")), None)
        laname = next((n for n in ds.coords if n.lower() in ("lat", "latitude")), None)
        if loname:
            lo, hi = float(ds[loname].min()), float(ds[loname].max())
            print(f"longitude   {lo:.2f} to {hi:.2f}")
            if hi > 180:
                print("  -> 0-360 convention. convert.py will roll it to -180/180.")
            else:
                print("  -> already -180/180. Nothing to fix.")
        if laname:
            v = ds[laname].values
            direction = "ascending" if v[0] < v[-1] else "descending"
            print(f"latitude    {float(v.min()):.2f} to {float(v.max()):.2f} ({direction})")
    else:
        print("Couldn't read a root dataset for the coordinate check.")
        print("If the store is grouped, read the group detail printed above.")

    print("\nNext: open convert.py, fill in GROUP and VARIABLES, then run it.\n")


if __name__ == "__main__":
    main()
