"""
Where is the time going? Times each stage separately and reports the chunk
layout, which is the usual culprit.

    python diagnose.py

Runs in well under a minute if the store is chunked sensibly. If the
"one single map" timing is slow, the chunking is the problem and that is
information to hand to Raphael.
"""

import time

import icechunk
import xarray as xr

BUCKET = "leap-pangeo-pipeline"
PREFIX = "CERES_EBAF/store.icechunk"
ENDPOINT = "https://nyu1.osn.mghpcc.org"
GROUP = None


class step:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"{self.label:<34}", end="", flush=True)
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        print(f"{time.time() - self.t:6.1f}s")


def main():
    with step("connect to OSN"):
        storage = icechunk.s3_storage(
            bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT,
            region="us-east-1", anonymous=True,
        )
        repo = icechunk.Repository.open(storage)
        store = repo.readonly_session(branch="main").store

    with step("open metadata"):
        kwargs = {"chunks": {}}
        if GROUP:
            kwargs["group"] = GROUP
        ds = xr.open_zarr(store, **kwargs)

    names = list(ds.data_vars)
    if not names:
        print("\nNo variables at this level — the store is grouped.")
        print("Set GROUP at the top of this file and rerun.")
        return

    var = names[0]
    v = ds[var]

    print(f"\nvariable    {var}")
    print(f"shape       {v.shape}")
    print(f"dims        {v.dims}")

    on_disk = v.encoding.get("chunks")
    print(f"disk chunks {on_disk}")

    if on_disk and "time" in v.dims:
        ti = list(v.dims).index("time")
        per = on_disk[ti]
        total = v.shape[ti]
        n = -(-total // per)
        print(f"\n  {per} timesteps per chunk, {n} chunks over {total} steps")
        if per >= total:
            print("  !! THE WHOLE TIME AXIS IS ONE CHUNK.")
            print("     Any single map requires downloading the entire record.")
            print("     This needs fixing in the store, not in the viewer.")
        elif per > 24:
            print("  !  large time chunks; one map pulls more than it needs")
        else:
            print("  ok for map viewing")

    size_mb = v.size * v.dtype.itemsize / 1e6
    print(f"\nvariable size in memory  {size_mb:.0f} MB")
    if on_disk:
        chunk_mb = 1
        for c in on_disk:
            chunk_mb *= c
        chunk_mb *= v.dtype.itemsize / 1e6
        print(f"one chunk                {chunk_mb:.1f} MB")

    # The number that actually matters: how long to fetch a single map?
    print()
    with step("read ONE single map"):
        first = {d: 0 for d in v.dims if d not in ("lat", "latitude", "lon", "longitude")}
        one = v.isel(first).load()
    print(f"  values {float(one.min()):.1f} to {float(one.max()):.1f}")

    print("\nIf that last read was slow, the store's chunking is the bottleneck.")
    print("Send Raphael the 'disk chunks' line above.")


if __name__ == "__main__":
    main()
