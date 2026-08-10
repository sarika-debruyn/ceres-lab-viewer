import icechunk, zarr, time

storage = icechunk.s3_storage(
    bucket="leap-pangeo-pipeline", prefix="CERES_EBAF/store.icechunk",
    endpoint_url="https://nyu1.osn.mghpcc.org",
    region="us-east-1", anonymous=True,
)
store = icechunk.Repository.open(storage).readonly_session(branch="main").store

t = time.time()
root = zarr.open_group(store, mode="r")
print(f"root opened in {time.time() - t:.1f}s")
print("groups:", list(root.group_keys()))
print("arrays:", list(root.array_keys()))