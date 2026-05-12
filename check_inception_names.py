import h5py, json
MODEL_PATH = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best.hdf5"

with h5py.File(MODEL_PATH, "r") as f:
    cfg = f.attrs["model_config"]
s = cfg.decode() if isinstance(cfg, (bytes, bytearray)) else cfg
obj = json.loads(s)

inc = next(L for L in obj["config"]["layers"] if L["class_name"] == "Functional")
sublayers = inc["config"]["layers"]
print("InceptionV3 sub-layers:", len(sublayers))
print("First 10:")
for L in sublayers[:10]:
    print(" ", L["class_name"], L["config"].get("name"))
print("Last 5:")
for L in sublayers[-5:]:
    print(" ", L["class_name"], L["config"].get("name"))

# Are names standard inception_v3 (no numeric suffix indicating shared session)?
names = [L["config"]["name"] for L in sublayers]
print("\nSample names look like fresh InceptionV3?", names[1])

# Inspect weight groups in the file
with h5py.File(MODEL_PATH, "r") as f:
    mw = f["model_weights"]
    top_groups = list(mw.keys())
    print("\nTop-level weight groups:", top_groups[:15], "...total", len(top_groups))
    # Check inside inception_v3 group
    if "inception_v3" in mw:
        inc_grp = mw["inception_v3"]
        sub = list(inc_grp.keys())
        print("inception_v3 has", len(sub), "weight subgroups; first 5:", sub[:5])
