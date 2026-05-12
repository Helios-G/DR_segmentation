import h5py, json, sys
MODEL_PATH = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best.hdf5"
with h5py.File(MODEL_PATH, "r") as f:
    cfg = f.attrs["model_config"]
    s = cfg.decode() if isinstance(cfg, (bytes, bytearray)) else cfg
obj = json.loads(s)

layers = obj["config"]["layers"]
print(f"top-level layers: {len(layers)}")
for L in layers:
    cls = L.get("class_name")
    cfg2 = L.get("config", {})
    name = cfg2.get("name")
    if cls == "Lambda":
        print("\n=== Lambda:", name, "===")
        print(json.dumps(cfg2, indent=2)[:3000])
    elif cls == "Functional":
        # nested model (InceptionV3) — just print its name
        inner = cfg2.get("layers", [])
        print(f"\nFunctional {name}: {len(inner)} sub-layers, input shape {cfg2.get('build_input_shape') or cfg2.get('layers')[0].get('config',{}).get('batch_input_shape')}")
    else:
        out = ""
        for k in ("filters", "kernel_size", "units", "activation", "padding", "rate", "axis"):
            if k in cfg2:
                out += f" {k}={cfg2[k]}"
        print(f"  {cls:24s} {name:36s}{out}")

# also print inbound nodes to understand graph
print("\n--- inbound graph (top-level) ---")
for L in layers:
    name = L["config"].get("name")
    inb = L.get("inbound_nodes", [])
    if inb:
        srcs = []
        for nodes in inb:
            for item in nodes:
                if isinstance(item, list) and item:
                    srcs.append(item[0])
        print(f"  {name} <- {srcs}")
