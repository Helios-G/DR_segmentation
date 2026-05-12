import gpu_bootstrap  # noqa: F401
import os
import h5py
import json

MODEL_PATH = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best.hdf5"

with h5py.File(MODEL_PATH, "r") as f:
    print("HDF5 root keys:", list(f.keys()))
    print("HDF5 root attrs:", list(f.attrs.keys()))
    cfg = f.attrs.get("model_config")
    keras_version = f.attrs.get("keras_version")
    backend = f.attrs.get("backend")
    print("keras_version:", keras_version)
    print("backend:", backend)

import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model(MODEL_PATH, compile=False)
    print("\nLoaded OK")
    model.summary(line_length=120)
    print("\nInput shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("\nConv layers (name, output_shape):")
    for layer in model.layers:
        cls = layer.__class__.__name__
        if "Conv" in cls or "Pool" in cls or "Global" in cls:
            print(f"  {layer.name:30s} {cls:18s} {layer.output_shape}")
except Exception as e:
    print("load_model FAILED:", type(e).__name__, e)
    print("\nFalling back to model_config JSON:")
    if cfg is not None:
        s = cfg.decode() if isinstance(cfg, (bytes, bytearray)) else cfg
        obj = json.loads(s)
        print("class_name:", obj.get("class_name"))
        for layer in obj.get("config", {}).get("layers", [])[:30]:
            print(" ", layer.get("class_name"), layer.get("config", {}).get("name"))
