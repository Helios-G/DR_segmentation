# Pretrained model

The DR classifier (`cnn_model_best.hdf5`, ~118 MB) is split into two 7-Zip
volumes so each part fits under GitHub's 100 MB single-file limit:

- `cnn_model_best.7z.001` (~80 MB)
- `cnn_model_best.7z.002` (~27 MB)

## Recombine

7-Zip auto-detects the split — keep both parts in this folder and run:

```
7z x cnn_model_best.7z.001
```

This produces `cnn_model_best.hdf5` in the same directory, which is what
`model_loader.py` expects.
