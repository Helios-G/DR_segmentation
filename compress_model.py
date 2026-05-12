import h5py, os, sys

SRC = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best.hdf5"
DST = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best_compressed.hdf5"


def copy_compressed(src, dst):
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        def walk(gin, gout):
            for name, item in gin.items():
                if isinstance(item, h5py.Group):
                    g = gout.create_group(name)
                    for k, v in item.attrs.items():
                        g.attrs[k] = v
                    walk(item, g)
                else:
                    data = item[()]
                    kwargs = {}
                    if data.ndim > 0 and data.size > 32:
                        kwargs.update(compression="gzip", compression_opts=9, shuffle=True)
                    ds = gout.create_dataset(name, data=data, **kwargs)
                    for k, v in item.attrs.items():
                        ds.attrs[k] = v

        walk(fin, fout)


copy_compressed(SRC, DST)
print("src:", os.path.getsize(SRC) // 1024, "KB")
print("dst:", os.path.getsize(DST) // 1024, "KB")
