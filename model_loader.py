import gpu_bootstrap  # noqa: F401
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\hyobi\PycharmProjects\DR_segmentation\model-other-default-v1\cnn_model_best.hdf5"


def rescale_gap(x):
    return x[0] / x[1]


class FixedLambda(Lambda):
    """Skip marshal-based function deserialization (broken across Py versions)."""

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if config.get("name") == "RescaleGAP":
            fn = rescale_gap
        else:
            fn = lambda x: x  # noqa: E731 — fallback no-op
        return cls(
            function=fn,
            output_shape=config.get("output_shape"),
            mask=config.get("mask"),
            arguments=config.get("arguments") or {},
            name=config.get("name"),
        )


def load_dr_model(path: str = MODEL_PATH):
    return load_model(
        path,
        custom_objects={"Lambda": FixedLambda},
        compile=False,
    )


if __name__ == "__main__":
    m = load_dr_model()
    print("Loaded OK")
    print("Input :", m.input_shape, "  Output:", m.output_shape)
    print("Total params:", m.count_params())
    print("\nTop-level layers:")
    for L in m.layers:
        print(f"  {L.name:35s} {L.__class__.__name__:22s} out={L.output_shape}")
