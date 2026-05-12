"""Import this BEFORE tensorflow on Windows so TF 2.10 finds CUDA 11.2/cuDNN 8.1
DLLs installed inside the conda env (Library\\bin)."""
import os
import sys
import ctypes

ENV_BIN = os.path.join(sys.prefix, "Library", "bin")

if os.name == "nt" and os.path.isdir(ENV_BIN):
    os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(ENV_BIN)
    except (AttributeError, OSError):
        pass
    for _dll in (
        "cudart64_110.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cufft64_10.dll",
        "cusparse64_11.dll",
        "cudnn64_8.dll",
    ):
        try:
            ctypes.WinDLL(os.path.join(ENV_BIN, _dll))
        except OSError:
            pass
