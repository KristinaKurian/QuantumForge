from __future__ import annotations

import os


def configure_environment() -> None:
    """
    Must be called before importing numpy / faiss / torch / transformers.

    The goal is to avoid native thread pools fighting each other on macOS
    and to disable tokenizer parallel workers. Task 3 is small enough that
    one native thread is more than sufficient.
    """
    values = {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)


def limit_native_threads(faiss_module=None) -> None:
    """
    Apply runtime limits after native libraries are imported.
    """
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows this to be configured only once per process.
            pass
    except Exception:
        pass

    if faiss_module is not None:
        try:
            faiss_module.omp_set_num_threads(1)
        except Exception:
            pass
