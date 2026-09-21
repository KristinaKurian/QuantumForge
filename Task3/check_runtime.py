from __future__ import annotations

from safe_runtime import configure_environment

configure_environment()

import platform
import sys

import faiss
import numpy as np
import sentence_transformers
import torch

from safe_runtime import limit_native_threads

limit_native_threads(faiss)

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("FAISS:", getattr(faiss, "__version__", "unknown"))
print("PyTorch:", torch.__version__)
print("SentenceTransformers:", sentence_transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Task3 execution device: CPU")
print("Native threads: 1")
print("Tokenizer parallelism: disabled")
