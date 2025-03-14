import random
import numpy as np
import torch
import torch_geometric
import os


def set_seed(seed: int, deterministic_algorithms: bool = False):
    """
    Set seeds for reproducibility and optionally enable deterministic algorithms.

    Args:
        seed: Integer seed for reproducibility
        deterministic_algorithms: If True, enables deterministic CUDA algorithms
            and disables cuDNN benchmarking. This will impact performance.
    """
    # Basic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyG specific seeding
    torch_geometric.seed_everything(seed)

    if deterministic_algorithms:
        # These settings impact performance but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        # Better performance settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.use_deterministic_algorithms(False)

    print(f"Set seed to {seed}")
    print(f"Deterministic mode: {deterministic_algorithms}")
