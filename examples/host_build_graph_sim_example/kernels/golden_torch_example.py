"""
Golden Computation Script for host_build_graph_example

Formula: f = (a + b + 1) * (a + b + 2)

Required components:
    - generate_inputs(params: dict) -> dict
    - compute_golden(tensors: dict, params: dict) -> None
    - TENSOR_ORDER: list of tensor names in orchestration parameter order

Optional components:
    - __outputs__: list of output tensor names (or use 'out_' prefix)
    - PARAMS_LIST: list of parameter dicts for multiple test cases
    - RTOL, ATOL: comparison tolerances
"""

import numpy as np
import torch


# =============================================================================
# Configuration - Modify these!
# =============================================================================

# Output tensor names (optional, framework can infer from generate_inputs)
__outputs__ = ["f"]

# Tensor order (REQUIRED - must match orchestration function signature)
# Orchestration expects: ptr(a), ptr(b), ptr(f), size(a), size(b), size(f), SIZE
TENSOR_ORDER = ["a", "b", "f"]

# Comparison tolerances
RTOL = 1e-4  # Relaxed for floating point computation
ATOL = 1e-4

# Multiple test cases (optional)
PARAMS_LIST = [
    {"size": 128 * 128, "dtype": "float32", "seed": 42},  # Fixed seed for reproducibility
    # {"size": 256 * 256, "dtype": "float32", "seed": 123},  # Large test
    # {"size": 32 * 32, "dtype": "float32", "seed": 456},    # Small test
]


# =============================================================================
# Input/Output Generation - Modify this!
# =============================================================================

def generate_inputs(params: dict) -> dict:
    """
    Generate input and output tensors.

    Args:
        params: Parameter dictionary (from PARAMS_LIST)

    Returns:
        Dictionary containing all tensors (inputs + outputs)
    """
    size = params.get("size", 16384)
    dtype = params.get("dtype", "float32")
    seed = params.get("seed", None)

    # Set seed if specified
    if seed is not None:
        np.random.seed(seed)

    return {
        # Input tensors
        "a": np.random.rand(size).astype(dtype),
        "b": np.random.rand(size).astype(dtype),

        # Output tensors (allocate with zeros)
        "f": np.zeros(size, dtype=dtype),
    }


# =============================================================================
# Golden Computation - Modify this!
# =============================================================================

def compute_golden(tensors: dict, params: dict) -> None:
    """
    Compute expected output (in-place modification).

    Args:
        tensors: Dictionary containing all tensors
        params: Parameter dictionary
    """
    a = torch.from_numpy(tensors["a"])
    b = torch.from_numpy(tensors["b"])

    # Compute formula: f = (a + b + 1) * (a + b + 2)
    f = (a + b + 1) * (a + b + 2)

    # Write result to output tensor (in-place)
    tensors["f"][:] = f.numpy()


# =============================================================================
# That's it! Just modify:
#   1. TENSOR_ORDER - match your orchestration signature
#   2. PARAMS_LIST - test parameters
#   3. generate_inputs - create tensors with correct shapes
#   4. compute_golden - implement your computation
# =============================================================================
