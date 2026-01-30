#!/usr/bin/env python3
"""
New-style Test Runner for Runtime Kernel Testing

Supports golden scripts with:
    - generate_inputs(params) -> dict
    - compute_golden(tensors, params) -> None (in-place)
    - TENSOR_ORDER: list (required)
    - __outputs__: list (optional)
    - PARAMS_LIST: list of dicts (optional)
    - RTOL, ATOL: floats (optional)

Usage:
    python test_runner_v2.py [--kernels DIR] [--golden FILE] [--device ID] [--platform PLATFORM]
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import importlib.util


class NewStyleTestRunner:
    """Test runner for new-style golden scripts."""

    def __init__(self, kernels_dir, golden_script_path, device_id=0, platform="a2a3"):
        """Initialize the test runner."""
        self.device_id = device_id
        self.platform = platform

        # Add runtime directory to path
        runtime_root = Path(__file__).parent.parent
        runtime_dir = runtime_root / "python"
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))

        # Import runtime modules
        from runtime_builder import RuntimeBuilder
        from bindings import bind_host_binary, register_kernel, set_device, launch_runtime
        from pto_compiler import PTOCompiler
        from elf_parser import extract_text_section

        self.RuntimeBuilder = RuntimeBuilder
        self.bind_host_binary = bind_host_binary
        self.register_kernel = register_kernel
        self.set_device = set_device
        self.launch_runtime = launch_runtime
        self.PTOCompiler = PTOCompiler
        self.extract_text_section = extract_text_section

        self.runtime_builder = RuntimeBuilder(platform=platform)
        self.pto_compiler = PTOCompiler(platform=platform)
        self.runtime_root = runtime_root

        # Load kernel configuration
        kernels_path = Path(kernels_dir)
        kernel_config_path = kernels_path / "kernel_config.py"

        kernel_module = self._load_module(kernel_config_path, 'kernel_config_module')
        self.KERNELS = kernel_module.KERNELS
        self.ORCHESTRATION = kernel_module.ORCHESTRATION

        # Auto-detect runtime name
        runtime_name = kernels_path.parent.name.replace('_example', '').replace('_sim', '')
        self.runtime_name = runtime_name

        # Load golden script
        golden_module = self._load_module(golden_script_path, 'golden_module')

        # Required components
        self.generate_inputs = golden_module.generate_inputs
        self.compute_golden = golden_module.compute_golden
        self.tensor_order = golden_module.TENSOR_ORDER

        # Optional components
        self.outputs_list = getattr(golden_module, '__outputs__', [])
        self.params_list = getattr(golden_module, 'PARAMS_LIST', [{}])
        self.rtol = getattr(golden_module, 'RTOL', 1e-5)
        self.atol = getattr(golden_module, 'ATOL', 1e-5)

    def _load_module(self, file_path, module_name):
        """Load a Python module from file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _build_func_args(self, tensors):
        """Build function arguments based on TENSOR_ORDER."""
        args = []

        # Add pointers in TENSOR_ORDER
        for name in self.tensor_order:
            if name not in tensors:
                raise ValueError(f"Tensor '{name}' in TENSOR_ORDER not found")
            args.append(tensors[name].ctypes.data)

        # Add sizes in TENSOR_ORDER
        for name in self.tensor_order:
            args.append(tensors[name].nbytes)

        # Add total SIZE (first tensor's size)
        first_tensor = tensors[self.tensor_order[0]]
        args.append(len(first_tensor))

        return args

    def run_test_case(self, params):
        """Run a single test case with given parameters."""
        print(f"\n{'='*70}")
        print(f"Test Parameters: {params}")
        print(f"{'='*70}")

        # Generate tensors
        print("\n=== Generating Tensors ===")
        tensors = self.generate_inputs(params)

        # Identify inputs and outputs
        inputs = {k: v for k, v in tensors.items() if k not in self.outputs_list}
        outputs = {k: v for k, v in tensors.items() if k in self.outputs_list}

        print("\nTensors:")
        for name in self.tensor_order:
            tensor_type = "OUTPUT" if name in self.outputs_list else "INPUT"
            if isinstance(tensors[name], np.ndarray):
                print(f"  {name} ({tensor_type}): shape={tensors[name].shape}, dtype={tensors[name].dtype}")

        # Compute golden results (in-place)
        print("\n=== Computing Golden Results ===")
        self.compute_golden(tensors, params)

        # Save golden results
        golden_results = {k: v.copy() for k, v in outputs.items()}

        # Reset outputs for runtime execution
        for name in self.outputs_list:
            tensors[name][:] = 0

        # Build and load runtime
        print("\n=== Building Runtime ===")
        host_binary, aicpu_binary, aicore_binary = self.runtime_builder.build(self.runtime_name)
        print(f"Build completed ({len(host_binary)} bytes)")

        Runtime = self.bind_host_binary(host_binary)

        # Set device
        print(f"\n=== Setting Device {self.device_id} ===")
        self.set_device(self.device_id)

        # Compile orchestration
        print("\n=== Compiling Orchestration ===")
        include_dirs = [
            str(self.runtime_root / "src" / "runtime" / self.runtime_name / "runtime"),
            str(self.runtime_root / "src" / "platform" / "a2a3" / "host"),
        ]
        orch_so_binary = self.pto_compiler.compile_orchestration(
            self.ORCHESTRATION["source"],
            extra_include_dirs=include_dirs
        )

        # Compile and register kernels
        print("\n=== Compiling and Registering Kernels ===")
        pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"
        for i, kernel in enumerate(self.KERNELS, 1):
            print(f"[{i}/{len(self.KERNELS)}] {kernel['source']}")
            incore_o = self.pto_compiler.compile_incore(
                kernel["source"],
                core_type=kernel["core_type"],
                pto_isa_root=pto_isa_root
            )
            kernel_bin = self.extract_text_section(incore_o)
            self.register_kernel(kernel["func_id"], kernel_bin)

        # Build func_args
        func_args = self._build_func_args(tensors)

        # Create and initialize runtime
        print("\n=== Initializing Runtime ===")
        runtime = Runtime()
        runtime.initialize(orch_so_binary, self.ORCHESTRATION["function_name"], func_args)

        # Execute
        print("\n=== Executing on Device ===")
        self.launch_runtime(runtime,
                           aicpu_thread_num=3,
                           block_dim=3,
                           device_id=self.device_id,
                           aicpu_binary=aicpu_binary,
                           aicore_binary=aicore_binary)

        # Finalize
        print("\n=== Finalizing ===")
        runtime.finalize()

        # Validate results
        print("\n=== Validating Results ===")
        all_passed = True
        for output_name in self.outputs_list:
            actual = tensors[output_name]
            expected = golden_results[output_name]

            print(f"\n{output_name}:")
            print(f"  First 5 elements: {actual[:5]}")
            print(f"  Expected: {expected[:5]}")

            try:
                assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)
                print(f"  ✓ PASS (rtol={self.rtol}, atol={self.atol})")
            except AssertionError:
                error_mask = ~np.isclose(actual, expected, rtol=self.rtol, atol=self.atol)
                error_count = np.sum(error_mask)
                print(f"  ✗ FAIL: {error_count}/{len(actual)} elements mismatch")
                all_passed = False

        return all_passed

    def run_all(self):
        """Run all test cases."""
        print(f"\n{'='*70}")
        print(f"Running {len(self.params_list)} test case(s)")
        print(f"{'='*70}")

        all_passed = True
        for i, params in enumerate(self.params_list):
            print(f"\n### Test Case {i+1}/{len(self.params_list)} ###")
            passed = self.run_test_case(params)
            all_passed = all_passed and passed

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='New-style test runner for runtime kernel testing'
    )

    default_kernels = 'examples/host_build_graph_example/kernels'
    default_golden = 'examples/host_build_graph_example/kernels/golden_torch_example.py'

    parser.add_argument('--kernels', default=default_kernels)
    parser.add_argument('--golden', default=default_golden)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--platform', default='a2a3', choices=['a2a3', 'a2a3sim'])

    args = parser.parse_args()

    # Validate paths
    if not Path(args.kernels).exists():
        print(f"Error: Kernels directory not found: {args.kernels}")
        return 1

    if not Path(args.golden).exists():
        print(f"Error: Golden script not found: {args.golden}")
        return 1

    try:
        runner = NewStyleTestRunner(
            kernels_dir=args.kernels,
            golden_script_path=args.golden,
            device_id=args.device,
            platform=args.platform
        )

        success = runner.run_all()

        print(f"\n{'='*70}")
        print(f"{'✓ ALL TESTS PASSED' if success else '✗ SOME TESTS FAILED'}")
        print(f"{'='*70}\n")

        return 0 if success else 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
