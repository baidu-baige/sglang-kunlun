#!/usr/bin/env python3
# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime patch for sglang parallel_state.py to inject XPU profiler and CPU binding setup.

This script is executed before sglang starts to patch the init_distributed_environment
function, adding:
1. XPU profiler environment variables based on local_rank
2. CPU affinity binding based on XPU NUMA topology

Usage:
    python3 sglang_xpu_patch.py [--search-root /path/to/search] [--dry-run]

The script will:
1. Find parallel_state.py in the aiak_sglang installation
2. Parse it using Python AST
3. Inject the XPU profiler and CPU binding code at the beginning of init_distributed_environment
4. Write the patched file back

Injected code:
    # XPU profiler environment setup
    device_id = local_rank % 8
    os.environ["XPU_CUPTI_ENABLE_DEVICE"] = str(device_id)
    os.environ["XPU_ENABLE_PROFILER_TRACING"] = "1"

    # CPU affinity binding based on XPU NUMA topology
    _bhta_bind_cpu(local_rank)

The _bhta_bind_cpu function parses `xpu-smi topo -m` output to determine CPU affinity
for each XPU device and binds the current process to the appropriate CPU cores.
"""

import argparse
import ast
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

# Primary search paths, tried in order (first hit wins)
PRIMARY_SGLANG_PATHS = (
    Path("/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages"),
    Path("/workspace/aiak_sglang/python"),
)
TARGET_FILE = "sglang/srt/distributed/parallel_state.py"
TARGET_FUNCTION = "init_distributed_environment"

# Code to inject (as Python source)
# This includes:
# 1. Helper functions for CPU binding (module-level)
# 2. XPU profiler environment setup (in function)
# 3. CPU binding call (in function)

# Helper functions to inject at module level
INJECT_MODULE_CODE = '''
import re
import subprocess


def _bhta_parse_cpu_ranges(range_str: str) -> list:
    """Parse CPU range string into list of (start, end) tuples for each segment."""
    ranges = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            val = int(part)
            ranges.append((val, val))
    return ranges


def _bhta_split_ranges_for_xpus(ranges: list, xpu_index: int, total_xpus: int) -> list:
    """Split each range evenly among XPUs and return cores for the given XPU index.

    For affinity '0-15,128-143' with 2 XPUs:
      - XPU0 gets: 0-7, 128-135 (first half of each range)
      - XPU1 gets: 8-15, 136-143 (second half of each range)
    """
    cores = []
    for start, end in ranges:
        range_size = end - start + 1
        chunk_size = range_size // total_xpus

        chunk_start = start + xpu_index * chunk_size
        if xpu_index == total_xpus - 1:
            chunk_end = end
        else:
            chunk_end = chunk_start + chunk_size - 1

        cores.extend(range(chunk_start, chunk_end + 1))
    return cores


def _bhta_parse_xpu_topo() -> dict:
    """Parse `xpu-smi topo -m` output to build XPU -> CPU affinity mapping.

    Returns a dict mapping local_rank (XPU index) to a list of CPU core IDs.
    XPUs sharing the same NUMA node split each CPU range segment evenly.
    """
    result = subprocess.run(
        ["xpu-smi", "topo", "-m"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"xpu-smi topo -m failed: {result.stderr.strip()}")

    lines = result.stdout.strip().splitlines()

    # Parse each XPU row using regex
    xpu_pattern = re.compile(r"^XPU(\\d+)\\s")
    # CPU affinity format: "0-15,128-143" followed by NUMA node number at line end
    affinity_pattern = re.compile(r"(\\d+-\\d+(?:,\\d+-\\d+)*)\\s+\\d+$")

    # Group XPUs by their raw affinity string (same NUMA node)
    affinity_groups = {}

    for line in lines:
        # Skip header line
        if "NUMA Affinity" in line:
            continue

        match = xpu_pattern.match(line)
        if not match:
            continue

        xpu_id = int(match.group(1))

        # Extract CPU affinity from line end
        aff_match = affinity_pattern.search(line)
        if not aff_match:
            continue

        affinity_str = aff_match.group(1)
        affinity_groups.setdefault(affinity_str, []).append(xpu_id)

    # Build the final mapping: split each range segment evenly among XPUs
    cpu_map = {}
    for affinity_str, xpu_ids in affinity_groups.items():
        ranges = _bhta_parse_cpu_ranges(affinity_str)
        xpu_ids_sorted = sorted(xpu_ids)
        n = len(xpu_ids_sorted)

        for i, xpu_id in enumerate(xpu_ids_sorted):
            cpu_map[xpu_id] = _bhta_split_ranges_for_xpus(ranges, i, n)

    return cpu_map


def _bhta_bind_cpu(local_rank: int) -> None:
    """Bind current process to CPU cores based on XPU NUMA topology."""
    try:
        cpu_map = _bhta_parse_xpu_topo()
        cores = cpu_map.get(local_rank)
        if cores:
            os.sched_setaffinity(0, cores)
            print(f"[rank {local_rank}] bind cpu {cores}")
        else:
            print(f"[rank {local_rank}] no CPU affinity found, skipping bind")
    except Exception as e:
        print(f"[rank {local_rank}] CPU binding failed: {e}, continuing without binding")
'''

# Code to inject at the beginning of init_distributed_environment
INJECT_CODE = '''
device_id = local_rank % 8
os.environ["XPU_CUPTI_ENABLE_DEVICE"] = str(device_id)
os.environ["XPU_ENABLE_PROFILER_TRACING"] = "1"
_bhta_bind_cpu(local_rank)
'''

# Marker comment to detect if already patched
PATCH_MARKER = "# BHTA_XPU_PROFILER_PATCH"


def find_parallel_state(search_root: Optional[Path] = None) -> Optional[Path]:
    """Find parallel_state.py in the filesystem.

    Strategy:
    1. Check primary paths first (conda site-packages, then aiak_sglang)
    2. Fallback: search from provided root or common locations
    """
    # Strategy 1: Primary paths
    for root in PRIMARY_SGLANG_PATHS:
        target = root / TARGET_FILE
        if target.exists():
            print(f"[BHTA] Found target at primary path: {target}", file=sys.stderr)
            return target

    # Strategy 2: Search from provided root
    search_paths = []
    if search_root:
        search_paths.append(search_root)

    # Add common fallback locations
    search_paths.extend([
        Path("/workspace"),
        Path("/opt"),
        Path("/usr/local"),
    ])

    print(f"[BHTA] Searching for parallel_state.py in fallback locations...", file=sys.stderr)
    for root in search_paths:
        if not root.exists():
            continue

        # Search for sglang installation
        for sglang_dir in root.rglob("sglang"):
            target = sglang_dir / "srt" / "distributed" / "parallel_state.py"
            if target.exists():
                print(f"[BHTA] Found target via search: {target}", file=sys.stderr)
                return target

    return None


def is_already_patched(source: str) -> bool:
    """Check if the file has already been patched."""
    return PATCH_MARKER in source


class FunctionPatcher(ast.NodeTransformer):
    """AST transformer that injects code at the beginning of a target function."""

    def __init__(self, target_func: str, inject_nodes: list):
        self.target_func = target_func
        self.inject_nodes = inject_nodes
        self.patched = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        """
        if node.name == self.target_func:
            # Create marker comment as a string expression
            marker = ast.Expr(value=ast.Constant(value=PATCH_MARKER))

            # Insert injected code at the beginning of the function body
            node.body = [marker] + self.inject_nodes + node.body
            self.patched = True
            print(f"[BHTA] Patched function: {self.target_func}", file=sys.stderr)

        return self.generic_visit(node)


def parse_inject_code() -> list:
    """Parse the function-level injection code string into AST nodes."""
    tree = ast.parse(INJECT_CODE)
    return tree.body


def parse_module_inject_code() -> list:
    """Parse the module-level injection code string into AST nodes."""
    tree = ast.parse(INJECT_MODULE_CODE)
    return tree.body


def patch_file(filepath: Path) -> Tuple[bool, str]:
    """Patch the target file with XPU profiler and CPU binding setup code.

    Returns:
        Tuple of (success, message)
    """
    print(f"[BHTA] ================================================", file=sys.stderr)
    print(f"[BHTA] SGLang XPU Patch Script", file=sys.stderr)
    print(f"[BHTA] Target: {filepath}", file=sys.stderr)
    print(f"[BHTA] ================================================", file=sys.stderr)

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read file: {e}"

    # Check if already patched
    if is_already_patched(source):
        print(f"[BHTA] File already patched, skipping", file=sys.stderr)
        return True, "File already patched, skipping"

    # Parse source into AST
    print(f"[BHTA] Parsing source file...", file=sys.stderr)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Failed to parse source: {e}"

    # Parse injection code (both module-level and function-level)
    module_inject_nodes = parse_module_inject_code()
    func_inject_nodes = parse_inject_code()

    # Find the position to insert module-level code (after imports)
    insert_pos = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_pos = i + 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Skip docstrings
            if i == 0:
                insert_pos = 1

    # Insert module-level code (helper functions)
    print(f"[BHTA] Inserting module-level helper functions...", file=sys.stderr)
    tree.body = tree.body[:insert_pos] + module_inject_nodes + tree.body[insert_pos:]

    # Apply function patch
    print(f"[BHTA] Patching function: {TARGET_FUNCTION}...", file=sys.stderr)
    patcher = FunctionPatcher(TARGET_FUNCTION, func_inject_nodes)
    patched_tree = patcher.visit(tree)

    if not patcher.patched:
        return False, f"Function '{TARGET_FUNCTION}' not found in {filepath}"

    # Fix line numbers for the modified AST
    ast.fix_missing_locations(patched_tree)

    # Generate patched source code
    try:
        # Use ast.unparse (Python 3.9+) for clean output
        patched_source = ast.unparse(patched_tree)
    except AttributeError:
        return False, "Python 3.9+ required for ast.unparse"

    # Create backup
    try:
        backup_path = filepath.with_suffix(".py.orig")
        if not backup_path.exists():
            shutil.copy2(filepath, backup_path)
            print(f"[BHTA] Created backup: {backup_path}", file=sys.stderr)
    except Exception as e:
        print(f"[BHTA] Warning: Failed to create backup: {e}", file=sys.stderr)

    # Write back
    try:
        filepath.write_text(patched_source, encoding="utf-8")
    except Exception as e:
        return False, f"Failed to write patched file: {e}"

    print(f"[BHTA] Patch applied successfully", file=sys.stderr)
    return True, f"Successfully patched {filepath}"


def main() -> int:
    """
    Main entry point of the script.
    """
    parser = argparse.ArgumentParser(
        description="Patch sglang parallel_state.py for XPU profiler and CPU binding"
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=None,
        help="Root directory to search for sglang installation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    print(f"[BHTA] ================================================", file=sys.stderr)
    print(f"[BHTA] SGLang XPU Patch - Finding Target File", file=sys.stderr)
    print(f"[BHTA] ================================================", file=sys.stderr)

    # Find target file
    target = find_parallel_state(args.search_root)
    if not target:
        print("[BHTA] ERROR: Could not find parallel_state.py", file=sys.stderr)
        print("[BHTA] Searched locations:", file=sys.stderr)
        for root in PRIMARY_SGLANG_PATHS:
            print(f"[BHTA]   - Primary: {root / TARGET_FILE}", file=sys.stderr)
        print("[BHTA]   - Fallback: /workspace, /opt, /usr/local", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[BHTA] Would patch: {target}", file=sys.stderr)
        return 0

    # Apply patch
    success, message = patch_file(target)
    print(f"[BHTA] {message}", file=sys.stderr)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
