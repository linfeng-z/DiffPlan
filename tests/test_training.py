"""
Test script for training different model configurations.
This script contains various test configurations that have been verified to work.
"""

import subprocess
import pytest


def test_mp_vin():
    """Test basic MP-VIN training."""
    cmd = [
        "python3", "-m", "diffplan.main",
        "--algorithm", "MP-VIN",
        "--cont_action",
        "--disable_wandb",
        "--epochs", "1",
        "--data_path", "data/m15_graph-cc_1k_cont_act.pth",
        "--task", "GraphWorld",
        "--has_obstacle",
        "--seed", "42"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    assert process.returncode == 0, f"MP-VIN training failed with error:\n{process.stderr}"


def test_sym_mp_vin():
    """Test Sym-MP-VIN training with D8 symmetry."""
    cmd = [
        "python3", "-m", "diffplan.main",
        "--algorithm", "Sym-MP-VIN",
        "--no_equiv_policy",
        "--cont_action",
        "--disable_wandb",
        "--epochs", "1",
        "--data_path", "data/m15_graph-cc_1k_cont_act.pth",
        "--task", "GraphWorld",
        "--has_obstacle",
        "--seed", "42",
        "--group", "d8"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    assert process.returncode == 0, f"Sym-MP-VIN training failed with error:\n{process.stderr}" 