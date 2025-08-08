#!/usr/bin/env python3
"""
Basic Training Tutorial for DiffPlan Models

This tutorial demonstrates how to:
1. Generate training data for different environments
2. Train basic planning models (VIN, SymVIN, MP-VIN, Sym-MP-VIN)
3. Evaluate model performance
4. Save and load trained models

Usage:
    python basic_training.py --help
    python basic_training.py --quick-demo  # Run a fast demo
    python basic_training.py --algorithm VIN --epochs 50
    python basic_training.py --algorithm Sym-MP-VIN --group d8 --continuous
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from diffplan.main import main as train_main
    from diffplan.envs.generate_dataset import main as generate_data_main
except ImportError:
    print("Error: Could not import diffplan modules. Make sure the package is installed:")
    print("pip install -e .")
    sys.exit(1)


def generate_toy_dataset(data_path: str, env_type: str = "grid", size: int = 15, 
                        train_size: int = 1000, continuous: bool = False):
    """Generate a small dataset for training demonstrations."""
    print(f"Generating toy dataset: {data_path}")
    
    if env_type == "grid":
        # Grid world dataset
        args = [
            "--output-path", data_path,
            "--mechanism", "4abs-cc",
            "--maze-size", str(size),
            "--train-size", str(train_size),
            "--valid-size", str(train_size // 5),
            "--test-size", str(train_size // 5),
            "--env", "RandomMaze"
        ]
    elif env_type == "graph":
        # Graph world dataset
        args = [
            "--output-path", data_path,
            "--mechanism", "4abs-cc", 
            "--maze-size", str(size),
            "--train-size", str(train_size),
            "--valid-size", str(train_size // 5),
            "--test-size", str(train_size // 5),
            "--env", "RandomMazeGraph",
            "--has_obstacle"
        ]
        if continuous:
            args.append("--cont_action")
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    # Mock the command line arguments for the data generation script
    old_argv = sys.argv
    try:
        sys.argv = ["generate_dataset.py"] + args
        generate_data_main()
        print(f"Dataset generated successfully: {data_path}")
    except Exception as e:
        print(f"Error generating dataset: {e}")
        raise
    finally:
        sys.argv = old_argv


def train_model(algorithm: str, data_path: str, epochs: int = 10, 
               group: str = None, continuous: bool = False, 
               output_dir: str = "tutorial_models"):
    """Train a planning model with the specified configuration."""
    print(f"Training {algorithm} model for {epochs} epochs")
    
    # Determine task type from data path
    if "graph" in data_path.lower():
        task = "GraphWorld"
        has_obstacle = True
    else:
        task = "GridWorld"
        has_obstacle = False
    
    # Base training arguments
    args = [
        "--algorithm", algorithm,
        "--data_path", data_path,
        "--task", task,
        "--epochs", str(epochs),
        "--disable_wandb",  # Disable logging for tutorial
        "--seed", "42"
    ]
    
    # Add algorithm-specific arguments
    if "Sym" in algorithm and group:
        args.extend(["--group", group])
    
    if continuous:
        args.append("--cont_action")
        if "Sym" in algorithm:
            args.append("--no_equiv_policy")
    
    if has_obstacle:
        args.append("--has_obstacle")
    
    # Mock command line arguments for training
    old_argv = sys.argv
    try:
        sys.argv = ["main.py"] + args
        train_main()
        print(f"Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        sys.argv = old_argv


def run_quick_demo():
    """Run a quick demonstration with minimal data and training."""
    print("=" * 60)
    print("Quick Demo: Training DiffPlan Models")
    print("=" * 60)
    
    demo_dir = "tutorial_output"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 1. Generate small grid world dataset
    grid_data_path = os.path.join(demo_dir, "demo_grid_data.npz")
    if not os.path.exists(grid_data_path):
        print("\n1. Generating Grid World dataset...")
        generate_toy_dataset(grid_data_path, "grid", size=15, train_size=500)
    else:
        print(f"\n1. Using existing dataset: {grid_data_path}")
    
    # 2. Train VIN model
    print("\n2. Training VIN (Value Iteration Network)...")
    train_model("VIN", grid_data_path, epochs=5)
    
    # 3. Generate graph world dataset
    graph_data_path = os.path.join(demo_dir, "demo_graph_data.pth")
    if not os.path.exists(graph_data_path):
        print("\n3. Generating Graph World dataset...")
        generate_toy_dataset(graph_data_path, "graph", size=15, train_size=500)
    else:
        print(f"\n3. Using existing dataset: {graph_data_path}")
    
    # 4. Train MP-VIN model
    print("\n4. Training MP-VIN (Message Passing VIN)...")
    train_model("MP-VIN", graph_data_path, epochs=5)
    
    # 5. Train Symmetric MP-VIN
    print("\n5. Training Sym-MP-VIN with D8 symmetry...")
    train_model("Sym-MP-VIN", graph_data_path, epochs=5, group="d8")
    
    print("\n" + "=" * 60)
    print("Quick demo completed successfully!")
    print("Check the tutorial_output/ directory for generated data and model checkpoints.")
    print("=" * 60)


def run_comprehensive_example(algorithm: str, epochs: int, group: str = None, continuous: bool = False):
    """Run a more comprehensive training example."""
    print("=" * 60)
    print(f"Comprehensive Training: {algorithm}")
    print("=" * 60)
    
    output_dir = "comprehensive_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine data requirements
    if "MP" in algorithm:
        env_type = "graph"
        data_extension = ".pth"
    else:
        env_type = "grid"
        data_extension = ".npz"
    
    # Generate larger dataset
    data_name = f"{algorithm.lower()}_{env_type}_data"
    if continuous:
        data_name += "_continuous"
    data_path = os.path.join(output_dir, data_name + data_extension)
    
    if not os.path.exists(data_path):
        print(f"\n1. Generating {env_type} world dataset for {algorithm}...")
        generate_toy_dataset(data_path, env_type, size=15, train_size=2000, continuous=continuous)
    else:
        print(f"\n1. Using existing dataset: {data_path}")
    
    # Train the model
    print(f"\n2. Training {algorithm} model...")
    train_model(algorithm, data_path, epochs=epochs, group=group, continuous=continuous)
    
    print(f"\n{algorithm} training completed!")
    print(f"Check {output_dir}/ for results.")


def print_available_algorithms():
    """Print information about available algorithms."""
    algorithms = {
        "VIN": "Value Iteration Network - Basic differentiable planning on grids",
        "SymVIN": "Symmetric VIN - VIN with equivariance to rotations and reflections",
        "MP-VIN": "Message Passing VIN - Graph-based planning with message passing",
        "Sym-MP-VIN": "Symmetric MP-VIN - Graph planning with geometric equivariance",
        "MP-VIN-NoT": "MP-VIN without transformations - Simpler graph variant",
        "Sym-MP-VIN-NoT": "Symmetric MP-VIN without transformations"
    }
    
    print("\nAvailable Algorithms:")
    print("-" * 40)
    for name, desc in algorithms.items():
        print(f"{name:15} - {desc}")
    
    print("\nSymmetry Groups (for symmetric models):")
    print("-" * 40)
    print("d4               - 4-fold rotational symmetry")
    print("d8               - 8-fold rotational + reflection symmetry")
    print("so2              - Continuous rotational symmetry")


def main():
    parser = argparse.ArgumentParser(
        description="Basic Training Tutorial for DiffPlan Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with multiple models
  python basic_training.py --quick-demo
  
  # Train VIN on grid world
  python basic_training.py --algorithm VIN --epochs 20
  
  # Train symmetric MP-VIN with continuous actions
  python basic_training.py --algorithm Sym-MP-VIN --group d8 --continuous --epochs 30
  
  # Train basic MP-VIN on graph world
  python basic_training.py --algorithm MP-VIN --epochs 25
        """
    )
    
    parser.add_argument('--quick-demo', action='store_true', 
                       help='Run a quick demo with multiple models (5 epochs each)')
    parser.add_argument('--algorithm', choices=['VIN', 'SymVIN', 'MP-VIN', 'Sym-MP-VIN', 
                                               'MP-VIN-NoT', 'Sym-MP-VIN-NoT'],
                       help='Algorithm to train')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--group', choices=['d4', 'd8', 'so2'], 
                       help='Symmetry group for equivariant models')
    parser.add_argument('--continuous', action='store_true',
                       help='Use continuous actions (for graph models)')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms and exit')
    
    args = parser.parse_args()
    
    if args.list_algorithms:
        print_available_algorithms()
        return 0
    
    if args.quick_demo:
        run_quick_demo()
        return 0
    
    if not args.algorithm:
        print("Error: Must specify --algorithm or use --quick-demo")
        parser.print_help()
        return 1
    
    # Validate symmetry group for symmetric models
    if "Sym" in args.algorithm and not args.group:
        print(f"Error: {args.algorithm} requires a symmetry group (--group)")
        print("Available groups: d4, d8, so2")
        return 1
    
    # Run comprehensive training
    run_comprehensive_example(
        args.algorithm, 
        args.epochs, 
        args.group, 
        args.continuous
    )
    
    return 0


if __name__ == "__main__":
    exit(main())