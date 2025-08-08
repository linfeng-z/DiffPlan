#!/usr/bin/env python3
"""
Model Inference Tutorial

This tutorial demonstrates how to load pre-trained DiffPlan models and run inference
on new data. It covers both grid-based and graph-based models.

Usage:
    python examples/tutorials/model_inference.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffplan.modules.Base import LitGridPlanner, LitGraphPlanner
from diffplan.modules.helpers import StandardReturn
from diffplan.envs.maze_env import MazeEnv
from diffplan.utils.vis_fields import plot_value_iteration


def load_pretrained_model(model_path: str, model_type: str = "grid"):
    """
    Load a pre-trained model from checkpoint.
    
    Args:
        model_path: Path to the .ckpt file
        model_type: "grid" or "graph"
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please ensure you have downloaded the pre-trained models.")
        return None
    
    try:
        if model_type == "grid":
            model = LitGridPlanner.load_from_checkpoint(model_path)
        elif model_type == "graph":
            model = LitGraphPlanner.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.eval()
        print(f"✅ Successfully loaded {model_type} model")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def run_grid_inference_example():
    """Demonstrate inference with a grid-based model (VIN/SymVIN)."""
    
    print("\n" + "="*50)
    print("GRID MODEL INFERENCE EXAMPLE")
    print("="*50)
    
    # Model path (update this with actual model path when available)
    model_path = "pretrained_models/grid/VIN_maze15.ckpt"
    
    # Load the model
    model = load_pretrained_model(model_path, "grid")
    if model is None:
        print("Skipping grid inference example - no model available")
        return
    
    # Generate sample maze data
    print("Generating sample maze environment...")
    env = MazeEnv(size=15, num_obstacles=20)
    maze_data = env.generate_random_maze()
    
    # Convert to tensor format expected by model
    obstacle_map = torch.FloatTensor(maze_data['obstacles']).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    goal_map = torch.FloatTensor(maze_data['goal']).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Combine obstacle and goal maps as input
        input_data = torch.cat([obstacle_map, goal_map], dim=1)  # [1, 2, H, W]
        
        # Forward pass
        result = model.net(input_data)
        
        if isinstance(result, StandardReturn):
            logits = result.logits
            probs = result.probs
        else:
            logits = result
            probs = torch.softmax(logits, dim=1)
    
    print(f"✅ Inference complete!")
    print(f"   Output shape: {logits.shape}")
    print(f"   Policy shape: {probs.shape}")
    
    # Visualize results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original maze
        axes[0, 0].imshow(maze_data['obstacles'], cmap='gray')
        axes[0, 0].set_title('Obstacles')
        axes[0, 0].axis('off')
        
        # Goal
        axes[0, 1].imshow(maze_data['goal'], cmap='Reds')
        axes[0, 1].set_title('Goal')
        axes[0, 1].axis('off')
        
        # Policy (argmax of action probabilities)
        policy = torch.argmax(probs[0], dim=0).cpu().numpy()
        axes[1, 0].imshow(policy, cmap='viridis')
        axes[1, 0].set_title('Predicted Policy')
        axes[1, 0].axis('off')
        
        # Action probabilities for a specific action (e.g., action 0)
        if probs.shape[1] > 0:
            action_prob = probs[0, 0].cpu().numpy()
            axes[1, 1].imshow(action_prob, cmap='plasma')
            axes[1, 1].set_title('Action 0 Probability')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('grid_inference_results.png', dpi=150, bbox_inches='tight')
        print("📊 Results saved to 'grid_inference_results.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def run_graph_inference_example():
    """Demonstrate inference with a graph-based model (MP-VIN/Sym-MP-VIN)."""
    
    print("\n" + "="*50)
    print("GRAPH MODEL INFERENCE EXAMPLE")
    print("="*50)
    
    # Model path (update this with actual model path when available)
    model_path = "pretrained_models/graph/MP-VIN_graph.ckpt"
    
    # Load the model
    model = load_pretrained_model(model_path, "graph")
    if model is None:
        print("Skipping graph inference example - no model available")
        return
    
    # Generate sample graph data
    print("Generating sample graph data...")
    num_nodes = 20
    
    # Create random node features (position + goal indicator)
    positions = torch.randn(num_nodes, 2)  # Random 2D positions
    goals = torch.zeros(num_nodes, 1)
    goals[-1] = 1.0  # Last node is the goal
    
    node_features = torch.cat([positions, goals], dim=1)  # [num_nodes, 3]
    
    # Create random edges (fully connected for simplicity)
    edge_indices = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Bidirectional
    
    edge_index = torch.LongTensor(edge_indices).t().contiguous()  # [2, num_edges]
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        try:
            # Note: Actual graph data format may vary based on model implementation
            # This is a simplified example
            result = model.net(node_features.unsqueeze(0), edge_index)
            
            if isinstance(result, StandardReturn):
                logits = result.logits
                probs = result.probs
            else:
                logits = result
                probs = torch.softmax(logits, dim=-1)
        
        except Exception as e:
            print(f"⚠️  Graph inference failed (expected without real pre-trained model): {e}")
            return
    
    print(f"✅ Inference complete!")
    print(f"   Output shape: {logits.shape}")
    print(f"   Policy shape: {probs.shape}")
    
    # Simple visualization
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Node positions
        pos_np = positions.cpu().numpy()
        axes[0].scatter(pos_np[:, 0], pos_np[:, 1], c='blue', s=50)
        axes[0].scatter(pos_np[-1, 0], pos_np[-1, 1], c='red', s=100, marker='*')  # Goal
        axes[0].set_title('Graph Layout')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        
        # Policy values (if available)
        if len(probs.shape) >= 2 and probs.shape[-1] > 1:
            policy_values = torch.max(probs[0], dim=-1)[0].cpu().numpy()
            scatter = axes[1].scatter(pos_np[:, 0], pos_np[:, 1], 
                                    c=policy_values, cmap='viridis', s=100)
            plt.colorbar(scatter, ax=axes[1])
            axes[1].set_title('Policy Confidence')
            axes[1].set_xlabel('X Position')
            axes[1].set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig('graph_inference_results.png', dpi=150, bbox_inches='tight')
        print("📊 Results saved to 'graph_inference_results.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def print_model_info(model_path: str):
    """Print information about a model checkpoint."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n📋 Model Information: {os.path.basename(model_path)}")
        print("-" * 40)
        
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            print(f"Algorithm: {hp.get('algorithm', 'Unknown')}")
            print(f"Learning Rate: {hp.get('lr', 'Unknown')}")
            print(f"Batch Size: {hp.get('batch_size', 'Unknown')}")
            print(f"K (iterations): {hp.get('k', 'Unknown')}")
        
        if 'epoch' in checkpoint:
            print(f"Training Epochs: {checkpoint['epoch']}")
        
        if 'state_dict' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['state_dict'].values())
            print(f"Total Parameters: {total_params:,}")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"Error reading model info: {e}")


def main():
    """Main function to run all inference examples."""
    
    print("🚀 DiffPlan Model Inference Tutorial")
    print("=" * 60)
    
    # List expected pre-trained models
    expected_models = [
        "pretrained_models/grid/VIN_maze15.ckpt",
        "pretrained_models/grid/SymVIN_maze15.ckpt", 
        "pretrained_models/graph/MP-VIN_graph.ckpt",
        "pretrained_models/graph/Sym-MP-VIN_graph.ckpt"
    ]
    
    print("\n📦 Checking for pre-trained models...")
    available_models = []
    for model_path in expected_models:
        if os.path.exists(model_path):
            available_models.append(model_path)
            print(f"✅ Found: {model_path}")
            print_model_info(model_path)
        else:
            print(f"⚠️  Missing: {model_path}")
    
    if not available_models:
        print("\n❌ No pre-trained models found!")
        print("Please download or train models first, then place them in the pretrained_models/ directory.")
        print("\nTo train models, run:")
        print("  python -m diffplan.main --algorithm VIN --data_path data/m15_4abs-cc_10k.npz")
        return
    
    # Run inference examples
    print("\n🎯 Running inference examples...")
    
    # Grid-based model inference
    run_grid_inference_example()
    
    # Graph-based model inference  
    run_graph_inference_example()
    
    print("\n✅ Tutorial completed!")
    print("Check the generated PNG files for visualization results.")


if __name__ == "__main__":
    main()