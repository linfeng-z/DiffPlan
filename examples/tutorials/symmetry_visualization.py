#!/usr/bin/env python3
"""
Symmetry and Equivariance Visualization Tutorial

This tutorial demonstrates:
1. How symmetric models handle rotations and reflections
2. Visualizing group actions on planning problems
3. Comparing equivariant vs non-equivariant models
4. Understanding the benefits of symmetry in planning

Usage:
    python symmetry_visualization.py --demo group-actions
    python symmetry_visualization.py --demo equivariance-test
    python symmetry_visualization.py --demo comparison
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from diffplan.utils.tensor_transform import *
    from diffplan.utils.tensor_transform_e2cnn import *
except ImportError:
    print("Warning: Could not import tensor transform utilities")


class SymmetryDemo:
    """Demonstration of symmetry and equivariance in planning."""
    
    def __init__(self, output_dir: str = "symmetry_demo"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define symmetry groups
        self.groups = {
            'd4': 'D4 (4-fold rotation)',
            'd8': 'D8 (4-fold rotation + reflection)',
            'so2': 'SO(2) (continuous rotation)'
        }
    
    def create_asymmetric_maze(self, size: int = 15):
        """Create an asymmetric maze to demonstrate symmetry effects."""
        maze = np.ones((size, size), dtype=np.float32)
        
        # Create L-shaped corridor
        maze[5:10, 2:4] = 0  # Vertical obstacle
        maze[2:5, 7:12] = 0  # Horizontal obstacle
        maze[10:13, 8:10] = 0  # Small obstacle
        
        # Ensure boundaries are walls
        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0
        
        # Create some free path
        maze[1, 1] = 1  # Start
        maze[size-2, size-2] = 1  # Goal
        
        return maze
    
    def apply_group_action(self, tensor, group: str, action_idx: int):
        """Apply group action to tensor."""
        if group == 'd4':
            return self.apply_d4_action(tensor, action_idx)
        elif group == 'd8':
            return self.apply_d8_action(tensor, action_idx)
        else:
            return tensor  # SO(2) would need continuous rotation
    
    def apply_d4_action(self, tensor, action_idx: int):
        """Apply D4 group action (rotations)."""
        # D4 has 4 elements: identity, 90°, 180°, 270° rotation
        if action_idx == 0:
            return tensor  # Identity
        elif action_idx == 1:
            return torch.rot90(tensor, k=1, dims=(-2, -1))  # 90° CCW
        elif action_idx == 2:
            return torch.rot90(tensor, k=2, dims=(-2, -1))  # 180°
        elif action_idx == 3:
            return torch.rot90(tensor, k=3, dims=(-2, -1))  # 270° CCW
        else:
            raise ValueError(f"Invalid D4 action index: {action_idx}")
    
    def apply_d8_action(self, tensor, action_idx: int):
        """Apply D8 group action (rotations + reflections)."""
        # D8 has 8 elements: 4 rotations + 4 reflections
        if action_idx < 4:
            return self.apply_d4_action(tensor, action_idx)
        else:
            # Apply reflection then rotation
            reflected = torch.flip(tensor, dims=[-1])  # Horizontal flip
            return self.apply_d4_action(reflected, action_idx - 4)
    
    def demonstrate_group_actions(self):
        """Visualize how group actions transform planning problems."""
        print("Demonstrating group actions on planning problems...")
        
        # Create test maze
        maze = self.create_asymmetric_maze(15)
        start = (1, 1)
        goal = (13, 13)
        
        # Create input tensor: [obstacle_map, start_map, goal_map]
        input_tensor = torch.zeros(3, 15, 15)
        input_tensor[0] = torch.from_numpy(1 - maze)  # Obstacles (1 = obstacle)
        input_tensor[1, start[0], start[1]] = 1.0
        input_tensor[2, goal[0], goal[1]] = 1.0
        
        # Test D4 and D8 groups
        groups_to_test = ['d4', 'd8']
        
        for group in groups_to_test:
            num_actions = 4 if group == 'd4' else 8
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'{self.groups[group]} Actions on Planning Problem', fontsize=16)
            
            for i in range(num_actions):
                row = i // 4
                col = i % 4
                
                # Apply group action
                transformed = self.apply_group_action(input_tensor, group, i)
                
                # Visualize
                ax = axes[row, col] if num_actions > 4 else axes[col]
                
                # Combine channels for visualization
                viz = torch.zeros(15, 15, 3)
                viz[:, :, 0] = transformed[0]  # Obstacles in red
                viz[:, :, 1] = transformed[1]  # Start in green
                viz[:, :, 2] = transformed[2]  # Goal in blue
                
                ax.imshow(viz.numpy())
                ax.set_title(f'Action {i}')
                ax.axis('off')
            
            # Hide unused subplots for D4
            if group == 'd4':
                for i in range(4, 8):
                    row = i // 4
                    col = i % 4
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f'{group}_actions.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved {group} actions to {output_path}")
            plt.show()
    
    def test_equivariance_property(self):
        """Test and visualize the equivariance property."""
        print("Testing equivariance property...")
        
        # This is a conceptual demonstration
        # In practice, you would need actual trained models
        
        maze = self.create_asymmetric_maze(12)
        
        # Create figure to show equivariance concept
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Equivariance Property: f(g·x) = g·f(x)', fontsize=16)
        
        # Original problem
        axes[0, 0].imshow(maze, cmap='gray')
        axes[0, 0].set_title('Original Problem')
        axes[0, 0].axis('off')
        
        # Rotated problem
        rotated_maze = np.rot90(maze)
        axes[0, 1].imshow(rotated_maze, cmap='gray') 
        axes[0, 1].set_title('Rotated Problem\n(g·x)')
        axes[0, 1].axis('off')
        
        # Expected equivariant response
        axes[0, 2].text(0.5, 0.5, 'Model should produce\nrotated solution\n(g·f(x))', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 2].axis('off')
        
        # Show what non-equivariant model might do
        axes[1, 0].imshow(maze, cmap='gray')
        axes[1, 0].set_title('Non-Equivariant Model')
        axes[1, 0].text(0.5, -0.1, 'Inconsistent with rotations', 
                       ha='center', transform=axes[1, 0].transAxes, color='red')
        axes[1, 0].axis('off')
        
        # Show what equivariant model does
        axes[1, 1].imshow(rotated_maze, cmap='gray')
        axes[1, 1].set_title('Equivariant Model')
        axes[1, 1].text(0.5, -0.1, 'Consistent behavior', 
                       ha='center', transform=axes[1, 1].transAxes, color='green')
        axes[1, 1].axis('off')
        
        # Benefits
        axes[1, 2].text(0.5, 0.5, 'Benefits:\n• Better generalization\n• Data efficiency\n• Principled inductive bias', 
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'equivariance_concept.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved equivariance concept to {output_path}")
        plt.show()
    
    def compare_symmetric_vs_regular(self):
        """Compare symmetric vs regular models conceptually."""
        print("Comparing symmetric vs regular models...")
        
        # Create diverse test scenarios
        scenarios = []
        
        # Scenario 1: Axis-aligned corridor
        maze1 = np.ones((10, 10))
        maze1[4:6, 2:8] = 0  # Horizontal corridor
        scenarios.append(("Horizontal Corridor", maze1))
        
        # Scenario 2: Rotated version
        maze2 = np.ones((10, 10))
        maze2[2:8, 4:6] = 0  # Vertical corridor  
        scenarios.append(("Vertical Corridor", maze2))
        
        # Scenario 3: Diagonal-ish pattern
        maze3 = np.ones((10, 10))
        for i in range(2, 8):
            maze3[i, i] = 0
        scenarios.append(("Diagonal Pattern", maze3))
        
        # Scenario 4: Complex asymmetric
        maze4 = self.create_asymmetric_maze(10)
        scenarios.append(("Complex Asymmetric", maze4))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Symmetric vs Regular Models: Generalization', fontsize=16)
        
        for i, (name, maze) in enumerate(scenarios):
            # Top row: scenarios
            axes[0, i].imshow(maze, cmap='gray')
            axes[0, i].set_title(name)
            axes[0, i].axis('off')
            
            # Bottom row: model performance illustration
            if i < 2:  # Similar scenarios
                performance = "Good" if i == 0 else "Poor (Regular)\nGood (Symmetric)"
                color = "lightgreen" if i == 0 else "lightyellow"
            else:
                performance = "Variable\n(Both models)"
                color = "lightcoral"
            
            axes[1, i].text(0.5, 0.5, f'Performance:\n{performance}', 
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {output_path}")
        plt.show()
    
    def create_symmetry_summary(self):
        """Create a summary visualization of symmetry concepts."""
        print("Creating symmetry summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Symmetry in Differentiable Planning - Summary', fontsize=16)
        
        # Quadrant 1: Group theory basics
        axes[0, 0].text(0.5, 0.7, 'Symmetry Groups', ha='center', fontsize=14, weight='bold')
        axes[0, 0].text(0.5, 0.5, 
                       'D4: 4-fold rotations\n' +
                       'D8: Rotations + reflections\n' +
                       'SO(2): Continuous rotations\n' +
                       'SE(2): Rotations + translations',
                       ha='center', va='center', fontsize=11)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        
        # Quadrant 2: Applications
        axes[0, 1].text(0.5, 0.7, 'Applications', ha='center', fontsize=14, weight='bold')
        axes[0, 1].text(0.5, 0.5,
                       '• Grid-world navigation\n' +
                       '• Graph-based planning\n' +
                       '• Visual navigation\n' +
                       '• Robotic manipulation',
                       ha='center', va='center', fontsize=11)
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        
        # Quadrant 3: Model architectures
        axes[1, 0].text(0.5, 0.7, 'Model Types', ha='center', fontsize=14, weight='bold')
        axes[1, 0].text(0.5, 0.5,
                       'SymVIN: Grid + Steerable CNNs\n' +
                       'Sym-MP-VIN: Graph + Equivariant GNNs\n' +
                       'E(2)-Equivariant: SE(2) symmetry',
                       ha='center', va='center', fontsize=11)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        # Quadrant 4: Benefits
        axes[1, 1].text(0.5, 0.7, 'Benefits', ha='center', fontsize=14, weight='bold')
        axes[1, 1].text(0.5, 0.5,
                       '✓ Sample efficiency\n' +
                       '✓ Better generalization\n' +
                       '✓ Principled inductive bias\n' +
                       '✓ Robust to viewpoint changes',
                       ha='center', va='center', fontsize=11, color='green')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'symmetry_summary.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved symmetry summary to {output_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Symmetry and Equivariance Visualization')
    parser.add_argument('--demo', choices=['group-actions', 'equivariance-test', 'comparison', 'summary', 'all'],
                       default='all', help='Which demonstration to run')
    parser.add_argument('--output-dir', default='symmetry_demo', help='Output directory')
    
    args = parser.parse_args()
    
    demo = SymmetryDemo(args.output_dir)
    
    print("=" * 60)
    print("Symmetry and Equivariance in Planning")
    print("=" * 60)
    
    if args.demo == 'group-actions' or args.demo == 'all':
        demo.demonstrate_group_actions()
    
    if args.demo == 'equivariance-test' or args.demo == 'all':
        demo.test_equivariance_property()
    
    if args.demo == 'comparison' or args.demo == 'all':
        demo.compare_symmetric_vs_regular()
    
    if args.demo == 'summary' or args.demo == 'all':
        demo.create_symmetry_summary()
    
    print(f"\nAll visualizations saved to {args.output_dir}/")
    print("These demonstrate key concepts in geometric equivariance for planning!")
    
    return 0


if __name__ == "__main__":
    exit(main())