#!/usr/bin/env python3
"""
Model Usage Tutorial for DiffPlan

This tutorial shows how to:
1. Load pre-trained models
2. Use models for inference on new planning problems
3. Understand model inputs and outputs
4. Visualize planning results
5. Compare different model architectures

Usage:
    python model_usage.py --model-path path/to/model.pth --test-data path/to/test_data
    python model_usage.py --interactive  # Interactive mode for testing
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from diffplan.modules.Dataset import *
    from diffplan.utils.dijkstra import dijkstra_np
except ImportError as e:
    print(f"Error importing diffplan modules: {e}")
    print("Make sure the package is installed: pip install -e .")
    sys.exit(1)


class ModelDemo:
    """Demonstration class for using trained DiffPlan models."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a pre-trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration from checkpoint
            if 'hyper_parameters' in checkpoint:
                config = checkpoint['hyper_parameters']
                self.model_type = config.get('algorithm', 'Unknown')
                
                # Create model based on configuration
                self.model = get_model(config)
                
                # Load state dict
                if 'state_dict' in checkpoint:
                    # Remove 'model.' prefix if present (Lightning checkpoint format)
                    state_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        if k.startswith('model.'):
                            state_dict[k[6:]] = v
                        else:
                            state_dict[k] = v
                    self.model.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                print(f"Successfully loaded {self.model_type} model")
                
            else:
                raise ValueError("Invalid checkpoint format - missing hyper_parameters")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_simple_maze(self, size: int = 15, obstacle_prob: float = 0.3):
        """Create a simple test maze for demonstration."""
        np.random.seed(42)  # For reproducibility
        
        # Create maze with obstacles
        maze = np.ones((size, size), dtype=np.float32)
        
        # Add random obstacles
        for i in range(size):
            for j in range(size):
                if np.random.random() < obstacle_prob:
                    maze[i, j] = 0.0
        
        # Ensure start and goal are free
        maze[1, 1] = 1.0  # Start
        maze[size-2, size-2] = 1.0  # Goal
        
        # Create connected path (simple)
        for i in range(1, size-1):
            maze[i, 1] = 1.0
            maze[size-2, i] = 1.0
        
        return maze
    
    def create_test_sample(self, maze_size: int = 15):
        """Create a test sample for model inference."""
        maze = self.create_simple_maze(maze_size)
        
        # Define start and goal positions
        start = (1, 1)
        goal = (maze_size-2, maze_size-2)
        
        # Create input format based on model type
        if "MP" in self.model_type:
            # Graph-based models need edge connectivity
            return self.create_graph_sample(maze, start, goal)
        else:
            # Grid-based models
            return self.create_grid_sample(maze, start, goal)
    
    def create_grid_sample(self, maze, start, goal):
        """Create grid-based input sample."""
        size = maze.shape[0]
        
        # Create input channels: [maze, start, goal]
        input_map = np.zeros((3, size, size), dtype=np.float32)
        input_map[0] = maze  # Obstacle map
        input_map[1, start[0], start[1]] = 1.0  # Start position
        input_map[2, goal[0], goal[1]] = 1.0   # Goal position
        
        # Compute optimal policy using Dijkstra
        optimal_policy = self.compute_optimal_policy_grid(maze, goal)
        
        return {
            'input': torch.from_numpy(input_map).unsqueeze(0),  # Add batch dimension
            'target': torch.from_numpy(optimal_policy).unsqueeze(0),
            'maze': maze,
            'start': start,
            'goal': goal
        }
    
    def create_graph_sample(self, maze, start, goal):
        """Create graph-based input sample (simplified)."""
        # This is a simplified version - in practice, you'd convert maze to graph
        print("Graph sample generation not fully implemented in this demo")
        print("For graph models, use pre-generated graph datasets")
        return None
    
    def compute_optimal_policy_grid(self, maze, goal):
        """Compute optimal policy for grid world using Dijkstra."""
        size = maze.shape[0]
        
        # Convert to format expected by dijkstra function
        distances = dijkstra_np(maze, goal)
        
        # Convert distances to policy (action directions)
        policy = np.zeros((4, size, size), dtype=np.float32)  # 4 actions: up, right, down, left
        
        for i in range(size):
            for j in range(size):
                if maze[i, j] == 0:  # Obstacle
                    continue
                
                best_action = 0
                best_value = float('inf')
                
                # Check all 4 neighbors
                actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
                for action_idx, (di, dj) in enumerate(actions):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and maze[ni, nj] > 0:
                        if distances[ni, nj] < best_value:
                            best_value = distances[ni, nj]
                            best_action = action_idx
                
                policy[best_action, i, j] = 1.0
        
        return policy
    
    def run_inference(self, sample):
        """Run model inference on a test sample."""
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        print("Running model inference...")
        
        with torch.no_grad():
            input_data = sample['input'].to(self.device)
            
            # Run forward pass
            if hasattr(self.model, 'forward'):
                output = self.model(input_data)
            else:
                # Some models might have different forward method names
                output = self.model.predict(input_data)
            
            # Convert output to numpy for analysis
            if isinstance(output, tuple):
                # Some models return multiple outputs (policy, value, etc.)
                policy = output[0].cpu().numpy()
                if len(output) > 1:
                    value = output[1].cpu().numpy()
                else:
                    value = None
            else:
                policy = output.cpu().numpy()
                value = None
            
            return {
                'policy': policy,
                'value': value,
                'input': input_data.cpu().numpy()
            }
    
    def visualize_results(self, sample, results, save_path: str = None):
        """Visualize planning results."""
        maze = sample['maze']
        start = sample['start']
        goal = sample['goal']
        
        policy = results['policy'][0]  # Remove batch dimension
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Environment
        axes[0].imshow(maze, cmap='gray')
        axes[0].plot(start[1], start[0], 'go', markersize=10, label='Start')
        axes[0].plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
        axes[0].set_title('Environment')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Predicted Policy (max action per cell)
        policy_vis = np.argmax(policy, axis=0)
        axes[1].imshow(maze, cmap='gray', alpha=0.5)
        axes[1].imshow(policy_vis, cmap='viridis', alpha=0.7)
        axes[1].set_title('Predicted Policy')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Policy arrows
        axes[2].imshow(maze, cmap='gray')
        
        # Draw policy arrows
        actions = [(-0.3, 0), (0, 0.3), (0.3, 0), (0, -0.3)]  # up, right, down, left
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] > 0:  # Only for free cells
                    action = np.argmax(policy[:, i, j])
                    di, dj = actions[action]
                    axes[2].arrow(j, i, dj, di, head_width=0.1, head_length=0.1, 
                                fc='red', ec='red', alpha=0.7)
        
        axes[2].plot(start[1], start[0], 'go', markersize=10, label='Start')
        axes[2].plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
        axes[2].set_title('Policy Arrows')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def interactive_demo(self):
        """Run an interactive demonstration."""
        print("=" * 60)
        print("Interactive Model Demo")
        print("=" * 60)
        
        if self.model is None:
            print("No model loaded. The demo will create a simple test environment.")
        
        while True:
            print("\nOptions:")
            print("1. Create test sample and run inference")
            print("2. Visualize results")
            print("3. Change maze size")
            print("4. Load different model")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                maze_size = int(input("Enter maze size (default 15): ") or "15")
                sample = self.create_test_sample(maze_size)
                
                if sample is None:
                    print("Cannot create sample for this model type")
                    continue
                
                if self.model is not None:
                    results = self.run_inference(sample)
                    print("Inference completed!")
                    
                    # Store for visualization
                    self.last_sample = sample
                    self.last_results = results
                else:
                    print("No model loaded - only showing environment")
                    self.last_sample = sample
                    self.last_results = None
            
            elif choice == '2':
                if hasattr(self, 'last_sample'):
                    if self.last_results is not None:
                        self.visualize_results(self.last_sample, self.last_results)
                    else:
                        # Just show the environment
                        plt.figure(figsize=(8, 6))
                        maze = self.last_sample['maze']
                        start = self.last_sample['start']
                        goal = self.last_sample['goal']
                        
                        plt.imshow(maze, cmap='gray')
                        plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
                        plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
                        plt.title('Test Environment')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.show()
                else:
                    print("No sample available. Run option 1 first.")
            
            elif choice == '3':
                maze_size = int(input("Enter new maze size: "))
                print(f"Maze size set to {maze_size}")
            
            elif choice == '4':
                model_path = input("Enter model path: ").strip()
                try:
                    self.load_model(model_path)
                except Exception as e:
                    print(f"Failed to load model: {e}")
            
            elif choice == '5':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-5.")


def main():
    parser = argparse.ArgumentParser(description='Model Usage Tutorial for DiffPlan')
    parser.add_argument('--model-path', help='Path to trained model checkpoint')
    parser.add_argument('--test-data', help='Path to test dataset')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive demo')
    parser.add_argument('--maze-size', type=int, default=15,
                       help='Size of test maze (default: 15)')
    parser.add_argument('--output-dir', default='demo_output',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize demo
    demo = ModelDemo(args.model_path)
    
    if args.interactive:
        demo.interactive_demo()
    else:
        # Run automated demo
        print("=" * 60)
        print("Model Usage Demo")
        print("=" * 60)
        
        # Create test sample
        print(f"Creating test sample (maze size: {args.maze_size})")
        sample = demo.create_test_sample(args.maze_size)
        
        if sample is None:
            print("Could not create test sample for this model type")
            return 1
        
        if demo.model is not None:
            # Run inference
            results = demo.run_inference(sample)
            
            # Visualize results
            output_path = os.path.join(args.output_dir, 'model_demo_results.png')
            demo.visualize_results(sample, results, output_path)
            
            print(f"Demo completed! Results saved to {args.output_dir}")
        else:
            print("No model loaded - showing environment only")
            
            # Just visualize the environment
            maze = sample['maze']
            start = sample['start'] 
            goal = sample['goal']
            
            plt.figure(figsize=(8, 6))
            plt.imshow(maze, cmap='gray')
            plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
            plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
            plt.title('Test Environment (No Model Loaded)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(args.output_dir, 'test_environment.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Environment visualization saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())