#!/usr/bin/env python3
"""
3D Visual Navigation Demo using Habitat-Sim

This script demonstrates how to:
1. Set up a Habitat environment for visual navigation
2. Extract panoramic images and build navigation graphs
3. Compute optimal paths and policies
4. Visualize the navigation environment and results

Usage:
    python habitat_demo.py --scene-path path/to/scene.glb --output-dir demo_output/
    
Requirements:
    - habitat-sim and habitat-lab
    - torch, torch-geometric
    - matplotlib, networkx
    - numpy, PIL
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import habitat_sim
    import habitat_sim.registry as registry
    from habitat_sim.utils.data import PoseExtractor
    from habitat_sim.utils.data.pose_extractor import TopdownView
    import torch_cluster
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx
    from torchvision.models import ResNet18_Weights
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install habitat-sim, torch-geometric, and torchvision")
    sys.exit(1)

try:
    from diffplan.envs.habitat.data_extractor import ImageExtractor
except ImportError:
    print("Warning: Could not import ImageExtractor. Make sure the project is properly installed.")


@registry.register_pose_extractor(name="demo_panorama_extractor")
class DemoPanoramaExtractor(PoseExtractor):
    """Custom pose extractor for generating navigation graphs."""
    
    def __init__(
        self,
        topdown_views: List[Tuple[TopdownView, str, Tuple[float, float, float]]],
        meters_per_pixel: float = 0.1,
        num_points: int = 256,
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)
        self.num_points = num_points

    def extract_poses(
        self, view: np.ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Extract camera poses for navigation graph construction."""
        height, width = view.shape
        dist = min(height, width) // 20

        # Generate random navigable points
        gridpoints = []
        while len(gridpoints) < self.num_points:
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            if self._valid_point(row, col, view) and (row, col) not in gridpoints:
                gridpoints.append((row, col))

        # Generate panoramic poses for each point
        poses = []
        for point in gridpoints:
            point_label_pairs = self._panorama_extraction(point, view, dist)
            poses.extend([(point, point_, fp) for point_, label in point_label_pairs])

        self.gridpoints = gridpoints
        return poses

    def _panorama_extraction(
        self, point: Tuple[int, int], view: np.ndarray, dist: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Generate 4 panoramic viewpoints around each navigation point."""
        r, c = point
        neighbor_dist = dist // 2
        neighbors = [
            (r - neighbor_dist, c - neighbor_dist),
            (r - neighbor_dist, c + neighbor_dist),
            (r + neighbor_dist, c - neighbor_dist),
            (r + neighbor_dist, c + neighbor_dist),
        ]
        return [(n, 0.0) for n in neighbors]


class HabitatNavigationDemo:
    """Demo class for 3D visual navigation using Habitat."""
    
    def __init__(self, scene_path: str, output_dir: str = "demo_output"):
        self.scene_path = scene_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.extractor = None
        self.graph = None
        self.positions = None
        self.goal_distances = None
        
    def setup_environment(self, img_size: Tuple[int, int] = (512, 512)):
        """Initialize the Habitat environment and image extractor."""
        print(f"Setting up Habitat environment with scene: {self.scene_path}")
        
        try:
            self.extractor = ImageExtractor(
                self.scene_path,
                img_size=img_size,
                output=["rgba", "depth", "semantic"],
                pose_extractor_name="demo_panorama_extractor",
                meters_per_pixel=0.1,
                shuffle=False
            )
            print(f"Extracted {len(self.extractor)} panoramic views")
            print(f"Number of navigation nodes: {len(self.extractor.pose_extractor.gridpoints)}")
            
        except Exception as e:
            print(f"Error setting up environment: {e}")
            raise
    
    def display_sample_images(self, num_samples: int = 3):
        """Display sample images from the environment."""
        print(f"Displaying {num_samples} sample images...")
        
        samples = self.extractor[0:num_samples]
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i, sample in enumerate(samples):
            # RGB image
            axes[i, 0].imshow(sample["rgba"][..., :3])
            axes[i, 0].set_title(f"RGB View {i+1}")
            axes[i, 0].axis('off')
            
            # Depth image
            axes[i, 1].imshow(sample["depth"], cmap='viridis')
            axes[i, 1].set_title(f"Depth {i+1}")
            axes[i, 1].axis('off')
            
            # Semantic segmentation
            axes[i, 2].imshow(sample["semantic"], cmap='tab20')
            axes[i, 2].set_title(f"Semantic {i+1}")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        sample_path = os.path.join(self.output_dir, "sample_images.png")
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        print(f"Sample images saved to {sample_path}")
        plt.show()
    
    def build_navigation_graph(self, connection_radius: float = 30, knn_k: int = 15):
        """Build a navigation graph from the extracted poses."""
        print("Building navigation graph...")
        
        # Get node positions
        positions = torch.tensor(
            list(self.extractor.pose_extractor.gridpoints), 
            dtype=torch.float32
        )
        
        # Create edges using radius and k-NN connections
        radius_edges = torch_cluster.radius_graph(positions, r=connection_radius)
        knn_edges = torch_cluster.knn_graph(positions, k=knn_k)
        
        # Combine edge types
        radius_data = Data(x=positions, edge_index=radius_edges)
        knn_data = Data(x=positions, edge_index=knn_edges)
        
        # Convert to NetworkX for easier manipulation
        G_radius = to_networkx(radius_data, node_attrs=["x"], to_undirected=True)
        G_knn = to_networkx(knn_data, node_attrs=["x"], to_undirected=True)
        
        # Create minimum spanning tree and combine with k-NN
        for edge in G_radius.edges:
            weight = np.linalg.norm(
                np.array(G_radius.nodes[edge[0]]["x"]) - 
                np.array(G_radius.nodes[edge[1]]["x"])
            )
            G_radius.edges[edge]["weight"] = weight
            
        for edge in G_knn.edges:
            weight = np.linalg.norm(
                np.array(G_knn.nodes[edge[0]]["x"]) - 
                np.array(G_knn.nodes[edge[1]]["x"])
            )
            G_knn.edges[edge]["weight"] = weight
        
        # Combine graphs
        mst = nx.minimum_spanning_tree(G_radius, weight="weight")
        self.graph = nx.compose(mst, G_knn)
        
        # Store positions for visualization
        self.positions = dict(zip(self.graph.nodes, positions.numpy()))
        
        print(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def remove_wall_crossing_edges(self):
        """Remove edges that cross walls in the environment."""
        print("Removing wall-crossing edges...")
        
        if self.extractor is None:
            raise ValueError("Environment not set up. Call setup_environment() first.")
        
        # Get the top-down view for wall detection
        view = self.extractor.tdv_fp_ref_triples[0][0].topdown_view
        
        edges_to_remove = []
        
        for edge in self.graph.edges:
            node_i = np.array(self.graph.nodes[edge[0]]["x"])
            node_j = np.array(self.graph.nodes[edge[1]]["x"])
            
            # Interpolate points along the edge
            if node_i[0] != node_j[0]:
                interp_x = np.arange(min(node_i[0], node_j[0]), max(node_i[0], node_j[0])).astype(int)
                interp_y = np.interp(interp_x, [node_i[0], node_j[0]], [node_i[1], node_j[1]]).astype(int)
            else:
                interp_y = np.arange(min(node_i[1], node_j[1]), max(node_i[1], node_j[1])).astype(int)
                interp_x = np.full_like(interp_y, int(node_i[0]))
            
            # Check if any interpolated point crosses a wall
            for x, y in zip(interp_x, interp_y):
                if 0 <= x < view.shape[0] and 0 <= y < view.shape[1]:
                    if view[x, y] != 1.0:  # Not navigable
                        edges_to_remove.append(edge)
                        break
        
        # Remove wall-crossing edges
        for edge in edges_to_remove:
            if self.graph.has_edge(*edge):
                self.graph.remove_edge(*edge)
        
        print(f"Removed {len(edges_to_remove)} wall-crossing edges")
    
    def find_goal_and_compute_distances(self, target_object: str = "refrigerator"):
        """Find goal location and compute shortest path distances."""
        print(f"Finding {target_object} and computing distances...")
        
        # Find instances of the target object
        target_ids = []
        for obj_id, obj_name in self.extractor.instance_id_to_name.items():
            if obj_name == target_object:
                target_ids.append(obj_id)
        
        if not target_ids:
            print(f"Warning: {target_object} not found in scene")
            return
        
        # Compute distance to target for each node
        node_distances = {}
        for idx, sample in enumerate(self.extractor):
            pose = self.extractor.pose_extractor.gridpoints[idx // 4]
            
            if pose not in node_distances:
                node_distances[pose] = np.inf
            
            # Create mask for target object pixels
            mask = np.zeros(sample['depth'].shape, dtype=bool)
            for obj_id in target_ids:
                mask = np.logical_or(mask, sample['semantic'] == obj_id)
            
            # Update distance if target is visible
            if np.sum(mask) > 30:  # Threshold to filter noise
                mean_depth = sample['depth'][mask].mean()
                if mean_depth > 0:
                    node_distances[pose] = min(node_distances[pose], mean_depth)
        
        # Convert to ordered array
        pos_to_idx = {pos: idx for idx, pos in enumerate(self.extractor.pose_extractor.gridpoints)}
        self.goal_distances = np.array([node_distances.get(pos, np.inf) for pos in self.extractor.pose_extractor.gridpoints])
        
        # Find goal node (closest to target)
        goal_idx = np.argmin(self.goal_distances)
        print(f"Goal node: {goal_idx}, distance to {target_object}: {self.goal_distances[goal_idx]:.2f}")
        
        # Compute shortest path distances from goal
        try:
            dijkstra_dist = nx.single_source_dijkstra_path_length(
                self.graph, goal_idx, weight="weight"
            )
            self.path_distances = np.full(len(self.graph), np.inf)
            for node_idx, dist in dijkstra_dist.items():
                self.path_distances[node_idx] = dist
                
        except Exception as e:
            print(f"Error computing shortest paths: {e}")
            self.path_distances = np.zeros(len(self.graph))
    
    def visualize_navigation_graph(self, show_goal: bool = True):
        """Visualize the navigation graph with optional goal highlighting."""
        print("Visualizing navigation graph...")
        
        if self.graph is None:
            raise ValueError("Graph not built. Call build_navigation_graph() first.")
        
        # Get the top-down view for background
        view = self.extractor.tdv_fp_ref_triples[0][0].topdown_view
        
        plt.figure(figsize=(12, 10))
        
        # Show environment map
        plt.imshow(view.T, cmap='gray', alpha=0.7)
        
        # Draw graph
        if hasattr(self, 'path_distances'):
            node_colors = self.path_distances
            cmap = 'viridis'
            colorbar_label = 'Distance to Goal'
        else:
            node_colors = 'blue'
            cmap = None
            colorbar_label = None
        
        nx.draw(self.graph, self.positions, 
                node_size=50, node_color=node_colors, 
                cmap=cmap, edge_color='red', alpha=0.6, width=0.5)
        
        if colorbar_label and hasattr(self, 'path_distances'):
            plt.colorbar(label=colorbar_label)
        
        # Highlight goal if available
        if show_goal and hasattr(self, 'goal_distances'):
            goal_idx = np.argmin(self.goal_distances)
            goal_pos = self.positions[goal_idx]
            plt.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, label='Goal')
            plt.legend()
        
        plt.title('Navigation Graph')
        plt.axis('equal')
        
        # Save visualization
        graph_path = os.path.join(self.output_dir, "navigation_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Navigation graph saved to {graph_path}")
        plt.show()
    
    def show_goal_view(self):
        """Display the view from the goal location."""
        if not hasattr(self, 'goal_distances'):
            print("Goal not computed yet. Call find_goal_and_compute_distances() first.")
            return
        
        goal_idx = np.argmin(self.goal_distances)
        goal_pos = self.extractor.pose_extractor.gridpoints[goal_idx]
        
        print(f"Showing view from goal node {goal_idx}")
        
        # Find sample from goal position
        for idx, sample in enumerate(self.extractor):
            pose = self.extractor.pose_extractor.gridpoints[idx // 4]
            if pose == goal_pos:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(sample["rgba"][..., :3])
                axes[0].set_title("Goal View - RGB")
                axes[0].axis('off')
                
                axes[1].imshow(sample["depth"], cmap='viridis')
                axes[1].set_title("Goal View - Depth") 
                axes[1].axis('off')
                
                axes[2].imshow(sample["semantic"], cmap='tab20')
                axes[2].set_title("Goal View - Semantic")
                axes[2].axis('off')
                
                plt.tight_layout()
                goal_view_path = os.path.join(self.output_dir, "goal_view.png")
                plt.savefig(goal_view_path, dpi=150, bbox_inches='tight')
                print(f"Goal view saved to {goal_view_path}")
                plt.show()
                break
    
    def cleanup(self):
        """Clean up resources."""
        if self.extractor:
            self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description='Habitat 3D Navigation Demo')
    parser.add_argument('--scene-path', required=True, help='Path to Habitat scene (.glb file)')
    parser.add_argument('--output-dir', default='demo_output', help='Output directory')
    parser.add_argument('--target-object', default='refrigerator', help='Target object to navigate to')
    parser.add_argument('--img-size', nargs=2, type=int, default=[512, 512], help='Image size (width height)')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of sample images to display')
    
    args = parser.parse_args()
    
    # Check if scene file exists
    if not os.path.exists(args.scene_path):
        print(f"Error: Scene file not found: {args.scene_path}")
        return 1
    
    print("=" * 60)
    print("Habitat 3D Visual Navigation Demo")
    print("=" * 60)
    
    try:
        # Initialize demo
        demo = HabitatNavigationDemo(args.scene_path, args.output_dir)
        
        # Setup environment
        demo.setup_environment(tuple(args.img_size))
        
        # Display sample images
        demo.display_sample_images(args.num_samples)
        
        # Build navigation graph
        demo.build_navigation_graph()
        
        # Remove wall-crossing edges
        demo.remove_wall_crossing_edges()
        
        # Find goal and compute distances
        demo.find_goal_and_compute_distances(args.target_object)
        
        # Visualize results
        demo.visualize_navigation_graph()
        demo.show_goal_view()
        
        print(f"Demo completed successfully! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        return 1
    
    finally:
        # Cleanup
        if 'demo' in locals():
            demo.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())