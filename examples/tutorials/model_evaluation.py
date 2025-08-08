#!/usr/bin/env python3
"""
Model Evaluation Tutorial

This tutorial shows how to evaluate pre-trained DiffPlan models on test datasets
and compute various performance metrics.

Usage:
    python examples/tutorials/model_evaluation.py --model_path pretrained_models/grid/VIN_maze15.ckpt
    python examples/tutorials/model_evaluation.py --model_path pretrained_models/graph/MP-VIN_graph.ckpt --model_type graph
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffplan.modules.Base import LitGridPlanner, LitGraphPlanner
from diffplan.modules.Dataset import GridDataModule, GraphDataModule
from diffplan.modules.helpers import StandardReturn


class ModelEvaluator:
    """Utility class for evaluating DiffPlan models."""
    
    def __init__(self, model, model_type: str = "grid"):
        self.model = model
        self.model_type = model_type
        self.model.eval()
        
    def evaluate_success_rate(self, dataloader, max_batches: int = None) -> Dict:
        """
        Evaluate model success rate on a dataset.
        
        Args:
            dataloader: DataLoader with test data
            max_batches: Maximum number of batches to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation metrics
        """
        total_samples = 0
        correct_predictions = 0
        total_loss = 0.0
        batch_count = 0
        
        print(f"🧪 Evaluating model on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                try:
                    # Forward pass
                    if self.model_type == "grid":
                        # Grid data format: (state, action, value_target)
                        state, action_target, value_target = batch
                        result = self.model(state)
                    else:
                        # Graph data format may vary
                        result = self.model(batch)
                    
                    # Extract predictions
                    if isinstance(result, StandardReturn):
                        logits = result.logits
                        probs = result.probs
                    else:
                        logits = result
                        probs = torch.softmax(logits, dim=-1)
                    
                    # Compute accuracy (for classification tasks)
                    if self.model_type == "grid" and action_target is not None:
                        predicted_actions = torch.argmax(logits, dim=1)
                        correct = (predicted_actions == action_target).sum().item()
                        correct_predictions += correct
                        total_samples += action_target.size(0)
                    
                    # Compute loss if possible
                    if self.model_type == "grid" and action_target is not None:
                        loss = torch.nn.functional.cross_entropy(logits, action_target)
                        total_loss += loss.item()
                    
                    batch_count += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(dataloader)}")
                        
                except Exception as e:
                    print(f"⚠️  Error processing batch {batch_idx}: {e}")
                    continue
        
        # Compute metrics
        metrics = {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0,
            'average_loss': total_loss / batch_count if batch_count > 0 else 0.0,
            'batches_processed': batch_count
        }
        
        return metrics
    
    def analyze_predictions(self, dataloader, num_samples: int = 5) -> List[Dict]:
        """
        Analyze individual predictions for debugging.
        
        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to analyze
        
        Returns:
            List of analysis results
        """
        analyses = []
        sample_count = 0
        
        print(f"🔍 Analyzing {num_samples} individual predictions...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                
                try:
                    if self.model_type == "grid":
                        state, action_target, value_target = batch
                        result = self.model(state)
                        
                        # Analyze first sample in batch
                        if isinstance(result, StandardReturn):
                            logits = result.logits[0]
                            probs = result.probs[0]
                        else:
                            logits = result[0]
                            probs = torch.softmax(logits, dim=-1)
                        
                        predicted_action = torch.argmax(logits, dim=0)
                        confidence = torch.max(probs, dim=0)[0]
                        
                        analysis = {
                            'sample_id': sample_count,
                            'predicted_action': predicted_action.item(),
                            'true_action': action_target[0].item() if action_target is not None else None,
                            'confidence': confidence.item(),
                            'action_probabilities': probs.cpu().numpy().tolist(),
                            'correct': predicted_action.item() == action_target[0].item() if action_target is not None else None
                        }
                        
                        analyses.append(analysis)
                        sample_count += 1
                        
                except Exception as e:
                    print(f"⚠️  Error analyzing sample: {e}")
                    continue
        
        return analyses


def load_test_data(data_path: str, model_type: str = "grid", batch_size: int = 32):
    """Load test dataset for evaluation."""
    
    print(f"📁 Loading test data from {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    try:
        if model_type == "grid":
            # Load grid data
            data_module = GridDataModule(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=0  # Set to 0 for debugging
            )
            data_module.setup("test")
            return data_module.test_dataloader()
            
        elif model_type == "graph":
            # Load graph data
            data_module = GraphDataModule(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=0
            )
            data_module.setup("test")
            return data_module.test_dataloader()
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def create_evaluation_report(model_path: str, metrics: Dict, analyses: List[Dict]) -> str:
    """Create a comprehensive evaluation report."""
    
    model_name = os.path.basename(model_path)
    
    report = f"""
# Model Evaluation Report

## Model Information
- **Model**: {model_name}
- **Path**: {model_path}
- **Evaluation Date**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

## Performance Metrics
- **Total Samples**: {metrics['total_samples']:,}
- **Correct Predictions**: {metrics['correct_predictions']:,}
- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- **Average Loss**: {metrics['average_loss']:.6f}
- **Batches Processed**: {metrics['batches_processed']}

## Sample Analysis
"""
    
    for analysis in analyses[:3]:  # Show first 3 samples
        report += f"""
### Sample {analysis['sample_id']}
- **Predicted Action**: {analysis['predicted_action']}
- **True Action**: {analysis['true_action']}
- **Confidence**: {analysis['confidence']:.4f}
- **Correct**: {'✅' if analysis['correct'] else '❌'}
- **Action Probabilities**: {[f'{p:.3f}' for p in analysis['action_probabilities']]}
"""
    
    return report


def visualize_results(metrics: Dict, analyses: List[Dict], save_path: str = "evaluation_results.png"):
    """Create visualizations of evaluation results."""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy visualization
        axes[0, 0].bar(['Correct', 'Incorrect'], 
                      [metrics['correct_predictions'], 
                       metrics['total_samples'] - metrics['correct_predictions']],
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title(f"Accuracy: {metrics['accuracy']:.2%}")
        axes[0, 0].set_ylabel('Number of Samples')
        
        # Confidence distribution
        if analyses:
            confidences = [a['confidence'] for a in analyses]
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='blue')
            axes[0, 1].set_title('Prediction Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
        
        # Action distribution (if available)
        if analyses and analyses[0]['action_probabilities']:
            num_actions = len(analyses[0]['action_probabilities'])
            avg_probs = np.mean([a['action_probabilities'] for a in analyses], axis=0)
            axes[1, 0].bar(range(num_actions), avg_probs, alpha=0.7)
            axes[1, 0].set_title('Average Action Probabilities')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Probability')
        
        # Correct vs Incorrect predictions confidence
        if analyses:
            correct_conf = [a['confidence'] for a in analyses if a['correct'] is True]
            incorrect_conf = [a['confidence'] for a in analyses if a['correct'] is False]
            
            if correct_conf and incorrect_conf:
                axes[1, 1].hist([correct_conf, incorrect_conf], 
                               label=['Correct', 'Incorrect'], 
                               alpha=0.7, bins=15)
                axes[1, 1].set_title('Confidence by Correctness')
                axes[1, 1].set_xlabel('Confidence')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained DiffPlan models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model checkpoint (.ckpt file)")
    parser.add_argument("--data_path", type=str, 
                       help="Path to test dataset (if not provided, will try to infer)")
    parser.add_argument("--model_type", type=str, choices=["grid", "graph"], default="grid",
                       help="Type of model (grid or graph)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches to evaluate")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    print("🎯 DiffPlan Model Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"📦 Loading model: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    try:
        if args.model_type == "grid":
            model = LitGridPlanner.load_from_checkpoint(args.model_path)
        else:
            model = LitGraphPlanner.load_from_checkpoint(args.model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Infer data path if not provided
    if args.data_path is None:
        if args.model_type == "grid":
            args.data_path = "data/m15_4abs-cc_10k.npz"
        else:
            args.data_path = "data/m15_graph-cc_10k.pth"
        print(f"🔍 Using inferred data path: {args.data_path}")
    
    # Load test data
    test_dataloader = load_test_data(args.data_path, args.model_type, args.batch_size)
    if test_dataloader is None:
        print("❌ Could not load test data")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, args.model_type)
    
    # Run evaluation
    print("\n🧪 Running evaluation...")
    metrics = evaluator.evaluate_success_rate(test_dataloader, args.max_batches)
    
    print("\n📊 Evaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Total Samples: {metrics['total_samples']:,}")
    print(f"  Correct Predictions: {metrics['correct_predictions']:,}")
    print(f"  Average Loss: {metrics['average_loss']:.6f}")
    
    # Analyze individual predictions
    analyses = evaluator.analyze_predictions(test_dataloader, num_samples=10)
    
    # Create report
    report = create_evaluation_report(args.model_path, metrics, analyses)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")
    
    # Save report
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to {report_path}")
    
    # Save visualizations
    viz_path = os.path.join(args.output_dir, "evaluation_results.png")
    visualize_results(metrics, analyses, viz_path)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()