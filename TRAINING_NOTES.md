# Training Notes & Model Release Checklist

## 🎯 Pre-trained Models to Create

### Grid-Based Models (Priority 1)

1. **VIN_maze15.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm VIN \
     --data_path data/m15_4abs-cc_10k.npz \
     --task GridWorld \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Standard VIN for 15x15 maze, release model"
   ```

2. **SymVIN_maze15.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm SymVIN \
     --data_path data/m15_4abs-cc_10k.npz \
     --task GridWorld \
     --group d4 \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Symmetric VIN with D4 group for 15x15 maze"
   ```

3. **DE-VIN_maze15.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm DE-VIN \
     --data_path data/m15_4abs-cc_10k.npz \
     --task GridWorld \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Deep Equilibrium VIN for 15x15 maze"
   ```

4. **DE-SymVIN_maze15.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm DE-SymVIN \
     --data_path data/m15_4abs-cc_10k.npz \
     --task GridWorld \
     --group d4 \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Deep Equilibrium Symmetric VIN"
   ```

### Graph-Based Models (Priority 2)

5. **MP-VIN_graph.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm MP-VIN \
     --data_path data/m15_graph-cc_10k.pth \
     --task GraphWorld \
     --has_obstacle \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Message Passing VIN for graph navigation"
   ```

6. **Sym-MP-VIN_graph.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm Sym-MP-VIN \
     --data_path data/m15_graph-cc_10k.pth \
     --task GraphWorld \
     --has_obstacle \
     --group d4 \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Symmetric Message Passing VIN"
   ```

7. **GAT-VIN_graph.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm GAT-VIN \
     --data_path data/m15_graph-cc_10k.pth \
     --task GraphWorld \
     --has_obstacle \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Graph Attention VIN for graph navigation"
   ```

8. **GCN-VIN-v2_graph.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm GCN-VIN-v2 \
     --data_path data/m15_graph-cc_10k.pth \
     --task GraphWorld \
     --has_obstacle \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "Graph Convolutional VIN v2"
   ```

### Continuous Action Models (Priority 3)

9. **MP-VIN_continuous.ckpt**
   ```bash
   python -m diffplan.main \
     --algorithm MP-VIN \
     --cont_action \
     --data_path data/m15_graph-cc_10k_cont_act.pth \
     --task GraphWorld \
     --has_obstacle \
     --epochs 100 \
     --batch_size 32 \
     --lr 1e-3 \
     --seed 42 \
     --notes "MP-VIN with continuous actions"
   ```

10. **Sym-MP-VIN_continuous.ckpt**
    ```bash
    python -m diffplan.main \
      --algorithm Sym-MP-VIN \
      --no_equiv_policy \
      --cont_action \
      --data_path data/m15_graph-cc_10k_cont_act.pth \
      --task GraphWorld \
      --has_obstacle \
      --group d4 \
      --epochs 100 \
      --batch_size 32 \
      --lr 1e-3 \
      --seed 42 \
      --notes "Symmetric MP-VIN with continuous actions"
    ```

## 📊 Expected Performance Targets

| Model | Environment | Target Accuracy | Expected Training Time |
|-------|-------------|-----------------|----------------------|
| VIN | 15x15 maze | >94% | 2-3 hours |
| SymVIN | 15x15 maze | >96% | 2-3 hours |
| DE-VIN | 15x15 maze | >93% | 3-4 hours |
| DE-SymVIN | 15x15 maze | >95% | 3-4 hours |
| MP-VIN | Graph | >91% | 3-4 hours |
| Sym-MP-VIN | Graph | >93% | 3-4 hours |
| GAT-VIN | Graph | >90% | 4-5 hours |
| GCN-VIN-v2 | Graph | >89% | 3-4 hours |

## 🚀 Training Process

### Step 1: Environment Setup
```bash
# Ensure environment is activated
conda activate diffplan
# or
source venv/bin/activate

# Verify all dependencies
pip list | grep -E "(torch|lightning|escnn|wandb)"
```

### Step 2: Data Preparation
```bash
# Generate/verify datasets exist
ls -la data/
# Should see:
# - m15_4abs-cc_10k.npz (for grid models)
# - m15_graph-cc_10k.pth (for graph models)
# - m15_graph-cc_10k_cont_act.pth (for continuous action models)

# If missing, generate:
python -m diffplan.envs.generate_dataset --help
```

### Step 3: Training Execution

**Parallel Training Strategy:**
- Run 2-3 models simultaneously on different GPUs if available
- Monitor GPU memory usage (`nvidia-smi`)
- Use `screen` or `tmux` for long training sessions

```bash
# Example parallel training
screen -S vin_training
python -m diffplan.main --algorithm VIN --data_path data/m15_4abs-cc_10k.npz --epochs 100 --seed 42
# Ctrl+A+D to detach

screen -S symvin_training  
python -m diffplan.main --algorithm SymVIN --data_path data/m15_4abs-cc_10k.npz --group d4 --epochs 100 --seed 42
# Ctrl+A+D to detach
```

### Step 4: Model Validation

For each trained model, run:
```bash
# 1. Basic inference test
python examples/tutorials/model_inference.py

# 2. Comprehensive evaluation
python examples/tutorials/model_evaluation.py \
  --model_path pretrained_models/grid/VIN_maze15.ckpt \
  --data_path data/m15_4abs-cc_10k.npz \
  --model_type grid

# 3. Check model size and parameters
python -c "
import torch
ckpt = torch.load('pretrained_models/grid/VIN_maze15.ckpt')
total_params = sum(p.numel() for p in ckpt['state_dict'].values())
print(f'Total parameters: {total_params:,}')
print(f'File size: {os.path.getsize('pretrained_models/grid/VIN_maze15.ckpt') / 1024 / 1024:.1f} MB')
"
```

## 📁 Model Organization

### Directory Structure
```
pretrained_models/
├── grid/
│   ├── VIN_maze15.ckpt
│   ├── SymVIN_maze15.ckpt
│   ├── DE-VIN_maze15.ckpt
│   ├── DE-SymVIN_maze15.ckpt
│   └── configs/
│       ├── VIN_maze15_config.yaml
│       └── ...
├── graph/
│   ├── MP-VIN_graph.ckpt
│   ├── Sym-MP-VIN_graph.ckpt
│   ├── GAT-VIN_graph.ckpt
│   ├── GCN-VIN-v2_graph.ckpt
│   └── configs/
│       ├── MP-VIN_graph_config.yaml
│       └── ...
└── continuous/
    ├── MP-VIN_continuous.ckpt
    ├── Sym-MP-VIN_continuous.ckpt
    └── configs/
```

### Model Metadata Template

For each model, create a `{model_name}_info.json`:
```json
{
  "model_name": "VIN_maze15",
  "algorithm": "VIN",
  "environment": "15x15 maze with obstacles",
  "training_data": "data/m15_4abs-cc_10k.npz",
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "RMSprop",
    "seed": 42
  },
  "performance": {
    "test_accuracy": 0.952,
    "validation_accuracy": 0.948,
    "final_loss": 0.123
  },
  "model_stats": {
    "total_parameters": 89432,
    "file_size_mb": 2.1,
    "training_time_hours": 2.5
  },
  "created_date": "2024-01-15",
  "notes": "Standard VIN baseline model for release"
}
```

## 🔍 Quality Assurance Checklist

### Before Release:
- [ ] All models train to completion without errors
- [ ] Performance meets or exceeds target accuracy
- [ ] Models can be loaded and run inference successfully
- [ ] File sizes are reasonable (<10MB per model)
- [ ] All model metadata files created
- [ ] Inference examples work with each model
- [ ] Evaluation scripts produce sensible results

### Testing Matrix:
```bash
# Test each model with:
# 1. Basic loading
# 2. Inference on sample data  
# 3. Performance evaluation
# 4. Memory usage check
# 5. Cross-platform compatibility (if needed)

./test_all_models.sh  # Create this script
```

## 📦 Release Package

### Final Structure:
```
pretrained_models/
├── README.md (comprehensive guide)
├── grid/ (4 models + configs + metadata)
├── graph/ (4 models + configs + metadata)  
├── continuous/ (2 models + configs + metadata)
├── evaluation_results/ (performance benchmarks)
└── quick_start_guide.md
```

### Archive for Distribution:
```bash
# Create release archive
tar -czf diffplan_pretrained_models_v1.0.tar.gz pretrained_models/
# or
zip -r diffplan_pretrained_models_v1.0.zip pretrained_models/
```

## 🎯 Success Criteria

- [ ] **10 high-quality pre-trained models** covering main algorithms
- [ ] **Comprehensive documentation** for each model
- [ ] **Working examples** that demonstrate usage
- [ ] **Performance benchmarks** for all models
- [ ] **Easy-to-use interface** for loading and inference
- [ ] **Reproducible results** with provided seeds

## 📝 Internal Notes

- Training priority: Grid models first (faster), then graph models
- Consider training variants with different seeds for robustness
- Monitor WandB logs for training curves and debugging
- Keep checkpoints of best-performing epochs
- Consider creating ensemble models if time permits
- Test memory requirements on different hardware configurations

## 🚨 Troubleshooting

**Common Issues:**
1. **OOM errors**: Reduce batch size, use gradient accumulation
2. **Slow convergence**: Check learning rate, data quality
3. **NaN losses**: Check for numerical instability, reduce LR
4. **Poor performance**: Verify data preprocessing, model configuration

**Emergency Contacts:**
- GPU cluster issues: [contact info]
- Dataset problems: [contact info]  
- Model architecture questions: [contact info]