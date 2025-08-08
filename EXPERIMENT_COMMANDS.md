# DiffPlan Experiment Commands Reference

This document contains all the training commands for different models and configurations in DiffPlan.

## Current Supported Models

These are the actively maintained models that can be run with the current codebase:

### Grid-Based Models

#### VIN (Value Iteration Network)
```bash
python3 -m diffplan.main \
    --algorithm VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42
```

#### SymVIN (Symmetric Value Iteration Network)
```bash
python3 -m diffplan.main \
    --algorithm SymVIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42 \
    --group d8
```

#### DE-VIN (Deep Equilibrium VIN)
```bash
python3 -m diffplan.main \
    --algorithm DE-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42
```

#### DE-SymVIN (Deep Equilibrium Symmetric VIN)
```bash
python3 -m diffplan.main \
    --algorithm DE-SymVIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42 \
    --group d8
```

#### GPPN Variants
```bash
# Basic GPPN
python3 -m diffplan.main \
    --algorithm GPPN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42

# Conv-GPPN
python3 -m diffplan.main \
    --algorithm Conv-GPPN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42

# DE-GPPN
python3 -m diffplan.main \
    --algorithm DE-GPPN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42

# SymGPPN
python3 -m diffplan.main \
    --algorithm SymGPPN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42 \
    --group d8
```

### Graph-Based Models

#### MP-VIN (Message Passing VIN) - Discrete Actions
```bash
python3 -m diffplan.main \
    --algorithm MP-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42
```

#### MP-VIN - Continuous Actions
```bash
python3 -m diffplan.main \
    --algorithm MP-VIN \
    --cont_action \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k_cont_act.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42
```

#### Sym-MP-VIN (Symmetric Message Passing VIN) - Discrete Actions
```bash
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42 \
    --group d8
```

#### Sym-MP-VIN - Continuous Actions
```bash
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --no_equiv_policy \
    --cont_action \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k_cont_act.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42 \
    --group d8
```

#### Sym-MP-VIN - Partial Observability
```bash
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --no_equiv_policy \
    --cont_action \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k_cont_act_partial.pth \
    --task GraphWorld \
    --has_obstacle \
    --obsv_mode partial \
    --seed 42 \
    --group d8
```

#### GCN-VIN (Graph Convolutional Network VIN)
```bash
python3 -m diffplan.main \
    --algorithm GCN-VIN-v2 \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42
```

#### GAT-VIN (Graph Attention Network VIN)
```bash
python3 -m diffplan.main \
    --algorithm GAT-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42
```

## Advanced Training Options

### With WandB Logging
Remove `--disable_wandb` from any command above:
```bash
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --no_equiv_policy \
    --cont_action \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k_cont_act.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42 \
    --group d8
```

### Hyperparameter Sweep
```bash
python3 -m diffplan.main --enable_sweep --sweep_id "your_sweep_id"
```

### Testing Mode
Add `--test` to any training command:
```bash
python3 -m diffplan.main \
    --algorithm MP-VIN \
    --test \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle
```

### Equivariance Error Testing
```bash
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --test_equiv_err \
    --data_path data/m15_graph-cc_10k_cont_act.pth \
    --task GraphWorld \
    --has_obstacle \
    --group d8
```

## Common Parameters

### Data Configuration
- `--data_path`: Path to training data
- `--task`: Task type (GridWorld, GraphWorld, VisNav, Habitat)
- `--has_obstacle`: Include obstacle nodes in graph
- `--obsv_mode`: Observation mode (full, partial)
- `--cont_action`: Enable continuous actions
- `--maze_size`: Size of maze environment

### Training Parameters
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--optimizer`: Optimizer choice (RMSprop, Adam, SGD)
- `--seed`: Random seed for reproducibility
- `--accumulate_grad_batches`: Gradient accumulation steps

### Model Parameters
- `--k`: Number of value iteration steps (default: 20)
- `--l_h`: Hidden layer size (default: 150)
- `--l_q`: Q-function layer size (default: 40)
- `--mp_h`: Message passing hidden size (default: 256)
- `--mp_q`: Message passing output size (default: 256)
- `--mp_agg`: Message passing aggregation (add, mean, max)

### Symmetry Parameters
- `--group`: Symmetry group (d4, d8, etc.)
- `--no_equiv_policy`: Disable equivariant policy layer
- `--num_orient`: Number of orientations
- `--num_actions`: Number of actions

### Logging and Debug
- `--disable_wandb`: Disable Weights & Biases logging
- `--notes`: Notes for experiment tracking
- `--test`: Testing mode
- `--test_equiv_err`: Test equivariance error
- `--no_weight_init`: Disable model weight initialization

## Experimental/Archive Models

The following models are archived in `archive/experimental_models/` and are not actively maintained but preserved for reference:

### Archived Experimental Models
- **BoundedGPPN**: Bounded version of GPPN
- **DE-SPT**: Deep Equilibrium Shortest Path Transformer
- **DE-VI-Recurrent**: Recurrent Deep Equilibrium VI
- **DE-VI-Transformer**: Transformer-based Deep Equilibrium VI
- **DE-VIN-Discount**: Discounted version of DE-VIN
- **DE-VIN-v2**, **DE-VIN-v3**: Older versions of DE-VIN
- **DecoupledSPT**: Decoupled Shortest Path Transformer
- **DecoupledVIN-v2**: Older version of DecoupledVIN
- **SPT**, **SPT-v1**: Shortest Path Transformer variants
- **VIN-Discount**: Discounted version of VIN

These archived models may require modification to work with the current codebase structure.

## Dataset Generation Commands

See the main README.md for comprehensive dataset generation commands for different environments and configurations.