# Differentiable Planning Library


**This is the official codebase for the following three papers:**


### **E(2)-Equivariant Graph Planning for Navigation**
- **RA-L, IROS 2024 (oral)**
- Linfeng Zhao*, Hongyu Li*, Taskin Padir, Huaizu Jiang, Lawson L.S. Wong
- [[arXiv]](https://arxiv.org/abs/2309.13043) [[Slides]](https://lfzhao.com/slides/slides-equiv-nav-iros2024-oral-ral.pdf) [[Program (IROS 2024 oral)]](https://ras.papercept.net/conferences/conferences/IROS24/program/IROS24_ContentListWeb_3.html#weat9_01)


### **Integrating Symmetry into Differentiable Planning with Steerable Convolutions**
- **ICLR 2023**
- Linfeng Zhao, Xupeng Zhu, Lingzhi Kong, Robin Walters, Lawson L.S. Wong
- [[Paper]](https://lfzhao.com/paper/paper-symplan-iclr2023.pdf) [[Poster]](https://lfzhao.com/poster/poster-symplan-iclr2023.pdf) [[Slides]](https://lfzhao.com/slides/slides-symplan-iclr2023.pdf) [[ICLR page]](https://iclr.cc/virtual/2023/poster/10993) [[OpenReview]](https://openreview.net/forum?id=n7CPzMPKQl) [[arXiv]](https://arxiv.org/abs/2206.03674)


### **Scaling up and Stabilizing Differentiable Planning with Implicit Differentiation**
- **ICLR 2023**
- Linfeng Zhao, Huazhe Xu, Lawson L.S. Wong
- [[Paper]](https://lfzhao.com/paper/paper-idplan-iclr2023.pdf) [[Poster]](https://lfzhao.com/poster/poster-idplan-iclr2023.pdf) [[Slides]](https://lfzhao.com/slides/slides-idplan-iclr2023.pdf) [[ICLR page]](https://iclr.cc/virtual/2023/poster/10976) [[OpenReview]](https://openreview.net/forum?id=PYbe4MoHf32) [[arXiv]](https://arxiv.org/abs/2210.13542)




## Installation

This project uses conda/mamba for environment management. We recommend using mamba for faster installation.

### Using the Library

1. First, install mamba if you haven't already:
```bash
conda install mamba -n base -c conda-forge
```

2. Create and activate the environment:
```bash
# Create environment
mamba env create -n release_diffplan -f environment.yml

# Activate environment
conda activate release_diffplan

# Install the package in development mode (basic installation)
uv pip install -e .

# Or, install with optional dependencies:
uv pip install -e ".[habitat]"  # For 3D visual navigation support
uv pip install -e ".[all]"      # Install all optional dependencies
```

3. Verify the installation:
```bash
# Check if PyTorch and PyTorch Geometric are properly installed
python -c "import torch; import torch_geometric; print(f'PyTorch version: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nPyG version: {torch_geometric.__version__}')"
```

### Development Setup

1. Create a development environment:
```bash
# Create environment
mamba env create -n dev_diffplan -f environment.yml

# Activate environment
conda activate dev_diffplan

# Install the package with development dependencies
uv pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Optional Dependencies

The package provides several optional dependency groups:

- `habitat`: For 3D visual navigation environments (includes `habitat-sim`, `habitat-lab`, and `magnum`)
- `dev`: Development tools (`pre-commit`, `black`, `isort`, `flake8`, `pytest`)
- `all`: Installs all optional dependencies

### Habitat Setup (Optional)

If you plan to use the 3D visual navigation features, you'll need to install some system dependencies first:

On Ubuntu:
```bash
sudo apt-get install -y \
    git \
    cmake \
    libglfw3-dev \
    libglm-dev \
    libegl1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libopenexr-dev
```

Then install the Habitat dependencies:
```bash
uv pip install -e ".[habitat]"
```

For other operating systems or if you encounter issues, please refer to the [Habitat-sim installation guide](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md).

### Troubleshooting

If you encounter any issues during installation:
   ```

1. If you don't have mamba installed and prefer using conda, you can replace `mamba env create` with `conda env create` in the commands above, but the installation might be slower.

2. For Habitat-related issues, make sure all system dependencies are installed and try reinstalling with:
   ```bash
   uv pip uninstall habitat-sim habitat-lab
   uv pip install -e ".[habitat]"
   ```

## Dataset Generation

Generate a dataset for training:

```bash
# Basic Grid-world dataset
python -m diffplan.envs.generate_dataset \
    --output-path data/m15_4abs-cc_10k.npz \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 10000 \
    --valid-size 2000 \
    --test-size 2000 \
    --env RandomMaze

# Graph version of Grid-world (Discrete actions)
python -m diffplan.envs.generate_dataset \
    --output-path data/m15_graph-cc_10k.pth \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 10000 \
    --valid-size 2000 \
    --test-size 2000 \
    --env RandomMazeGraph \
    --has_obstacle

# Graph version of Grid-world (Continuous actions)
python -m diffplan.envs.generate_dataset \
    --output-path data/m15_graph-cc_10k_cont_act.pth \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 10000 \
    --valid-size 2000 \
    --test-size 2000 \
    --env RandomMazeGraph \
    --cont_action \
    --has_obstacle

# Graph version of Grid-world (Continuous actions & Partial observable)
python -m diffplan.envs.generate_dataset \
    --output-path data/m15_graph-cc_10k_cont_act_partial.pth \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 10000 \
    --valid-size 2000 \
    --test-size 2000 \
    --env RandomMazeGraph \
    --cont_action \
    --obsv_mode partial \
    --has_obstacle

# Graph-world (Continuous actions)
python -m diffplan.envs.generate_dataset \
    --output-path data/graph_128_10k_cont_act.pth \
    --mechanism 4abs-cc \
    --maze-size 128 \
    --train-size 10000 \
    --valid-size 2000 \
    --test-size 2000 \
    --env RandomGraph \
    --cont_action \
    --has_obstacle

# 3D Visual Navigation
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    python -m diffplan.envs.generate_dataset \
    --env Visual3DNav \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 1000 \
    --valid-size 200 \
    --test-size 200 \
    --output-path 'data/Visual3DNav_1k_15_4abs-cc.npz'

# Graph version of 3D Visual Navigation
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    python -m diffplan.envs.generate_dataset \
    --env Visual3DNavGraph \
    --mechanism 4abs-cc \
    --maze-size 15 \
    --train-size 1000 \
    --valid-size 200 \
    --test-size 200 \
    --output-path 'data/graph-Visual3DNav_1k_15_4abs-cc_cont_act.pth' \
    --cont_action \
    --has_obstacle
```

**Note for 3D Visual Navigation**: Dataset generation takes several hours and requires:
- ~10GB storage space
- ~20GB memory during generation
- For HPC servers without xvfb, use Singularity:
  ```bash
  # Pull the container
  singularity pull corl.sif library://lhy0807/lhy0807/corl:latest
  
  # Run inside container
  singularity exec --nv --network=host --bind /work/riverlab/hongyu/corl/:/work/riverlab/hongyu/corl/ corl.sif /bin/bash
  ```

## Training

Run training with different configurations:

```bash
# Grid-based Models

## VIN (Value Iteration Network)
python3 -m diffplan.main \
    --algorithm VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42

## SymVIN (Symmetric Value Iteration Network)
python3 -m diffplan.main \
    --algorithm SymVIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_4abs-cc_10k.npz \
    --task GridWorld \
    --seed 42 \
    --group d8

# Graph-based Models

## MP-VIN (Message Passing VIN) with discrete actions
python3 -m diffplan.main \
    --algorithm MP-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42

## MP-VIN with continuous actions
python3 -m diffplan.main \
    --algorithm MP-VIN \
    --cont_action \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k_cont_act.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42

## Sym-MP-VIN (Symmetric Message Passing VIN) with discrete actions
python3 -m diffplan.main \
    --algorithm Sym-MP-VIN \
    --disable_wandb \
    --epochs 100 \
    --data_path data/m15_graph-cc_10k.pth \
    --task GraphWorld \
    --has_obstacle \
    --seed 42 \
    --group d8

## Sym-MP-VIN with continuous actions
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

## Sym-MP-VIN with partial observability
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

# Advanced Training Options

## Training with WandB logging enabled (remove --disable_wandb)
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

## Training with hyperparameter sweep
python3 -m diffplan.main --enable_sweep --sweep_id "your_sweep_id"
```

### Training Options

You can customize the training with various options:

- **Algorithm Selection**:
  - `--algorithm`: Choose between models (VIN, SymVIN, MP-VIN, Sym-MP-VIN)
  - `--group`: Symmetry group for equivariant models (e.g., d8 for dihedral group)
  - `--no_equiv_policy`: Disable equivariant layer in policy network

- **Task Configuration**:
  - `--task`: Choose task type (GridWorld, GraphWorld)
  - `--has_obstacle`: Enable obstacle handling
  - `--obsv_mode`: Observation mode (full, partial)
  - `--cont_action`: Enable continuous actions

- **Training Parameters**:
  - `--epochs`: Number of training epochs
  - `--batch_size`: Batch size for training (default: 32)
  - `--lr`: Learning rate (default: 1e-3)
  - `--seed`: Random seed for reproducibility
  - `--optimizer`: Choose optimizer (RMSprop, Adam, SGD)

- **Logging and Monitoring**:
  - `--disable_wandb`: Disable Weights & Biases logging
  - `--enable_sweep`: Enable hyperparameter sweep
  - `--sweep_id`: WandB sweep ID for hyperparameter tuning

For more examples and detailed configurations, please refer to the papers linked above.

## Testing

The project includes tests to verify the functionality of different training configurations. To run the tests:

1. Install test dependencies:
```bash
uv pip install -e ".[dev]"  # Installs pytest and other development tools
```

2. Run tests:
```bash
# Run all tests
pytest tests/test_training.py

# Run specific test
pytest tests/test_training.py::test_mp_vin      # Test MP-VIN training
pytest tests/test_training.py::test_sym_mp_vin  # Test Sym-MP-VIN training
```

The tests verify that:
- Basic MP-VIN training works with continuous actions
- Symmetric MP-VIN training works with D8 symmetry group
- Both configurations can handle obstacle environments

Each test runs a quick training (1 epoch) to ensure the setup is working correctly.


## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{zhao2024e2,
  title={E(2)-Equivariant Graph Planning for Navigation},
  author={Zhao, Linfeng and Li, Hongyu and Padir, Taskin and Jiang, Huaizu and Wong, Lawson LS},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

@inproceedings{zhao2023integrating,
  title={Integrating Symmetry into Differentiable Planning with Steerable Convolutions},
  author={Zhao, Linfeng and Zhu, Xupeng and Kong, Lingzhi and Walters, Robin and Wong, Lawson LS},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}

@inproceedings{zhao2023scaling,
  title={Scaling up and Stabilizing Differentiable Planning with Implicit Differentiation},
  author={Zhao, Linfeng and Xu, Huazhe and Wong, Lawson LS},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

## Contributing

We welcome contributions! Please see the development setup instructions above for setting up a development environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
