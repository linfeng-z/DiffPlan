"""
DiffPlan Modules

This package contains the core modules for the DiffPlan library:

- Base: Lightning modules for training (LitBase, LitGraphPlanner, LitGridPlanner)
- Dataset: Data loading and preprocessing (GridDataModule, GraphDataModule, HabitatDataModule)
- helpers: Common utilities (StandardReturn, NormalDebugReturn, Timer)
- DEQ: Deep Equilibrium Network implementations
- MPNN: Message Passing Neural Network components
- pooling: Group-equivariant pooling operations
- mapper: Navigation mapping utilities
- ConvRNN_modules: Convolutional RNN components
- SPT_modules: Shortest Path Transformer components
"""

# Core training modules
from .Base import LitBase, LitGraphPlanner, LitGridPlanner

# Data modules
from .Dataset import GridDataModule, GraphDataModule, HabitatDataModule

# Common utilities and return types
from .helpers import (
    StandardReturn,
    StandardReturnWithAuxInfo, 
    NormalDebugReturn,
    EquivariantDebugReturn,
    TransformedOutput,
    Timer,
    get_solver,
    get_deq_layer
)

# Most commonly used components
__all__ = [
    # Training modules
    "LitBase",
    "LitGraphPlanner", 
    "LitGridPlanner",
    
    # Data modules
    "GridDataModule",
    "GraphDataModule", 
    "HabitatDataModule",
    
    # Return types and utilities
    "StandardReturn",
    "StandardReturnWithAuxInfo",
    "NormalDebugReturn", 
    "EquivariantDebugReturn",
    "TransformedOutput",
    "Timer",
    "get_solver",
    "get_deq_layer",
]