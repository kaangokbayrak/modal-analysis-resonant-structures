"""
PyVib: Computational Modal Analysis Workbench

A custom-built FEM solver for modal analysis of resonant structures.
Built from the ground up to demonstrate understanding of structural dynamics.

Author: Kaan Gokbayrak, Purdue University
Date: December 2025 - January 2026
"""

__version__ = "1.0.0"
__author__ = "Kaan Gokbayrak"

from .beam import Material, Beam
from .analytical import AnalyticalSolver
from .fem_solver import FEMSolver
from .signal_processing import SignalProcessor
from .parametric_study import ParametricStudy
from .visualization import Visualizer

__all__ = [
    'Material',
    'Beam',
    'AnalyticalSolver',
    'FEMSolver',
    'SignalProcessor',
    'ParametricStudy',
    'Visualizer',
]
