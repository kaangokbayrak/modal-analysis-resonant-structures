"""
PyVib: Computational Modal Analysis Workbench

A custom-built FEM solver for modal analysis of resonant structures.
Built from the ground up to demonstrate understanding of structural dynamics.

Author: Kaan Gokbayrak, Purdue University
Date: December 2025 - January 2026
"""

__version__ = "1.0.0"
__author__ = "Kaan Gokbayrak"

from .beam import (
    Material,
    Beam,
    RectangularSection,
    IBeamSection,
    HollowRectSection,
    CircularSection,
    HollowCircularSection,
    TSection,
    SectionType,
)
from .analytical import AnalyticalSolver
from .fem_solver import FEMSolver
from .signal_processing import SignalProcessor
from .parametric_study import ParametricStudy
from .visualization import Visualizer
from .modal_analysis import ModalAnalyzer, from_fem_results, from_analytical_results

__all__ = [
    'Material',
    'Beam',
    'RectangularSection',
    'IBeamSection',
    'HollowRectSection',
    'CircularSection',
    'HollowCircularSection',
    'TSection',
    'SectionType',
    'AnalyticalSolver',
    'FEMSolver',
    'SignalProcessor',
    'ParametricStudy',
    'Visualizer',
    'ModalAnalyzer',
    'from_fem_results',
    'from_analytical_results',
]
