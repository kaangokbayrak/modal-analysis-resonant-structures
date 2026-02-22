"""
Parametric Study and Optimization Module

This module performs parametric sweeps and optimization studies to understand
how design parameters affect natural frequencies and to find optimal designs
that avoid resonance conditions.

Author: Kaan Gokbayrak, Purdue University
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Dict, Tuple
import copy
import csv
from .beam import Beam
from .analytical import AnalyticalSolver


class ParametricStudy:
    """
    Parametric study and optimization for beam design.
    
    This class enables:
    - Single parameter sweeps (thickness, length, width)
    - 2D design space exploration
    - Optimization to avoid resonance conditions
    
    Parameters
    ----------
    base_beam : Beam
        Reference beam configuration
    bc : str, optional
        Boundary condition (default: 'cantilever')
    """
    
    def __init__(self, base_beam: Beam, bc: str = 'cantilever'):
        """
        Initialize parametric study.
        
        Parameters
        ----------
        base_beam : Beam
            Base beam configuration
        bc : str
            Boundary condition
        """
        self.base_beam = base_beam
        self.bc = bc
    
    def sweep_parameter(self, param: str, values: np.ndarray,
                        n_modes: int = 3) -> Dict:
        """
        Sweep a single parameter and compute natural frequencies.
        
        This function varies one geometric parameter (thickness, length, or width)
        while keeping all others constant, then computes how natural frequencies
        change. This helps understand sensitivity to design parameters.
        
        Parameters
        ----------
        param : str
            Parameter name: 'thickness', 'length', or 'width'
        values : np.ndarray
            Parameter values to test [m]
        n_modes : int, optional
            Number of modes to analyze (default: 3)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'parameter': Parameter name
            - 'values': Parameter values tested
            - 'frequencies': Natural frequencies [Hz], shape (n_values, n_modes)
            - 'masses': Total beam mass [kg], shape (n_values,)
        """
        if param not in ['thickness', 'length', 'width']:
            raise ValueError(f"Invalid parameter: {param}. "
                           f"Must be 'thickness', 'length', or 'width'")
        
        frequencies = []
        masses = []
        
        for value in values:
            # Create modified beam
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, value)
            
            # Compute natural frequencies
            solver = AnalyticalSolver(beam_copy)
            freqs = solver.natural_frequencies(n_modes, self.bc)
            
            frequencies.append(freqs)
            masses.append(beam_copy.total_mass)
        
        return {
            'parameter': param,
            'values': values,
            'frequencies': np.array(frequencies),
            'masses': np.array(masses)
        }
    
    def design_space_map(self, param1: str, values1: np.ndarray,
                          param2: str, values2: np.ndarray,
                          mode: int = 1) -> Dict:
        """
        Create 2D design space map showing frequency variation.
        
        This explores the design space by varying two parameters simultaneously
        and computing a specific mode's frequency. Results can be visualized as
        a contour plot to identify safe/unsafe design regions.
        
        Parameters
        ----------
        param1 : str
            First parameter name
        values1 : np.ndarray
            First parameter values [m]
        param2 : str
            Second parameter name
        values2 : np.ndarray
            Second parameter values [m]
        mode : int, optional
            Mode number to map (1-indexed, default: 1 for first mode)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'param1': First parameter name
            - 'param2': Second parameter name
            - 'values1': First parameter values
            - 'values2': Second parameter values
            - 'frequency_map': Frequency grid [Hz], shape (n_values2, n_values1)
            - 'mass_map': Mass grid [kg], shape (n_values2, n_values1)
            - 'mode': Mode number
        """
        if param1 not in ['thickness', 'length', 'width']:
            raise ValueError(f"Invalid param1: {param1}")
        if param2 not in ['thickness', 'length', 'width']:
            raise ValueError(f"Invalid param2: {param2}")
        if param1 == param2:
            raise ValueError("Parameters must be different")
        
        # Initialize grids
        freq_map = np.zeros((len(values2), len(values1)))
        mass_map = np.zeros((len(values2), len(values1)))
        
        # Sweep parameter space
        for i, val2 in enumerate(values2):
            for j, val1 in enumerate(values1):
                # Create modified beam
                beam_copy = copy.deepcopy(self.base_beam)
                setattr(beam_copy, param1, val1)
                setattr(beam_copy, param2, val2)
                
                # Compute natural frequencies
                solver = AnalyticalSolver(beam_copy)
                freqs = solver.natural_frequencies(mode, self.bc)
                
                freq_map[i, j] = freqs[mode - 1]  # mode is 1-indexed
                mass_map[i, j] = beam_copy.total_mass
        
        return {
            'param1': param1,
            'param2': param2,
            'values1': values1,
            'values2': values2,
            'frequency_map': freq_map,
            'mass_map': mass_map,
            'mode': mode
        }
    
    def optimize_for_frequency(self, target_freq: float,
                                operational_range: Tuple[float, float] = (50.0, 70.0),
                                param: str = 'thickness',
                                bounds: Tuple[float, float] = (0.001, 0.020)) -> Dict:
        """
        Optimize beam geometry to avoid resonance while minimizing mass.
        
        Engineering Scenario:
        A beam's first natural frequency currently falls within the operating range
        of a pump/motor (e.g., 50 Hz). This creates a resonance risk that can lead
        to excessive vibration and potential failure. The goal is to redesign the
        beam by adjusting a parameter (typically thickness) to:
        
        1. Shift the first natural frequency outside the operational range (> 65 Hz)
        2. Minimize total mass (material cost, inertia)
        
        This is formulated as a constrained optimization problem:
        
        Objective: minimize mass
        Constraint: f1 >= target_freq
        
        Parameters
        ----------
        target_freq : float
            Target minimum frequency [Hz] to avoid resonance
        operational_range : tuple of float, optional
            Operating frequency range [Hz] to avoid (default: (50.0, 70.0))
        param : str, optional
            Parameter to optimize: 'thickness', 'length', or 'width' (default: 'thickness')
        bounds : tuple of float, optional
            Bounds on parameter [m] (default: (0.001, 0.020))
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'optimal_value': Optimal parameter value [m]
            - 'optimal_frequency': Achieved first natural frequency [Hz]
            - 'optimal_mass': Achieved total mass [kg]
            - 'initial_value': Initial parameter value [m]
            - 'initial_frequency': Initial first natural frequency [Hz]
            - 'initial_mass': Initial total mass [kg]
            - 'mass_reduction_percent': Mass change [%] (negative = reduction)
            - 'optimization_result': scipy OptimizeResult object
            - 'history': Optimization history (parameter values and frequencies)
        """
        if param not in ['thickness', 'length', 'width']:
            raise ValueError(f"Invalid parameter: {param}")
        
        # Store initial state
        initial_value = getattr(self.base_beam, param)
        solver_initial = AnalyticalSolver(self.base_beam)
        initial_freq = solver_initial.natural_frequencies(1, self.bc)[0]
        initial_mass = self.base_beam.total_mass
        
        # Optimization history
        history = {'values': [], 'frequencies': [], 'masses': []}
        
        def objective(x):
            """Objective function: minimize mass, and record full history."""
            # Create modified beam
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, x[0])
            
            mass = beam_copy.total_mass
            
            # Compute frequency here to keep history arrays aligned
            solver = AnalyticalSolver(beam_copy)
            freq = solver.natural_frequencies(1, self.bc)[0]
            
            # Store history (all three arrays updated together)
            history['values'].append(x[0])
            history['masses'].append(mass)
            history['frequencies'].append(freq)
            
            return mass
        
        def constraint_frequency(x):
            """Constraint: first natural frequency >= target_freq."""
            # Create modified beam
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, x[0])
            
            # Compute first natural frequency
            solver = AnalyticalSolver(beam_copy)
            freq = solver.natural_frequencies(1, self.bc)[0]
            
            # Return constraint value (>= 0 is satisfied)
            return freq - target_freq
        
        # Define optimization problem
        x0 = np.array([initial_value])
        
        # Constraints: frequency constraint
        constraints = {'type': 'ineq', 'fun': constraint_frequency}
        
        # Bounds
        param_bounds = [(bounds[0], bounds[1])]
        
        # Solve optimization problem
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=param_bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        # Extract optimal solution
        optimal_value = result.x[0]
        
        # Compute optimal beam properties
        beam_optimal = copy.deepcopy(self.base_beam)
        setattr(beam_optimal, param, optimal_value)
        solver_optimal = AnalyticalSolver(beam_optimal)
        optimal_freq = solver_optimal.natural_frequencies(1, self.bc)[0]
        optimal_mass = beam_optimal.total_mass
        
        # Compute mass reduction
        mass_change_percent = 100 * (optimal_mass - initial_mass) / initial_mass
        
        return {
            'optimal_value': optimal_value,
            'optimal_frequency': optimal_freq,
            'optimal_mass': optimal_mass,
            'initial_value': initial_value,
            'initial_frequency': initial_freq,
            'initial_mass': initial_mass,
            'mass_change_percent': mass_change_percent,
            'optimization_result': result,
            'history': history,
            'parameter': param,
            'target_frequency': target_freq,
            'operational_range': operational_range
        }
    
    def export_results(self, results: Dict, filepath: str) -> None:
        """
        Export parametric study results to CSV.
        
        Parameters
        ----------
        results : dict
            Results dictionary from sweep_parameter or design_space_map
        filepath : str
            Output CSV file path
        """
        if 'parameter' in results:
            # Single parameter sweep
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                n_modes = results['frequencies'].shape[1]
                header = [results['parameter'], 'mass_kg'] + [f'f{i+1}_Hz' for i in range(n_modes)]
                writer.writerow(header)
                
                # Data rows
                for i, val in enumerate(results['values']):
                    row = [val, results['masses'][i]] + list(results['frequencies'][i, :])
                    writer.writerow(row)
        
        elif 'param1' in results:
            # 2D parameter sweep
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = [results['param1'], results['param2'], f'f{results["mode"]}_Hz', 'mass_kg']
                writer.writerow(header)
                
                # Data rows
                for i, val2 in enumerate(results['values2']):
                    for j, val1 in enumerate(results['values1']):
                        row = [val1, val2, results['frequency_map'][i, j], results['mass_map'][i, j]]
                        writer.writerow(row)
        
        print(f"Results exported to {filepath}")
