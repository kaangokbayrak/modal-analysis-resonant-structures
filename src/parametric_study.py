"""
Parametric Study and Optimization Module

This module performs parametric sweeps and optimization studies to understand
how design parameters affect natural frequencies and to find optimal designs
that avoid resonance conditions.

Author: Kaan Gokbayrak, Purdue University
"""

import logging
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Dict, Tuple
import copy
import csv
from .beam import Beam, Material
from .analytical import AnalyticalSolver

logger = logging.getLogger(__name__)


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
        
        logger.info("Results exported to %s", filepath)

    def optimize_multivariate(self, target_freq: float, params: list,
                               bounds_dict: dict, n_modes: int = 1) -> dict:
        """
        Optimize multiple parameters simultaneously.

        Minimizes total mass subject to the constraint that the first n_modes
        natural frequencies are all above target_freq.

        Parameters
        ----------
        target_freq : float
            Minimum acceptable natural frequency [Hz]
        params : list of str
            Parameters to optimize, e.g. ['thickness', 'width']
        bounds_dict : dict
            Mapping param_name -> (min, max)
        n_modes : int, optional
            Number of modes that must exceed target_freq (default 1)

        Returns
        -------
        dict
            Keys: 'optimal_values' (dict param->value), 'optimal_frequencies',
                  'optimal_mass', 'initial_values' (dict), 'initial_frequencies',
                  'initial_mass', 'optimization_result'
        """
        x0 = [getattr(self.base_beam, p) for p in params]
        initial_values = {p: getattr(self.base_beam, p) for p in params}

        solver_initial = AnalyticalSolver(self.base_beam)
        initial_frequencies = solver_initial.natural_frequencies(n_modes, self.bc)
        initial_mass = self.base_beam.total_mass

        def _make_beam(x):
            beam_copy = copy.deepcopy(self.base_beam)
            for p, val in zip(params, x):
                setattr(beam_copy, p, val)
            return beam_copy

        def objective(x):
            return _make_beam(x).total_mass

        def constraint_min_freq(x):
            beam_copy = _make_beam(x)
            solver = AnalyticalSolver(beam_copy)
            freqs = solver.natural_frequencies(n_modes, self.bc)
            return min(freqs[:n_modes]) - target_freq

        bounds = [(bounds_dict[p][0], bounds_dict[p][1]) for p in params]
        constraints = {'type': 'ineq', 'fun': constraint_min_freq}

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6}
        )

        optimal_beam = _make_beam(result.x)
        solver_opt = AnalyticalSolver(optimal_beam)
        optimal_frequencies = solver_opt.natural_frequencies(n_modes, self.bc)
        optimal_mass = optimal_beam.total_mass
        optimal_values = {p: result.x[i] for i, p in enumerate(params)}

        return {
            'optimal_values': optimal_values,
            'optimal_frequencies': optimal_frequencies,
            'optimal_mass': optimal_mass,
            'initial_values': initial_values,
            'initial_frequencies': initial_frequencies,
            'initial_mass': initial_mass,
            'optimization_result': result
        }

    def pareto_study(self, param: str, values: np.ndarray,
                     excitation_freq: float, n_modes: int = 3) -> dict:
        """
        Pareto study: mass vs minimum frequency separation from excitation_freq.

        Sweeps param over values and identifies Pareto-optimal points for:
        - Objective 1: minimize mass
        - Objective 2: maximize minimum separation = min(|f_n - excitation_freq|) over n_modes

        A point i dominates point j if mass_i <= mass_j AND sep_i >= sep_j
        (with at least one strict).

        Parameters
        ----------
        param : str
        values : np.ndarray
        excitation_freq : float
        n_modes : int, optional

        Returns
        -------
        dict
            Keys: 'values', 'masses', 'min_separations', 'pareto_indices',
                  'pareto_values', 'pareto_masses', 'pareto_separations'
        """
        masses = []
        separations = []

        for value in values:
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, value)
            solver = AnalyticalSolver(beam_copy)
            freqs = solver.natural_frequencies(n_modes, self.bc)
            masses.append(beam_copy.total_mass)
            separations.append(float(np.min(np.abs(freqs - excitation_freq))))

        masses = np.array(masses)
        separations = np.array(separations)

        n = len(values)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if masses[j] <= masses[i] and separations[j] >= separations[i]:
                        if masses[j] < masses[i] or separations[j] > separations[i]:
                            is_dominated[i] = True
                            break
        pareto_indices = np.where(~is_dominated)[0]

        return {
            'values': values,
            'masses': masses,
            'min_separations': separations,
            'pareto_indices': pareto_indices,
            'pareto_values': values[pareto_indices],
            'pareto_masses': masses[pareto_indices],
            'pareto_separations': separations[pareto_indices]
        }

    def optimize_avoid_harmonics(self, excitation_harmonics: list,
                                  param: str = 'thickness',
                                  bounds: tuple = (0.001, 0.020),
                                  separation_hz: float = 10.0) -> dict:
        """
        Optimize to keep all natural frequencies away from all harmonics.

        Minimizes mass subject to: for every mode n and harmonic h,
        |f_n - h| >= separation_hz.

        Parameters
        ----------
        excitation_harmonics : list of float
        param : str
        bounds : tuple
        separation_hz : float

        Returns
        -------
        dict
            Keys: 'optimal_value', 'optimal_frequencies', 'optimal_mass',
                  'initial_frequencies', 'initial_mass', 'optimization_result'
        """
        n_modes = max(len(excitation_harmonics), 3)

        initial_value = getattr(self.base_beam, param)
        solver_initial = AnalyticalSolver(self.base_beam)
        initial_frequencies = solver_initial.natural_frequencies(n_modes, self.bc)
        initial_mass = self.base_beam.total_mass

        def get_freqs(x):
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, x[0])
            solver = AnalyticalSolver(beam_copy)
            return solver.natural_frequencies(n_modes, self.bc)

        def objective(x):
            beam_copy = copy.deepcopy(self.base_beam)
            setattr(beam_copy, param, x[0])
            return beam_copy.total_mass

        constraints = []
        for m in range(n_modes):
            for h in excitation_harmonics:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, m=m, h=h: (
                        abs(get_freqs(x)[m] - h) - separation_hz
                    )
                })

        result = minimize(
            objective,
            [initial_value],
            method='SLSQP',
            bounds=[(bounds[0], bounds[1])],
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6}
        )

        optimal_value = result.x[0]
        beam_optimal = copy.deepcopy(self.base_beam)
        setattr(beam_optimal, param, optimal_value)
        solver_opt = AnalyticalSolver(beam_optimal)
        optimal_frequencies = solver_opt.natural_frequencies(n_modes, self.bc)
        optimal_mass = beam_optimal.total_mass

        return {
            'optimal_value': optimal_value,
            'optimal_frequencies': optimal_frequencies,
            'optimal_mass': optimal_mass,
            'initial_frequencies': initial_frequencies,
            'initial_mass': initial_mass,
            'optimization_result': result
        }

    def optimize_material(self, target_freq: float,
                          materials_library: list = None,
                          param: str = 'thickness',
                          bounds: tuple = (0.001, 0.020)) -> dict:
        """
        Select optimal material by trying all materials in library.

        For each material, runs optimize_for_frequency and picks the one
        with minimum achievable mass while meeting target_freq.

        Parameters
        ----------
        target_freq : float
        materials_library : list of Material, optional
            Defaults to [steel, aluminum, titanium, carbon_fiber]
        param : str
        bounds : tuple

        Returns
        -------
        dict
            Keys: 'best_material', 'best_value', 'best_mass', 'best_frequency',
                  'results_by_material'
        """
        if materials_library is None:
            materials_library = [
                Material.steel(),
                Material.aluminum(),
                Material.titanium(),
                Material.carbon_fiber()
            ]

        results_by_material = {}
        best_material = None
        best_value = None
        best_mass = np.inf
        best_frequency = None

        for material in materials_library:
            beam_copy = copy.deepcopy(self.base_beam)
            beam_copy.material = material
            study = ParametricStudy(beam_copy, self.bc)
            result = study.optimize_for_frequency(target_freq, param=param, bounds=bounds)
            results_by_material[material.name] = result

            if (result['optimal_frequency'] >= target_freq - 0.1
                    and result['optimal_mass'] < best_mass):
                best_mass = result['optimal_mass']
                best_material = material
                best_value = result['optimal_value']
                best_frequency = result['optimal_frequency']

        return {
            'best_material': best_material,
            'best_value': best_value,
            'best_mass': best_mass,
            'best_frequency': best_frequency,
            'results_by_material': results_by_material
        }
