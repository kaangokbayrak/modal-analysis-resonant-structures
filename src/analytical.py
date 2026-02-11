"""
Analytical Solutions for Euler-Bernoulli Beam Modal Analysis

This module implements closed-form solutions for natural frequencies and mode shapes
of Euler-Bernoulli beams with various boundary conditions.

Author: Kaan Gokbayrak, Purdue University
"""

import numpy as np
from typing import Dict, Tuple
from .beam import Beam


class AnalyticalSolver:
    """
    Analytical solver for Euler-Bernoulli beam natural frequencies and mode shapes.
    
    Supports four boundary conditions:
    - cantilever: fixed at one end, free at the other
    - simply-supported: pinned at both ends
    - fixed-fixed: clamped at both ends
    - fixed-pinned: fixed at one end, pinned at the other
    
    Parameters
    ----------
    beam : Beam
        Beam object with material and geometric properties
    """
    
    # Characteristic equation roots (β values) for different boundary conditions
    # These are dimensionless eigenvalues satisfying the characteristic equation
    BETA_VALUES = {
        'cantilever': np.array([1.8751, 4.6941, 7.8548, 10.9955, 14.1372, 
                                17.2788, 20.4204, 23.5619, 26.7035, 29.8451]),
        'simply-supported': None,  # β_n = nπ (computed dynamically)
        'fixed-fixed': np.array([4.7300, 7.8532, 10.9956, 14.1372, 17.2788,
                                20.4204, 23.5619, 26.7035, 29.8451, 32.9867]),
        'fixed-pinned': np.array([3.9266, 7.0686, 10.2102, 13.3518, 16.4934,
                                 19.6350, 22.7766, 25.9182, 29.0598, 32.2014])
    }
    
    def __init__(self, beam: Beam):
        """
        Initialize analytical solver with a beam.
        
        Parameters
        ----------
        beam : Beam
            Beam object to analyze
        """
        self.beam = beam
        
    def _get_beta_values(self, n_modes: int, bc: str) -> np.ndarray:
        """
        Get characteristic equation roots for specified boundary condition.
        
        Parameters
        ----------
        n_modes : int
            Number of modes
        bc : str
            Boundary condition identifier
            
        Returns
        -------
        np.ndarray
            Array of β values for the first n_modes
        """
        if bc not in self.BETA_VALUES:
            raise ValueError(f"Unknown boundary condition: {bc}. "
                           f"Valid options: {list(self.BETA_VALUES.keys())}")
        
        if bc == 'simply-supported':
            # For simply-supported: β_n = nπ
            return np.pi * np.arange(1, n_modes + 1)
        else:
            beta = self.BETA_VALUES[bc]
            if n_modes > len(beta):
                raise ValueError(f"Only {len(beta)} modes available for {bc}, "
                               f"requested {n_modes}")
            return beta[:n_modes]
    
    def natural_frequencies(self, n_modes: int = 5, bc: str = 'cantilever') -> np.ndarray:
        """
        Compute natural frequencies using closed-form Euler-Bernoulli solution.
        
        The natural frequency formula is:
        f_n = (β_n²) / (2πL²) × √(EI / ρA)
        
        or in angular frequency:
        ω_n = (β_n²) / (L²) × √(EI / ρA)
        
        Parameters
        ----------
        n_modes : int, optional
            Number of natural frequencies to compute (default: 5)
        bc : str, optional
            Boundary condition: 'cantilever', 'simply-supported', 
            'fixed-fixed', or 'fixed-pinned' (default: 'cantilever')
            
        Returns
        -------
        np.ndarray
            Natural frequencies in Hz, shape (n_modes,)
        """
        beta = self._get_beta_values(n_modes, bc)
        
        # Material and geometric properties
        E = self.beam.material.E
        I = self.beam.I
        rho = self.beam.material.rho
        A = self.beam.area
        L = self.beam.length
        
        # Angular frequencies: ω_n = (β_n / L)² × √(EI / ρA)
        omega_n = (beta / L)**2 * np.sqrt(E * I / (rho * A))
        
        # Convert to Hz: f = ω / (2π)
        freq_hz = omega_n / (2 * np.pi)
        
        return freq_hz
    
    def mode_shapes(self, x: np.ndarray, n_modes: int = 5, 
                    bc: str = 'cantilever') -> np.ndarray:
        """
        Compute mode shape functions φ_n(x) at specified locations.
        
        Mode shapes are the spatial distribution of displacement for each natural mode.
        Each mode shape is normalized so that the maximum absolute value is 1.
        
        Parameters
        ----------
        x : np.ndarray
            Positions along beam where mode shapes are evaluated [m], shape (n_points,)
        n_modes : int, optional
            Number of mode shapes to compute (default: 5)
        bc : str, optional
            Boundary condition (default: 'cantilever')
            
        Returns
        -------
        np.ndarray
            Mode shapes, shape (n_points, n_modes)
            Each column is a mode shape φ_n(x)
        """
        beta = self._get_beta_values(n_modes, bc)
        L = self.beam.length
        
        # Normalize x to dimensionless coordinate ξ = x/L
        xi = x / L
        
        # Initialize mode shape array
        mode_shapes = np.zeros((len(x), n_modes))
        
        for i, beta_n in enumerate(beta):
            if bc == 'cantilever':
                # Cantilever mode shape (fixed at x=0, free at x=L)
                # φ(ξ) = cosh(β_n ξ) - cos(β_n ξ) - σ_n [sinh(β_n ξ) - sin(β_n ξ)]
                # where σ_n = (cosh(β_n) - cos(β_n)) / (sinh(β_n) - sin(β_n))
                sigma_n = (np.cosh(beta_n) - np.cos(beta_n)) / (np.sinh(beta_n) - np.sin(beta_n))
                phi = (np.cosh(beta_n * xi) - np.cos(beta_n * xi) - 
                       sigma_n * (np.sinh(beta_n * xi) - np.sin(beta_n * xi)))
                       
            elif bc == 'simply-supported':
                # Simply-supported mode shape (pinned at both ends)
                # φ(ξ) = sin(β_n ξ) where β_n = nπ
                phi = np.sin(beta_n * xi)
                
            elif bc == 'fixed-fixed':
                # Fixed-fixed mode shape (clamped at both ends)
                # φ(ξ) = cosh(β_n ξ) - cos(β_n ξ) - σ_n [sinh(β_n ξ) - sin(β_n ξ)]
                # where σ_n = (cosh(β_n) - cos(β_n)) / (sinh(β_n) - sin(β_n))
                sigma_n = (np.cosh(beta_n) - np.cos(beta_n)) / (np.sinh(beta_n) - np.sin(beta_n))
                phi = (np.cosh(beta_n * xi) - np.cos(beta_n * xi) - 
                       sigma_n * (np.sinh(beta_n * xi) - np.sin(beta_n * xi)))
                       
            elif bc == 'fixed-pinned':
                # Fixed-pinned mode shape (fixed at x=0, pinned at x=L)
                # φ(ξ) = cosh(β_n ξ) - cos(β_n ξ) - σ_n [sinh(β_n ξ) - sin(β_n ξ)]
                # where σ_n = (cosh(β_n) - cos(β_n)) / (sinh(β_n) - sin(β_n))
                sigma_n = (np.cosh(beta_n) - np.cos(beta_n)) / (np.sinh(beta_n) - np.sin(beta_n))
                phi = (np.cosh(beta_n * xi) - np.cos(beta_n * xi) - 
                       sigma_n * (np.sinh(beta_n * xi) - np.sin(beta_n * xi)))
            
            # Normalize mode shape so that max|φ| = 1
            phi = phi / np.max(np.abs(phi))
            mode_shapes[:, i] = phi
        
        return mode_shapes
    
    def full_analysis(self, n_modes: int = 5, bc: str = 'cantilever', 
                      n_points: int = 100) -> Dict:
        """
        Perform complete analytical modal analysis.
        
        Parameters
        ----------
        n_modes : int, optional
            Number of modes to analyze (default: 5)
        bc : str, optional
            Boundary condition (default: 'cantilever')
        n_points : int, optional
            Number of points for mode shape evaluation (default: 100)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'frequencies_hz': Natural frequencies [Hz], shape (n_modes,)
            - 'x': Position array [m], shape (n_points,)
            - 'mode_shapes': Mode shapes, shape (n_points, n_modes)
            - 'bc': Boundary condition string
            - 'beam': Beam object
            - 'beta_values': Characteristic equation roots
        """
        # Compute natural frequencies
        frequencies = self.natural_frequencies(n_modes, bc)
        
        # Create position array
        x = np.linspace(0, self.beam.length, n_points)
        
        # Compute mode shapes
        mode_shapes = self.mode_shapes(x, n_modes, bc)
        
        # Get beta values
        beta = self._get_beta_values(n_modes, bc)
        
        return {
            'frequencies_hz': frequencies,
            'x': x,
            'mode_shapes': mode_shapes,
            'bc': bc,
            'beam': self.beam,
            'beta_values': beta
        }
