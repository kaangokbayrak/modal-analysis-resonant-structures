"""
Tests for Analytical Solver

This test suite verifies the analytical solutions for Euler-Bernoulli beams
against known values from vibration textbooks and theoretical relationships.

Author: Kaan Gokbayrak, Purdue University
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.beam import Material, Beam
from src.analytical import AnalyticalSolver


class TestMaterial:
    """Test Material class and presets."""
    
    def test_steel_preset(self):
        """Test steel material preset values."""
        steel = Material.steel()
        assert steel.name == "Steel"
        assert steel.E == 200e9
        assert steel.rho == 7850
        assert steel.nu == 0.3
    
    def test_aluminum_preset(self):
        """Test aluminum material preset values."""
        aluminum = Material.aluminum()
        assert aluminum.name == "Aluminum"
        assert aluminum.E == 69e9
        assert aluminum.rho == 2700
        assert aluminum.nu == 0.33
    
    def test_invalid_material(self):
        """Test that invalid material properties raise errors."""
        with pytest.raises(ValueError):
            Material("Invalid", E=-1, rho=1000, nu=0.3)
        with pytest.raises(ValueError):
            Material("Invalid", E=200e9, rho=-1, nu=0.3)
        with pytest.raises(ValueError):
            Material("Invalid", E=200e9, rho=1000, nu=0.6)


class TestBeam:
    """Test Beam class."""
    
    def test_beam_properties(self):
        """Test beam geometric property calculations."""
        steel = Material.steel()
        beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)
        
        # Cross-sectional area
        assert beam.area == pytest.approx(0.025 * 0.003)
        
        # Second moment of area: I = b*h³/12
        expected_I = 0.025 * 0.003**3 / 12
        assert beam.I == pytest.approx(expected_I)
        
        # Mass per length
        expected_mass_per_length = steel.rho * beam.area
        assert beam.mass_per_length == pytest.approx(expected_mass_per_length)
        
        # Total mass
        expected_total_mass = expected_mass_per_length * 0.3
        assert beam.total_mass == pytest.approx(expected_total_mass)
    
    def test_invalid_beam(self):
        """Test that invalid beam dimensions raise errors."""
        steel = Material.steel()
        with pytest.raises(ValueError):
            Beam(material=steel, length=-0.3, width=0.025, thickness=0.003)
        with pytest.raises(ValueError):
            Beam(material=steel, length=0.3, width=-0.025, thickness=0.003)
        with pytest.raises(ValueError):
            Beam(material=steel, length=0.3, width=0.025, thickness=0)


class TestAnalyticalSolver:
    """Test analytical solutions for natural frequencies and mode shapes."""
    
    @pytest.fixture
    def steel_cantilever(self):
        """Create a standard steel cantilever beam."""
        steel = Material.steel()
        beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)
        return beam
    
    def test_cantilever_frequencies(self, steel_cantilever):
        """Test cantilever natural frequencies against expected values.
        
        For a steel cantilever (300mm × 25mm × 3mm):
        - First mode should be around 30-40 Hz
        """
        solver = AnalyticalSolver(steel_cantilever)
        freqs = solver.natural_frequencies(n_modes=5, bc='cantilever')
        
        # Check that we get 5 frequencies
        assert len(freqs) == 5
        
        # Check that frequencies are positive and increasing
        assert np.all(freqs > 0)
        assert np.all(np.diff(freqs) > 0)
        
        # Check first frequency is in reasonable range
        assert 20 < freqs[0] < 60
    
    def test_simply_supported_frequencies(self, steel_cantilever):
        """Test simply-supported natural frequencies."""
        solver = AnalyticalSolver(steel_cantilever)
        freqs = solver.natural_frequencies(n_modes=5, bc='simply-supported')
        
        # Check that frequencies are positive and increasing
        assert len(freqs) == 5
        assert np.all(freqs > 0)
        assert np.all(np.diff(freqs) > 0)
    
    def test_fixed_fixed_frequencies(self, steel_cantilever):
        """Test fixed-fixed natural frequencies."""
        solver = AnalyticalSolver(steel_cantilever)
        freqs = solver.natural_frequencies(n_modes=5, bc='fixed-fixed')
        
        # Fixed-fixed should have higher frequencies than cantilever
        freqs_cant = solver.natural_frequencies(n_modes=5, bc='cantilever')
        assert freqs[0] > freqs_cant[0]
    
    def test_frequency_scaling_with_E(self, steel_cantilever):
        """Test that frequencies scale as √E."""
        solver1 = AnalyticalSolver(steel_cantilever)
        freqs1 = solver1.natural_frequencies(n_modes=3, bc='cantilever')
        
        # Double Young's modulus
        material2 = Material("HighE", E=2*steel_cantilever.material.E, 
                            rho=steel_cantilever.material.rho, nu=0.3)
        beam2 = Beam(material=material2, length=steel_cantilever.length,
                     width=steel_cantilever.width, thickness=steel_cantilever.thickness)
        solver2 = AnalyticalSolver(beam2)
        freqs2 = solver2.natural_frequencies(n_modes=3, bc='cantilever')
        
        # Frequencies should scale as √2
        ratio = freqs2 / freqs1
        assert np.allclose(ratio, np.sqrt(2), rtol=1e-10)
    
    def test_frequency_scaling_with_length(self, steel_cantilever):
        """Test that frequencies scale as 1/L²."""
        solver1 = AnalyticalSolver(steel_cantilever)
        freqs1 = solver1.natural_frequencies(n_modes=3, bc='cantilever')
        
        # Double length
        beam2 = Beam(material=steel_cantilever.material, 
                     length=2*steel_cantilever.length,
                     width=steel_cantilever.width, 
                     thickness=steel_cantilever.thickness)
        solver2 = AnalyticalSolver(beam2)
        freqs2 = solver2.natural_frequencies(n_modes=3, bc='cantilever')
        
        # Frequencies should scale as 1/4
        ratio = freqs2 / freqs1
        assert np.allclose(ratio, 0.25, rtol=1e-10)
    
    def test_mode_shapes_boundary_conditions(self, steel_cantilever):
        """Test that mode shapes satisfy boundary conditions."""
        solver = AnalyticalSolver(steel_cantilever)
        x = np.linspace(0, steel_cantilever.length, 100)
        
        # Cantilever: w(0) = 0 (fixed at root)
        modes = solver.mode_shapes(x, n_modes=3, bc='cantilever')
        assert np.allclose(modes[0, :], 0, atol=1e-10)
        
        # Simply-supported: w(0) = w(L) = 0 (pinned at both ends)
        modes_ss = solver.mode_shapes(x, n_modes=3, bc='simply-supported')
        assert np.allclose(modes_ss[0, :], 0, atol=1e-10)
        assert np.allclose(modes_ss[-1, :], 0, atol=1e-10)
    
    def test_mode_shape_normalization(self, steel_cantilever):
        """Test that mode shapes are normalized to max = 1."""
        solver = AnalyticalSolver(steel_cantilever)
        x = np.linspace(0, steel_cantilever.length, 100)
        modes = solver.mode_shapes(x, n_modes=3, bc='cantilever')
        
        for i in range(3):
            max_val = np.max(np.abs(modes[:, i]))
            assert np.isclose(max_val, 1.0, atol=1e-10)
    
    def test_full_analysis(self, steel_cantilever):
        """Test full analysis returns complete results dictionary."""
        solver = AnalyticalSolver(steel_cantilever)
        results = solver.full_analysis(n_modes=5, bc='cantilever', n_points=100)
        
        # Check all required keys are present
        assert 'frequencies_hz' in results
        assert 'x' in results
        assert 'mode_shapes' in results
        assert 'bc' in results
        assert 'beam' in results
        assert 'beta_values' in results
        
        # Check shapes
        assert len(results['frequencies_hz']) == 5
        assert len(results['x']) == 100
        assert results['mode_shapes'].shape == (100, 5)
    
    def test_invalid_boundary_condition(self, steel_cantilever):
        """Test that invalid boundary condition raises error."""
        solver = AnalyticalSolver(steel_cantilever)
        with pytest.raises(ValueError):
            solver.natural_frequencies(n_modes=3, bc='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
