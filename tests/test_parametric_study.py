"""
Tests for Parametric Study Module

Validates parameter sweeps, design-space mapping, optimisation routines,
Pareto analysis, and CSV export.

Author: Kaan Gokbayrak, Purdue University
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.beam import Material, Beam
from src.parametric_study import ParametricStudy


@pytest.fixture
def steel_beam():
    """Steel cantilever beam 300 mm × 25 mm × 3 mm."""
    steel = Material.steel()
    return Beam(material=steel, length=0.3, width=0.025, thickness=0.003)


class TestParametricStudy:
    """Tests for ParametricStudy."""

    def test_sweep_thickness_increases_frequency(self, steel_beam):
        """Thicker beam → higher first natural frequency."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        thicknesses = np.linspace(0.002, 0.006, 4)
        result = study.sweep_parameter('thickness', thicknesses, n_modes=3)
        freqs = result['frequencies']
        # First mode should increase monotonically with thickness
        assert np.all(np.diff(freqs[:, 0]) > 0)

    def test_sweep_length_decreases_frequency(self, steel_beam):
        """Longer beam → lower first natural frequency."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        lengths = np.linspace(0.2, 0.5, 4)
        result = study.sweep_parameter('length', lengths, n_modes=3)
        freqs = result['frequencies']
        # First mode should decrease monotonically with length
        assert np.all(np.diff(freqs[:, 0]) < 0)

    def test_sweep_returns_correct_shape(self, steel_beam):
        """Sweep over 5 values with 3 modes → expected array shapes."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        thicknesses = np.linspace(0.002, 0.006, 5)
        result = study.sweep_parameter('thickness', thicknesses, n_modes=3)
        assert result['frequencies'].shape == (5, 3)
        assert result['masses'].shape == (5,)

    def test_design_space_map_shape(self, steel_beam):
        """3 × 3 design space → frequency_map and mass_map of shape (3, 3)."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        thicknesses = np.array([0.002, 0.003, 0.005])
        lengths = np.array([0.25, 0.30, 0.40])
        result = study.design_space_map(
            'thickness', thicknesses,
            'length', lengths,
            mode=1
        )
        assert result['frequency_map'].shape == (3, 3)
        assert result['mass_map'].shape == (3, 3)

    def test_optimize_for_frequency_keys(self, steel_beam):
        """optimize_for_frequency result contains expected keys."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        result = study.optimize_for_frequency(
            target_freq=40.0,
            param='thickness',
            bounds=(0.001, 0.010)
        )
        for key in ('optimal_value', 'optimal_frequency', 'optimal_mass'):
            assert key in result, f"Missing key: {key}"

    def test_optimize_for_frequency_achieves_target(self, steel_beam):
        """Optimised beam first frequency should meet or exceed target."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        result = study.optimize_for_frequency(
            target_freq=40.0,
            param='thickness',
            bounds=(0.001, 0.010)
        )
        assert result['optimal_frequency'] >= 40.0 - 0.1

    def test_optimize_multivariate_keys(self, steel_beam):
        """optimize_multivariate result contains expected keys."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        result = study.optimize_multivariate(
            target_freq=40.0,
            params=['thickness', 'width'],
            bounds_dict={
                'thickness': (0.001, 0.010),
                'width': (0.010, 0.050),
            },
            n_modes=1
        )
        for key in ('optimal_values', 'optimal_frequencies', 'optimal_mass'):
            assert key in result, f"Missing key: {key}"

    def test_pareto_study_structure(self, steel_beam):
        """pareto_study result has required keys and at least one Pareto point."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        thicknesses = np.linspace(0.002, 0.006, 5)
        result = study.pareto_study(
            param='thickness',
            values=thicknesses,
            excitation_freq=50.0,
            n_modes=3
        )
        for key in ('pareto_indices', 'pareto_values', 'masses', 'min_separations'):
            assert key in result, f"Missing key: {key}"
        assert len(result['pareto_values']) > 0

    def test_optimize_avoid_harmonics_keys(self, steel_beam):
        """optimize_avoid_harmonics result contains expected keys."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        result = study.optimize_avoid_harmonics(
            excitation_harmonics=[50.0, 100.0],
            param='thickness',
            bounds=(0.001, 0.010),
            separation_hz=5.0
        )
        for key in ('optimal_value', 'optimal_mass'):
            assert key in result, f"Missing key: {key}"

    def test_optimize_material_returns_best(self, steel_beam):
        """optimize_material returns a Material instance and results dict."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        result = study.optimize_material(
            target_freq=40.0,
            param='thickness',
            bounds=(0.001, 0.010)
        )
        assert 'best_material' in result
        assert 'best_mass' in result
        assert 'results_by_material' in result
        assert isinstance(result['best_material'], Material)

    def test_export_csv(self, steel_beam, tmp_path):
        """Exported CSV file exists and is non-empty."""
        study = ParametricStudy(steel_beam, bc='cantilever')
        thicknesses = np.linspace(0.002, 0.006, 4)
        result = study.sweep_parameter('thickness', thicknesses, n_modes=3)

        out_file = tmp_path / 'out.csv'
        study.export_results(result, str(out_file))

        assert out_file.exists()
        assert out_file.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
