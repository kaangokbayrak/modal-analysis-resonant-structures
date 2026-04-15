"""
Tests for Modal Analysis Post-Processor

Validates FRF computation, MAC matrix, participation factors, effective mass
fractions, steady-state and impulse responses, and the summary dictionary.

Author: Kaan Gokbayrak, Purdue University
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.beam import Material, Beam
from src.fem_solver import FEMSolver
from src.analytical import AnalyticalSolver
from src.modal_analysis import ModalAnalyzer, from_fem_results, from_analytical_results

N_MODES = 3
N_ELEMENTS = 20


@pytest.fixture
def fem_results():
    """FEM solve results for a steel cantilever beam."""
    steel = Material.steel()
    beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)
    solver = FEMSolver(beam, n_elements=N_ELEMENTS, bc='cantilever')
    return solver.solve(n_modes=N_MODES)


@pytest.fixture
def analyzer(fem_results):
    """ModalAnalyzer built from FEM results."""
    return from_fem_results(fem_results)


class TestModalAnalyzer:
    """Tests for ModalAnalyzer."""

    # ------------------------------------------------------------------
    # Factory functions
    # ------------------------------------------------------------------

    def test_from_fem_results_factory(self, fem_results):
        """from_fem_results returns a ModalAnalyzer with correct mode count."""
        ma = from_fem_results(fem_results)
        assert isinstance(ma, ModalAnalyzer)
        assert len(ma.frequencies_hz) == N_MODES

    def test_from_analytical_results_factory(self):
        """from_analytical_results returns a ModalAnalyzer with correct mode count."""
        steel = Material.steel()
        beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)
        analytical = AnalyticalSolver(beam)
        results = analytical.full_analysis(n_modes=N_MODES, bc='cantilever')
        ma = from_analytical_results(results)
        assert isinstance(ma, ModalAnalyzer)
        assert len(ma.frequencies_hz) == N_MODES

    # ------------------------------------------------------------------
    # Frequency Response Function
    # ------------------------------------------------------------------

    def test_frf_shape(self, analyzer):
        """compute_frf returns two arrays of length 200; H is complex."""
        omega_range = np.linspace(1.0, 5000.0, 200)
        omega_out, H = analyzer.compute_frf(omega_range, dof_out=2, dof_in=2)
        assert omega_out.shape == (200,)
        assert H.shape == (200,)
        assert np.iscomplexobj(H)

    def test_frf_peaks_at_natural_frequencies(self, analyzer):
        """Magnitude of H should peak near each natural frequency."""
        # Dense grid covering all modes
        f_max = analyzer.frequencies_hz[-1] * 2.0
        omega_range = np.linspace(1.0, 2.0 * np.pi * f_max, 5000)
        _, H = analyzer.compute_frf(omega_range, dof_out=2, dof_in=2)
        H_abs = np.abs(H)
        median_H = np.median(H_abs)

        for omega_n in analyzer.omega_n:
            # Find the index closest to this natural frequency
            idx = np.argmin(np.abs(omega_range - omega_n))
            # Allow a small window around the peak
            lo = max(0, idx - 5)
            hi = min(len(H_abs) - 1, idx + 5)
            local_peak = np.max(H_abs[lo:hi + 1])
            assert local_peak > median_H, (
                f"No peak near omega_n={omega_n:.1f} rad/s"
            )

    # ------------------------------------------------------------------
    # MAC matrix
    # ------------------------------------------------------------------

    def test_mac_diagonal_is_one(self, analyzer):
        """MAC of mode shapes vs themselves should have unit diagonal."""
        mac = analyzer.compute_mac_matrix(analyzer.mode_shapes)
        np.testing.assert_allclose(np.diag(mac), np.ones(N_MODES), atol=1e-10)

    def test_mac_off_diagonal_below_threshold(self, analyzer):
        """Off-diagonal MAC values for well-separated modes should be < 0.9."""
        mac = analyzer.compute_mac_matrix(analyzer.mode_shapes)
        mask = ~np.eye(N_MODES, dtype=bool)
        assert np.all(mac[mask] < 0.9)

    # ------------------------------------------------------------------
    # Participation factors and effective mass
    # ------------------------------------------------------------------

    def test_participation_factors_shape(self, analyzer):
        """modal_participation_factors returns array of length n_modes."""
        gamma = analyzer.modal_participation_factors()
        assert gamma.shape == (N_MODES,)

    def test_effective_mass_fractions_bounded(self, analyzer):
        """effective_mass_fractions values are non-negative and sum ≤ 1."""
        emf = analyzer.effective_mass_fractions()
        assert emf.shape == (N_MODES,)
        assert np.all(emf >= 0)
        assert np.sum(emf) <= 1.0 + 1e-9

    # ------------------------------------------------------------------
    # Steady-state and impulse responses
    # ------------------------------------------------------------------

    def test_steady_state_response_shape(self, analyzer):
        """steady_state_response returns complex array of shape (n_dofs, n_freq)."""
        omega = np.linspace(1.0, 5000.0, 100)
        force = np.zeros(analyzer.n_dofs, dtype=complex)
        force[2] = 1.0
        X = analyzer.steady_state_response(omega, force)
        assert X.shape == (analyzer.n_dofs, 100)
        assert np.iscomplexobj(X)

    def test_impulse_response_shape(self, analyzer):
        """impulse_response returns a real array of the correct length."""
        t = np.linspace(0, 1.0, 500)
        h = analyzer.impulse_response(t, dof_out=2, dof_in=2)
        assert h.shape == (500,)
        assert not np.iscomplexobj(h)

    def test_impulse_response_decays(self, analyzer):
        """Impulse response RMS should decrease over time (damped system)."""
        t = np.linspace(0, 2.0, 2000)
        h = analyzer.impulse_response(t, dof_out=2, dof_in=2)
        quarter = len(t) // 4
        rms_first = np.sqrt(np.mean(h[:quarter] ** 2))
        rms_last = np.sqrt(np.mean(h[3 * quarter:] ** 2))
        assert rms_first > rms_last, (
            "Impulse response does not decay: "
            f"rms_first={rms_first:.4e}, rms_last={rms_last:.4e}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def test_summary_keys(self, analyzer):
        """summary() contains expected keys and cumulative fraction is non-decreasing."""
        info = analyzer.summary()
        for key in ('frequencies_hz', 'participation_factors',
                    'effective_mass_fractions', 'cumulative_mass_fraction'):
            assert key in info, f"Missing key: {key}"

        # cumulative_mass_fraction should be a sensible non-negative number
        assert info['cumulative_mass_fraction'] >= 0.0

        # Cumulative sum of effective_mass_fractions should be non-decreasing
        cumulative = np.cumsum(info['effective_mass_fractions'])
        assert np.all(np.diff(cumulative) >= -1e-12), (
            "Cumulative mass fractions are not monotonically non-decreasing"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
