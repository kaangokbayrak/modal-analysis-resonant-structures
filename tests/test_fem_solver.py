"""
Tests for FEM Solver

This test suite validates the custom FEM implementation, including:
- Element matrix formulation
- Global matrix assembly
- Boundary condition application
- Eigenvalue solution
- Convergence properties

Author: Kaan Gokbayrak, Purdue University
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.beam import Material, Beam
from src.fem_solver import FEMSolver
from src.analytical import AnalyticalSolver


class TestFEMSolver:
    """Test FEM solver implementation."""
    
    @pytest.fixture
    def steel_cantilever(self):
        """Create a standard steel cantilever beam."""
        steel = Material.steel()
        beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)
        return beam
    
    def test_element_stiffness_matrix_symmetry(self, steel_cantilever):
        """Test that element stiffness matrix is symmetric."""
        solver = FEMSolver(steel_cantilever, n_elements=10, bc='cantilever')
        k_e = solver.element_stiffness_matrix(solver.element_length)
        
        # Check symmetry
        assert np.allclose(k_e, k_e.T)
    
    def test_element_mass_matrix_symmetry(self, steel_cantilever):
        """Test that element mass matrix is symmetric."""
        solver = FEMSolver(steel_cantilever, n_elements=10, bc='cantilever')
        m_e = solver.element_mass_matrix(solver.element_length)
        
        # Check symmetry
        assert np.allclose(m_e, m_e.T)
    
    def test_element_mass_matrix_positive_definite(self, steel_cantilever):
        """Test that element mass matrix is positive semi-definite."""
        solver = FEMSolver(steel_cantilever, n_elements=10, bc='cantilever')
        m_e = solver.element_mass_matrix(solver.element_length)
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(m_e)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical error
    
    def test_global_matrix_dimensions(self, steel_cantilever):
        """Test that global matrices have correct dimensions."""
        n_elem = 10
        solver = FEMSolver(steel_cantilever, n_elements=n_elem, bc='cantilever')
        K, M = solver.assemble_global_matrices()
        
        # Total DOFs = 2 * (n_elements + 1)
        expected_dofs = 2 * (n_elem + 1)
        assert K.shape == (expected_dofs, expected_dofs)
        assert M.shape == (expected_dofs, expected_dofs)
    
    def test_global_matrix_symmetry(self, steel_cantilever):
        """Test that global matrices are symmetric."""
        solver = FEMSolver(steel_cantilever, n_elements=10, bc='cantilever')
        K, M = solver.assemble_global_matrices()
        
        assert np.allclose(K, K.T)
        assert np.allclose(M, M.T)
    
    def test_boundary_condition_application(self, steel_cantilever):
        """Test that boundary conditions reduce matrix size correctly."""
        solver = FEMSolver(steel_cantilever, n_elements=10, bc='cantilever')
        K, M = solver.assemble_global_matrices()
        K_reduced, M_reduced = solver.apply_boundary_conditions(K, M)
        
        # Cantilever: eliminate 2 DOFs (w and θ at node 0)
        expected_size = K.shape[0] - 2
        assert K_reduced.shape == (expected_size, expected_size)
        assert M_reduced.shape == (expected_size, expected_size)
    
    def test_fem_frequencies_positive(self, steel_cantilever):
        """Test that FEM produces positive natural frequencies."""
        solver = FEMSolver(steel_cantilever, n_elements=50, bc='cantilever')
        results = solver.solve(n_modes=5)
        
        freqs = results['frequencies_hz']
        assert len(freqs) == 5
        assert np.all(freqs > 0)
        assert np.all(np.diff(freqs) > 0)  # Frequencies should be increasing
    
    def test_fem_vs_analytical_convergence(self, steel_cantilever):
        """Test that FEM converges to analytical solution with mesh refinement."""
        # Analytical solution
        analytical = AnalyticalSolver(steel_cantilever)
        freq_analytical = analytical.natural_frequencies(n_modes=3, bc='cantilever')
        
        # FEM with 100 elements should be within 2% of analytical
        solver = FEMSolver(steel_cantilever, n_elements=100, bc='cantilever')
        results = solver.solve(n_modes=3)
        freq_fem = results['frequencies_hz']
        
        # Compute relative error
        error = np.abs(freq_fem - freq_analytical) / freq_analytical
        
        # Should be within 2% for all modes
        assert np.all(error < 0.02), f"Errors: {error*100}%"
    
    def test_fem_increasing_accuracy_with_mesh(self, steel_cantilever):
        """Test that finer mesh improves accuracy."""
        # Analytical reference
        analytical = AnalyticalSolver(steel_cantilever)
        freq_analytical = analytical.natural_frequencies(n_modes=1, bc='cantilever')[0]
        
        # Test with increasing mesh density
        element_counts = [10, 20, 50, 100]
        errors = []
        
        for n_elem in element_counts:
            solver = FEMSolver(steel_cantilever, n_elements=n_elem, bc='cantilever')
            results = solver.solve(n_modes=1)
            freq_fem = results['frequencies_hz'][0]
            error = abs(freq_fem - freq_analytical) / freq_analytical
            errors.append(error)
        
        # Errors should decrease with mesh refinement
        assert errors[-1] < errors[0]
        assert errors[-1] < errors[1]
    
    def test_all_boundary_conditions(self, steel_cantilever):
        """Test that all boundary conditions produce valid results."""
        bcs = ['cantilever', 'simply-supported', 'fixed-fixed', 'fixed-pinned']
        
        for bc in bcs:
            solver = FEMSolver(steel_cantilever, n_elements=50, bc=bc)
            results = solver.solve(n_modes=3)
            freqs = results['frequencies_hz']
            
            # Check valid frequencies
            assert len(freqs) == 3
            assert np.all(freqs > 0)
            assert np.all(np.diff(freqs) > 0)
    
    def test_fem_mode_shapes_normalized(self, steel_cantilever):
        """Test that mode shapes are properly normalized."""
        solver = FEMSolver(steel_cantilever, n_elements=50, bc='cantilever')
        results = solver.solve(n_modes=3)
        mode_shapes = results['mode_shapes']
        
        # Extract displacement DOFs (every other DOF)
        for i in range(3):
            displacements = mode_shapes[::2, i]
            max_disp = np.max(np.abs(displacements))
            # Should be normalized to 1
            assert np.isclose(max_disp, 1.0, atol=1e-6)
    
    def test_mesh_convergence_study(self, steel_cantilever):
        """Test mesh convergence study functionality."""
        solver = FEMSolver(steel_cantilever, n_elements=50, bc='cantilever')
        convergence_data = solver.mesh_convergence_study(
            element_counts=[10, 20, 50],
            n_modes=3
        )
        
        # Check returned data structure
        assert 'element_counts' in convergence_data
        assert 'frequencies' in convergence_data
        assert 'errors' in convergence_data
        assert 'analytical_frequencies' in convergence_data
        
        # Check shapes
        assert len(convergence_data['element_counts']) == 3
        assert convergence_data['frequencies'].shape == (3, 3)
        assert convergence_data['errors'].shape == (3, 3)
        
        # Errors should generally decrease with refinement
        errors_first_mode = convergence_data['errors'][:, 0]
        assert errors_first_mode[-1] < errors_first_mode[0]


class TestFEMSpecialCases:
    """Test FEM solver special cases and edge conditions."""
    
    def test_very_stiff_beam(self):
        """Test FEM with very high stiffness beam."""
        material = Material("VeryStiff", E=1e12, rho=7850, nu=0.3)
        beam = Beam(material=material, length=0.3, width=0.025, thickness=0.003)
        
        solver = FEMSolver(beam, n_elements=50, bc='cantilever')
        results = solver.solve(n_modes=3)
        
        # Should still produce valid frequencies
        assert np.all(results['frequencies_hz'] > 0)
    
    def test_very_long_beam(self):
        """Test FEM with very long beam."""
        steel = Material.steel()
        beam = Beam(material=steel, length=10.0, width=0.025, thickness=0.003)
        
        solver = FEMSolver(beam, n_elements=50, bc='cantilever')
        results = solver.solve(n_modes=3)
        
        # Should produce lower frequencies
        assert results['frequencies_hz'][0] < 10  # Much lower first mode


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
