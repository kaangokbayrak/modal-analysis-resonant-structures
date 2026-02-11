"""
Custom Finite Element Method Solver for Euler-Bernoulli Beams

This module implements a complete FEM solver from scratch for modal analysis of beams.
No ready-made FEA libraries are used - all matrices are assembled manually to demonstrate
deep understanding of the finite element method.

Key Features:
- Element stiffness and mass matrix formulation
- Global matrix assembly with proper DOF mapping
- Boundary condition application via matrix partitioning
- Generalized eigenvalue problem solution
- Mesh convergence studies

Author: Kaan Gokbayrak, Purdue University
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, Tuple, List
from .beam import Beam


class FEMSolver:
    """
    Custom Finite Element Method solver for Euler-Bernoulli beam modal analysis.
    
    This solver discretizes the beam into 2-node elements with 2 DOFs per node:
    - w: transverse displacement
    - θ: rotation (slope)
    
    Total DOFs per element: 4 [w1, θ1, w2, θ2]
    
    Parameters
    ----------
    beam : Beam
        Beam object with material and geometric properties
    n_elements : int, optional
        Number of finite elements (default: 50)
    bc : str, optional
        Boundary condition: 'cantilever', 'simply-supported', 
        'fixed-fixed', or 'fixed-pinned' (default: 'cantilever')
    """
    
    def __init__(self, beam: Beam, n_elements: int = 50, bc: str = 'cantilever'):
        """
        Initialize FEM solver.
        
        Parameters
        ----------
        beam : Beam
            Beam object to analyze
        n_elements : int
            Number of finite elements
        bc : str
            Boundary condition identifier
        """
        self.beam = beam
        self.n_elements = n_elements
        self.bc = bc
        
        # Derived quantities
        self.n_nodes = n_elements + 1
        self.dofs_per_node = 2  # [w, θ]
        self.total_dofs = self.n_nodes * self.dofs_per_node
        self.element_length = beam.length / n_elements
        
        # Material and geometric properties
        self.E = beam.material.E
        self.I = beam.I
        self.rho = beam.material.rho
        self.A = beam.area
        
    def element_stiffness_matrix(self, Le: float) -> np.ndarray:
        """
        Construct 4×4 element stiffness matrix for Euler-Bernoulli beam element.
        
        The stiffness matrix relates element nodal forces to nodal displacements
        through the elastic bending energy. For a 2-node beam element:
        
        DOFs: [w1, θ1, w2, θ2]
        
        k_e = (EI/Le³) × [[ 12,    6Le,  -12,    6Le  ],
                          [  6Le,  4Le², -6Le,  2Le²  ],
                          [ -12,  -6Le,   12,  -6Le   ],
                          [  6Le,  2Le², -6Le,  4Le²  ]]
        
        Parameters
        ----------
        Le : float
            Element length [m]
            
        Returns
        -------
        np.ndarray
            4×4 element stiffness matrix, shape (4, 4)
        """
        EI = self.E * self.I
        Le2 = Le * Le
        Le3 = Le2 * Le
        
        # Element stiffness matrix (symmetric)
        k_e = (EI / Le3) * np.array([
            [ 12,      6*Le,   -12,      6*Le  ],
            [ 6*Le,    4*Le2,  -6*Le,    2*Le2 ],
            [-12,     -6*Le,    12,     -6*Le  ],
            [ 6*Le,    2*Le2,  -6*Le,    4*Le2 ]
        ])
        
        return k_e
    
    def element_mass_matrix(self, Le: float) -> np.ndarray:
        """
        Construct 4×4 consistent mass matrix for Euler-Bernoulli beam element.
        
        The consistent mass matrix is derived from the kinetic energy using the
        same shape functions as the stiffness matrix. This gives better accuracy
        than a lumped mass matrix for modal analysis.
        
        DOFs: [w1, θ1, w2, θ2]
        
        m_e = (ρALe/420) × [[ 156,    22Le,    54,   -13Le  ],
                            [  22Le,   4Le²,   13Le,  -3Le²  ],
                            [  54,    13Le,   156,   -22Le  ],
                            [ -13Le,  -3Le²,  -22Le,   4Le²  ]]
        
        Parameters
        ----------
        Le : float
            Element length [m]
            
        Returns
        -------
        np.ndarray
            4×4 element mass matrix, shape (4, 4)
        """
        rho_A_Le = self.rho * self.A * Le
        Le2 = Le * Le
        
        # Consistent mass matrix (symmetric)
        m_e = (rho_A_Le / 420.0) * np.array([
            [ 156,      22*Le,    54,     -13*Le  ],
            [ 22*Le,    4*Le2,    13*Le,   -3*Le2 ],
            [ 54,       13*Le,   156,     -22*Le  ],
            [-13*Le,   -3*Le2,   -22*Le,    4*Le2 ]
        ])
        
        return m_e
    
    def assemble_global_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble global stiffness and mass matrices.
        
        This function iterates over all elements, computes element matrices,
        and assembles them into global matrices using the connectivity mapping.
        
        For element e connecting nodes n1 and n2:
        - Element DOFs: [w1, θ1, w2, θ2] (local numbering)
        - Global DOFs: [2*n1, 2*n1+1, 2*n2, 2*n2+1] (global numbering)
        
        The assembly process adds each element's contribution to the corresponding
        locations in the global matrices.
        
        Returns
        -------
        K : np.ndarray
            Global stiffness matrix, shape (total_dofs, total_dofs)
        M : np.ndarray
            Global mass matrix, shape (total_dofs, total_dofs)
        """
        # Initialize global matrices
        K = np.zeros((self.total_dofs, self.total_dofs))
        M = np.zeros((self.total_dofs, self.total_dofs))
        
        # Get element matrices (same for all elements since uniform beam)
        k_e = self.element_stiffness_matrix(self.element_length)
        m_e = self.element_mass_matrix(self.element_length)
        
        # Assemble element by element
        for elem in range(self.n_elements):
            # Node numbers for this element (0-indexed)
            node1 = elem
            node2 = elem + 1
            
            # Global DOF indices for this element
            # Each node has 2 DOFs: [w, θ]
            dof_indices = np.array([
                2*node1,      # w at node 1
                2*node1 + 1,  # θ at node 1
                2*node2,      # w at node 2
                2*node2 + 1   # θ at node 2
            ])
            
            # Add element matrices to global matrices
            # This is the assembly process: K_global[i,j] += k_element[local_i, local_j]
            for i in range(4):
                for j in range(4):
                    K[dof_indices[i], dof_indices[j]] += k_e[i, j]
                    M[dof_indices[i], dof_indices[j]] += m_e[i, j]
        
        return K, M
    
    def apply_boundary_conditions(self, K: np.ndarray, 
                                   M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions by eliminating constrained DOFs.
        
        Boundary conditions are enforced by removing rows and columns corresponding
        to constrained DOFs from the global matrices. This is called matrix partitioning.
        
        Boundary Conditions:
        - Cantilever: Fixed at x=0 (node 0): w=0, θ=0 → eliminate DOFs 0, 1
        - Simply-supported: Pinned at both ends: w=0 at nodes 0 and n_nodes-1 → eliminate DOFs 0, 2*n_nodes-2
        - Fixed-fixed: Clamped at both ends: w=0, θ=0 at both ends → eliminate DOFs 0, 1, 2*n_nodes-2, 2*n_nodes-1
        - Fixed-pinned: Fixed at x=0, pinned at x=L → eliminate DOFs 0, 1, 2*n_nodes-2
        
        Parameters
        ----------
        K : np.ndarray
            Global stiffness matrix before BC application
        M : np.ndarray
            Global mass matrix before BC application
            
        Returns
        -------
        K_reduced : np.ndarray
            Stiffness matrix with constrained DOFs removed
        M_reduced : np.ndarray
            Mass matrix with constrained DOFs removed
        """
        # Identify DOFs to eliminate based on boundary condition
        if self.bc == 'cantilever':
            # Fixed at x=0: eliminate DOFs 0 (w) and 1 (θ) at node 0
            constrained_dofs = [0, 1]
            
        elif self.bc == 'simply-supported':
            # Pinned at both ends: eliminate w at nodes 0 and n_nodes-1
            constrained_dofs = [0, self.total_dofs - 2]
            
        elif self.bc == 'fixed-fixed':
            # Clamped at both ends: eliminate w and θ at nodes 0 and n_nodes-1
            constrained_dofs = [0, 1, self.total_dofs - 2, self.total_dofs - 1]
            
        elif self.bc == 'fixed-pinned':
            # Fixed at x=0, pinned at x=L: eliminate w,θ at node 0 and w at node n_nodes-1
            constrained_dofs = [0, 1, self.total_dofs - 2]
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc}")
        
        # Create list of free (unconstrained) DOFs
        all_dofs = np.arange(self.total_dofs)
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
        
        # Partition matrices to keep only free DOFs
        # This is equivalent to solving the reduced system [K_ff]{u_f} = ω²[M_ff]{u_f}
        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        M_reduced = M[np.ix_(free_dofs, free_dofs)]
        
        # Store free DOFs for later reconstruction
        self.free_dofs = free_dofs
        self.constrained_dofs = constrained_dofs
        
        return K_reduced, M_reduced
    
    def solve(self, n_modes: int = 5) -> Dict:
        """
        Solve the generalized eigenvalue problem for natural frequencies and mode shapes.
        
        The equation of motion for free vibration is:
        [K]{φ} = ω²[M]{φ}
        
        This is a generalized eigenvalue problem where:
        - ω² are the eigenvalues (squared angular frequencies)
        - {φ} are the eigenvectors (mode shapes)
        
        We solve this using scipy.linalg.eigh which is optimized for symmetric matrices.
        
        Parameters
        ----------
        n_modes : int, optional
            Number of modes to extract (default: 5)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'frequencies_hz': Natural frequencies [Hz], shape (n_modes,)
            - 'frequencies_rad': Angular frequencies [rad/s], shape (n_modes,)
            - 'mode_shapes': Mode shapes mapped to full DOF set, shape (total_dofs, n_modes)
            - 'n_elements': Number of elements used
            - 'bc': Boundary condition
            - 'element_length': Element length [m]
        """
        # Assemble global matrices
        K, M = self.assemble_global_matrices()
        
        # Apply boundary conditions
        K_reduced, M_reduced = self.apply_boundary_conditions(K, M)
        
        # Solve generalized eigenvalue problem: K φ = ω² M φ
        # eigh returns eigenvalues in ascending order (perfect for modal analysis)
        eigenvalues, eigenvectors = eigh(K_reduced, M_reduced)
        
        # Extract first n_modes
        eigenvalues = eigenvalues[:n_modes]
        eigenvectors = eigenvectors[:, :n_modes]
        
        # Convert eigenvalues to frequencies
        omega_n = np.sqrt(eigenvalues)  # Angular frequencies [rad/s]
        freq_hz = omega_n / (2 * np.pi)  # Frequencies [Hz]
        
        # Map reduced mode shapes back to full DOF set
        mode_shapes_full = np.zeros((self.total_dofs, n_modes))
        mode_shapes_full[self.free_dofs, :] = eigenvectors
        
        # Normalize mode shapes (maximum displacement = 1)
        for i in range(n_modes):
            # Extract displacement DOFs (every other DOF, starting from 0)
            displacements = mode_shapes_full[::2, i]
            max_disp = np.max(np.abs(displacements))
            if max_disp > 0:
                mode_shapes_full[:, i] /= max_disp
        
        return {
            'frequencies_hz': freq_hz,
            'frequencies_rad': omega_n,
            'mode_shapes': mode_shapes_full,
            'n_elements': self.n_elements,
            'bc': self.bc,
            'element_length': self.element_length
        }
    
    def mesh_convergence_study(self, element_counts: List[int] = None,
                                n_modes: int = 5) -> Dict:
        """
        Perform mesh convergence study by solving with increasing mesh densities.
        
        This demonstrates that the FEM solution converges to the analytical solution
        as the mesh is refined (more elements → more accurate).
        
        Parameters
        ----------
        element_counts : list of int, optional
            List of element counts to test (default: [5, 10, 20, 50, 100, 200])
        n_modes : int, optional
            Number of modes to analyze (default: 5)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'element_counts': List of element counts tested
            - 'frequencies': Frequencies for each mesh, shape (n_meshes, n_modes)
            - 'errors': Relative errors vs analytical solution [%], shape (n_meshes, n_modes)
            - 'analytical_frequencies': Reference analytical frequencies [Hz]
        """
        if element_counts is None:
            element_counts = [5, 10, 20, 50, 100, 200]
        
        # Import analytical solver
        from .analytical import AnalyticalSolver
        
        # Compute analytical reference solution
        analytical = AnalyticalSolver(self.beam)
        freq_analytical = analytical.natural_frequencies(n_modes, self.bc)
        
        # Store results
        frequencies = []
        errors = []
        
        # Test each mesh density
        for n_elem in element_counts:
            # Create solver with this mesh
            solver = FEMSolver(self.beam, n_elements=n_elem, bc=self.bc)
            
            # Solve
            results = solver.solve(n_modes)
            freq_fem = results['frequencies_hz']
            
            # Compute relative error [%]
            error = 100 * np.abs(freq_fem - freq_analytical) / freq_analytical
            
            frequencies.append(freq_fem)
            errors.append(error)
        
        return {
            'element_counts': element_counts,
            'frequencies': np.array(frequencies),
            'errors': np.array(errors),
            'analytical_frequencies': freq_analytical
        }
