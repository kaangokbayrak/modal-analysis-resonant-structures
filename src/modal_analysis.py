"""
Advanced Modal Analysis Post-Processor

Provides frequency response functions, modal participation factors,
effective mass fractions, MAC matrices, steady-state and impulse responses
for structures characterised by natural frequencies and mode shapes.

Author: Kaan Gokbayrak, Purdue University
Date: December 2025 - January 2026
"""

import numpy as np
from typing import Tuple, Union, Optional


class ModalAnalyzer:
    """
    Advanced modal analysis post-processor.

    Parameters
    ----------
    frequencies_hz : np.ndarray
        Natural frequencies in Hz, shape (n_modes,).
    mode_shapes : np.ndarray
        Mode shape matrix, shape (n_dofs, n_modes).  Each column is a mode.
    mass_matrix : np.ndarray, optional
        Consistent or lumped mass matrix, shape (n_dofs, n_dofs).
        Defaults to the identity matrix when not provided.
    damping_ratios : np.ndarray, optional
        Viscous damping ratio per mode, shape (n_modes,).
        Defaults to 0.02 for all modes when not provided.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        mode_shapes: np.ndarray,
        mass_matrix: Optional[np.ndarray] = None,
        damping_ratios: Optional[np.ndarray] = None,
    ):
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        self.mode_shapes = np.asarray(mode_shapes, dtype=float)

        self.n_dofs = self.mode_shapes.shape[0]
        self.n_modes = self.mode_shapes.shape[1]

        if mass_matrix is None:
            self.mass_matrix = np.eye(self.n_dofs)
        else:
            self.mass_matrix = np.asarray(mass_matrix, dtype=float)

        if damping_ratios is None:
            self.damping_ratios = np.full(self.n_modes, 0.02)
        else:
            self.damping_ratios = np.asarray(damping_ratios, dtype=float)

        self.omega_n = 2.0 * np.pi * self.frequencies_hz

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _modal_masses(self) -> np.ndarray:
        """Return modal mass M_n = phi_n @ M @ phi_n for each mode."""
        M = self.mass_matrix
        modal_masses = np.array(
            [self.mode_shapes[:, n] @ M @ self.mode_shapes[:, n]
             for n in range(self.n_modes)]
        )
        return modal_masses

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute_frf(
        self,
        omega_range: np.ndarray,
        dof_out: int,
        dof_in: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the scalar frequency response function H(ω) between two DOFs.

        H(ω) = Σ_n  φ_n[out] * φ_n[in]
                     ─────────────────────────────────────────────────────
                     M_n * (ω_n² - ω² + 2j·ζ_n·ω_n·ω)

        Parameters
        ----------
        omega_range : np.ndarray
            Circular frequency array (rad/s), shape (n_freq,).
        dof_out : int
            Output (response) degree of freedom index.
        dof_in : int, optional
            Input (excitation) degree of freedom index.  Default 0.

        Returns
        -------
        omega_range : np.ndarray
            The same frequency array passed in, shape (n_freq,).
        H_complex : np.ndarray
            Complex FRF values, shape (n_freq,).
        """
        omega = np.asarray(omega_range, dtype=float)
        modal_masses = self._modal_masses()
        H = np.zeros(len(omega), dtype=complex)

        for n in range(self.n_modes):
            phi_out = self.mode_shapes[dof_out, n]
            phi_in = self.mode_shapes[dof_in, n]
            M_n = modal_masses[n]
            wn = self.omega_n[n]
            zn = self.damping_ratios[n]
            denom = M_n * (wn**2 - omega**2 + 2j * zn * wn * omega)
            H += phi_out * phi_in / denom

        return omega, H

    def compute_frf_matrix(
        self,
        omega_range: np.ndarray,
        dofs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the FRF matrix for a selected subset of DOFs.

        H[i, j, k] is the response at dofs[i] due to excitation at dofs[j]
        at circular frequency omega_range[k].

        Parameters
        ----------
        omega_range : np.ndarray
            Circular frequency array (rad/s), shape (n_freq,).
        dofs : array-like
            Indices of the selected degrees of freedom, length n_dofs_sel.

        Returns
        -------
        H_matrix : np.ndarray
            Complex FRF matrix, shape (n_dofs_sel, n_dofs_sel, n_freq).
        """
        omega = np.asarray(omega_range, dtype=float)
        dofs = np.asarray(dofs, dtype=int)
        n_sel = len(dofs)
        n_freq = len(omega)

        H_matrix = np.zeros((n_sel, n_sel, n_freq), dtype=complex)
        modal_masses = self._modal_masses()

        for i, dof_i in enumerate(dofs):
            for j, dof_j in enumerate(dofs):
                for n in range(self.n_modes):
                    phi_i = self.mode_shapes[dof_i, n]
                    phi_j = self.mode_shapes[dof_j, n]
                    M_n = modal_masses[n]
                    wn = self.omega_n[n]
                    zn = self.damping_ratios[n]
                    denom = M_n * (wn**2 - omega**2 + 2j * zn * wn * omega)
                    H_matrix[i, j, :] += phi_i * phi_j / denom

        return H_matrix

    def modal_participation_factors(
        self,
        direction: Union[str, np.ndarray] = 'vertical',
    ) -> np.ndarray:
        """
        Compute modal participation factors Γ_n = φ_n @ M @ r.

        Parameters
        ----------
        direction : str or np.ndarray
            Excitation direction vector.
            - ``'vertical'``: unit translational excitation (1 at even-indexed
              DOFs, 0 at odd-indexed / rotational DOFs).
            - np.ndarray: custom influence vector of length n_dofs.

        Returns
        -------
        gamma : np.ndarray
            Participation factors, shape (n_modes,).
        """
        if isinstance(direction, str):
            if direction == 'vertical':
                r = np.zeros(self.n_dofs)
                r[0::2] = 1.0
            else:
                raise ValueError(
                    f"Unknown direction '{direction}'. "
                    "Use 'vertical' or supply a custom np.ndarray."
                )
        else:
            r = np.asarray(direction, dtype=float)

        M = self.mass_matrix
        gamma = np.array(
            [self.mode_shapes[:, n] @ M @ r for n in range(self.n_modes)]
        )
        return gamma

    def effective_mass_fractions(
        self,
        direction: Union[str, np.ndarray] = 'vertical',
    ) -> np.ndarray:
        """
        Compute the effective mass fraction for each mode.

        Effective mass fraction = Γ_n² / (M_n · r @ M @ r)

        The sum over all modes is ≤ 1 (equals 1 when all modes are included).

        Parameters
        ----------
        direction : str or np.ndarray
            Excitation direction vector (same semantics as
            :meth:`modal_participation_factors`).

        Returns
        -------
        eff_mass : np.ndarray
            Effective mass fractions, shape (n_modes,).
        """
        if isinstance(direction, str):
            if direction == 'vertical':
                r = np.zeros(self.n_dofs)
                r[0::2] = 1.0
            else:
                raise ValueError(
                    f"Unknown direction '{direction}'. "
                    "Use 'vertical' or supply a custom np.ndarray."
                )
        else:
            r = np.asarray(direction, dtype=float)

        M = self.mass_matrix
        gamma = self.modal_participation_factors(r)
        modal_masses = self._modal_masses()
        total_mass = r @ M @ r

        eff_mass = np.where(
            modal_masses != 0.0,
            gamma**2 / (modal_masses * total_mass),
            0.0,
        )
        return eff_mass

    def compute_mac_matrix(
        self,
        other_mode_shapes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Modal Assurance Criterion (MAC) matrix.

        MAC[i, j] = |φ_i · ψ_j|² / ((φ_i · φ_i) · (ψ_j · ψ_j))

        Parameters
        ----------
        other_mode_shapes : np.ndarray
            Second set of mode shapes, shape (n_dofs, n_modes2).

        Returns
        -------
        mac : np.ndarray
            MAC matrix, shape (n_modes, n_modes2).
        """
        psi = np.asarray(other_mode_shapes, dtype=float)
        n_modes2 = psi.shape[1]
        mac = np.zeros((self.n_modes, n_modes2))

        for i in range(self.n_modes):
            phi_i = self.mode_shapes[:, i]
            norm_i = phi_i @ phi_i
            for j in range(n_modes2):
                psi_j = psi[:, j]
                norm_j = psi_j @ psi_j
                cross = phi_i @ psi_j
                mac[i, j] = cross**2 / (norm_i * norm_j)

        return mac

    def steady_state_response(
        self,
        omega_range: np.ndarray,
        force_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the steady-state displacement response to a harmonic force.

        X(ω) = Σ_n  φ_n · (φ_n @ F)
                     ─────────────────────────────────────────────────────
                     M_n · (ω_n² - ω² + 2j·ζ_n·ω_n·ω)

        Parameters
        ----------
        omega_range : np.ndarray
            Circular frequency array (rad/s), shape (n_freq,).
        force_vector : np.ndarray
            Complex force amplitude vector, shape (n_dofs,).

        Returns
        -------
        X : np.ndarray
            Complex displacement response, shape (n_dofs, n_freq).
        """
        omega = np.asarray(omega_range, dtype=float)
        F = np.asarray(force_vector, dtype=complex)
        n_freq = len(omega)
        X = np.zeros((self.n_dofs, n_freq), dtype=complex)
        modal_masses = self._modal_masses()

        for n in range(self.n_modes):
            phi_n = self.mode_shapes[:, n]
            M_n = modal_masses[n]
            wn = self.omega_n[n]
            zn = self.damping_ratios[n]
            modal_force = phi_n @ F
            denom = M_n * (wn**2 - omega**2 + 2j * zn * wn * omega)
            X += np.outer(phi_n, modal_force / denom)

        return X

    def impulse_response(
        self,
        t: np.ndarray,
        dof_out: int,
        dof_in: int = 0,
    ) -> np.ndarray:
        """
        Compute the unit-impulse response function h(t) between two DOFs.

        h(t) = Σ_n  φ_n[out] · φ_n[in]
                     ───────────────────── · exp(−ζ_n·ω_n·t) · sin(ω_d_n·t)
                     M_n · ω_d_n

        where ω_d_n = ω_n · √(1 − ζ_n²).

        Parameters
        ----------
        t : np.ndarray
            Time array (seconds), shape (n_t,).  Must contain non-negative values.
        dof_out : int
            Output (response) degree of freedom index.
        dof_in : int, optional
            Input (excitation) degree of freedom index.  Default 0.

        Returns
        -------
        h : np.ndarray
            Real impulse response, shape (n_t,).
        """
        t = np.asarray(t, dtype=float)
        modal_masses = self._modal_masses()
        h = np.zeros(len(t))

        for n in range(self.n_modes):
            phi_out = self.mode_shapes[dof_out, n]
            phi_in = self.mode_shapes[dof_in, n]
            M_n = modal_masses[n]
            wn = self.omega_n[n]
            zn = self.damping_ratios[n]
            wd = wn * np.sqrt(max(1.0 - zn**2, 0.0))
            if M_n == 0.0 or wd == 0.0:
                continue
            h += (phi_out * phi_in / (M_n * wd)
                  * np.exp(-zn * wn * t)
                  * np.sin(wd * t))

        return h

    def summary(self) -> dict:
        """
        Return a summary dictionary of key modal properties.

        Returns
        -------
        info : dict
            Keys:
            - ``'frequencies_hz'``: np.ndarray, shape (n_modes,)
            - ``'participation_factors'``: np.ndarray, shape (n_modes,)
            - ``'effective_mass_fractions'``: np.ndarray, shape (n_modes,)
            - ``'cumulative_mass_fraction'``: float
        """
        pf = self.modal_participation_factors()
        emf = self.effective_mass_fractions()
        return {
            'frequencies_hz': self.frequencies_hz.copy(),
            'participation_factors': pf,
            'effective_mass_fractions': emf,
            'cumulative_mass_fraction': float(np.sum(emf)),
        }


# ---------------------------------------------------------------------------
# Module-level factory functions
# ---------------------------------------------------------------------------

def from_fem_results(
    fem_results: dict,
    mass_matrix: Optional[np.ndarray] = None,
    damping_ratios: Optional[np.ndarray] = None,
) -> ModalAnalyzer:
    """
    Create a :class:`ModalAnalyzer` from the results dict of
    :meth:`FEMSolver.solve`.

    Parameters
    ----------
    fem_results : dict
        Dictionary returned by ``FEMSolver.solve()``.  Must contain:
        ``'frequencies_hz'`` and ``'mode_shapes'``.
    mass_matrix : np.ndarray, optional
        Global mass matrix.  Forwarded to :class:`ModalAnalyzer`.
    damping_ratios : np.ndarray, optional
        Per-mode damping ratios.  Forwarded to :class:`ModalAnalyzer`.

    Returns
    -------
    ModalAnalyzer
    """
    return ModalAnalyzer(
        frequencies_hz=fem_results['frequencies_hz'],
        mode_shapes=fem_results['mode_shapes'],
        mass_matrix=mass_matrix,
        damping_ratios=damping_ratios,
    )


def from_analytical_results(
    analytical_results: dict,
    damping_ratios: Optional[np.ndarray] = None,
) -> ModalAnalyzer:
    """
    Create a :class:`ModalAnalyzer` from the results dict of
    :meth:`AnalyticalSolver.full_analysis`.

    Parameters
    ----------
    analytical_results : dict
        Dictionary returned by ``AnalyticalSolver.full_analysis()``.  Must
        contain ``'frequencies_hz'`` and ``'mode_shapes'``.
    damping_ratios : np.ndarray, optional
        Per-mode damping ratios.  Forwarded to :class:`ModalAnalyzer`.

    Returns
    -------
    ModalAnalyzer
    """
    return ModalAnalyzer(
        frequencies_hz=analytical_results['frequencies_hz'],
        mode_shapes=analytical_results['mode_shapes'],
        damping_ratios=damping_ratios,
    )
