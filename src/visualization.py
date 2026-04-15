"""
Visualization module for modal analysis of resonant structures.

This module provides comprehensive plotting capabilities for visualizing
mode shapes, frequency analysis, convergence studies, and parametric studies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import seaborn as sns
from scipy.signal import hilbert


class Visualizer:
    """
    Comprehensive visualization class for modal analysis results.
    
    This class provides methods for creating publication-quality plots
    including mode shapes, FFT spectra, convergence studies, and
    parametric analyses.
    
    Parameters
    ----------
    save_dir : str or Path, optional
        Directory to save figures. Default is 'results/figures'
    style : str, optional
        Matplotlib style to use. Default is 'seaborn-v0_8-darkgrid'
    dpi : int, optional
        Resolution for saved figures. Default is 300
        
    Attributes
    ----------
    save_dir : Path
        Directory where figures are saved
    dpi : int
        Resolution for saved figures
        
    Examples
    --------
    >>> viz = Visualizer(save_dir='results/figures')
    >>> viz.plot_mode_shapes(x, mode_shapes, frequencies)
    >>> viz.plot_fft_spectrum(frequencies, amplitudes)
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path] = 'results/figures',
        style: str = 'seaborn-v0_8-darkgrid',
        dpi: int = 300
    ):
        """Initialize the Visualizer with style settings."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default seaborn style
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                # Use seaborn directly if style not available
                sns.set_style('darkgrid')
        
        # Set seaborn context for publication quality
        sns.set_context('paper', font_scale=1.2)
        
    def plot_mode_shapes(
        self,
        x: np.ndarray,
        mode_shapes: np.ndarray,
        frequencies: np.ndarray,
        n_modes: int = 4,
        length: float = 1.0,
        filename: str = 'mode_shapes.png'
    ) -> None:
        """
        Plot mode shapes overlaid on beam geometry.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates along beam (m)
        mode_shapes : np.ndarray
            Mode shape matrix (n_points x n_modes)
        frequencies : np.ndarray
            Natural frequencies (Hz)
        n_modes : int, optional
            Number of modes to plot. Default is 4
        length : float, optional
            Beam length (m). Default is 1.0
        filename : str, optional
            Output filename. Default is 'mode_shapes.png'
            
        Notes
        -----
        Mode shapes are normalized and offset vertically for clarity.
        Each mode is labeled with its frequency.
        """
        n_modes = min(n_modes, mode_shapes.shape[1], len(frequencies))

        # Per-mode colour palette (warm tones for positive, cool for negative)
        palette = plt.cm.tab10(np.linspace(0, 0.9, n_modes))

        fig, axes = plt.subplots(n_modes, 1, figsize=(11, 2.8 * n_modes),
                                 gridspec_kw={'hspace': 0.45},
                                 constrained_layout=False)
        if n_modes == 1:
            axes = [axes]

        fig.suptitle('Mode Shapes — Euler-Bernoulli Beam',
                     fontsize=14, fontweight='bold', y=1.01)

        for i, ax in enumerate(axes):
            color = palette[i]

            # Normalize mode shape
            mode = mode_shapes[:, i]
            mode_normalized = mode / np.max(np.abs(mode))

            # Beam centreline
            ax.plot([0, length], [0, 0], 'k--', linewidth=0.8, alpha=0.4,
                    zorder=1)

            # Filled area: positive (warm) and negative (cool) separately
            ax.fill_between(x, 0, mode_normalized,
                            where=(mode_normalized >= 0),
                            alpha=0.25, color=color, zorder=2)
            ax.fill_between(x, 0, mode_normalized,
                            where=(mode_normalized < 0),
                            alpha=0.25, color=color, zorder=2)

            # Mode shape curve
            ax.plot(x, mode_normalized, '-', linewidth=2.2,
                    color=color, label=f'Mode {i+1}', zorder=3)

            # Node points (zero crossings)
            signs = np.sign(mode_normalized)
            zero_crossings = np.where(np.diff(signs) != 0)[0]
            for zc in zero_crossings:
                x_node = x[zc] + (x[zc + 1] - x[zc]) * (
                    -mode_normalized[zc] / (mode_normalized[zc + 1] - mode_normalized[zc])
                )
                ax.plot(x_node, 0, 'o', color='white', markersize=7,
                        markeredgecolor=color, markeredgewidth=1.8, zorder=5)

            # Fixed-end wall indicator
            ax.add_patch(Rectangle((-0.035 * length, -1.15), 0.035 * length,
                                   2.3, facecolor='#555555', alpha=0.7,
                                   zorder=4, clip_on=False))

            # Boundary lines
            ax.axvline(0, color='k', linewidth=2.5, zorder=4)
            ax.axvline(length, color='k', linewidth=1.2, alpha=0.5, zorder=4)

            # Frequency label box
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color,
                              alpha=0.15, edgecolor=color)
            ax.text(0.98, 0.97,
                    f'f({i+1}) = {frequencies[i]:.2f} Hz',
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    ha='right', va='top', bbox=bbox_props, color=color)

            ax.set_ylabel('Norm. Disp.', fontsize=9)
            ax.set_title(f'Mode {i + 1}', fontsize=11, fontweight='bold',
                         color=color, pad=4)
            ax.set_ylim(-1.3, 1.3)
            ax.set_xlim(-0.06 * length, 1.05 * length)
            ax.grid(True, alpha=0.25, linestyle=':')
            ax.tick_params(axis='both', labelsize=9)

            if i == n_modes - 1:
                ax.set_xlabel('Position along beam (m)', fontsize=10)
            else:
                ax.set_xticklabels([])

        plt.subplots_adjust(hspace=0.45)
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def animate_mode_shapes(
        self,
        x: np.ndarray,
        mode_shapes: np.ndarray,
        frequencies: np.ndarray,
        mode_index: int = 0,
        n_frames: int = 60,
        length: float = 1.0,
        filename: str = 'mode_animation.gif'
    ) -> None:
        """
        Create animated GIF of vibrating mode shapes.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates along beam (m)
        mode_shapes : np.ndarray
            Mode shape matrix (n_points x n_modes)
        frequencies : np.ndarray
            Natural frequencies (Hz)
        mode_index : int, optional
            Index of mode to animate. Default is 0
        n_frames : int, optional
            Number of frames in animation. Default is 60
        length : float, optional
            Beam length (m). Default is 1.0
        filename : str, optional
            Output filename. Default is 'mode_animation.gif'
            
        Notes
        -----
        The animation shows one complete oscillation cycle of the mode.
        Frame rate is adjusted to show smooth motion.
        """
        mode = mode_shapes[:, mode_index]
        mode_normalized = mode / np.max(np.abs(mode))
        freq = frequencies[mode_index]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Set up the plot
        line, = ax.plot([], [], 'b-', linewidth=2)
        fill = ax.fill_between(x, 0, mode_normalized, alpha=0.3)
        ax.plot([0, length], [0, 0], 'k--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linewidth=2)
        ax.axvline(length, color='k', linewidth=2)
        
        ax.set_xlim(-0.05 * length, 1.05 * length)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Position (m)', fontsize=12)
        ax.set_ylabel('Normalized Displacement', fontsize=12)
        ax.set_title(f'Mode {mode_index+1} Animation: f = {freq:.2f} Hz',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(frame):
            t = frame / n_frames  # Time normalized to one period
            phase = 2 * np.pi * t
            y = mode_normalized * np.sin(phase)
            line.set_data(x, y)
            time_text.set_text(f't/T = {t:.2f}')
            return line, time_text
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=50, blit=True
        )
        
        anim.save(self.save_dir / filename, writer='pillow', fps=20, dpi=100)
        plt.close()
        
    def plot_fft_spectrum(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        theoretical_freqs: Optional[np.ndarray] = None,
        peak_freqs: Optional[np.ndarray] = None,
        filename: str = 'fft_spectrum.png',
        freq_range: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot FFT spectrum with peaks and theoretical frequencies.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array (Hz)
        amplitudes : np.ndarray
            FFT amplitude spectrum
        theoretical_freqs : np.ndarray, optional
            Theoretical natural frequencies to overlay
        peak_freqs : np.ndarray, optional
            Detected peak frequencies
        filename : str, optional
            Output filename. Default is 'fft_spectrum.png'
        freq_range : tuple of float, optional
            Frequency range to display (f_min, f_max)
            
        Notes
        -----
        The spectrum is plotted on a logarithmic scale for clarity.
        Peaks are marked with vertical lines and annotations.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply frequency range filter
        if freq_range is not None:
            mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            frequencies = frequencies[mask]
            amplitudes = amplitudes[mask]
        
        # Plot FFT spectrum
        ax.semilogy(frequencies, amplitudes, 'b-', linewidth=1.5, label='FFT Spectrum')
        
        # Mark detected peaks
        if peak_freqs is not None:
            for i, f in enumerate(peak_freqs):
                if freq_range is None or (freq_range[0] <= f <= freq_range[1]):
                    ax.axvline(f, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.text(f, ax.get_ylim()[1] * 0.5, f'  f{i+1}={f:.1f} Hz',
                           rotation=90, verticalalignment='bottom', fontsize=9)
        
        # Mark theoretical frequencies
        if theoretical_freqs is not None:
            for i, f in enumerate(theoretical_freqs):
                if freq_range is None or (freq_range[0] <= f <= freq_range[1]):
                    ax.axvline(f, color='green', linestyle=':', linewidth=2, alpha=0.7)
                    ax.text(f, ax.get_ylim()[1] * 0.7, f'  Theory={f:.1f} Hz',
                           rotation=90, verticalalignment='bottom', fontsize=9,
                           color='green')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude (log scale)', fontsize=12)
        ax.set_title('FFT Spectrum Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_mesh_convergence(
        self,
        n_elements: np.ndarray,
        errors: np.ndarray,
        labels: Optional[List[str]] = None,
        filename: str = 'mesh_convergence.png',
        log_scale: bool = True
    ) -> None:
        """
        Plot convergence of frequencies vs number of elements.
        
        Parameters
        ----------
        n_elements : np.ndarray
            Array of element counts
        errors : np.ndarray
            Error values (can be 1D or 2D for multiple modes)
        labels : list of str, optional
            Labels for each error curve
        filename : str, optional
            Output filename. Default is 'mesh_convergence.png'
        log_scale : bool, optional
            Use log scale for y-axis. Default is True
            
        Notes
        -----
        Errors are typically computed as relative percentage errors
        compared to analytical solutions or fine mesh reference.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle both 1D and 2D error arrays
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)
        
        # Plot each error curve
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
        colors = plt.cm.tab10(np.linspace(0, 1, errors.shape[1]))
        
        for i in range(errors.shape[1]):
            label = labels[i] if labels and i < len(labels) else f'Mode {i+1}'
            ax.plot(n_elements, errors[:, i], marker=markers[i % len(markers)],
                   linewidth=2, markersize=8, label=label, color=colors[i])
        
        if log_scale:
            ax.set_yscale('log')
            
        ax.set_xlabel('Number of Elements', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title('Mesh Convergence Study', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best')
        
        # Add convergence reference line
        if log_scale and len(n_elements) > 1:
            # Show theoretical O(h^2) convergence
            x_ref = np.array([n_elements[0], n_elements[-1]])
            y_ref = errors[0, 0] * (x_ref[0] / x_ref) ** 2
            ax.plot(x_ref, y_ref, 'k--', linewidth=1, alpha=0.5,
                   label='O(h²) reference')
            ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_frequency_comparison(
        self,
        frequencies_dict: Dict[str, np.ndarray],
        mode_numbers: Optional[np.ndarray] = None,
        filename: str = 'frequency_comparison.png'
    ) -> None:
        """
        Create grouped bar chart comparing frequencies from different methods.
        
        Parameters
        ----------
        frequencies_dict : dict
            Dictionary mapping method names to frequency arrays
            e.g., {'Analytical': [f1, f2, ...], 'FEM': [...], 'FFT': [...]}
        mode_numbers : np.ndarray, optional
            Mode numbers for x-axis. If None, uses 1, 2, 3, ...
        filename : str, optional
            Output filename. Default is 'frequency_comparison.png'
            
        Examples
        --------
        >>> freqs = {'Analytical': [10.2, 40.5], 'FEM': [10.1, 40.8]}
        >>> viz.plot_frequency_comparison(freqs)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine number of modes
        n_modes = max(len(freqs) for freqs in frequencies_dict.values())
        if mode_numbers is None:
            mode_numbers = np.arange(1, n_modes + 1)
        
        # Set up bar positions
        n_methods = len(frequencies_dict)
        bar_width = 0.8 / n_methods
        x = np.arange(n_modes)
        
        # Plot bars for each method
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        for i, (method, freqs) in enumerate(frequencies_dict.items()):
            offset = (i - n_methods / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, freqs[:n_modes], bar_width,
                  label=method, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for j, freq in enumerate(freqs[:n_modes]):
                ax.text(x[j] + offset, freq, f'{freq:.1f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Mode Number', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title('Natural Frequency Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_numbers)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_parametric_sweep(
        self,
        parameter_values: np.ndarray,
        frequencies: np.ndarray,
        parameter_name: str = 'Parameter',
        parameter_unit: str = '',
        mode_labels: Optional[List[str]] = None,
        hazard_range: Optional[Tuple[float, float]] = None,
        filename: str = 'parametric_sweep.png'
    ) -> None:
        """
        Plot frequencies vs a varying parameter.
        
        Parameters
        ----------
        parameter_values : np.ndarray
            Values of the parameter being swept
        frequencies : np.ndarray
            Frequency matrix (n_parameters x n_modes)
        parameter_name : str, optional
            Name of the parameter. Default is 'Parameter'
        parameter_unit : str, optional
            Unit of the parameter
        mode_labels : list of str, optional
            Labels for each mode
        hazard_range : tuple of float, optional
            Frequency band to highlight as a resonance-risk zone (f_low, f_high)
        filename : str, optional
            Output filename. Default is 'parametric_sweep.png'
            
        Examples
        --------
        >>> lengths = np.linspace(0.5, 2.0, 50)
        >>> freqs = np.array([...])  # shape (50, 4)
        >>> viz.plot_parametric_sweep(lengths, freqs, 'Length', 'm')
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle both 1D and 2D frequency arrays
        if frequencies.ndim == 1:
            frequencies = frequencies.reshape(-1, 1)
        
        # Hazard band (resonance risk zone)
        if hazard_range is not None:
            f_low, f_high = hazard_range
            ax.axhspan(f_low, f_high, color='red', alpha=0.12, zorder=0,
                       label=f'Resonance risk zone ({f_low}–{f_high} Hz)')
            ax.axhline(f_low, color='red', linewidth=1.2, linestyle='--', alpha=0.6)
            ax.axhline(f_high, color='red', linewidth=1.2, linestyle='--', alpha=0.6)

        # Plot each mode
        colors = plt.cm.tab10(np.linspace(0, 0.9, frequencies.shape[1]))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
        
        for i in range(frequencies.shape[1]):
            label = mode_labels[i] if mode_labels and i < len(mode_labels) else f'Mode {i+1}'
            ax.plot(parameter_values, frequencies[:, i],
                   marker=markers[i % len(markers)], markersize=6,
                   linewidth=2, label=label, color=colors[i], alpha=0.9)
        
        # Labels and formatting
        x_label = f'{parameter_name}'
        if parameter_unit:
            x_label += f' ({parameter_unit})'
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Natural Frequency (Hz)', fontsize=12)
        ax.set_title(f'Parametric Study: {parameter_name} Variation',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_design_space(
        self,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
        frequency_map: np.ndarray,
        param1_name: str = 'Parameter 1',
        param2_name: str = 'Parameter 2',
        safe_region: Optional[Tuple[float, float]] = None,
        filename: str = 'design_space.png'
    ) -> None:
        """
        Create 2D contour plot of design space with safe/unsafe regions.
        
        Parameters
        ----------
        param1_values : np.ndarray
            First parameter values (x-axis)
        param2_values : np.ndarray
            Second parameter values (y-axis)
        frequency_map : np.ndarray
            2D array of frequencies (n_param2 x n_param1)
        param1_name : str, optional
            Name of first parameter
        param2_name : str, optional
            Name of second parameter
        safe_region : tuple of float, optional
            Safe frequency range (f_min, f_max) to highlight
        filename : str, optional
            Output filename. Default is 'design_space.png'
            
        Notes
        -----
        Safe regions are areas where the frequency falls outside
        specified constraints (e.g., avoiding resonance with excitation).
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid
        P1, P2 = np.meshgrid(param1_values, param2_values)
        
        # Contour plot
        levels = 20
        contour = ax.contourf(P1, P2, frequency_map, levels=levels,
                             cmap='viridis', alpha=0.8)
        contour_lines = ax.contour(P1, P2, frequency_map, levels=levels,
                                   colors='k', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f Hz')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Natural Frequency (Hz)', fontsize=12)
        
        # Highlight safe/unsafe regions
        if safe_region is not None:
            f_min, f_max = safe_region
            # Unsafe region (within the range)
            unsafe_mask = (frequency_map >= f_min) & (frequency_map <= f_max)
            if np.any(unsafe_mask):
                ax.contourf(P1, P2, unsafe_mask.astype(float),
                           levels=[0.5, 1.5], colors='red', alpha=0.3)
                ax.contour(P1, P2, unsafe_mask.astype(float),
                          levels=[0.5], colors='red', linewidths=2,
                          linestyles='--')
        
        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)
        ax.set_title('Design Space Exploration', fontsize=14, fontweight='bold')
        
        # Add legend for safe/unsafe regions
        if safe_region is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.3, label='Unsafe Region'),
                Patch(facecolor='white', label='Safe Region')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_damping_estimation(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        peak_freq: float,
        damping_ratio: float,
        bandwidth: Tuple[float, float],
        filename: str = 'damping_estimation.png'
    ) -> None:
        """
        Illustrate half-power bandwidth method for damping estimation.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array (Hz)
        amplitudes : np.ndarray
            Amplitude spectrum
        peak_freq : float
            Peak frequency (Hz)
        damping_ratio : float
            Estimated damping ratio (dimensionless)
        bandwidth : tuple of float
            Half-power frequencies (f1, f2)
        filename : str, optional
            Output filename. Default is 'damping_estimation.png'
            
        Notes
        -----
        The half-power bandwidth method estimates damping from the
        frequency range where amplitude drops to 1/sqrt(2) of peak.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot amplitude spectrum
        ax.plot(frequencies, amplitudes, 'b-', linewidth=2, label='Amplitude Spectrum')
        
        # Mark peak
        peak_amp = np.max(amplitudes)
        ax.plot(peak_freq, peak_amp, 'ro', markersize=10,
               label=f'Peak: f₀ = {peak_freq:.2f} Hz')
        
        # Mark half-power points
        half_power_amp = peak_amp / np.sqrt(2)
        f1, f2 = bandwidth
        
        ax.axhline(half_power_amp, color='green', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Half-Power Level')
        ax.plot([f1, f2], [half_power_amp, half_power_amp],
               'go', markersize=8)
        
        # Add vertical lines at bandwidth frequencies
        ax.axvline(f1, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(f2, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        
        # Add annotations
        ax.annotate(f'f1 = {f1:.2f} Hz', xy=(f1, half_power_amp),
                   xytext=(f1 - 5, half_power_amp * 1.2),
                   fontsize=10, ha='right',
                   arrowprops=dict(arrowstyle='->', color='orange'))
        ax.annotate(f'f2 = {f2:.2f} Hz', xy=(f2, half_power_amp),
                   xytext=(f2 + 5, half_power_amp * 1.2),
                   fontsize=10, ha='left',
                   arrowprops=dict(arrowstyle='->', color='orange'))
        
        # Add text box with results
        textstr = f'Bandwidth Method:\n'
        textstr += f'Δf = {f2 - f1:.2f} Hz\n'
        textstr += f'ζ = {damping_ratio:.4f}\n'
        textstr += f'Q = {1/(2*damping_ratio):.1f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Damping Estimation: Half-Power Bandwidth Method',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_validation_table(
        self,
        data: Dict[str, List[float]],
        column_labels: List[str],
        title: str = 'Validation Results',
        filename: str = 'validation_table.png'
    ) -> None:
        """
        Create formatted table figure for comparison of results.
        
        Parameters
        ----------
        data : dict
            Dictionary mapping row labels to data lists
        column_labels : list of str
            Labels for table columns
        title : str, optional
            Table title
        filename : str, optional
            Output filename. Default is 'validation_table.png'
            
        Examples
        --------
        >>> data = {
        ...     'Mode 1': [10.2, 10.1, 10.3, 1.0],
        ...     'Mode 2': [40.5, 40.8, 40.2, 0.7]
        ... }
        >>> labels = ['Analytical (Hz)', 'FEM (Hz)', 'FFT (Hz)', 'Error (%)']
        >>> viz.plot_validation_table(data, labels)
        """
        fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.6)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        row_labels = list(data.keys())
        table_data = [data[label] for label in row_labels]
        
        # Format numbers
        formatted_data = []
        for row in table_data:
            formatted_row = []
            for val in row:
                if isinstance(val, (int, float)):
                    formatted_row.append(f'{val:.2f}')
                else:
                    formatted_row.append(str(val))
            formatted_data.append(formatted_row)
        
        # Create table
        table = ax.table(cellText=formatted_data,
                        rowLabels=row_labels,
                        colLabels=column_labels,
                        cellLoc='center',
                        rowLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(column_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Color row labels
        for i in range(len(row_labels)):
            cell = table[(i + 1, -1)]
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
        
        # Alternate row colors
        for i in range(len(row_labels)):
            for j in range(len(column_labels)):
                cell = table[(i + 1, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F5F5F5')
                else:
                    cell.set_facecolor('#FFFFFF')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_window_comparison(
        self,
        frequencies: np.ndarray,
        spectra: Dict[str, np.ndarray],
        freq_range: Optional[Tuple[float, float]] = None,
        filename: str = 'window_comparison.png'
    ) -> None:
        """
        Compare FFT results with different window functions.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array (Hz)
        spectra : dict
            Dictionary mapping window names to amplitude spectra
            e.g., {'Rectangular': [...], 'Hanning': [...], 'Hamming': [...]}
        freq_range : tuple of float, optional
            Frequency range to display (f_min, f_max)
        filename : str, optional
            Output filename. Default is 'window_comparison.png'
            
        Notes
        -----
        Different window functions trade off between frequency resolution
        and spectral leakage. This plot helps visualize these trade-offs.
        """
        fig, axes = plt.subplots(len(spectra), 1,
                                figsize=(10, 3 * len(spectra)),
                                sharex=True)
        
        if len(spectra) == 1:
            axes = [axes]
        
        # Apply frequency range filter
        if freq_range is not None:
            mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            freq_plot = frequencies[mask]
        else:
            freq_plot = frequencies
            mask = slice(None)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectra)))
        
        for i, (window_name, spectrum) in enumerate(spectra.items()):
            ax = axes[i]
            spectrum_plot = spectrum[mask]
            
            # Plot spectrum
            ax.semilogy(freq_plot, spectrum_plot, linewidth=1.5,
                       color=colors[i], label=window_name)
            
            # Find and mark peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(spectrum_plot, height=np.max(spectrum_plot) * 0.1)
            if len(peaks) > 0:
                ax.plot(freq_plot[peaks], spectrum_plot[peaks],
                       'ro', markersize=6)
            
            ax.set_ylabel(f'{window_name}\nAmplitude', fontsize=10)
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(loc='upper right', fontsize=10)
            
            if i == len(spectra) - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=12)
        
        fig.suptitle('Window Function Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()


    def plot_frf(
        self,
        omega: np.ndarray,
        H: np.ndarray,
        frequencies_hz: Optional[np.ndarray] = None,
        title: str = 'Frequency Response Function (FRF)',
        filename: str = 'frf_plot.png'
    ) -> None:
        """
        Plot FRF magnitude (dB) and phase (degrees) as a two-panel figure.

        Parameters
        ----------
        omega : np.ndarray
            Circular frequency array (rad/s), shape (n_freq,).
        H : np.ndarray
            Complex FRF values, shape (n_freq,).
        frequencies_hz : np.ndarray, optional
            Natural frequencies (Hz) to mark as resonance lines.
        title : str, optional
            Figure title. Default is 'Frequency Response Function (FRF)'.
        filename : str, optional
            Output filename. Default is 'frf_plot.png'.

        Notes
        -----
        Magnitude is displayed in decibels: 20·log₁₀|H(ω)|.
        Phase wraps to [−180°, +180°].
        """
        freq_hz = omega / (2.0 * np.pi)
        magnitude_db = 20.0 * np.log10(np.abs(H) + 1e-30)
        phase_deg = np.degrees(np.angle(H))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'hspace': 0.08, 'height_ratios': [3, 2]},
            constrained_layout=False
        )
        fig.subplots_adjust(hspace=0.08)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # ── Magnitude ──────────────────────────────────────────────────────
        ax1.plot(freq_hz, magnitude_db, color='#1565C0', linewidth=1.8,
                 label='|H(ω)| [dB]')
        ax1.fill_between(freq_hz, magnitude_db, magnitude_db.min() - 5,
                         alpha=0.08, color='#1565C0')
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.grid(True, alpha=0.3, which='both', linestyle=':')

        if frequencies_hz is not None:
            for i, f in enumerate(frequencies_hz):
                ax1.axvline(f, color='#C62828', linestyle='--',
                            linewidth=1.2, alpha=0.75)
                idx = int(np.argmin(np.abs(freq_hz - f)))
                db_val = magnitude_db[idx]
                ax1.annotate(
                    f'  f{i+1}={f:.0f} Hz',
                    xy=(f, db_val),
                    xytext=(f + freq_hz[-1] * 0.015, db_val + 2),
                    fontsize=8, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=0.8),
                    clip_on=True
                )

        resonance_patch = Line2D([0], [0], color='#C62828', linewidth=1.5,
                                 linestyle='--', label='Resonance')
        ax1.legend(handles=[
            Line2D([0], [0], color='#1565C0', linewidth=2, label='|H(ω)| [dB]'),
            resonance_patch
        ], fontsize=10, loc='upper right')

        # ── Phase ───────────────────────────────────────────────────────────
        ax2.plot(freq_hz, phase_deg, color='#2E7D32', linewidth=1.5,
                 label='Phase H(\u03c9) [\u00b0]')
        ax2.axhline(0, color='k', linewidth=0.6, alpha=0.4)
        ax2.set_yticks([-180, -90, 0, 90, 180])
        ax2.set_ylim(-200, 200)
        ax2.set_ylabel('Phase (°)', fontsize=12)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(fontsize=10, loc='upper right')

        if frequencies_hz is not None:
            for f in frequencies_hz:
                ax2.axvline(f, color='#C62828', linestyle='--',
                            linewidth=1.2, alpha=0.6)

        fig.subplots_adjust(bottom=0.10, top=0.92)
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_mac_matrix(
        self,
        mac: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = 'Modal Assurance Criterion (MAC)',
        filename: str = 'mac_matrix.png'
    ) -> None:
        """
        Plot the MAC matrix as an annotated colour-coded heatmap.

        Parameters
        ----------
        mac : np.ndarray
            MAC matrix, shape (n_modes_row, n_modes_col).  Values in [0, 1].
        row_labels : list of str, optional
            Labels for rows (first set of mode shapes).
        col_labels : list of str, optional
            Labels for columns (second set of mode shapes).
        title : str, optional
            Figure title.
        filename : str, optional
            Output filename. Default is 'mac_matrix.png'.

        Notes
        -----
        Diagonal entries equal to 1 indicate perfect correlation between mode
        pairs.  Off-diagonal entries near 0 confirm mode orthogonality.
        """
        n_rows, n_cols = mac.shape
        if row_labels is None:
            row_labels = [f'Mode {i+1}' for i in range(n_rows)]
        if col_labels is None:
            col_labels = [f'Mode {i+1}' for i in range(n_cols)]

        fig, ax = plt.subplots(figsize=(max(6, n_cols + 2), max(5, n_rows + 1)))

        im = ax.imshow(mac, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto',
                       interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label('MAC Value', fontsize=11)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        # Cell annotations
        for i in range(n_rows):
            for j in range(n_cols):
                val = mac[i, j]
                text_color = 'white' if (val < 0.25 or val > 0.80) else 'black'
                ax.text(j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_xlabel('FEM Mode Shapes', fontsize=12)
        ax.set_ylabel('Analytical Mode Shapes', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_impulse_response(
        self,
        t: np.ndarray,
        h: np.ndarray,
        frequencies_hz: Optional[np.ndarray] = None,
        title: str = 'Unit Impulse Response Function h(t)',
        filename: str = 'impulse_response.png'
    ) -> None:
        """
        Plot the unit impulse response function with its Hilbert envelope.

        Parameters
        ----------
        t : np.ndarray
            Time array (s), shape (n_t,).
        h : np.ndarray
            Impulse response values, shape (n_t,).
        frequencies_hz : np.ndarray, optional
            Natural frequencies (Hz) shown in a subtitle annotation.
        title : str, optional
            Figure title. Default is 'Unit Impulse Response Function h(t)'.
        filename : str, optional
            Output filename. Default is 'impulse_response.png'.
        """
        envelope = np.abs(hilbert(h))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, h, color='#1565C0', linewidth=0.9, alpha=0.85,
                label='h(t)', zorder=2)
        ax.plot(t, envelope, color='#C62828', linewidth=2.2,
                label='Envelope |h(t)|', zorder=3)
        ax.plot(t, -envelope, color='#C62828', linewidth=2.2,
                linestyle='--', alpha=0.7, zorder=3)
        ax.fill_between(t, -envelope, envelope, alpha=0.07, color='#C62828')
        ax.axhline(0, color='k', linewidth=0.6, alpha=0.4)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')

        if frequencies_hz is not None:
            freq_str = ', '.join(f'{f:.1f} Hz' for f in frequencies_hz)
            ax.text(0.02, 0.05, f'Natural frequencies: {freq_str}',
                    transform=ax.transAxes, fontsize=9, color='gray',
                    verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_time_signal(
        self,
        time: np.ndarray,
        signal_data: np.ndarray,
        n_show_samples: int = 1000,
        title: str = 'Synthetic Vibration Signal',
        filename: str = 'time_signal.png'
    ) -> None:
        """
        Plot the time-domain vibration signal.

        Parameters
        ----------
        time : np.ndarray
            Time array (s).
        signal_data : np.ndarray
            Vibration signal values.
        n_show_samples : int, optional
            Number of samples to display. Default is 1000.
        title : str, optional
            Figure title.
        filename : str, optional
            Output filename. Default is 'time_signal.png'.
        """
        t_show = time[:n_show_samples]
        s_show = signal_data[:n_show_samples]
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        t_end_ms = t_show[-1] * 1000

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_show * 1000, s_show, color='#0D47A1', linewidth=0.8,
                alpha=0.9)
        ax.fill_between(t_show * 1000, s_show, 0,
                        where=(s_show >= 0), alpha=0.15, color='#1E88E5')
        ax.fill_between(t_show * 1000, s_show, 0,
                        where=(s_show < 0), alpha=0.15, color='#E53935')
        ax.axhline(0, color='k', linewidth=0.6, alpha=0.5)

        # RMS annotation
        rms = np.sqrt(np.mean(signal_data**2))
        ax.axhline(rms, color='orange', linewidth=1.2, linestyle='--',
                   alpha=0.8, label=f'RMS = {rms:.3f}')
        ax.axhline(-rms, color='orange', linewidth=1.2, linestyle='--',
                   alpha=0.8)

        total_duration = time[-1]
        fs = 1.0 / dt
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(
            f'{title}  (first {t_end_ms:.0f} ms of {total_duration:.1f} s, '
            f'fs = {fs/1000:.0f} kHz)',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.25, linestyle=':')

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_optimization_history(
        self,
        frequencies: List[float],
        masses: List[float],
        target_frequency: float,
        parameter_name: str = 'thickness',
        filename: str = 'optimization_convergence.png'
    ) -> None:
        """
        Plot the optimization convergence history (frequency and mass).

        Parameters
        ----------
        frequencies : list of float
            First natural frequency at each iteration (Hz).
        masses : list of float
            Total beam mass at each iteration (kg).
        target_frequency : float
            Target frequency constraint (Hz).
        parameter_name : str, optional
            Name of the optimised parameter.
        filename : str, optional
            Output filename. Default is 'optimization_convergence.png'.
        """
        n = min(len(frequencies), len(masses))
        iters = list(range(1, n + 1))
        freqs = frequencies[:n]
        mass_g = [m * 1000 for m in masses[:n]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Optimization Convergence — {parameter_name.capitalize()} Sweep',
                     fontsize=13, fontweight='bold')

        # Frequency panel
        ax1.plot(iters, freqs, 'o-', color='#1565C0', linewidth=2,
                 markersize=5, label='f1 (Hz)')
        ax1.axhline(target_frequency, color='#C62828', linewidth=1.8,
                    linestyle='--', label=f'Target: {target_frequency} Hz')
        ax1.fill_between(iters, freqs, target_frequency,
                         where=[f < target_frequency for f in freqs],
                         alpha=0.15, color='#C62828', label='Below target')
        ax1.fill_between(iters, freqs, target_frequency,
                         where=[f >= target_frequency for f in freqs],
                         alpha=0.12, color='#2E7D32', label='Above target')

        # Annotate start and end
        ax1.annotate(f'Start\n{freqs[0]:.1f} Hz', xy=(iters[0], freqs[0]),
                     xytext=(iters[0] + max(1, n * 0.05), freqs[0]),
                     fontsize=9, color='#1565C0',
                     arrowprops=dict(arrowstyle='->', color='#1565C0', lw=0.8))
        ax1.annotate(f'End\n{freqs[-1]:.1f} Hz', xy=(iters[-1], freqs[-1]),
                     xytext=(iters[-1] - max(1, n * 0.15), freqs[-1] + (max(freqs) - min(freqs)) * 0.08),
                     fontsize=9, color='#2E7D32',
                     arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))

        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('First Natural Frequency (Hz)', fontsize=11)
        ax1.set_title('Frequency Convergence', fontsize=12)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3, linestyle=':')

        # Mass panel
        ax2.plot(iters, mass_g, 's-', color='#2E7D32', linewidth=2,
                 markersize=5, label='Mass (g)')
        ax2.fill_between(iters, mass_g, min(mass_g),
                         alpha=0.12, color='#2E7D32')

        ax2.annotate(f'Start\n{mass_g[0]:.1f} g', xy=(iters[0], mass_g[0]),
                     xytext=(iters[0] + max(1, n * 0.05), mass_g[0]),
                     fontsize=9, color='#2E7D32',
                     arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))
        ax2.annotate(f'End\n{mass_g[-1]:.1f} g', xy=(iters[-1], mass_g[-1]),
                     xytext=(iters[-1] - max(1, n * 0.15), mass_g[-1] + (max(mass_g) - min(mass_g)) * 0.08),
                     fontsize=9, color='#388E3C',
                     arrowprops=dict(arrowstyle='->', color='#388E3C', lw=0.8))

        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Total Mass (g)', fontsize=11)
        ax2.set_title('Mass Convergence', fontsize=12)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_summary_dashboard(
        self,
        x: np.ndarray,
        mode_shapes: np.ndarray,
        frequencies_analytical: np.ndarray,
        freq_hz: np.ndarray,
        H_mag_db: np.ndarray,
        fft_frequencies: np.ndarray,
        fft_amplitudes: np.ndarray,
        peak_freqs: np.ndarray,
        n_elements: np.ndarray,
        errors: np.ndarray,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
        frequency_map: np.ndarray,
        hazard_range: Optional[Tuple[float, float]] = None,
        length: float = 1.0,
        filename: str = 'summary_dashboard.png'
    ) -> None:
        """
        Create a comprehensive 2×3 summary dashboard of key analysis results.

        Panels (left→right, top→bottom):
        1. First 3 mode shapes (stacked)
        2. FRF magnitude spectrum
        3. FFT spectrum with detected peaks
        4. Mesh convergence study
        5. 2-D design space map
        6. Validation text card (frequencies table)

        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates along beam (m).
        mode_shapes : np.ndarray
            Mode shape matrix (n_points × n_modes).
        frequencies_analytical : np.ndarray
            Analytical natural frequencies (Hz).
        freq_hz : np.ndarray
            Frequency axis for FRF (Hz).
        H_mag_db : np.ndarray
            FRF magnitude in dB.
        fft_frequencies : np.ndarray
            FFT frequency axis (Hz).
        fft_amplitudes : np.ndarray
            FFT amplitude spectrum.
        peak_freqs : np.ndarray
            Detected FFT peak frequencies (Hz).
        n_elements : np.ndarray
            Element counts for mesh convergence.
        errors : np.ndarray
            Convergence errors, shape (n_meshes, n_modes).
        param1_values : np.ndarray
            First parameter values for design space (mm).
        param2_values : np.ndarray
            Second parameter values for design space (mm).
        frequency_map : np.ndarray
            2-D frequency map (n_param2 × n_param1).
        hazard_range : tuple of float, optional
            Resonance-risk frequency band (f_low, f_high).
        length : float, optional
            Beam length (m). Default is 1.0.
        filename : str, optional
            Output filename. Default is 'summary_dashboard.png'.
        """
        palette = plt.cm.tab10(np.linspace(0, 0.9, 5))

        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('#F8F9FA')

        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                              left=0.06, right=0.97, top=0.92, bottom=0.06)

        # ── Panel 1: Mode shapes (first 3) ───────────────────────────────
        gs_modes = gs[0, 0].subgridspec(3, 1, hspace=0.15)
        axes_mode = [fig.add_subplot(gs_modes[k]) for k in range(3)]
        n_show = min(3, mode_shapes.shape[1])
        for k in range(n_show):
            ax = axes_mode[k]
            m = mode_shapes[:, k]
            m_norm = m / np.max(np.abs(m))
            ax.fill_between(x, 0, m_norm, alpha=0.25, color=palette[k])
            ax.plot(x, m_norm, color=palette[k], linewidth=1.8)
            ax.axvline(0, color='k', linewidth=2)
            ax.plot([0, length], [0, 0], 'k--', linewidth=0.6, alpha=0.4)
            ax.set_ylim(-1.35, 1.35)
            ax.set_xlim(-0.05 * length, 1.05 * length)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=7)
            ax.text(0.97, 0.88, f'f{k+1}={frequencies_analytical[k]:.1f} Hz',
                    transform=ax.transAxes, fontsize=7, ha='right', color=palette[k],
                    fontweight='bold')
            if k < n_show - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('x (m)', fontsize=8)
        axes_mode[0].set_title('Mode Shapes', fontsize=11, fontweight='bold')

        # ── Panel 2: FRF magnitude ────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(freq_hz, H_mag_db, color='#1565C0', linewidth=1.4)
        ax2.fill_between(freq_hz, H_mag_db, H_mag_db.min() - 2,
                         alpha=0.07, color='#1565C0')
        for i, f in enumerate(frequencies_analytical):
            ax2.axvline(f, color='#C62828', linewidth=1.0, linestyle='--', alpha=0.7)
            ax2.text(f, ax2.get_ylim()[0] + 1, f' f{i+1}', fontsize=7,
                     color='#C62828', rotation=90, va='bottom')
        ax2.set_xlabel('Frequency (Hz)', fontsize=9)
        ax2.set_ylabel('|H(ω)| (dB)', fontsize=9)
        ax2.set_title('FRF Magnitude', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.25, linestyle=':')
        ax2.tick_params(labelsize=8)

        # ── Panel 3: FFT spectrum ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        if len(fft_frequencies) > 0 and len(fft_amplitudes) > 0:
            ax3.semilogy(fft_frequencies, fft_amplitudes + 1e-30,
                         color='#4A148C', linewidth=1.3, alpha=0.9)
        for i, f in enumerate(peak_freqs):
            ax3.axvline(f, color='#E65100', linewidth=1.2, linestyle='--', alpha=0.8)
            ax3.text(f, ax3.get_ylim()[1] * 0.8 if ax3.get_ylim()[1] > 0 else 1,
                     f' f{i+1}', fontsize=7, color='#E65100', rotation=90, va='top')
        ax3.set_xlabel('Frequency (Hz)', fontsize=9)
        ax3.set_ylabel('Amplitude (log)', fontsize=9)
        ax3.set_title('FFT Spectrum', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.25, linestyle=':', which='both')
        ax3.tick_params(labelsize=8)

        # ── Panel 4: Mesh convergence ─────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        err_arr = np.asarray(errors)
        if err_arr.ndim == 1:
            err_arr = err_arr.reshape(-1, 1)
        for k in range(min(err_arr.shape[1], 3)):
            ax4.semilogy(n_elements, err_arr[:, k], 'o-',
                         color=palette[k], linewidth=1.8, markersize=5,
                         label=f'Mode {k+1}')
        ax4.set_xlabel('Number of Elements', fontsize=9)
        ax4.set_ylabel('Rel. Error (%)', fontsize=9)
        ax4.set_title('Mesh Convergence', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8, loc='upper right')
        ax4.grid(True, alpha=0.25, linestyle=':', which='both')
        ax4.tick_params(labelsize=8)

        # ── Panel 5: 2-D design space ─────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        P1, P2 = np.meshgrid(param1_values, param2_values)
        cf = ax5.contourf(P1, P2, frequency_map, levels=16,
                          cmap='plasma', alpha=0.85)
        plt.colorbar(cf, ax=ax5, fraction=0.04, pad=0.04,
                     label='f1 (Hz)').ax.tick_params(labelsize=7)
        if hazard_range is not None:
            unsafe = ((frequency_map >= hazard_range[0]) &
                      (frequency_map <= hazard_range[1])).astype(float)
            if np.any(unsafe):
                ax5.contourf(P1, P2, unsafe, levels=[0.5, 1.5],
                             colors='red', alpha=0.25)
        ax5.set_xlabel('Thickness (mm)', fontsize=9)
        ax5.set_ylabel('Length (mm)', fontsize=9)
        ax5.set_title('Design Space Map', fontsize=11, fontweight='bold')
        ax5.tick_params(labelsize=8)

        # ── Panel 6: Validation summary card ─────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        n_modes_show = min(5, len(frequencies_analytical))
        rows = [['Mode', 'Analytical\n(Hz)', 'Description']]
        descriptions = ['1st bending', '2nd bending', '3rd bending',
                        '4th bending', '5th bending']
        for k in range(n_modes_show):
            rows.append([
                str(k + 1),
                f'{frequencies_analytical[k]:.2f}',
                descriptions[k] if k < len(descriptions) else ''
            ])
        tbl = ax6.table(cellText=rows[1:], colLabels=rows[0],
                        cellLoc='center', loc='upper center',
                        bbox=[0.0, 0.15, 1.0, 0.80])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for j in range(3):
            tbl[(0, j)].set_facecolor('#1565C0')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, n_modes_show + 1):
            bg = '#E3F2FD' if i % 2 == 0 else '#FFFFFF'
            for j in range(3):
                tbl[(i, j)].set_facecolor(bg)
        ax6.set_title('Natural Frequencies', fontsize=11, fontweight='bold')

        # ── Super-title ───────────────────────────────────────────────────
        fig.text(0.5, 0.96,
                 'Modal Analysis Dashboard — Steel Cantilever Beam',
                 ha='center', va='top', fontsize=15, fontweight='bold',
                 color='#1A237E')

        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()


if __name__ == '__main__':
    """
    Demonstration of visualization capabilities.
    """
    # Create sample data
    x = np.linspace(0, 1, 100)
    n_modes = 4
    
    # Generate sample mode shapes (sine functions)
    mode_shapes = np.zeros((len(x), n_modes))
    frequencies = np.zeros(n_modes)
    
    for i in range(n_modes):
        mode_shapes[:, i] = np.sin((i + 1) * np.pi * x)
        frequencies[i] = (i + 1) ** 2 * 10.0
    
    # Initialize visualizer
    viz = Visualizer(save_dir='results/demo_figures')
    
    print("Generating demonstration plots...")
    
    # Plot mode shapes
    viz.plot_mode_shapes(x, mode_shapes, frequencies, n_modes=4)
    print("✓ Mode shapes plot created")
    
    # Generate FFT spectrum demo
    freq_array = np.linspace(0, 200, 1000)
    fft_spectrum = np.zeros_like(freq_array)
    for f in frequencies:
        fft_spectrum += 1000 * np.exp(-((freq_array - f) / 2) ** 2)
    fft_spectrum += np.random.random(len(freq_array)) * 10
    
    viz.plot_fft_spectrum(freq_array, fft_spectrum,
                         theoretical_freqs=frequencies,
                         peak_freqs=frequencies)
    print("✓ FFT spectrum plot created")
    
    # Mesh convergence demo
    n_elements = np.array([5, 10, 20, 40, 80, 160])
    errors = 100 * (n_elements[0] / n_elements) ** 2
    errors = np.column_stack([errors, errors * 1.5, errors * 2])
    
    viz.plot_mesh_convergence(n_elements, errors,
                             labels=['Mode 1', 'Mode 2', 'Mode 3'])
    print("✓ Mesh convergence plot created")
    
    # Frequency comparison demo
    freq_dict = {
        'Analytical': frequencies,
        'FEM': frequencies * 1.02,
        'FFT': frequencies * 0.98
    }
    viz.plot_frequency_comparison(freq_dict)
    print("✓ Frequency comparison plot created")
    
    print(f"\nAll demonstration plots saved to: {viz.save_dir}")
