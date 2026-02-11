"""
Visualization module for modal analysis of resonant structures.

This module provides comprehensive plotting capabilities for visualizing
mode shapes, frequency analysis, convergence studies, and parametric studies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import seaborn as sns


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
        
        fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2.5 * n_modes))
        if n_modes == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            # Normalize mode shape
            mode = mode_shapes[:, i]
            mode_normalized = mode / np.max(np.abs(mode))
            
            # Plot beam centerline
            ax.plot([0, length], [0, 0], 'k--', linewidth=0.5, alpha=0.5)
            
            # Plot mode shape
            ax.plot(x, mode_normalized, 'b-', linewidth=2, label=f'Mode {i+1}')
            ax.fill_between(x, 0, mode_normalized, alpha=0.3)
            
            # Add beam boundaries
            ax.axvline(0, color='k', linewidth=2)
            ax.axvline(length, color='k', linewidth=2)
            
            # Labels and formatting
            ax.set_ylabel('Normalized\nDisplacement', fontsize=10)
            ax.set_title(f'Mode {i+1}: f = {frequencies[i]:.2f} Hz', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05 * length, 1.05 * length)
            
            if i == n_modes - 1:
                ax.set_xlabel('Position (m)', fontsize=10)
            else:
                ax.set_xticklabels([])
                
        plt.tight_layout()
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
        
        # Plot each mode
        colors = plt.cm.tab10(np.linspace(0, 1, frequencies.shape[1]))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
        
        for i in range(frequencies.shape[1]):
            label = mode_labels[i] if mode_labels and i < len(mode_labels) else f'Mode {i+1}'
            ax.plot(parameter_values, frequencies[:, i],
                   marker=markers[i % len(markers)], markersize=6,
                   linewidth=2, label=label, color=colors[i], alpha=0.8)
        
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
        ax.annotate(f'f₁ = {f1:.2f} Hz', xy=(f1, half_power_amp),
                   xytext=(f1 - 5, half_power_amp * 1.2),
                   fontsize=10, ha='right',
                   arrowprops=dict(arrowstyle='->', color='orange'))
        ax.annotate(f'f₂ = {f2:.2f} Hz', xy=(f2, half_power_amp),
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
