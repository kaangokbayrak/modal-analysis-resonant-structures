"""
Signal Processing Module for Vibration Analysis

This module implements FFT-based frequency analysis and damping estimation techniques
for experimental (or synthetic) vibration data.

Key Features:
- Synthetic vibration signal generation with realistic damping
- FFT with multiple windowing options
- Peak detection in frequency domain
- Damping ratio estimation (half-power bandwidth & log decrement methods)

Author: Kaan Gokbayrak, Purdue University
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple


class SignalProcessor:
    """
    Signal processor for vibration analysis using FFT and time-domain methods.
    
    Parameters
    ----------
    fs : float, optional
        Sampling frequency [Hz] (default: 10000 Hz)
    """
    
    def __init__(self, fs: float = 10000):
        """
        Initialize signal processor.
        
        Parameters
        ----------
        fs : float
            Sampling frequency [Hz]
        """
        self.fs = fs
        self.dt = 1.0 / fs
    
    def generate_synthetic_vibration(self, frequencies: np.ndarray,
                                      damping_ratios: np.ndarray,
                                      duration: float = 2.0,
                                      snr_db: float = 20.0,
                                      amplitudes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic synthetic vibration signal with exponential decay.
        
        The signal is a sum of damped sinusoids representing structural modes:
        x(t) = Σ A_n exp(-ζ_n ω_n t) sin(ω_d_n t) + noise
        
        where:
        - A_n: mode amplitude
        - ζ_n: damping ratio
        - ω_n: natural frequency [rad/s]
        - ω_d_n = ω_n √(1 - ζ_n²): damped natural frequency
        
        Parameters
        ----------
        frequencies : np.ndarray
            Natural frequencies [Hz], shape (n_modes,)
        damping_ratios : np.ndarray
            Damping ratios (dimensionless), shape (n_modes,)
        duration : float, optional
            Signal duration [s] (default: 2.0)
        snr_db : float, optional
            Signal-to-noise ratio [dB] (default: 20.0)
        amplitudes : np.ndarray, optional
            Mode amplitudes (default: decreasing with mode number)
            
        Returns
        -------
        time : np.ndarray
            Time array [s], shape (n_samples,)
        signal : np.ndarray
            Vibration signal, shape (n_samples,)
        """
        # Generate time array
        time = np.arange(0, duration, self.dt)
        n_samples = len(time)
        
        # Initialize signal
        x = np.zeros(n_samples)
        
        # Default amplitudes: decrease with mode number
        if amplitudes is None:
            amplitudes = 1.0 / (np.arange(len(frequencies)) + 1)
        
        # Add each mode as a damped sinusoid
        for i, (f_n, zeta, A) in enumerate(zip(frequencies, damping_ratios, amplitudes)):
            # Convert to angular frequency
            omega_n = 2 * np.pi * f_n
            
            # Damped natural frequency
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            
            # Exponential decay envelope
            decay = np.exp(-zeta * omega_n * time)
            
            # Damped oscillation
            oscillation = A * decay * np.sin(omega_d * time)
            
            x += oscillation
        
        # Add white noise to achieve desired SNR
        signal_power = np.mean(x**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(n_samples)
        
        x_noisy = x + noise
        
        return time, x_noisy
    
    def compute_fft(self, signal_data: np.ndarray, 
                    window: str = 'hanning') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute single-sided FFT spectrum with windowing.
        
        Windowing reduces spectral leakage in FFT analysis by tapering the signal
        at the boundaries. Different windows have different tradeoffs between
        frequency resolution and amplitude accuracy.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time-domain signal, shape (n_samples,)
        window : str, optional
            Window function: 'hanning', 'hamming', 'blackman', 'flattop', or 'rectangular'
            (default: 'hanning')
            
        Returns
        -------
        frequencies : np.ndarray
            Frequency array [Hz], shape (n_freq,)
        amplitudes : np.ndarray
            Single-sided amplitude spectrum, shape (n_freq,)
        """
        n_samples = len(signal_data)
        
        # Apply window function
        if window == 'hanning':
            w = np.hanning(n_samples)
        elif window == 'hamming':
            w = np.hamming(n_samples)
        elif window == 'blackman':
            w = np.blackman(n_samples)
        elif window == 'flattop':
            w = signal.windows.flattop(n_samples)
        elif window == 'rectangular':
            w = np.ones(n_samples)
        else:
            raise ValueError(f"Unknown window type: {window}")
        
        # Apply window
        signal_windowed = signal_data * w
        
        # Compute FFT
        fft_result = np.fft.fft(signal_windowed)
        
        # Single-sided spectrum (keep only positive frequencies)
        n_freq = n_samples // 2
        fft_single = fft_result[:n_freq]
        
        # Compute amplitude spectrum (scale by 2 to account for negative frequencies)
        amplitudes = 2.0 * np.abs(fft_single) / n_samples
        
        # Frequency array
        frequencies = np.fft.fftfreq(n_samples, self.dt)[:n_freq]
        
        return frequencies, amplitudes
    
    def detect_peaks(self, freqs: np.ndarray, amplitudes: np.ndarray,
                     min_height_ratio: float = 0.05,
                     min_distance_hz: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect peaks in FFT spectrum.
        
        Peaks in the frequency spectrum correspond to natural frequencies of the structure.
        This function identifies local maxima that exceed a threshold.
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency array [Hz]
        amplitudes : np.ndarray
            Amplitude spectrum
        min_height_ratio : float, optional
            Minimum peak height as fraction of max amplitude (default: 0.05)
        min_distance_hz : float, optional
            Minimum distance between peaks [Hz] (default: 20.0)
            
        Returns
        -------
        peak_frequencies : np.ndarray
            Frequencies of detected peaks [Hz]
        peak_amplitudes : np.ndarray
            Amplitudes at detected peaks
        """
        # Minimum peak height
        min_height = min_height_ratio * np.max(amplitudes)
        
        # Convert min distance to samples
        df = freqs[1] - freqs[0]  # Frequency resolution
        min_distance_samples = int(min_distance_hz / df)
        
        # Find peaks using scipy
        peak_indices, properties = signal.find_peaks(
            amplitudes,
            height=min_height,
            distance=min_distance_samples
        )
        
        peak_frequencies = freqs[peak_indices]
        peak_amplitudes = amplitudes[peak_indices]
        
        return peak_frequencies, peak_amplitudes
    
    def estimate_damping_halfpower(self, freqs: np.ndarray, amplitudes: np.ndarray,
                                    peak_freq: float, search_width: float = 50.0) -> float:
        """
        Estimate damping ratio using half-power bandwidth method.
        
        The half-power bandwidth method estimates damping from the width of a resonance
        peak in the frequency response. The frequencies f1 and f2 where the amplitude
        drops to 1/√2 (≈ 0.707) of the peak value are related to damping by:
        
        ζ ≈ (f2 - f1) / (2 * f_n)
        
        This method assumes light damping (ζ < 0.2) and well-separated modes.
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency array [Hz]
        amplitudes : np.ndarray
            Amplitude spectrum
        peak_freq : float
            Peak frequency [Hz] around which to estimate damping
        search_width : float, optional
            Search width around peak [Hz] (default: 50.0)
            
        Returns
        -------
        float
            Estimated damping ratio (dimensionless)
        """
        # Find index of peak frequency
        idx_peak = np.argmin(np.abs(freqs - peak_freq))
        peak_amp = amplitudes[idx_peak]
        
        # Half-power amplitude (3 dB down = 1/√2 of peak)
        half_power_amp = peak_amp / np.sqrt(2)
        
        # Define search region around peak
        df = freqs[1] - freqs[0]
        search_samples = int(search_width / df)
        idx_min = max(0, idx_peak - search_samples)
        idx_max = min(len(freqs), idx_peak + search_samples)
        
        # Find half-power points (frequencies where amplitude crosses half-power level)
        # Search to the left of peak for f1
        left_region = amplitudes[idx_min:idx_peak]
        left_indices = np.where(left_region <= half_power_amp)[0]
        if len(left_indices) > 0:
            idx_f1 = idx_min + left_indices[-1]  # Closest to peak
            f1 = freqs[idx_f1]
        else:
            f1 = freqs[idx_min]
        
        # Search to the right of peak for f2
        right_region = amplitudes[idx_peak:idx_max]
        right_indices = np.where(right_region <= half_power_amp)[0]
        if len(right_indices) > 0:
            idx_f2 = idx_peak + right_indices[0]  # Closest to peak
            f2 = freqs[idx_f2]
        else:
            f2 = freqs[idx_max-1]
        
        # Compute damping ratio
        zeta = (f2 - f1) / (2 * peak_freq)
        
        # Store for potential visualization
        self._last_halfpower = {
            'f1': f1,
            'f2': f2,
            'peak_freq': peak_freq,
            'half_power_amp': half_power_amp,
            'zeta': zeta
        }
        
        return max(0.0, zeta)  # Ensure non-negative
    
    def estimate_damping_log_decrement(self, time: np.ndarray, signal_data: np.ndarray,
                                        freq: float, n_cycles: int = 5) -> float:
        """
        Estimate damping ratio using logarithmic decrement method.
        
        The logarithmic decrement method uses the decay rate of oscillation peaks
        in the time domain. For a damped sinusoid, the damping ratio is:
        
        δ = (1/n) ln(x_i / x_{i+n})  (logarithmic decrement)
        ζ = δ / √((2π)² + δ²)        (damping ratio)
        
        where n is the number of cycles between peaks x_i and x_{i+n}.
        
        Parameters
        ----------
        time : np.ndarray
            Time array [s]
        signal_data : np.ndarray
            Time-domain signal
        freq : float
            Dominant frequency [Hz] for bandpass filtering
        n_cycles : int, optional
            Number of cycles to average over (default: 5)
            
        Returns
        -------
        float
            Estimated damping ratio (dimensionless)
        """
        # Bandpass filter around frequency of interest
        period = 1.0 / freq
        lowcut = freq * 0.8
        highcut = freq * 1.2
        nyquist = 0.5 * self.fs
        
        # Design Butterworth bandpass filter
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        signal_filtered = signal.filtfilt(b, a, signal_data)
        
        # Find peaks in filtered signal
        peak_indices, _ = signal.find_peaks(signal_filtered, distance=int(0.8 * period / self.dt))
        
        if len(peak_indices) < n_cycles + 1:
            # Not enough peaks, return default value
            return 0.01
        
        # Get amplitudes of first n_cycles + 1 peaks
        peak_amplitudes = np.abs(signal_filtered[peak_indices])
        
        # Compute logarithmic decrement
        x_i = peak_amplitudes[0]
        x_n = peak_amplitudes[n_cycles]
        
        if x_n <= 0 or x_i <= 0:
            return 0.01
        
        delta = (1.0 / n_cycles) * np.log(x_i / x_n)
        
        # Convert to damping ratio
        zeta = delta / np.sqrt((2 * np.pi)**2 + delta**2)
        
        return max(0.0, min(0.5, zeta))  # Clamp to reasonable range
    
    def window_comparison(self, signal_data: np.ndarray) -> Dict:
        """
        Compare FFT results with different window functions.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Time-domain signal
            
        Returns
        -------
        dict
            Dictionary with window types as keys and (frequencies, amplitudes) as values
        """
        windows = ['rectangular', 'hanning', 'hamming', 'blackman', 'flattop']
        results = {}
        
        for window_type in windows:
            freqs, amps = self.compute_fft(signal_data, window=window_type)
            results[window_type] = {'frequencies': freqs, 'amplitudes': amps}
        
        return results
    
    def full_analysis(self, frequencies: np.ndarray,
                      damping_ratios: np.ndarray,
                      duration: float = 2.0,
                      snr_db: float = 20.0) -> Dict:
        """
        Run complete signal processing pipeline.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Target natural frequencies [Hz]
        damping_ratios : np.ndarray
            Target damping ratios
        duration : float, optional
            Signal duration [s] (default: 2.0)
        snr_db : float, optional
            Signal-to-noise ratio [dB] (default: 20.0)
            
        Returns
        -------
        dict
            Comprehensive results dictionary containing:
            - 'time': Time array
            - 'signal': Generated signal
            - 'frequencies': FFT frequency array
            - 'amplitudes': FFT amplitude spectrum
            - 'peak_frequencies': Detected peak frequencies
            - 'peak_amplitudes': Peak amplitudes
            - 'target_frequencies': Input frequencies
            - 'estimated_damping': Estimated damping ratios
        """
        # Generate synthetic signal
        time, sig = self.generate_synthetic_vibration(
            frequencies, damping_ratios, duration, snr_db
        )
        
        # Compute FFT
        freqs, amps = self.compute_fft(sig, window='hanning')
        
        # Detect peaks
        peak_freqs, peak_amps = self.detect_peaks(freqs, amps)
        
        # Estimate damping for detected peaks
        estimated_damping = []
        for pf in peak_freqs:
            zeta = self.estimate_damping_halfpower(freqs, amps, pf)
            estimated_damping.append(zeta)
        
        return {
            'time': time,
            'signal': sig,
            'frequencies': freqs,
            'amplitudes': amps,
            'peak_frequencies': peak_freqs,
            'peak_amplitudes': peak_amps,
            'target_frequencies': frequencies,
            'estimated_damping': np.array(estimated_damping)
        }
