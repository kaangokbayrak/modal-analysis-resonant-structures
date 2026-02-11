"""
Tests for Signal Processing Module

This test suite validates FFT analysis, peak detection, and damping estimation methods.

Author: Kaan Gokbayrak, Purdue University
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signal_processing import SignalProcessor


class TestSignalProcessor:
    """Test signal processing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a signal processor with standard sampling rate."""
        return SignalProcessor(fs=10000)
    
    def test_synthetic_signal_generation(self, processor):
        """Test synthetic vibration signal generation."""
        frequencies = np.array([50.0, 150.0, 300.0])
        damping = np.array([0.01, 0.02, 0.03])
        
        time, signal = processor.generate_synthetic_vibration(
            frequencies, damping, duration=1.0, snr_db=30.0
        )
        
        # Check output shapes
        assert len(time) == len(signal)
        assert len(time) == int(1.0 * processor.fs)
        
        # Check signal has reasonable amplitude
        assert np.max(np.abs(signal)) > 0
        assert np.max(np.abs(signal)) < 10  # Should be reasonable scale
    
    def test_fft_detects_single_frequency(self, processor):
        """Test that FFT correctly identifies a single frequency."""
        # Generate clean sine wave
        freq_target = 100.0
        duration = 1.0
        time = np.arange(0, duration, processor.dt)
        signal = np.sin(2 * np.pi * freq_target * time)
        
        # Compute FFT
        freqs, amps = processor.compute_fft(signal, window='rectangular')
        
        # Find peak
        peak_idx = np.argmax(amps)
        detected_freq = freqs[peak_idx]
        
        # Should detect frequency within resolution
        df = 1.0 / duration
        assert abs(detected_freq - freq_target) < 2 * df
    
    def test_fft_detects_multiple_frequencies(self, processor):
        """Test that FFT identifies multiple frequencies."""
        # Generate signal with two clear frequencies
        freqs_target = [50.0, 200.0]
        duration = 2.0
        time = np.arange(0, duration, processor.dt)
        signal = (np.sin(2 * np.pi * freqs_target[0] * time) + 
                 np.sin(2 * np.pi * freqs_target[1] * time))
        
        # Compute FFT
        freqs, amps = processor.compute_fft(signal, window='hanning')
        
        # Detect peaks
        peak_freqs, peak_amps = processor.detect_peaks(
            freqs, amps, min_height_ratio=0.1, min_distance_hz=50.0
        )
        
        # Should detect 2 peaks
        assert len(peak_freqs) >= 2
        
        # Check that detected peaks are close to target frequencies
        for target in freqs_target:
            min_error = min(abs(peak_freqs - target))
            assert min_error < 5.0  # Within 5 Hz
    
    def test_different_windows(self, processor):
        """Test FFT with different window functions."""
        freq_target = 100.0
        duration = 1.0
        time = np.arange(0, duration, processor.dt)
        signal = np.sin(2 * np.pi * freq_target * time)
        
        windows = ['rectangular', 'hanning', 'hamming', 'blackman', 'flattop']
        
        for window in windows:
            freqs, amps = processor.compute_fft(signal, window=window)
            
            # Should produce valid spectrum
            assert len(freqs) == len(amps)
            assert np.all(amps >= 0)
            
            # Peak should be near target frequency
            peak_idx = np.argmax(amps)
            detected_freq = freqs[peak_idx]
            assert abs(detected_freq - freq_target) < 10.0
    
    def test_peak_detection(self, processor):
        """Test peak detection algorithm."""
        # Create spectrum with known peaks
        freqs = np.linspace(0, 500, 5000)
        amps = np.zeros_like(freqs)
        
        # Add peaks at specific frequencies
        peak_locs = [50, 150, 300]
        for loc in peak_locs:
            idx = np.argmin(np.abs(freqs - loc))
            amps[idx] = 1.0
        
        # Add some noise
        amps += 0.05 * np.random.randn(len(amps))
        amps = np.abs(amps)
        
        # Detect peaks
        detected_freqs, detected_amps = processor.detect_peaks(
            freqs, amps, min_height_ratio=0.1, min_distance_hz=30.0
        )
        
        # Should find the 3 peaks
        assert len(detected_freqs) >= 3
    
    def test_damping_estimation_halfpower(self, processor):
        """Test half-power bandwidth damping estimation."""
        # Generate signal with known damping
        freq_target = 100.0
        zeta_target = 0.05
        duration = 2.0
        
        time = np.arange(0, duration, processor.dt)
        omega = 2 * np.pi * freq_target
        signal = np.exp(-zeta_target * omega * time) * np.sin(omega * time)
        
        # Compute FFT
        freqs, amps = processor.compute_fft(signal, window='hanning')
        
        # Estimate damping
        zeta_estimated = processor.estimate_damping_halfpower(
            freqs, amps, peak_freq=freq_target, search_width=50.0
        )
        
        # Should be reasonably close to target (within factor of 2)
        # Note: This method is approximate for transient signals
        assert 0 < zeta_estimated < 0.2
    
    def test_damping_estimation_log_decrement(self, processor):
        """Test logarithmic decrement damping estimation."""
        # Generate damped sinusoid
        freq_target = 100.0
        zeta_target = 0.05
        duration = 0.5
        
        time = np.arange(0, duration, processor.dt)
        omega = 2 * np.pi * freq_target
        omega_d = omega * np.sqrt(1 - zeta_target**2)
        signal = np.exp(-zeta_target * omega * time) * np.sin(omega_d * time)
        
        # Estimate damping
        zeta_estimated = processor.estimate_damping_log_decrement(
            time, signal, freq=freq_target, n_cycles=5
        )
        
        # Should be within reasonable range
        assert 0 < zeta_estimated < 0.2
        
        # For clean signal, should be close to target
        # (Allow wide tolerance due to method limitations)
        assert abs(zeta_estimated - zeta_target) < 0.05
    
    def test_window_comparison(self, processor):
        """Test window function comparison."""
        freq_target = 100.0
        duration = 1.0
        time = np.arange(0, duration, processor.dt)
        signal = np.sin(2 * np.pi * freq_target * time)
        
        results = processor.window_comparison(signal)
        
        # Should return results for all window types
        assert 'rectangular' in results
        assert 'hanning' in results
        assert 'hamming' in results
        assert 'blackman' in results
        assert 'flattop' in results
        
        # Each result should have frequencies and amplitudes
        for window_type, data in results.items():
            assert 'frequencies' in data
            assert 'amplitudes' in data
    
    def test_full_analysis_pipeline(self, processor):
        """Test complete signal processing pipeline."""
        frequencies = np.array([30.0, 100.0, 250.0])
        damping_ratios = np.array([0.01, 0.02, 0.03])
        
        results = processor.full_analysis(
            frequencies, damping_ratios, duration=2.0, snr_db=25.0
        )
        
        # Check all expected keys are present
        assert 'time' in results
        assert 'signal' in results
        assert 'frequencies' in results
        assert 'amplitudes' in results
        assert 'peak_frequencies' in results
        assert 'peak_amplitudes' in results
        assert 'target_frequencies' in results
        assert 'estimated_damping' in results
        
        # Should detect at least some peaks
        assert len(results['peak_frequencies']) > 0
    
    def test_snr_effect(self, processor):
        """Test effect of SNR on signal quality."""
        frequencies = np.array([100.0])
        damping = np.array([0.01])
        
        # High SNR signal
        _, signal_high_snr = processor.generate_synthetic_vibration(
            frequencies, damping, duration=1.0, snr_db=40.0
        )
        
        # Low SNR signal
        _, signal_low_snr = processor.generate_synthetic_vibration(
            frequencies, damping, duration=1.0, snr_db=5.0
        )
        
        # High SNR should have lower noise
        # (approximate test - just check they're different)
        assert not np.array_equal(signal_high_snr, signal_low_snr)


class TestSignalProcessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_window_type(self):
        """Test that invalid window type raises error."""
        processor = SignalProcessor(fs=10000)
        signal = np.random.randn(1000)
        
        with pytest.raises(ValueError):
            processor.compute_fft(signal, window='invalid_window')
    
    def test_short_signal(self):
        """Test handling of very short signals."""
        processor = SignalProcessor(fs=10000)
        frequencies = np.array([100.0])
        damping = np.array([0.01])
        
        # Very short duration
        time, signal = processor.generate_synthetic_vibration(
            frequencies, damping, duration=0.1, snr_db=20.0
        )
        
        # Should still work
        freqs, amps = processor.compute_fft(signal)
        assert len(freqs) > 0
        assert len(amps) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
