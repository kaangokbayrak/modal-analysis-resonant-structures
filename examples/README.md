# Examples Directory

This directory contains example scripts demonstrating the modal analysis workflow.

## Main Entry Point: `run_full_analysis.py`

This comprehensive script demonstrates the entire modal analysis workflow for a steel
cantilever beam and generates **16 publication-quality figures**.

### Quick Start

```bash
# From the project root directory
python examples/run_full_analysis.py
```

### What It Does

The script performs a complete modal analysis workflow in 11 steps:

1. **Beam Definition** — Creates a steel cantilever beam (300 mm × 25 mm × 3 mm)
2. **Analytical Solution** — Computes natural frequencies and mode shapes using Euler-Bernoulli theory
3. **FEM Solution** — Solves using a custom finite element implementation (100 elements, consistent mass matrix)
4. **Mesh Convergence** — Tests convergence with 5, 10, 20, 50, 100, 200 elements, confirming O(h²) error reduction
5. **Signal Processing** — Generates a synthetic multi-modal vibration signal (2 s, 10 kHz, SNR = 40 dB) and performs FFT analysis
6. **Three-Way Validation** — Compares Analytical vs FEM vs FFT results with error table
7. **Parametric Studies** — Sweeps thickness (2–5 mm) and length (200–400 mm) parameters
8. **Design Optimization** — Finds the minimum-mass thickness that shifts f₁ > 65 Hz (avoiding the 50 Hz pump)
9. **Modal Post-Processing** — Computes FRF, MAC matrix, and impulse response using `ModalAnalyzer`
10. **Visualization** — Generates all 16 figures (see table below)
11. **Summary** — Prints a comprehensive text summary with key findings

### Engineering Application

The script addresses a real-world resonance avoidance problem:

> A mechanical system contains a 50 Hz pump that creates vibration.
> The beam's first natural frequency lies within the operational range (50–70 Hz),
> creating a resonance risk.  The optimiser redesigns the beam by adjusting
> thickness to shift f₁ above 65 Hz while tracking the mass penalty.

### Output Figures

All figures are saved to `docs/figures/`:

| Filename | Description |
|---|---|
| `mode_shapes.png` | First 4 mode shapes with per-mode colour scheme, node-point markers, and a fixed-end wall indicator |
| `mode_animation.gif` | Animated GIF of Mode 1 oscillating through one full cycle |
| `fft_spectrum.png` | FFT amplitude spectrum (log scale) with detected peaks and analytical frequency overlays |
| `mesh_convergence.png` | Relative FEM error vs element count for each mode; includes O(h²) reference line |
| `frequency_comparison.png` | Grouped bar chart comparing Analytical / FEM / FFT natural frequencies |
| `parametric_thickness.png` | Natural frequencies vs thickness, with 50–70 Hz resonance-risk zone shaded |
| `parametric_length.png` | Natural frequencies vs length, with 50–70 Hz resonance-risk zone shaded |
| `design_space_map.png` | 2-D contour map of first natural frequency over thickness × length design space, unsafe region highlighted |
| `validation_table.png` | Three-way validation table (Analytical, FEM, FFT frequencies and % errors) |
| `time_signal.png` | Time-domain synthetic vibration signal (first 100 ms) with RMS annotation |
| `optimization_convergence.png` | Two-panel plot: frequency and mass convergence over SLSQP iterations |
| `frf_plot.png` | Frequency Response Function — magnitude (dB) and phase (°) with resonance peak annotations |
| `mac_matrix.png` | Modal Assurance Criterion heatmap comparing analytical vs FEM mode shapes |
| `impulse_response.png` | Unit impulse response function with Hilbert envelope, illustrating damped multi-modal vibration |
| `summary_dashboard.png` | 2×3 overview dashboard combining mode shapes, FRF, FFT, convergence, design space, and frequency table |
| `damping_estimation.png` | Half-power bandwidth method illustrated on the first resonance peak |

### Dependencies

```bash
pip install -r ../requirements.txt
```

### Execution Time

The complete analysis takes approximately 2–4 minutes to run, including:
- Mesh convergence study (6 mesh densities)
- Parametric sweeps (20 points each, 2 sweeps)
- 2-D design space mapping (15 × 15 grid)
- Optimization (SLSQP iterations)
- FRF computation (8 000 frequency points)
- All 16 visualizations

### Expected Console Output

The script prints:
- Beam properties and material data
- Natural frequencies from all three methods
- Three-way validation table
- Optimization results (initial vs optimal design)
- Comprehensive summary with key findings

Exit status 0 indicates successful completion.
