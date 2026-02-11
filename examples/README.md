# Examples Directory

This directory contains example scripts demonstrating the modal analysis workflow.

## Main Entry Point: `run_full_analysis.py`

This comprehensive script demonstrates the entire modal analysis workflow for a steel cantilever beam.

### Quick Start

```bash
# From the project root directory
python examples/run_full_analysis.py
```

### What It Does

The script performs a complete modal analysis workflow including:

1. **Beam Definition** - Creates a steel cantilever beam (300mm × 25mm × 3mm)
2. **Analytical Solution** - Computes natural frequencies using Euler-Bernoulli theory
3. **FEM Solution** - Solves using custom finite element implementation (100 elements)
4. **Mesh Convergence** - Tests convergence with 5, 10, 20, 50, 100, 200 elements
5. **Signal Processing** - Generates synthetic vibration data and performs FFT analysis
6. **Three-Way Validation** - Compares Analytical vs FEM vs FFT results
7. **Parametric Studies** - Sweeps thickness (2-5mm) and length (200-400mm) parameters
8. **Design Optimization** - Finds optimal thickness to avoid 50 Hz resonance
9. **Comprehensive Visualization** - Generates 11 publication-quality figures

### Output

All generated figures are saved to `docs/figures/`:

- `mode_shapes.png` - First 4 mode shapes
- `mode_animation.gif` - Animated mode shapes (first 3 modes)
- `fft_spectrum.png` - FFT spectrum with detected peaks
- `mesh_convergence.png` - Convergence study results
- `frequency_comparison.png` - Bar chart comparing all methods
- `parametric_thickness.png` - Thickness parametric sweep
- `parametric_length.png` - Length parametric sweep
- `design_space_map.png` - 2D design space contour map
- `validation_table.png` - Three-way validation table
- `time_signal.png` - Time-domain vibration signal
- `optimization_convergence.png` - Optimization iteration history

### Engineering Application

The script demonstrates a real-world engineering scenario:

**Problem**: A mechanical system has a 50 Hz pump that creates vibration. The beam's first natural frequency falls within the operational range, creating a resonance risk.

**Solution**: The optimization routine redesigns the beam by adjusting thickness to shift the first natural frequency above 65 Hz, successfully avoiding resonance while tracking the mass penalty.

### Dependencies

Ensure all requirements are installed:

```bash
pip install -r ../requirements.txt
```

### Execution Time

The complete analysis takes approximately 2-3 minutes to run, including:
- Mesh convergence study
- Parametric sweeps
- Optimization
- All visualizations

### Expected Output

The script will print:
- Beam properties and material data
- Natural frequencies from all three methods
- Three-way validation table
- Optimization results
- Comprehensive summary with key findings

Exit status 0 indicates successful completion.
