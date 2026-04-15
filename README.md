# modal-analysis-resonant-structures

[![CI](https://github.com/kaangokbayrak/modal-analysis-resonant-structures/actions/workflows/ci.yml/badge.svg)](https://github.com/kaangokbayrak/modal-analysis-resonant-structures/actions/workflows/ci.yml)

Computational modal analysis workbench for resonant structures — custom
Euler-Bernoulli FEM solver, analytical closed-form solutions, FRF/MAC
post-processing, FFT-based signal processing, and parametric optimisation,
all in pure Python.

---

## Features

| Capability | Details |
|---|---|
| **Analytical solver** | Euler-Bernoulli closed-form natural frequencies & mode shapes (cantilever, simply-supported, fixed-fixed, fixed-pinned) |
| **FEM solver** | Hermitian beam elements, consistent mass matrix, mesh-convergence study |
| **Boundary conditions** | Cantilever, simply-supported, fixed-fixed, fixed-pinned |
| **Modal post-processing** | FRF, FRF matrix, MAC matrix, modal participation factors, effective mass fractions |
| **Dynamic responses** | Steady-state harmonic response, impulse response function |
| **Signal processing** | FFT (Welch PSD), windowing, peak detection, half-power & log-decrement damping estimation |
| **Parametric study** | 1-D and 2-D parameter sweeps, constrained mass minimisation (SLSQP), Pareto analysis, harmonic avoidance, material selection |
| **Export** | CSV export of sweep / design-space results |
| **Visualisation** | 16 publication-quality figures including mode shapes, FRF, MAC, impulse response, convergence, design-space, and a summary dashboard |

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): `numpy`, `scipy`, `matplotlib`, `seaborn`, `pillow`.

---

## Quick Start

```python
from src.beam import Material, Beam
from src.analytical import AnalyticalSolver
from src.fem_solver import FEMSolver
from src.modal_analysis import from_fem_results

# --- Define beam ---
steel = Material.steel()
beam = Beam(material=steel, length=0.3, width=0.025, thickness=0.003)

# --- Analytical solution ---
analytical = AnalyticalSolver(beam)
freqs_hz = analytical.natural_frequencies(n_modes=3, bc='cantilever')
print("Analytical [Hz]:", freqs_hz)

# --- FEM solution ---
fem = FEMSolver(beam, n_elements=50, bc='cantilever')
results = fem.solve(n_modes=3)
print("FEM [Hz]:", results['frequencies_hz'])

# --- Modal post-processing ---
ma = from_fem_results(results)

import numpy as np
omega = np.linspace(1, 20_000, 5000)
_, H = ma.compute_frf(omega, dof_out=2, dof_in=2)

emf = ma.effective_mass_fractions()
print("Effective mass fractions:", emf)
```

---

## Full Analysis Workflow

Run the complete end-to-end analysis:

```bash
python examples/run_full_analysis.py
```

This produces **16 publication-quality figures** in `docs/figures/`:

| Figure | Description |
|---|---|
| `mode_shapes.png` | First 4 mode shapes with per-mode colouring, node-point markers, and fixed-end indicator |
| `mode_animation.gif` | Animated GIF of the first mode oscillating through one full cycle |
| `fft_spectrum.png` | FFT amplitude spectrum (log scale) with detected peaks and theoretical frequency markers |
| `mesh_convergence.png` | FEM convergence study — relative error vs element count for each mode |
| `frequency_comparison.png` | Grouped bar chart comparing Analytical / FEM / FFT frequencies |
| `parametric_thickness.png` | Natural frequencies vs beam thickness, with resonance-risk zone band |
| `parametric_length.png` | Natural frequencies vs beam length, with resonance-risk zone band |
| `design_space_map.png` | 2-D contour map of first natural frequency over the thickness × length design space |
| `validation_table.png` | Three-way validation table (Analytical vs FEM vs FFT, with % errors) |
| `time_signal.png` | Time-domain synthetic vibration signal with RMS annotation |
| `optimization_convergence.png` | Optimization history: frequency and mass convergence over iterations |
| `frf_plot.png` | Frequency Response Function — magnitude (dB) + phase (°) two-panel figure |
| `mac_matrix.png` | Modal Assurance Criterion heatmap — Analytical vs FEM mode shapes |
| `impulse_response.png` | Unit impulse response function with Hilbert envelope |
| `summary_dashboard.png` | 2×3 dashboard combining mode shapes, FRF, FFT, convergence, design space, and frequency table |
| `damping_estimation.png` | Half-power bandwidth method illustrated on first resonance peak |

---

## Project Structure

```
modal-analysis-resonant-structures/
├── src/
│   ├── __init__.py          # Package exports + version
│   ├── beam.py              # Material, Beam, cross-section helpers
│   ├── analytical.py        # Closed-form Euler-Bernoulli solver
│   ├── fem_solver.py        # Custom FEM solver (Hermitian elements)
│   ├── modal_analysis.py    # ModalAnalyzer: FRF, MAC, impulse response
│   ├── parametric_study.py  # Sweeps, optimisation, Pareto analysis
│   ├── signal_processing.py # FFT / Welch PSD, damping estimation
│   └── visualization.py     # Matplotlib plotting helpers (16 plot types)
├── tests/
│   ├── test_analytical.py
│   ├── test_fem_solver.py
│   ├── test_modal_analysis.py
│   ├── test_parametric_study.py
│   └── test_signal_processing.py
├── examples/
│   ├── run_full_analysis.py # Complete 16-figure workflow
│   └── README.md            # Examples documentation
├── docs/figures/            # Generated output figures
├── results/                 # CSV exports and other results
├── .github/workflows/ci.yml # GitHub Actions CI
└── requirements.txt
```

---

## Theory Background

### Engineering Problem

The simulation models a **steel cantilever beam** (300 mm × 25 mm × 3 mm) in a mechanical
system with a 50 Hz pump.  The workflow answers three questions:

1. **Where are the resonances?** — solved analytically (closed-form) and with FEM, cross-validated by FFT peak detection on synthetic data.
2. **Is the design safe?** — a parametric sweep maps how thickness and length shift frequencies into or out of the 50–70 Hz danger zone.
3. **How do we fix it?** — a constrained SLSQP optimiser finds the minimum-mass design that pushes the first natural frequency above 65 Hz.

---

### Euler-Bernoulli vs Timoshenko Beam Theory

The **Euler-Bernoulli** model assumes plane sections remain plane and shear
deformation is negligible — accurate when the slenderness ratio L/h ≫ 10.
For deep or thick beams, **Timoshenko** theory adds a shear correction factor
κ and rotatory inertia, reducing the over-prediction of natural frequencies.
This codebase implements the Euler-Bernoulli formulation; the thin beams
targeted (e.g. 300 mm × 3 mm, L/h = 100) fall squarely in the valid regime.

---

### FEM Formulation

Each two-node beam element has four DOFs (v₁, θ₁, v₂, θ₂).  The stiffness
and consistent mass matrices are derived from the Hermitian cubic shape
functions:

```
K_e = EI/L³ · [[12, 6L, -12, 6L],
                [6L, 4L², -6L, 2L²], ...]

M_e = ρAL/420 · [[156, 22L, 54, -13L], ...]
```

Global matrices are assembled by standard DOF mapping, boundary conditions
are enforced by row/column elimination, and natural frequencies are obtained
from the generalised eigenvalue problem **K φ = ω² M φ**.

The mesh convergence study confirms O(h²) error reduction, validating the
Hermitian element implementation.

---

### FRF and MAC

The **Frequency Response Function** H(ω) is reconstructed from the modal
superposition:

```
H(ω) = Σ_n  φₙ[out] φₙ[in] / [Mₙ (ωₙ² − ω² + 2j ζₙ ωₙ ω)]
```

The FRF plot shows **magnitude in dB** (20 log₁₀|H|) and **phase in degrees**
in a two-panel figure.  Resonance peaks appear as local maxima with 180° phase
transitions — a signature of lightly damped structural modes.

The **Modal Assurance Criterion** quantifies similarity between two mode
shape vectors:

```
MAC[i,j] = |φᵢᵀ ψⱼ|² / [(φᵢᵀ φᵢ)(ψⱼᵀ ψⱼ)]  ∈ [0, 1]
```

A diagonal MAC matrix with values near 1 (and near-zero off-diagonals) confirms
that FEM and analytical mode shapes agree and are mutually orthogonal.

---

### Impulse Response Function

The unit impulse response h(t) is the inverse Fourier transform of H(ω):

```
h(t) = Σ_n  φₙ[out] φₙ[in] / (Mₙ ωdₙ)  · exp(−ζₙ ωₙ t) · sin(ωdₙ t)
```

where ωdₙ = ωₙ √(1 − ζₙ²) is the damped natural frequency.  The plot shows
the multi-modal free-decay signal and its Hilbert envelope — a visual
confirmation of modal superposition and damping.

---

### Signal Processing

`SignalProcessor` wraps NumPy/SciPy FFT routines with Hann, Hamming,
Blackman, and rectangular windows.  The synthetic vibration signal is:

```
x(t) = Σₙ Aₙ exp(−ζₙ ωₙ t) sin(ωdₙ t) + noise
```

Damping ratios are estimated via the **half-power bandwidth** method
(Δf / 2f₀) or the **logarithmic decrement** from free-decay records.

---

### Design Optimisation

The optimiser minimises beam mass subject to the constraint that the first
natural frequency exceeds a target value:

```
minimise   m = ρ A L
subject to f₁(t) ≥ f_target    (t = thickness)
           t_min ≤ t ≤ t_max
```

Solved via SciPy's SLSQP algorithm.  The convergence history shows frequency
and mass evolving over iterations until the constraint is met.

---

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

63 tests covering analytical frequencies, FEM convergence, modal
post-processing, parametric optimisation, and signal processing.

---

## License

MIT — see [LICENSE](LICENSE).
