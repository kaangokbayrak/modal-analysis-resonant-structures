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
| **Visualisation** | Mode shape, FRF, convergence, and design-space plots via `Visualizer` |

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): `numpy`, `scipy`, `matplotlib`.

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
│   └── visualization.py     # Matplotlib plotting helpers
├── tests/
│   ├── test_analytical.py
│   ├── test_fem_solver.py
│   ├── test_modal_analysis.py
│   ├── test_parametric_study.py
│   └── test_signal_processing.py
├── examples/                # Jupyter notebooks / scripts
├── results/                 # Generated output files
├── .github/workflows/ci.yml # GitHub Actions CI
└── requirements.txt
```

---

## Theory Background

### Euler-Bernoulli vs Timoshenko Beam Theory

The **Euler-Bernoulli** model assumes plane sections remain plane and shear
deformation is negligible — accurate when the slenderness ratio L/h ≫ 10.
For deep or thick beams, **Timoshenko** theory adds a shear correction factor
κ and rotatory inertia, reducing the over-prediction of natural frequencies.
This codebase implements the Euler-Bernoulli formulation; the thin beams
targeted (e.g. 300 mm × 3 mm) fall squarely in the valid regime.

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

### FRF and MAC

The **Frequency Response Function** H(ω) is reconstructed from the modal
superposition:

```
H(ω) = Σ_n  φₙ[out] φₙ[in] / [Mₙ (ωₙ² − ω² + 2j ζₙ ωₙ ω)]
```

The **Modal Assurance Criterion** quantifies similarity between two mode
shape vectors:

```
MAC[i,j] = |φᵢᵀ ψⱼ|² / [(φᵢᵀ φᵢ)(ψⱼᵀ ψⱼ)]  ∈ [0, 1]
```

A value of 1 indicates identical (or co-linear) shapes; values near 0
indicate orthogonality.

### Signal Processing

`SignalProcessor` wraps NumPy/SciPy FFT routines with Hann, Hamming,
Blackman, and rectangular windows.  Damping ratios are estimated via the
**half-power bandwidth** method or the **logarithmic decrement** from
free-decay records.

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
