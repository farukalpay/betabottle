# β-Optimization in the Information Bottleneck Framework

[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--2207--6528-brightgreen)](https://orcid.org/0009-0009-2207-6528)

**Author:** Faruk Alpay

This repository contains code and documentation accompanying the manuscript:

**Title:** β-Optimization in the Information Bottleneck Framework: A Theoretical Analysis

**Date:** May 7, 2025 - May 11, 2025

**DOI V1:** [10.22541/au.174664105.57850297/v1](https://doi.org/10.22541/au.174664105.57850297/v1)

**DOI V2:** [10.5281/zenodo.15384382](https://doi.org/10.5281/zenodo.15384382)

**Paper V1:** `Code_v1/paper/enhanced_ib_framework.pdf` (contained within version-specific directories)

**Paper V2:** `code_v2_Multi_Path/paper/enhanced_ib_framework.pdf` (contained within version-specific directories)

The research focuses on the rigorous determination and practical application of the critical parameter $\beta$ in the Information Bottleneck (IB) framework. This repository documents two major stages of the project: an initial validation of the critical phase transition point ($\beta^*$) and an enhanced multi-path optimization strategy.

A key finding of this research is the deterministic identification of the critical phase transition point:
$$ \beta^* = 4.14144 $$
This value was originally formulated using Alpay Algebra—a novel symbolic mathematical system optimized for analyzing phase transitions—and subsequently validated using conventional mathematical notation and computational frameworks.

---

## 📖 Contents and Evolution

This project has evolved through two main code versions:

1.  **Code_v1: Enhanced Information Bottleneck Framework: $\beta^*$-Optimization Validation**
    * Focus: Initial theoretical and computational validation of the critical $\beta^*$.
    * [Jump to Code_v1 Details](#code_v1-enhanced-information-bottleneck-framework-beta-optimization-validation)

2.  **Code_v2: Multi-Path Incremental-$\beta$ Information Bottleneck**
    * Focus: An improved method addressing stability and convergence, building upon the findings of Code_v1.
    * [Jump to Code_v2 Details](#code_v2-multi-path-incremental-beta-information-bottleneck)

---

## Code_v1: Enhanced Information Bottleneck Framework: $\beta^*$-Optimization Validation

*(Based on the initial implementation, located in the `ib-beta-star-validation/Code_v1/` directory)*

This version presented the first deterministic and fully validated computational framework for identifying the critical phase transition point ($\beta^* = 4.14144$) within the Information Bottleneck (IB) methodology.

### 🎯 Project Overview (Code_v1)

* **Objective:** Precise identification of $\beta^*$ through rigorous theoretical and statistical validation.
* **Methodology:** Employed advanced methods including multi-stage optimization, symbolic spline detection, and $\Lambda++$ ensemble initialization.
* **Implementation:** A self-contained Python script (`validate_beta_star.py`) requiring `numpy`, `scipy`, `scikit-learn`, and `matplotlib`.
* **Nature:** Primarily a computational proof-of-concept for theoretical validation.

### 🛡️ Validation and Verification Protocols (Code_v1)

The `validate_beta_star.py` script rigorously confirmed:
* Identification of $\beta^*$ within a computational tolerance of < 0.00001% error from the theoretically derived value (4.14144).
* Successful completion of a stringent validation suite evaluating:
    * Phase Transition Sharpness
    * $\Delta$-Violation Verification
    * Theoretical Alignment
    * Curve Concavity
    * Encoder Stability
    * Information-Theoretic Consistency
* Comprehensive verification through:
    * Confidence interval precision
    * Theoretical alignment with error < 0.01%
    * Monotonicity checks
    * Reproducibility across multiple random seeds
    * Sharpness and clarity of the phase transition
    * Adherence to theoretical predictions above and below $\beta^*$.

### 📊 Graphical Outputs (Code_v1)

Generated plots were stored in `ib-beta-star-validation/ib_plots/`:
* `multiscale_phase_transition.png`
* `information_plane_dynamics.png`
* `gradient_landscape.png`
* `statistical_validation.png`

### 📁 Repository Structure (Code_v1)

```
Code_v1/
├── ib_plots/                   # Auto-generated plots from Code_v1
│   ├── multiscale_phase_transition.png
│   ├── information_plane_dynamics.png
│   ├── gradient_landscape.png
│   └── statistical_validation.png
├── paper/
│   └── enhanced_ib_framework.pdf # Formal manuscript (relevant to Code_v1 context)
├── validate_beta_star.py       # Core β* validation script for Code_v1
```

### ✨ Significance of Results (Code_v1)

* Concretely demonstrated the theoretical and practical efficacy of employing Alpay Algebra to pinpoint $\beta^*$.
* Validated the precise critical value $\beta^* = 4.14144$.
* Introduced computational techniques like symbolic spline detection and $\Lambda++$ hybrid initialization.
* Offered enhanced accuracy, comparable or superior to state-of-the-art IB implementations at the time (e.g., DeepBI).

---

## Code_v2: Multi-Path Incremental-$\beta$ Information Bottleneck

*(Based on the `code_v2_Multi_Path/` repository)*

This second-stage implementation introduces the **Multi-Path Incremental-$\beta$ Optimization Strategy**. It addresses convergence failures observed in standard single-path iterative methods, particularly under conditions of partial correlation or noise. Code_v2 aims for enhanced stability, relevance preservation, and smoother traversal through critical $\beta$ regions, preventing premature collapse to trivial encoders that Code_v1 might experience under certain conditions.

### 🚀 Key Features (Code_v2)

-   **Multi-path evolution:** Parallel paths explore distinct encoder regimes.
-   **Incremental $\beta$ scheduling:** Prevents premature convergence by gradually increasing $\beta$.
-   **Adaptive damping & trimming:** Ensures stable encoder trajectories across the $\beta$-spectrum.
-   **Deterministic convergence:** Reproducible optimization that aligns with theoretical predictions, including the symbolic $\beta^* = 4.14144$ transition.
-   **Refined plots:** Includes information plane and $\beta$-trajectory visualizations for all paths.
-   **JAX acceleration:** Utilizes `jax` for improved computational performance.

### 📁 Directory Structure (Code_v2)

```
code_v2_Multi_Path/
├── multi_path_ib.py            # Core multi-path IB framework
├── paper/
│   └── enhanced_ib_framework.pdf # Associated paper (reflecting Code_v2 advancements)
├── ib_plots/                   # Auto-generated plots from Code_v2
│   ├── critical_region_info_plane_bsc_corrected.png
│   ├── critical_region_beta_trajectories_bsc_corrected.png
│   ├── multi_path_info_plane_demo_corrected.png
│   └── multi_path_beta_trajectories_demo_corrected.png
```

### 📦 Dependencies (Code_v2)

-   Python 3.8+
-   `jax` (with GPU support recommended)
-   `numpy`
-   `scipy`
-   `matplotlib`

Install with:
```bash
pip install jax jaxlib numpy scipy matplotlib
```

### 🧪 Example Usage (Code_v2)

To run the full demonstration for Code_v2 (including visualizations):
```bash
python Code_v2_Multi_Path/run_demo.py
```
Output will include:
* Multi-path evolution over the $\beta$ range.
* Critical $\beta^*$ convergence plots.
* Info plane trajectory visualizations.
* Comparative logs vs. standard IB baseline.

---

## 🔄 Improvements: Code_v1 vs. code_v2_Multi_Path

| Feature                         | Code_v1 (Validation Framework)         | Code_v2 (Multi-Path Framework)         |
|---------------------------------|----------------------------------------|----------------------------------------|
| **Primary Goal** | Validate symbolic $\beta^*$ (4.14144)  | Robust IB optimization across $\beta$ spectrum |
| $\beta$ Scheduling              | Static / Focused on $\beta^*$          | Incremental & adaptive                 |
| Encoder Collapse Prevention   | Less emphasis, potential for collapse  | ✅ Multi-path stability                |
| Critical $\beta^*$ Detection    | Deterministic, high-precision          | Convergent & reproducible (validates $\beta^*$) |
| Information Plane Path Tracking | Basic dynamics plot                    | ✅ Multi-path plot support             |
| Damping & Stabilization       | Implicit or fixed                      | Adaptive per path                      |
| Noise/Partial Correlation       | Not explicitly addressed               | Designed for robustness under these conditions |
| Core Algorithm                  | Multi-stage opt, symbolic splines      | Multi-path incremental evolution       |
| **Dependencies** | numpy, scipy, scikit-learn, matplotlib | numpy, scipy, matplotlib, **jax, jaxlib** |
| JAX Acceleration                | ❌ No                                  | ✅ Yes                                 |
| Plot Generation                 | Validation-focused                     | Full $\beta$-dynamics + multi-path info plane |

---

## 🧠 General Notes & Future Directions

* **Theoretical Validation:** Both codebase versions serve primarily as computational proof-of-concept and theoretical validation frameworks rather than general-purpose libraries. They aim to accurately reproduce theoretical findings, particularly the symbolic $\beta^* = 4.14144$ transition, and test encoder stability.
* **Alpay Algebra:** The underlying theoretical insights from Alpay Algebra drive the methodologies. These repositories serve as groundwork for future, potentially more user-friendly, Alpay Algebra–driven IB modules.
* **Development Focus:**
    * (Done in v1) Rigorously validate $\beta^*$.
    * (Focus of v2) Enhance stability and practical convergence across a range of $\beta$ values and data conditions.
    * (Future) Develop and release a modular IB framework based on Alpay Algebra for broader computational research.
* **Publication:** Findings and associated validation data are intended for dissemination, e.g., via preprint platforms (arXiv categories `cs.IT` or `math.IT`).

---

## 📜 Citation

Users referencing this repository, its code (either Code_v1 or Code_v2), or the associated research in academic or professional contexts should cite the following publication. This paper details the theoretical analysis and findings that are implemented and validated across the different versions of the codebase presented here:

```bibtex
@article{alpay2025beta,
  author  = {Faruk Alpay},
  title   = {{\textgreek{b}}-Optimization in the Information Bottleneck Framework: A Theoretical Analysis},
  journal = {Authorea},
  year    = {2025},
  doi     = {10.22541/au.174664105.57850297/v1},
  url     = {[https://doi.org/10.22541/au.174664105.57850297/v1](https://doi.org/10.22541/au.174664105.57850297/v1)}
}

---

## 📄 Licensing

This project is distributed under the **MIT License**, permitting unrestricted academic and educational use. Commercial inquiries should be directed to the author. See the `LICENSE` file in the respective code directories for details.

---

## 📬 Contact Information

**Faruk Alpay**
* ORCID: [https://orcid.org/0009-0009-2207-6528](https://orcid.org/0009-0009-2207-6528)
* Email: alpay@lightcap.ai
