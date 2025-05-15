# β-Optimization in the Information Bottleneck Framework  
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--2207--6528-brightgreen)](https://orcid.org/0009-0009-2207-6528)

**Author:** Faruk Alpay

| Version | Title | Date | DOI / License |
|---------|-------|------|---------------|
| **V1** | *β-Optimization in the Information Bottleneck Framework: A Theoretical Analysis* | 7 – 11 May 2025 | 10.22541/au.174664105.57850297 / MIT |
| **V2** | *β-Optimization … Multi-Path Extension* | 12 – 27 May 2025 | 10.5281/zenodo.15384382 / MIT |
| **V3 (current)** | *Stable and Convexified Information Bottleneck Optimization via Symbolic Continuation and Entropy-Regularized Trajectories* | ≥ 12 May 2025 | 10.13140/RG.2.2.12135.15521 / CC-BY-4.0 |
| **V4 (planned)** | *Proof-Tight & Large-Scale Continuation IB* | Q4 2025 (target) | T B A |

> **Please cite V3** for new work; older DOIs remain valid for archival purposes.

---

## 📂 Repository map

```
Code_v1/                      # β* validation framework
code_v2_Multi_Path/           # multi-path incremental-β solver
code_v3_Stable_Continuation/  # NEW: convex + entropy + continuation
docs/                         # legacy citations, notes
LICENSE
README.md
```

### Quick PDFs

| Version | Path |
|---------|------|
| **V1** | `Code_v1/paper/enhanced_ib_framework.pdf` |
| **V2** | `code_v2_Multi_Path/paper/enhanced_ib_framework.pdf` |
| **V3** | `code_v3_Stable_Continuation/paper/stable_convex_ib.pdf` |

---

## Code_v3 — Stable Continuation IB (Convex + Entropy)  

*(directory `code_v3_Stable_Continuation/`)*

| File | Role |
|------|------|
| `stable_continuation_ib.py` | Predictor–corrector solver implementing \(u(t)=t^2\) and small entropy penalty |
| `requirements.txt` | `numpy`, `scipy`, `jax` (GPU optional), `matplotlib` |
| `ib_plots/` | `bsc_critical_region.png`, `bsc_phase_transition_detection.png`, `continuation_ib_results.png`, `encoder_comparison.png`, `encoder_evolution.png`, `enhanced_multipath_best_encoder.png`, `enhanced_multipath_beta_trajectories.png`, `enhanced_multipath_convergence.png`, `enhanced_multipath_info_plane.png`, `ib_curve_comparison.png`, `izy_vs_beta_continuation.png` |
| `paper/stable_convex_ib.pdf` | V3 manuscript (same as DOI) |

Run the demo:

```bash
python code_v3_Stable_Continuation/stable_continuation_ib.py 
```

Outputs the figures above and reproduces the BSC & 8×8 experiments (see Figures 1–5 in the PDF).

---

## 🔄 Improvements: Version Comparison

### Code_v1 vs. code_v2_Multi_Path vs. code_v3_Stable_Continuation
| Feature                         | Code_v1 (Validation Framework)         | Code_v2 (Multi-Path Framework)         | Code_v3 (Stable Continuation)          |
|---------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| **Primary Goal** | Validate symbolic $\beta^*$ (4.14144)  | Prevent encoder collapse & robust IB optimization across $\beta$ spectrum | Eliminate phase jumps via symbolic continuation & convexification |
| $\beta$ Scheduling              | Static / Focused on $\beta^*$          | Incremental & adaptive with gradual increase | Predictor-corrector ODE with continuation |
| Encoder Collapse Prevention     | Structural KL convergence criteria     | ✅ Multi-path stability (multiple parallel solutions) | ✅ Entropy regularization + convex surrogate |
| Critical $\beta^*$ Detection    | Deterministic, high-precision          | Multi-method estimation with gradient tracking | Guaranteed via Hessian eigenvalue monitoring |
| Information Plane Path Tracking | Basic dynamics plot                    | Multi-path visualization with solution trajectories | Continuous trajectory & bifurcation visualization |
| Damping & Stabilization         | Adaptive based on convergence behavior | Adaptive per path with local iterations | ✅ Automatic via ODE continuation |
| Convex Surrogate Function       | ❌ None                                | ❌ None                                | ✅ $u(t)=t^2$ |
| Entropy Regularization          | ❌ None                                | ❌ None                                | ✅ Constant small $\varepsilon$ |
| Bifurcation Handling            | ❌ Limited                             | Path selection & merging               | ✅ Explicit detection via Hessian eigenvalues |
| Core Algorithm                  | Staged optimization, symbolic β*       | JIT-compiled multi-path incremental evolution | Predictor-corrector ODE with implicit function continuation |
| **Dependencies** | numpy, scipy, scikit-learn, matplotlib | numpy, scipy, matplotlib, **jax, jaxlib**, (sympy optional) | numpy, scipy, **jax, jaxlib**, matplotlib |
| JAX Acceleration                | ❌ No                                  | ✅ Yes (JIT-compiled core functions)   | ✅ Yes (64-bit precision enabled) |
| Visualization                   | Static plots, convergence tracking     | Solution paths, β trajectories, multi-path info plane | Solution trajectories & bifurcation visualization |

---

## 🔄 Improvements across versions

| Feature | V1 | V2 | V3 | V4 (planned) |
|---------|----|----|----|--------------| 
| Goal | β* proof | Multi-path robustness | Eliminate phase jumps | Proof-tight, large-scale |
| Convex surrogate (u(t)) | — | — | (t^2) | Adaptive slope |
| Entropy regulator (ε) | — | — | constant small | Annealed ε(β) |
| Continuation | — | β-grid multi-path | Predictor-corrector ODE | Arc-length continuation |
| Dataset scale | 2×2, 8×8 | 8×8 | 2×2, 8×8 | MNIST, CIFAR-10 |
| JAX / GPU | — | ✅ | ✅ | ✅+TPU |
| Package | script | script | script | pip package |
| Proof rigor | β* lemma | empirical | convexity lemma | full theorem set |
| Target venue | Authorea | Zenodo | arXiv | Springer-Nature |

---

## 🔮 v4 Roadmap (Q4 2025)

- Full formal proof of global convexity + uniqueness.
- Adaptive entropy schedule linked to Hessian condition number.
- Gaussian/Variational IB demo on MNIST & CIFAR-10.
- Arc-length continuation for automatic step control.
- Package ib-continuation on PyPI with CLI ib-trace.
- Submit Springer-Nature manuscript (sn-article.cls).

---

## 📜 Citation

```
@article{alpay2025stableIB,
  author  = {Faruk Alpay},
  title   = {Stable and Convexified Information Bottleneck Optimization via Symbolic Continuation and Entropy-Regularized Trajectories},
  journal = {arXiv preprint arXiv:2505.09239},
  year    = {2025},
  note    = {Version 1. Please cite this version unless a newer version appears on arXiv},
  url     = {https://doi.org/10.48550/arXiv.2505.09239}
}
```

(Legacy BibTeX for V1 and V2 lives in docs/old_citations.bib.)

---

## 📄 License

MIT for academic/educational use.
Commercial enquiries → alpay@lightcap.ai

---

## 📬 Contact

Faruk Alpay · ORCID 0009-0009-2207-6528 · alpay@lightcap.ai
