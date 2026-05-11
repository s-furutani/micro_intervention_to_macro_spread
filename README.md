# Herd Immunity Against Misinformation

Code for reproducing the results in the paper on "The Impact of Micro-level User Interventions on Macro-level Misinformation Spread". This repository implements a **Continuous-Time Independent Cascade (CTIC)** model for misinformation spread on social networks and evaluates three intervention strategies: **Prebunking**, **Contextualization**, and **Nudging**, both via simulation and quenched mean-field theory.

## Directory Structure

```
.
├── CTIC.py                              # CTIC model (simulation engine)
├── load_graph.py                        # Raw Nikolov data loaders (also caches as .pkl)
├── graphs.py                            # Unified `load_graph_by_name(name)`
├── params.py                            # Best-fit (η, λ) and seed user IDs
├── paths.py                             # `results/*.npy` path / suffix builders
├── prebunking_targets.py                # Shared prebunking target selection
├── seed_users.py                        # `get_seed_users` (re-exported via util.py)
├── plot_style.py                        # Matplotlib rcParams shared across plots
├── plot_violin.py                       # Violin / boxplot helpers (re-exported via util.py)
├── plot_heatmap.py                      # Heatmap helpers (re-exported via util.py)
├── util.py                              # Backward-compatible facade for the above
├── intervention_analysis_common.py      # Shared engine for fixed-η / varying-η heatmaps
├── intervention_analysis.py             # Heatmaps with fixed η (CLI)
├── intervention_analysis_varying_eta.py # Heatmaps with varying η (CLI)
├── single_vs_combined.py                # Single vs. combined intervention comparison
├── estimate_eta_lam.py                  # Estimation of η, λ (fitting to Twitter cascades)
├── estimate_eta_lam_cma.py              # CMA-ES variant of η, λ estimation
├── estimate_eta_lam_abc.py              # ABC-SMC variant of η, λ estimation
├── estimate_epsilon.py                  # Estimation of intervention effect ε
├── quenched_mean_field.py               # Critical threshold via quenched mean-field theory
├── preprocess_intervention_dataset.py   # Preprocessing of intervention datasets
├── plot_results.py                      # Visualization of results (Jupyter notebook style)
├── requirements.txt
├── data/
│   ├── nikolov/                         # Nikolov social network data
│   ├── twitter_diffusion_dataset/       # Twitter cascade data (for parameter estimation)
│   ├── intervention_dataset/            # Intervention experiment datasets (for ε estimation)
│   └── politifact/                      # PolitiFact article data
└── results/
    ├── nikolov/                         # Results on Nikolov graph
    ├── randomized_nikolov/              # Results on randomized Nikolov graph
    ├── uniform_nikolov/                 # Results on uniform-susceptibility Nikolov topology
    └── test/                            # Results on test graph
```

## Setup

```bash
pip install -r requirements.txt
```

## File Descriptions

### Core Model

| File | Description |
|------|-------------|
| `CTIC.py` | CTIC model implementation. Event-driven simulation of misinformation spread with support for Prebunking (pre-exposure), Contextualization (post-exposure), and Nudging (population-wide) interventions. |
| `load_graph.py` | Builds and caches the Nikolov graph. `Nikolov_susceptibility_graph()` returns the original graph; `randomized_nikolov_graph()` shuffles susceptibilities; `uniform_nikolov_graph()` keeps the same topology with every node’s susceptibility set to the population mean. |

### Parameter Estimation

| File | Description |
|------|-------------|
| `estimate_eta_lam.py` | Estimates misinformation contagiousness η and delay rate λ by fitting simulated cascade curves to Twitter cascade data. |
| `estimate_epsilon.py` | Estimates intervention effect ε from intervention experiment data (Pennycook 2020/2021, Fazio 2020, Basol 2021, Drolsbach 2024). |
| `quenched_mean_field.py` | Computes critical transmission threshold from the adjacency matrix and susceptibility vector using quenched mean-field theory. |

### Intervention Simulation

| File | Description |
|------|-------------|
| `intervention_analysis.py` | Generates heatmaps over intervention parameter space with fixed η. Explores (ε_pre, δ_pre) for Prebunking and (ε_ctx, φ_ctx) for Contextualization. |
| `intervention_analysis_varying_eta.py` | Generates heatmaps over (ε, η) space. Supports Prebunking, Contextualization, and Nudging. |
| `single_vs_combined.py` | Compares five conditions: no intervention, Prebunking, Contextualization, Nudging, and combined intervention; saves prevalence distributions. |

### Preprocessing and Visualization

| File | Description |
|------|-------------|
| `preprocess_intervention_dataset.py` | Converts multiple intervention datasets (Pennycook, Fazio, Basol, Drolsbach) into a unified format. |
| `plot_results.py` | Visualization script (Jupyter notebook style) for heatmaps, violin plots, line plots, and critical curves. |
| `util.py` | Backward-compatible facade re-exporting the helpers below. Existing `import util` keeps working. |

### Shared helper modules (introduced by the refactor)

| File | Description |
|------|-------------|
| `graphs.py` | Single source of truth for `load_graph_by_name(name)` (used by `CTIC.py`, the intervention scripts, etc.). |
| `params.py` | Best-fit `(η, λ)` for Nikolov-family graphs, plus the canonical seed user IDs. |
| `paths.py` | `results/*.npy` filename construction (seed-mode / eta-scale / γ suffix rules). |
| `prebunking_targets.py` | `select_prebunking_targets()` shared by the simulator and the QMF analysis. |
| `seed_users.py` | `get_seed_users()` (re-exported via `util.py`). |
| `plot_style.py` | Matplotlib `rcParams` shared across plotting modules. |
| `plot_violin.py` | Violin / boxplot helpers (re-exported via `util.py`). |
| `plot_heatmap.py` | Heatmap helpers including `plot_multiple_heatmaps` (re-exported via `util.py`). |
| `intervention_analysis_common.py` | Shared `create_heatmap` / `save_heatmap_data` engine used by both intervention CLIs. |

## Usage

### Intervention Simulation (Fixed η)

```bash
# Prebunking (ε_pre × δ_pre heatmap)
python intervention_analysis.py -i prebunking -g nikolov -s 10 --save_data --target_selection random

# Contextualization (ε_ctx × φ_ctx heatmap)
python intervention_analysis.py -i contextualization -g nikolov -s 10 --save_data

# Prebunking (ε_pre × η heatmap)
python intervention_analysis_varying_eta.py -i prebunking -g nikolov -s 10 --save_data

# Contextualization (ε_ctx × η heatmap)
python intervention_analysis_varying_eta.py -i contextualization -g nikolov -s 10 --save_data

# Nudging (ε_nud × η heatmap)
python intervention_analysis_varying_eta.py -i nudging -g nikolov -s 10 --save_data

# Ten random initial spreaders (same --seed fixes which 10 nodes are chosen)
python intervention_analysis.py -i prebunking -g nikolov -s 10 --save_data --seed_mode random10 --seed 42
python intervention_analysis_varying_eta.py -i contextualization -g nikolov -s 10 --save_data --seed_mode random10
```

**Arguments (both scripts):**

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--intervention` | Intervention type | required |
| `-g`, `--graph` | Graph name (`test`, `nikolov`, `randomized_nikolov`, `uniform_nikolov`) | `test` |
| `-s`, `--simulations` | Number of simulations per grid point | `10` (`intervention_analysis.py`), `100` (`intervention_analysis_varying_eta.py`) |
| `--seed` | Random seed for simulations and (for `random10`) for choosing the 10 seed nodes | `42` |
| `--save_data` | Save results as `.npy` files | `false` |
| `--target_selection` | Target selection strategy (`random`, `high_degree`, `high_susceptible`, `cocoon`); prebunking only | `random` |
| `--seed_mode` | Initial spreaders passed to `util.get_seed_users`: `single_largest_degree` (alias `single`), `multiple_moderate_degree`, or `random10` (uses `--seed` for sampling) | `single_largest_degree` |
| `--eta_scale` | η scale `0.5` / `1` / `2`. Saving suffix: `_half` / `_double` | `1` |

**Saved filenames:** Non-default modes add a suffix before `.npy`: `_seed10` for `random10`, `_seed_mod3` for `multiple_moderate_degree` (e.g. `prebunking_random_seed10.npy`). Default `single_largest_degree` keeps paths unchanged. Plot scripts must match the chosen `--seed_mode`.


## Datasets

| Directory | Description |
|-----------|-------------|
| `data/nikolov/` | Nikolov social network: friend links (`anonymized-friends.json`) and susceptibility scores (`measures.tab`). A cached graph is written as `nikolov_graph.pkl`. |
| `data/twitter_diffusion_dataset/` | Twitter information diffusion cascades, used for estimating η and λ. |
| `data/intervention_dataset/` | Intervention experiment datasets (Pennycook 2020/2021, Fazio 2020, Basol 2021, Drolsbach 2024), used for estimating ε. |

## Notation

| Symbol | Variable | Meaning |
|--------|----------|---------|
| η | `eta` | Misinformation contagiousness |
| λ | `lam` | Exponential delay parameter |
| ε_pre | `epsilon_pre` | Prebunking effect strength |
| ε_ctx | `epsilon_ctx` | Contextualization effect strength |
| ε_nud | `epsilon_nud` | Nudging effect strength |
| δ_pre | `delta_pre` | Fraction of nodes targeted by Prebunking |
| φ_ctx | `intervention_timing` | Contextualization trigger parameter |
