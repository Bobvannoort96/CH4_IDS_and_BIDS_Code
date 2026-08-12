# Misclosure Space Partitioning of Iterative Datasnooping

Code accompanying

> Chapter 4 of the dissertation by B.G. van Noort, on the yet to be published paper:
> B.G. van Noort, P.J.G. Teunissen and C.C.J.M. Tiberius,
> *Misclosure Space Partitioning of Iterative Datasnooping*.

The code computes the **misclosure space partitioning (MSP)** of outlier-identification
testing procedures within the DIA estimator framework, and the testing probabilities that
follow from it (P<sub>CA</sub>, P<sub>FA</sub>, P<sub>CD</sub>, P<sub>MD</sub>,
P<sub>CI</sub>, P<sub>WI</sub>, P<sub>UA</sub>), together with masking and swamping
probabilities.

Four testing procedures are implemented:

| Procedure in the paper | Keyword in the code | Types |
|---|---|---|
| Datasnooping (DS) | `"DS"` | A, B, C, D |
| Iterative datasnooping (IDS) | `"IDS"` | A, B, C |
| Backward IDS (BIDS) | `"R_IDS"` | A, B, C |
| Traditional DIA | `"classical DIA"` | — |

> **Naming.** Backward IDS is called *reverse* IDS (`R_IDS`, `RIDS`) throughout the source
> code. The two terms mean the same thing due to it initially being called reverse IDS.

---

## 1. Before you run anything

### 1.1 Requirements

```bash
pip install -r requirements.txt
```

MATLAB (R2024b or newer) is needed only for the misclosure-sphere figures.

### 1.2 Hard-coded paths — read this first

**All input and output paths in this code are absolute paths on the author's machine**, of
the form

```python
r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\B_transpose_matrix.txt"
```

Nothing will run until you replace these with paths on your own system. They occur in:

| File | Lines containing an absolute path |
|---|---|
| `Functions.py` | 187, 195, 466, 496, 558, 586, 1983, 2277 |
| `IDS_partitioning_on_2D_sphere.py` | 67, 79, 125, 128, 190 |
| `RIDS_partitioning_on_3D_sphere.py` | 103, 108, 120, 125 |
| `ordinary_partitioning_example_3D.py` | 23, 149, 421 |
| `Ordinary_DS_partitioning_example.py` | 19 |
| `Evaluate_mean_penalties.py` | 321 |
| `Plot_true_alpha_RIDS.py` | 80, 130 |

The MATLAB scripts contain the same kind of literal at the top of each file
(`pathname`, `pathname_full_grid`); change those too.

The fastest way to get running is to reproduce the directory tree of Section 2 on your own
machine, and then replace the common prefix `C:\Users\bgvannoort\Documents\IDS` everywhere
with your own root directory.

### 1.3 Platform

The path handling assumes **Windows**: directories are built by string concatenation with
`"\\"`, for example inside `write_data_for_matlab`. On Linux or macOS these do not create
nested directories but single files whose names contain backslashes. To run on another
platform, replace those concatenations with `os.path.join`.

---

## 2. Assumed directory structure

The routines do not take a root directory as an argument; they derive it from the current
working directory. `write_data_for_matlab` takes `os.getcwd()`, strips a trailing `Code`,
and appends `Sim Data`. `Performance_eval_PCI_PWI.py` writes to `<cwd>/../Results`.

**Run all Python scripts from a directory named `Code`**, placed like this:

```
<ROOT>/                                   e.g. C:\Users\<you>\Documents\IDS
├── Code/                                 all .py files; run the scripts from here
│
├── Sim Data/                             input geometries + partition grids for MATLAB to plot the 3D MSP
│   ├── grid_x.txt, grid_y.txt, grid_z.txt        unit-sphere grid, shared by all figures
│   ├── fault_vectors.txt                         for the 'simple' example
│   └── SPP_GNSS/
│       ├── B_transpose_matrix.txt                B^T, comma separated (r x m)
│       ├── fault_vectors.txt                     c_ti = B^T c_i, one per column
│       ├── Qyy_diag.txt                          diagonal of Qyy
│       ├── indices_partition_no_separate_order.json
│       ├── indices_partition_separate_order.json
│       ├── colors_partitions_dict.json
│       ├── separate_colors_partitions_dict.json  (only if separate_partitions = true in MATLAB)
│       │
│       ├── IDS/<A|B|C>/no_separate_partitionings/lastOMT/alpha_type_<method>/
│       │       └── Partitioned_grid/factor_<R>/{P0,P1,...,P99}_{xx,yy,zz}.txt
│       ├── R_IDS/<A|B|C>/no_separate_partitionings/lastOMT/alpha_type_<method>/
│       │       └── Partitioned_grid/factor_<R>/...
│       ├── ordinary_DIA/no_separate_partitionings/
│       │       └── Partitioned_grid/factor_<R>/...
│       └── DS_DIA/<A|B|C|D>/no_separate_partitionings/
│               └── Partitioned_grid/factor_<R>/...
│
└── Results/
    └── TestingProbabilities/SPP_GNSS/
        ├── <IDS|DS|R_IDS>/<type>/<alpha method>/Hypothesis<k>/
        │       ├── PCA_FA_CI_WI_bval_<b>_sim_<i>.csv
        │       └── IR_estimates_bval_<b>_sim_<i>.csv
        └── classical DIA/Hypothesis<k>/
                └── ...
```

The `Sim Data` sub-path is assembled inside `write_data_for_matlab` from the run settings:

```
Sim Data \ <example> \ <IDS|R_IDS|ordinary_DIA|DS_DIA> \ <DS type>
          \ <no_>separate_partitionings [\ inPlane] [\ lastOMT]
          [\ alpha_type_<method>] [\ alpha_0=<value>]
          \ Partitioned_grid \ factor_<R> \ P<idx>_{xx,yy,zz}.txt
```

so a change in `separate_order_IDS`, `lastOMT`, `alpha_method` or `alpha_0` writes into a
different folder. When pointing MATLAB at the results, make sure its `pathname` matches the
settings you actually ran with. The `alpha_type_...` level is added for `IDS` only, and
`ordinary_DIA` has no DS-type level, since ordinary DIA complies with traditional DIA which only has one type. 
The BIDS export (`write_data_for_matlab_RIDS`) does not
build this path itself: the target folder is passed in as `manualPath` from
`RIDS_partitioning_on_3D_sphere.py`, following the same convention.

### The partition files

For every partition P<sub>i</sub> and every sphere radius `factor`, three files are written:
`P<idx>_xx.txt`, `P<idx>_yy.txt`, `P<idx>_zz.txt`. Each has the shape of the full spherical
grid, with `NaN` at every grid point that does **not** belong to that partition. MATLAB
overlays all of them, each in its own color, which produces the partitioned sphere. `P0` is
the acceptance region of H<sub>0</sub> and `P99` is the undecided region P<sub>Ω</sub>.

---

## 3. What the routines do

### 3.1 `Functions.py` — the library

All of the method is in this one module; every script imports from it with
`from Functions import *`. Grouped by purpose:

**Problem setup**

| Routine | What it does |
|---|---|
| `load_setup_parameters(type_of_example, alpha_method, ...)` | Dispatcher. Returns `m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B^T, Qtt, Qtt_inv` for the requested example. Every script starts with this call. |
| `setup(...)` | The simple example: m = 4, n = 1, A = **1**, Q<sub>yy</sub> = I. |
| `setup_SPP_GNSS_example(...)` | The single-constellation GNSS SPP example of the paper (m = 7, n = 4) with the elevation-dependent Q<sub>yy</sub> of Eq. (58). Loads B<sup>T</sup> from file, so that the orientation of the sphere plots is reproducible. |
| `setup_GNSS_example_ARAIM(...)` | ARAIM variant with one satellite relocated (not used in the paper). |
| `RIDS_example_setup_alpha_prime(...)` | Synthetic example used to study the α actually achieved by BIDS (not used in the paper). |
| `modify_alpha_prime(alpha, alpha_0, m, method, df, beta_0)` | Derives α′ from α₀. `method` is `'Kok_IDS'` (Baarda's B-method, β₀ = 0.2), `'Bonferonni'`, `'iteration'` (α′/m per iteration) or `'manual'`. |
| `import_DS_types(type_of_DS)` | Returns the four thresholds of Table 1 for DS types A–D. |

**Misclosure grid** (Appendix C)

| Routine | What it does |
|---|---|
| `generate_t_grid(nx, ny)` | Uniform grid on the unit sphere in ℝ³; returns the `x, y, z` meshes and the stacked misclosure vectors. Multiply by a radius `factor` to sample a sphere of that radius. |
| `generate_t_grid_in_plane(a1, a2, nx, ny)` | The same, but sampled inside the plane spanned by `a1` and `a2` — used for cross sections such as a fault plane. |

**Testing procedures.** Each returns, per sampled misclosure vector, the index of the
partition it falls into.

| Routine | What it does |
|---|---|
| `compute_Pis(t, Qtt, B_T, alpha_prime, type_of_testing, ...)` | **The single entry point.** Assigns to one of the procedures below according to `type_of_testing` ∈ {`'DS'`, `'IDS'`, `'R_IDS'`, `'classical DIA'`} and evaluates the decision function H(t) of Eq. (9) at every column (=sample) of `t`. Variable Pis is a list/array of indices corresponding to the decision, where 0 is H0, and -1 is the undecided region. Everything else in this repository is post-processing of its output. |
| `ordinary_DS(...)` | Computes the data for Baarda's datasnooping, one iteration, types A–D (Section 3). |
| `IDS_typeA_mult_its`, `IDS_typeB_mult_its`, `IDS_typeC_mult_its` | Forward IDS (Section 4.1). Recursive: each call identifies one outlier, re-projects the fault vectors following Proposition 1, and calls itself with the enlarged set S. |
| `RIDS_mult_its_type_A`, `..._type_B`, `..._type_C` | Backward IDS (Section 4.3). Selects the initial model H<sub>S</sub> by Eq. (48), then removes bias parameters one at a time. Also a recursive function. |
| `ordinary_DIA_testing_only(...)` | Traditional DIA: all hypotheses with q ≤ q<sub>max</sub> tested simultaneously, decision by the largest S<sub>j</sub> of Eq. (20). |

**Linear algebra helpers.** `plusmat`, `P_mat`, `P_perp`, and the `numba`-compiled
`plusmat_invQ`, `P_mat_invQ`: BLUE-inverse and (orthogonal complement) projectors in the Q
metric. The first call in a run pays a few seconds of `numba` compilation.

**Bookkeeping**

| Routine | What it does |
|---|---|
| `load_indices_partitions_GNSS_ex(sep_order, load_colors)` | Loads the dictionary mapping a partition name (`'P13'`) to the integer index used inside `compute_Pis`, optionally together with the plotting colors. Required for every run and determines the index in the variable Pis to the decision of the hypothesis. |
| `get_idx_hypt_RIDS(m, subset, qmax)` | The same mapping, generated on the fly for a given `qmax`. |
| `make_partition_string(indices)` | Turns a set of outlier indices into a partition name. |

**Export to MATLAB**

| Routine | What it does |
|---|---|
| `write_data_for_matlab(...)` | Writes the `P*_xx/_yy/_zz.txt` triplets for the DS, IDS and traditional DIA partitions into the tree of Section 2. |
| `write_data_for_matlab_RIDS(...)` | The same for BIDS; takes the target directory as `manualPath`. |

`timing`, `plot_3Dline`, `plot_faultline` and `save_interactive_3D` are small utilities.

### 3.2 Common settings

Every script is configured by editing the variables at the top of its `__main__` block;
there are no command-line arguments.

| Variable | Values | Meaning |
|---|---|---|
| `type_of_example` | `'simple'`, `'SPP_GNSS'`, `'ARAIM_UNDEC_GNSS'`, `'RIDS_EXAMPLE_ALPHA_PRIME'` | which measurement setup to load |
| `type_of_testing` | `'DS'`, `'IDS'`, `'R_IDS'`, `'classical DIA'` | the testing procedure |
| `type_of_DS` | `'A'`, `'B'`, `'C'`, `'D'` | the type within that procedure (Tables 1, 5, 8); type D exists for DS only |
| `type_of_alpha` / `alpha_method` | `'Kok_IDS'`, `'Bonferonni'`, `'iteration'`, `'manual'` | how α′ is derived and updated |
| `alpha_0` | float | level of significance of the w-test |
| `alpha_prime` | float | level of significance of the overall model test |
| `qmax` | int | maximum number of outliers monitored, q<sub>max</sub> < r |
| `lastOMT` | bool | perform a final OMT after adaptation |
| `separate_order_IDS` | bool | keep P<sub>{1,2}</sub> and P<sub>{2,1}</sub> separate instead of merging them by Eq. (41) |
| `factor` | float | radius of the sampled misclosure sphere |

---

## 4. Generating results

### 4.1 Misclosure space partitionings (Figure 8)

Two stages: Python classifies a grid of misclosure vectors and writes the result, MATLAB
renders it.

```bash
cd <ROOT>/Code

python IDS_partitioning_on_2D_sphere.py       # forward IDS      -> Fig. 8, left
python RIDS_partitioning_on_3D_sphere.py      # backward IDS     -> Fig. 8, middle
python ordinary_partitioning_example_3D.py    # traditional DIA  -> Fig. 8, right
python Ordinary_DS_partitioning_example.py    # datasnooping, one iteration
```

Set `type_of_example`, `type_of_DS`, `alpha_0`, `qmax`, `lastOMT` and the list of `factor`
values at the top of each script. The paper uses type C, α₀ = 0.01, q<sub>max</sub> = 2, at
radii R = 6 and R = 11. The grid resolution is `n_samples` (1000 × 1000 in the paper); cost
grows quadratically with it.

Each script writes into `Sim Data/...` as described in Section 2. Then, in MATLAB, set
`pathname` at the top of each script and run:

```matlab
Plot_3D_partitioning_IDS_GNSS_ex             % Fig. 8, left panel
plot_3D_sphere_RIDS_GNSS_ex                  % Fig. 8, middle panel
Plot_3D_partitioning_classical_DIA_GNSS_ex   % Fig. 8, right panel
plot_3D_partitioning_stepwise_GNSS_ex        % partitioning per iteration
Plot_3D_partitioning_DS_GNSS_ex              % datasnooping partitioning
```

### 4.2 Testing probabilities (Figure 9, and the input for Tables 11–13)

```bash
python Performance_eval_PCI_PWI.py
```

This is the Monte Carlo driver. For a chosen true hypothesis H<sub>i</sub> it samples
misclosure vectors over a range of bias magnitudes, calls `compute_Pis`, and counts how
often each partition is selected. Per bias value and per simulation run it writes

```
Results/TestingProbabilities/<example>/<procedure>/<type>/<alpha method>/Hypothesis<k>/
        PCA_FA_CI_WI_bval_<b>_sim_<i>.csv        P_CA, P_FA, P_CI, P_WI per partition
        IR_estimates_bval_<b>_sim_<i>.csv        integrity-risk estimates
```

Set `Nsims`, the number of samples, `D1_hypothesis` (the hypothesis that is true), `qmax`
and the bias range in the `__main__` block. This is the expensive step: for the GNSS example
roughly 1–10 minutes per procedure per bias value at 10<sup>5</sup> samples, so a full sweep
over all hypotheses and bias values is best run as a batch of independent jobs. The results
in the paper were obtained by running this script as an array of such jobs on a compute
cluster; the job submission scripts are cluster-specific and are not included.
---

## 5. Analyzing results

The analysis scripts do not simulate; they read the CSV files produced in Section 4.2.
Point their `main_dir` at your own `Results/TestingProbabilities/...` directory first.

### 5.1 Mean penalties (Tables 11 and 12)

```bash
python Evaluate_mean_penalties.py
```

Reads the probability CSVs for every hypothesis, weights one minus P<sub>CI</sub> by the
prior probabilities π<sub>i</sub>, and evaluates the mean penalty of Eq. (26). Set at the
top of the script:

- `scenario_type` = 1 or 2 — the two prior-probability scenarios of Table 10;
- `at_x_sigma` = 3 or 5 — the bias magnitude at which the penalty is evaluated
  (Table 11 uses 3σ, Table 12 uses 5σ);
- `type_of_testing`, `type_of_DS`, `qmax`, and the matching `alpha_0_dictionary` entry.

Run it once per testing procedure and collect the printed totals into the table. Its helpers
`get_data` (one-outlier hypotheses) and `get_data_2Dhypt` (two-outlier hypotheses) glob the
per-run CSVs and return the mean and standard deviation over the simulation runs.

### 5.2 P<sub>CI</sub> and P<sub>WI</sub> (Figure 9)

```bash
python Plot_PCI_PWI_results_from_DelftBlue.py          # one-outlier hypotheses
python Plot_PCI_PWI_results_2Dhypts_from_DelftBlue.py  # two-outlier hypotheses
```

These read the same CSVs and plot P<sub>CI</sub> and P<sub>WI</sub> as a function of the
bias. The second produces the (b₁, b₃) maps of Figure 9 under H<sub>1,3</sub>.

### 5.3 Masking and swamping (Table 13)

```bash
python Masking_swamping_for_SPP_GNSS.py
```

Applies definition (54) to the wrong-identification probabilities: for a true hypothesis
H<sub>P</sub> it sums P(t ∈ P<sub>R</sub> | H<sub>P</sub>) over all R with
R<sup>c</sup> ∩ P ≠ ∅ (masking) and over all R with R ∩ P<sup>c</sup> ≠ ∅ (swamping),
following Eq. (55). It therefore needs the P<sub>WI</sub> output of Section 4.2 for the
hypothesis of interest, at the bias values to be reported.

### 5.4 Angles between fault vectors and fault spaces (Table 9)

```bash
python Compute_table_angles_fault_vectors.py
```

Self-contained: it needs the setup only, not any simulation output. For every fault vector
c<sub>t<sub>j</sub></sub> and every subset S of size q<sub>max</sub> it computes
γ<sub>j;S</sub>, the diagnostic of Section 6 for predicting masking and swamping. Set
`type_of_example` and `qmax` at the top.

---

[
## 6. Which script produces which result

| Paper item | Script |
|---|---|
| Fig. 1 — type A DS partitioning in ℝ², m = 3 | `Type_D_datasnooping.m` (MATLAB) contains this m = 3, r = 2 example, implemented for type D |
| Fig. 2–5 — flow diagrams and set diagrams | drawn, not generated |
| Fig. 6 — fault vectors and fault plane | sketch |
| Fig. 7 — skyplot | `generate_skyplot_from_A.m` |
| Fig. 8 — MSP for type C IDS, type C BIDS and traditional DIA | `IDS_partitioning_on_2D_sphere.py`, `RIDS_partitioning_on_3D_sphere.py`, `ordinary_partitioning_example_3D.py` → `Plot_3D_partitioning_IDS_GNSS_ex.m`, `plot_3D_sphere_RIDS_GNSS_ex.m`, `Plot_3D_partitioning_classical_DIA_GNSS_ex.m` |
| Fig. 9 — P<sub>CI</sub> and P<sub>WI</sub> over (b₁, b₃) | `Performance_eval_PCI_PWI.py` → `Plot_PCI_PWI_results_2Dhypts_from_DelftBlue.py` |
| Tab. 9 — angles γ<sub>j;S</sub> | `Compute_table_angles_fault_vectors.py` |
| Tab. 10 — prior probabilities π<sub>i</sub> | set in `Evaluate_mean_penalties.py` (`scenario_type`) |
| Tab. 11, 12 — mean penalties at 3σ and 5σ | `Performance_eval_PCI_PWI.py` → `Evaluate_mean_penalties.py` |
| Tab. 13 — masking and swamping probabilities | `Performance_eval_PCI_PWI.py` → masking and swamping script |
| Tab. B1 — computation times | measured with the `timing` decorator in `Functions.py` |
| App. C — construction of the sphere grid | `Functions.generate_t_grid` |

---


## 7. Known limitations

- Test statistics beyond the first iteration are conditioned on earlier decisions and are
  therefore not exactly χ²- or normally distributed, while the thresholds are computed from
  those nominal distributions. The realized levels of significance consequently differ from
  the nominal ones. This is a property of iterative model selection, discussed in Section 4
  of the paper, not a defect of the implementation.
- Inseparable hypotheses (parallel projected fault vectors c̄<sub>t</sub>) are not detected
  automatically. If the maximum among the w-statistics is not unique, the identification is
  arbitrary; the user should monitor this.
- Path handling is Windows-specific, see Section 1.3.

---
]:#

## 8. Citation

Please cite the paper when using this code.

## 9. Acknowledgement

The authors acknowledge the Netherlands Space Office (NSO) for funding this research.

## 10. Contact

b.g.vannoort@tudelft.nl
