# Code and reproducibility resources for deep learning-based pancreatic lesion risk prediction

This repository contains evaluation scripts, precomputed test-set predictions, Bootstrap results, plotting scripts, and selected pretrained model artifacts for longitudinal risk prediction experiments.

## Reproducibility scope

**The current release supports reproducing evaluation metrics, patient-level Bootstrap resampling results, 95% confidence intervals (CIs), and figures from the already provided `Test_set_result.csv` files.**

The recommended order is:

1. Recalculate evaluation metrics using the **existing validation-derived thresholds** in each `Metrics.csv`.
2. Generate 200 valid patient-level Bootstrap iterations using random seed **42**.
3. Calculate 95% percentile Bootstrap CIs from the saved Bootstrap iterations.
4. Reproduce the subfigures and composite figures using the scripts in `plot/`.

> **Important limitation:** This release does **not** currently support reconstructing `Test_set_result.csv` from raw inputs or rerunning the full model inference/training pipeline. The `Model/` folder is provided for reference and includes code and selected pretrained artifacts, but the required embedding files have not yet been released. See [Model artifacts and limitations](#model-artifacts-and-limitations).

## Repository structure

```text
.
├── requirements.txt
├── Model/
│   ├── GRU-D.py
│   ├── label_windows.py
│   ├── post_train_missing_data.py
│   ├── transformer.py
│   ├── XGBoost.py
│   ├── best_model_TemporalTransformer_missing_20.pth
│   └── FT_transformer/
│       ├── FT_transformer_standard package_missing_*.pth
│       └── Preprocess_standard package_missing_*.pkl
├── plot/
│   ├── Broader_outcome_definitions.py
│   ├── Data_exclusion_result.py
│   ├── Feature_tiers.py
│   ├── Independent_hold-out_datasets.py
│   ├── Minimum_sequence_length.py
│   ├── Missingness_tolerance.py
│   ├── Model_types.py
│   ├── Prediction_horizons.py
│   ├── plot_editable_fig2.py
│   ├── plot_editable_fig3.py
│   ├── subfigure/
│   ├── figure/
│   └── backup/
└── test_data/
    ├── metric.py
    ├── result_200_bootstrap.py
    ├── confidence.py
    ├── Broader outcome definitions/
    ├── Data exclusion result/
    ├── Feature tiers/
    ├── Independent hold-out datasets/
    ├── Minimum sequence length/
    ├── Missingness tolerance/
    ├── Model types/
    └── Prediction horizons/
```

There are **23 experiment subdirectories** under `test_data/`. Each contains the following four files:

```text
<experiment-directory>/
├── Test_set_result.csv
├── Metrics.csv
├── Bootstrap_200_Iterations_Results.csv
└── Metrics_with_95CI_Overwritten.csv
```

The two expanded feature-tier experiments additionally contain `Feature_Attributions_Rank_IG.csv`.

The `plot/backup/` directory contains archived figure versions and is **not required** for the reproduction workflow.

## 1. Environment setup

Run the following commands **from the repository root** (the directory containing `requirements.txt`, `test_data/`, and `plot/`).

### Option A: Using a fresh Python environment

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Option B: Using the original Conda environment

If you already have the environment used for the experiments:

```powershell
conda activate plot
python -m pip install -r requirements.txt
```

The primary evaluation and plotting libraries include `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and (where required) `tqdm`. Use the versions recorded in the committed `requirements.txt`.

> The `requirements.txt` file may have been exported from an existing environment using `pip freeze`. Before publishing it, check for machine-specific paths (for example, `package @ file:///...`) and unnecessary packages. A full Conda environment or GPU training environment may require additional setup beyond pip.

## 2. Reproduce the results in `test_data/`

**Do this stage first.** All three scripts automatically discover experiment folders under `test_data/` and are designed to be invoked from the repository root.

### 2.1 Required input files and columns

Each experiment's `Test_set_result.csv` is the **existing test-set prediction output**. It must contain:

- `pid`: patient identifier used for patient-level aggregation and resampling;
- `target_12m`, `target_18m`, `target_24m`: binary ground-truth labels;
- `prob_12m`, `prob_18m`, `prob_24m`: prediction scores;
- `mask_12m`, `mask_18m`, `mask_24m`: indicators of valid records (`1` means valid).

Each experiment's **existing** `Metrics.csv` contains the evaluation structure and saved `Threshold` values. The `ValidatedOptTh` thresholds were chosen previously on validation data and **are reused without re-optimizing them on the test set**.

**Do not delete `Metrics.csv` before Step 2.2**: it is both an input (saved thresholds) and an output (recalculated metrics). The script overwrites this file. Keep your own separate copy of the original files if you wish to retain them.

### 2.2 Recalculate `Metrics.csv`

Run:

```powershell
python test_data/metric.py
```

**Reads (per experiment):**

- `Test_set_result.csv`
- Existing `Metrics.csv` (including saved validation-derived thresholds)

**Generates/updates (per experiment):**

- `Metrics.csv` — **overwrites** the existing file with recalculated evaluation metrics.

The script reports window-level metrics for 12-, 18-, and 24-month prediction horizons as well as the overall patient-level evaluation. It reports default-threshold (`0.5`) and saved optimized-threshold results, including AUROC, AUPRC, F1, accuracy, precision, recall, relative risk (RR), confusion-matrix counts, and Top-*k* PPV/RR.

The original metric table layout and ordering are retained for evaluable rows. Numeric metrics are written to **four decimal places**; TN/FP/FN/TP remain integer counts.

**Special case — `Data exclusion result/Exclusion 12 months`:** The valid 12-month data do not contain both target classes. As AUROC is undefined for a single-class outcome, the 12-month evaluation rows are skipped. The 18-month, 24-month, and evaluable patient-level results remain. Consequently, this experiment normally has **6** evaluation rows rather than the usual **8**.

No fresh threshold search is performed, and no separate comparison report or automatic backup is produced.

### 2.3 Generate 200 patient-level Bootstrap iterations

Run:

```powershell
python test_data/result_200_bootstrap.py
```

**Reads (per experiment):**

- `Test_set_result.csv`
- Recalculated `Metrics.csv` (to obtain the saved thresholds)

**Generates/overwrites (per experiment):**

- `Bootstrap_200_Iterations_Results.csv`

**Bootstrap settings:**

| Setting | Value |
| --- | --- |
| Random seed | `42` |
| Resampling unit | Patient (`pid`) |
| Sampling | With replacement |
| Sample size per iteration | Number of unique patients in the experiment |
| Number of iterations | 200 **valid** iterations |
| Prediction horizons | 12, 18, 24 months, where evaluable |
| Threshold types | `Default` (0.5) and `OptTh` (saved threshold) |
| Top-*k* cutoff | Integer truncation (floor), at least one record; Top 0.1%, 1%, and 10% |

An iteration is retained only when the active prediction horizons meet the script's validity requirements. The CSV records individual iteration metrics (including `Iteration`, `Window`, and `Threshold_Type`) so the reported uncertainty estimates can be independently checked.

**Expected row counts:**

- Standard experiment: **200 × 3 horizons × 2 threshold types = 1,200 rows**.
- `Exclusion 12 months`: **200 × 2 horizons × 2 threshold types = 800 rows** (12m excluded).

Random seed 42 makes the new resampling workflow repeatable **when the input data/order, code, NumPy random-number implementation, and environment are kept consistent**. It does not recreate an earlier unseeded Bootstrap run.

### 2.4 Calculate 95% Bootstrap confidence intervals

Run:

```powershell
python test_data/confidence.py
```

**Reads (per experiment):**

- `Metrics.csv` (point estimates and evaluation levels)
- `Bootstrap_200_Iterations_Results.csv` (the 200 saved iterations)

**Generates/overwrites (per experiment):**

- `Metrics_with_95CI_Overwritten.csv`

The confidence intervals use the **2.5th and 97.5th percentiles** of the saved Bootstrap metric distribution. This step **does not perform another random resampling**.

For the active windows, the output combines each original point estimate with its corresponding 95% CI, for example:

```text
0.8500 (0.7900-0.9000)
```

This is a formatting example, **not a reported result from this repository**.

CIs are added to the **window-level** metrics supported by the Bootstrap output. **Patient-level rows are retained as point estimates only**; the current workflow does not calculate patient-level Bootstrap CIs.

Expected updated window/threshold rows:

- Most experiments: **6** rows (3 windows × 2 thresholds).
- `Exclusion 12 months`: **4** rows (2 windows × 2 thresholds).

### 2.5 Check the evaluation outputs

After these commands, each experiment folder should contain:

| File | Role | Produced in |
| --- | --- | --- |
| `Test_set_result.csv` | Input prediction scores, ground truth, masks, and patient IDs | Provided; **not regenerated** |
| `Metrics.csv` | Recalculated point estimates using the existing validation thresholds | Step 2.2 |
| `Bootstrap_200_Iterations_Results.csv` | Individual results for 200 valid patient-level resamples | Step 2.3 |
| `Metrics_with_95CI_Overwritten.csv` | Point estimates combined with 95% percentile CIs | Step 2.4 |

Check that every expected directory was processed, that Bootstrap row counts are consistent with its valid windows, and that each percentile CI agrees with the underlying Bootstrap records (up to the output rounding precision).

## 3. Reproduce the figures in `plot/`

Complete **Section 2** before running the plotting scripts. The plots use the supplied predictions and/or evaluation outputs under `test_data/`.

### 3.1 Working directory and output paths

Run the commands **from the repository root**, not from inside `plot/`, unless you first update the scripts' relative paths.

Some plotting code uses relative paths such as:

```python
pd.read_csv(r".\test_data\Prediction horizons\Pretrain + Transformer\Test_set_result.csv")
```

The location written by `savefig(...)` is also relative to the **current working directory**. For example, a legacy `r".\figure\...pdf"` path will write under `<repository-root>/figure/`, **not** under `plot/subfigure/`, when run from the repository root. To reproduce the directory layout shown above, verify or adjust the `savefig(...)` paths so that subfigure outputs go to `plot/subfigure/` and composite outputs go to `plot/figure/`. Create these directories if needed:

```powershell
New-Item -ItemType Directory -Force "plot/subfigure", "plot/figure" | Out-Null
```

The exact saved location is determined by the paths in your local copies of the plotting scripts. Files already in these folders are existing outputs, not proof that every script currently writes to those paths without adjustment.

### 3.2 Generate the eight subfigure groups

Run these commands from the repository root:

```powershell
python plot/Broader_outcome_definitions.py
python plot/Data_exclusion_result.py
python plot/Feature_tiers.py
python plot/Independent_hold-out_datasets.py
python plot/Minimum_sequence_length.py
python plot/Missingness_tolerance.py
python plot/Model_types.py
python plot/Prediction_horizons.py
```

Each script corresponds to one experiment family. The existing repository layout contains the following expected subfigure filenames in `plot/subfigure/`:

| Script | Figure files (PDF and SVG) |
| --- | --- |
| `Broader_outcome_definitions.py` | `Broader_outcome_definitions_AUC.*`, `Broader_outcome_definitions_RR.*` |
| `Data_exclusion_result.py` | `Data_exclusion_result_AUC.*`, `Data_exclusion_result_RR.*` |
| `Feature_tiers.py` | `Feature_tiers_AUC.*`, `Feature_tiers_RR.*` |
| `Independent_hold-out_datasets.py` | `Independent_hold-out_datasets_AUC.*`, `Independent_hold-out_datasets_RR.*` |
| `Minimum_sequence_length.py` | `Minimum_sequence_length_AUC.*`, `Minimum_sequence_length_RR.*` |
| `Missingness_tolerance.py` | `Missing_tolerance_AUC.*`, `Missing_tolerance_RR.*` |
| `Model_types.py` | `Model_types_AUC.*`, `Model_types_RR.*` |
| `Prediction_horizons.py` | `Prediction_horizons_AUC.*`, `Prediction_horizons_RR.*` |

Here `.*` indicates two separately saved formats: `.pdf` and `.svg`, **not** a literal filename extension. The repository also contains `plot/subfigure/exclude.pdf`; its generation step is not identified by the filename listing and should not be assumed to be produced by any specific command above.

### 3.3 Generate the final composite figures

After the subfigures have been generated, run:

```powershell
python plot/plot_editable_fig2.py
python plot/plot_editable_fig3.py
```

The `plot/figure/` directory contains these final PDF artifacts:

```text
plot/figure/
├── fig2.pdf
├── fig3.pdf
```

These are the final figure filenames currently organized in the repository. Confirm which files each composite script writes by inspecting its save/export calls; naming alone does not establish whether `fig21.pdf` or `fig31.pdf` is an alternate export or a separate figure.

### 3.4 Figure reproduction checklist

- Verify that the paths inside the plotting scripts point to this repository's `test_data/`.
- Check that the output directories exist.
- Confirm that the expected PDFs and SVGs were created or updated.
- Inspect the resulting plots and compare them with the committed versions.
- Ensure any fonts required by the plotting scripts are installed, or substitute an available font while noting that typography may differ.
- The files in `plot/backup/` are archival references, not required intermediate outputs.

## 4. Model artifacts and limitations

The `Model/` directory provides selected model-related source code and pretrained artifacts, including:

- `best_model_TemporalTransformer_missing_20.pth`
- `FT_transformer/FT_transformer_standard package_missing_10.pth`
- `FT_transformer/FT_transformer_standard package_missing_20.pth`
- `FT_transformer/FT_transformer_standard package_missing_30.pth`
- The corresponding `Preprocess_standard package_missing_*.pkl` files
- `GRU-D.py`, `label_windows.py`, `post_train_missing_data.py`, `transformer.py`, and `XGBoost.py`

**These files are shared as reference artifacts and are not a currently runnable, complete end-to-end reproduction pipeline.** In particular, the embedding files required to reproduce test-set prediction outputs are **not included** in the present release.

Therefore:

- **Supported now:** Reproduction of reported evaluation metrics, 200-iteration Bootstrap results, percentile CIs, and figures, **starting from the supplied `Test_set_result.csv` files**.
- **Not supported now:** Independently regenerating `Test_set_result.csv` from raw input data by rerunning the full model inference/training workflow.
- **Future extension:** To support reproducing the test-set prediction files themselves, the required embedding files will be released separately as part of an expanded reproduction package.

Do not interpret the presence of `.pth` or `.pkl` files as evidence that the complete upstream inference pipeline can be executed with the current repository contents. Avoid loading untrusted serialized `.pkl` or `.pth` files.

## 5. Reproduction order at a glance

From the repository root:

```powershell
# Evaluation
python test_data/metric.py
python test_data/result_200_bootstrap.py
python test_data/confidence.py

# Subfigures
python plot/Broader_outcome_definitions.py
python plot/Data_exclusion_result.py
python plot/Feature_tiers.py
python plot/Independent_hold-out_datasets.py
python plot/Minimum_sequence_length.py
python plot/Missingness_tolerance.py
python plot/Model_types.py
python plot/Prediction_horizons.py

# Final figures
python plot/plot_editable_fig2.py
python plot/plot_editable_fig3.py
```

The three evaluation scripts **overwrite existing result files**. Plotting scripts may also overwrite existing figures at their configured output paths.

## 6. Data sharing and scientific reproducibility

The supplied prediction CSV files may include patient-level identifiers, labels, or prediction scores. Before publication, confirm that their release is authorized and that any necessary de-identification and data-use checks have been completed. Similarly, review model checkpoints and preprocessing artifacts for restricted information.

This release supports **downstream evaluation and plotting reproducibility** from precomputed test predictions; it should not be described as full reproducibility of model training or test-set prediction generation.

For strict numerical reproduction, retain the exact input CSV contents, row order, saved threshold values, source code, Python package versions, and random-number implementation.
