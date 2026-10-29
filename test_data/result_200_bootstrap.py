import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Place this script directly inside test_data. All paths are anchored to this file.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_FILENAME = "Test_set_result.csv"
METRICS_FILENAME = "Metrics.csv"
RAW_BOOTSTRAP_FILENAME = "Bootstrap_200_Iterations_Results.csv"
N_BOOTSTRAP = 200
SEED = 42
WINDOWS = [12, 18, 24]
KS = [0.1, 1, 10]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv"}


def read_csv_auto(path):
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to identify CSV encoding: {path}")


def find_experiment_dirs():
    found = []
    for current_dir, subdirs, filenames in os.walk(ROOT_DIR):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)
        if PREDICTIONS_FILENAME in filenames:
            found.append(current_dir)
    return sorted(found)


def calculate_all_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None

    y_pred = (y_score > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    incidence = float(np.mean(y_true))

    results = {
        "AUC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Acc": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # Match Prediction_horizons.py: descending score order, floor cutoff, minimum one.
    sorted_df = pd.DataFrame({"target": y_true, "prob": y_score}).sort_values(
        by="prob", ascending=False
    )
    sorted_targets = sorted_df["target"].to_numpy()
    n = len(sorted_targets)
    for k in KS:
        k_str = "0.1" if k == 0.1 else str(int(k))
        n_top = max(1, int((k / 100.0) * n))
        ppv_k = float(np.mean(sorted_targets[:n_top]))
        rr_k = ppv_k / incidence if incidence > 0 else 0.0
        results[f"PPV@{k_str}%"] = ppv_k
        results[f"RR@{k_str}%"] = rr_k
    return results


def get_opt_threshold(metrics_df, window):
    level = f"Window_{window}m_ValidatedOptTh_Imbalanced"
    matches = metrics_df[metrics_df["Evaluation_Level"] == level]
    if matches.empty:
        raise ValueError(f"metric.csv has no saved validation threshold row: {level}")
    threshold = pd.to_numeric(matches.iloc[0]["Threshold"], errors="coerce")
    if pd.isna(threshold):
        raise ValueError(f"Invalid Threshold in metric.csv row: {level}")
    return float(threshold)


def valid_window_data(df, window):
    mask = pd.to_numeric(df[f"mask_{window}m"], errors="coerce")
    target = pd.to_numeric(df[f"target_{window}m"], errors="coerce")
    prob = pd.to_numeric(df[f"prob_{window}m"], errors="coerce")
    valid = (mask == 1) & target.isin([0, 1]) & prob.notna() & np.isfinite(prob)
    return target.loc[valid].astype(int).to_numpy(), prob.loc[valid].astype(float).to_numpy()


def process_directory(base_dir):
    predictions_path = os.path.join(base_dir, PREDICTIONS_FILENAME)
    metrics_path = os.path.join(base_dir, METRICS_FILENAME)
    output_path = os.path.join(base_dir, RAW_BOOTSTRAP_FILENAME)

    if not os.path.isfile(metrics_path):
        print(f"  [Skip] Missing metric.csv: {metrics_path}")
        return

    df = read_csv_auto(predictions_path)
    if "pid" not in df.columns or df["pid"].isna().any():
        raise ValueError("Test_set_result.csv must contain a non-missing pid column.")
    for w in WINDOWS:
        for col in (f"target_{w}m", f"prob_{w}m", f"mask_{w}m"):
            if col not in df.columns:
                raise KeyError(f"Test_set_result.csv is missing required column: {col}")

    metrics_df = read_csv_auto(metrics_path)
    if "Evaluation_Level" not in metrics_df.columns or "Threshold" not in metrics_df.columns:
        raise ValueError("metric.csv must contain Evaluation_Level and Threshold columns.")

    active_windows = []
    for w in WINDOWS:
        y_true, _ = valid_window_data(df, w)
        if len(y_true) > 0 and len(np.unique(y_true)) >= 2:
            active_windows.append(w)
        else:
            print(f"  [Warning] {w}m has no valid data or only one class; skipping this window.")
    if not active_windows:
        raise ValueError("No prediction window has valid samples from both classes.")

    opt_thresholds = {w: get_opt_threshold(metrics_df, w) for w in active_windows}
    pid_to_df = {pid: group for pid, group in df.groupby("pid", sort=False)}
    pid_list = list(pid_to_df.keys())
    n_patients = len(pid_list)
    if n_patients == 0:
        raise ValueError("No patients available for bootstrap sampling.")

    # A fresh deterministic generator per experiment directory makes each directory reproducible.
    rng = np.random.RandomState(SEED)
    bootstrap_results = {
        "Default": {w: [] for w in active_windows},
        "OptTh": {w: [] for w in active_windows},
    }
    valid_iterations = 0
    attempts = 0
    max_attempts = N_BOOTSTRAP * 10

    with tqdm(total=N_BOOTSTRAP, desc="  Bootstrapping", leave=False) as pbar:
        while valid_iterations < N_BOOTSTRAP and attempts < max_attempts:
            attempts += 1
            sampled_pids = rng.choice(pid_list, size=n_patients, replace=True)
            sampled_df = pd.concat([pid_to_df[pid] for pid in sampled_pids], ignore_index=True)
            iteration_metrics = {"Default": {}, "OptTh": {}}
            iteration_valid = True

            for w in active_windows:
                y_true, y_score = valid_window_data(sampled_df, w)
                if len(y_true) == 0:
                    iteration_valid = False
                    break
                res_default = calculate_all_metrics(y_true, y_score, threshold=0.5)
                res_opt = calculate_all_metrics(y_true, y_score, threshold=opt_thresholds[w])
                if res_default is None or res_opt is None:
                    iteration_valid = False
                    break
                iteration_metrics["Default"][w] = res_default
                iteration_metrics["OptTh"][w] = res_opt

            if iteration_valid:
                for threshold_type in ("Default", "OptTh"):
                    for w in active_windows:
                        bootstrap_results[threshold_type][w].append(iteration_metrics[threshold_type][w])
                valid_iterations += 1
                pbar.update(1)

    if valid_iterations != N_BOOTSTRAP:
        raise RuntimeError(
            f"Only {valid_iterations}/{N_BOOTSTRAP} valid iterations after {max_attempts} attempts; output not written."
        )

    records = []
    for threshold_type in ("Default", "OptTh"):
        for w in active_windows:
            threshold = 0.5 if threshold_type == "Default" else opt_thresholds[w]
            for iteration, values in enumerate(bootstrap_results[threshold_type][w], start=1):
                record = {
                    "Iteration": iteration,
                    "Window": f"{w}m",
                    "Threshold_Type": threshold_type,
                    "Threshold": threshold,
                }
                record.update(values)
                records.append(record)

    pd.DataFrame(records).to_csv(output_path, index=False, float_format="%.8f", encoding="utf-8-sig")
    print(f"  [Done] Saved {len(records)} bootstrap rows to {output_path}")
    print(f"  [Settings] seed={SEED}; valid iterations={N_BOOTSTRAP}; patient-level sampling with replacement.")


def main():
    directories = find_experiment_dirs()
    print(f"Root directory: {ROOT_DIR}")
    print(f"Found {len(directories)} directories containing {PREDICTIONS_FILENAME}.")
    for i, directory in enumerate(directories, start=1):
        print(f"\n[{i}/{len(directories)}] {os.path.relpath(directory, ROOT_DIR)}")
        try:
            process_directory(directory)
        except Exception as exc:
            print(f"  [Error] {exc}")
    print("\nFinished Bootstrap generation.")


if __name__ == "__main__":
    main()
