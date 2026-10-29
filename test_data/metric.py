import os
import re
import shutil
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

warnings.filterwarnings("ignore")

# Place this script directly inside test_data. All paths are anchored to this file.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_FILENAME = "Test_set_result.csv"
METRICS_FILENAME = "Metrics.csv"
BACKUP_FILENAME = "metric_original_backup.csv"
COMPARISON_FILENAME = "Metric_Reproduction_Comparison.csv"
WINDOWS = [12, 18, 24]
TOP_PERCENTS = [0.1] + list(range(1, 10)) + list(range(10, 101, 10))
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


def calculate_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("AUC and this evaluation row require both target classes (0 and 1).")

    y_pred = (y_score > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    precision = precision_score(y_true, y_pred, zero_division=0)
    incidence = float(np.mean(y_true)) if total else 0.0
    rr = precision / incidence if incidence > 0 else 0.0

    results = {
        "AUC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Acc": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "RR": rr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # Match Prediction_horizons.py: sort scores descending, floor the cutoff,
    # and always select at least one record.
    sorted_df = pd.DataFrame({"target": y_true, "prob": y_score}).sort_values(
        by="prob", ascending=False
    )
    sorted_targets = sorted_df["target"].to_numpy()
    n = len(sorted_targets)
    for p in TOP_PERCENTS:
        n_top = max(1, int((p / 100.0) * n))
        ppv_k = float(np.mean(sorted_targets[:n_top]))
        rr_k = ppv_k / incidence if incidence > 0 else 0.0
        results[f"PPV@{p:g}%"] = ppv_k
        results[f"RR@{p:g}%"] = rr_k
    return results


def get_numeric_arrays(df):
    required = ["pid"]
    for w in WINDOWS:
        required.extend([f"target_{w}m", f"prob_{w}m", f"mask_{w}m"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Test_set_result.csv is missing required columns: {missing}")

    probs = df[[f"prob_{w}m" for w in WINDOWS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    targets = df[[f"target_{w}m" for w in WINDOWS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    masks = df[[f"mask_{w}m" for w in WINDOWS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pids = df["pid"].to_numpy()
    return targets, probs, masks, pids


def patient_level_arrays(targets, probs, masks, pids):
    patient_targets = []
    patient_probs = []
    for pid in pd.unique(pids):
        idx = np.where(pids == pid)[0]
        p_targets = targets[idx]
        p_probs = probs[idx]
        p_masks = masks[idx]
        valid = (p_masks == 1) & np.isin(p_targets, [0, 1]) & np.isfinite(p_probs)
        if not np.any(valid):
            continue
        # Preserve the original evaluation definition: combine valid entries across horizons.
        patient_targets.append(int(np.any(p_targets[valid] == 1)))
        patient_probs.append(float(np.max(p_probs[valid])))
    return np.asarray(patient_targets, dtype=int), np.asarray(patient_probs, dtype=float)


def parse_level(level):
    match = re.fullmatch(
        r"Window_(12|18|24)m_(DefaultTh\(0\.5\)|ValidatedOptTh)_Imbalanced",
        str(level),
    )
    if match:
        return "window", int(match.group(1)), match.group(2)
    if level == "Patient_Level_Overall_DefaultTh(0.5)_Imbalanced":
        return "patient", None, "DefaultTh(0.5)"
    if level == "Patient_Level_Overall_ValidatedOptTh_Imbalanced":
        return "patient", None, "ValidatedOptTh"
    return None, None, None


def values_match(old_value, new_value, column):
    if pd.isna(old_value) and pd.isna(new_value):
        return True
    if pd.isna(old_value) or pd.isna(new_value):
        return False
    try:
        old_num = float(old_value)
        new_num = float(new_value)
        if column in {"TN", "FP", "FN", "TP"}:
            return int(round(old_num)) == int(round(new_num))
        return round(old_num, 4) == round(new_num, 4)
    except (TypeError, ValueError):
        return str(old_value) == str(new_value)


def process_directory(base_dir):
    predictions_path = os.path.join(base_dir, PREDICTIONS_FILENAME)
    metrics_path = os.path.join(base_dir, METRICS_FILENAME)
    backup_path = os.path.join(base_dir, BACKUP_FILENAME)
    comparison_path = os.path.join(base_dir, COMPARISON_FILENAME)

    if not os.path.isfile(metrics_path):
        print(f"  [Skip] Missing threshold source: {metrics_path}")
        return

    threshold_source_metrics = read_csv_auto(metrics_path)
    if "Evaluation_Level" not in threshold_source_metrics.columns or "Threshold" not in threshold_source_metrics.columns:
        raise ValueError("metric.csv must contain Evaluation_Level and Threshold columns.")
    if not os.path.isfile(backup_path):
        shutil.copy2(metrics_path, backup_path)
        print(f"  [Backup] Original metric.csv saved to {backup_path}")
    # Keep a stable first-run baseline for comparisons on later runs too.
    baseline_metrics = read_csv_auto(backup_path)
    if "Evaluation_Level" not in baseline_metrics.columns:
        raise ValueError("metric_original_backup.csv must contain Evaluation_Level.")

    df = read_csv_auto(predictions_path)
    targets, probs, masks, pids = get_numeric_arrays(df)
    new_metrics = threshold_source_metrics.copy()
    skip_indices = []

    for idx, row in threshold_source_metrics.iterrows():
        level = str(row["Evaluation_Level"])
        level_type, window, threshold_type = parse_level(level)
        if level_type is None:
            continue

        if threshold_type == "DefaultTh(0.5)":
            threshold = 0.5
        else:
            threshold = pd.to_numeric(pd.Series([row["Threshold"]]), errors="coerce").iloc[0]
            if pd.isna(threshold):
                raise ValueError(f"No valid saved Threshold in metric.csv for {level}")
            threshold = float(threshold)

        if level_type == "window":
            col_idx = WINDOWS.index(window)
            valid = (
                (masks[:, col_idx] == 1)
                & np.isin(targets[:, col_idx], [0, 1])
                & np.isfinite(probs[:, col_idx])
            )
            y_true = targets[valid, col_idx].astype(int)
            y_score = probs[valid, col_idx].astype(float)
        else:
            y_true, y_score = patient_level_arrays(targets, probs, masks, pids)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print(f"  [Skip] {level}: only one target class")
            skip_indices.append(idx)
            continue

        metric_values = calculate_metrics(y_true, y_score, threshold=threshold)
        for column, value in metric_values.items():
            if column in new_metrics.columns:
                new_metrics.at[idx, column] = value
        # Preserve the exact threshold from metric.csv for optimized rows.
        new_metrics.at[idx, "Threshold"] = threshold

    comparison_rows = []
    common_columns = [c for c in baseline_metrics.columns if c in new_metrics.columns]
    for _, baseline_row in baseline_metrics.iterrows():
        level = str(baseline_row["Evaluation_Level"])
        current_matches = new_metrics.index[new_metrics["Evaluation_Level"].astype(str) == level].tolist()
        if not current_matches:
            comparison_rows.append({
                "Evaluation_Level": level,
                "Metric": "ROW",
                "Original_Value": "present in original baseline",
                "Recalculated_Value": "missing",
                "Match": False,
            })
            continue
        current_idx = current_matches[0]
        for column in common_columns:
            if column == "Evaluation_Level":
                continue
            old_value = baseline_row[column]
            new_value = new_metrics.at[current_idx, column]
            comparison_rows.append(
                {
                    "Evaluation_Level": level,
                    "Metric": column,
                    "Original_Value": old_value,
                    "Recalculated_Value": new_value,
                    "Match": values_match(old_value, new_value, column),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    if skip_indices:
        new_metrics = new_metrics.drop(
            index=skip_indices
        ).reset_index(drop=True)

    new_metrics.to_csv(metrics_path, index=False, float_format="%.4f", encoding="utf-8-sig")

    all_match = bool(comparison_df["Match"].all()) if not comparison_df.empty else False
    matched = int(comparison_df["Match"].sum()) if not comparison_df.empty else 0
    total = len(comparison_df)
    print(f"  [Done] metric.csv regenerated using saved thresholds. Comparison: {matched}/{total}; overall={'YES' if all_match else 'NO'}")
    print(f"  [Report] {comparison_path}")


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
    print("\nFinished metric reproduction check.")


if __name__ == "__main__":
    main()
