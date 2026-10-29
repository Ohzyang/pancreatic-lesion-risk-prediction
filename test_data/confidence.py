import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Place this script directly inside test_data. All paths are anchored to this file.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILENAME = "Metrics.csv"
RAW_BOOTSTRAP_FILENAME = "Bootstrap_200_Iterations_Results.csv"
OUTPUT_FILENAME = "Metrics_with_95CI_Overwritten.csv"
N_BOOTSTRAP = 200
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv"}
COUNT_METRICS = {"TN", "FP", "FN", "TP"}
META_COLUMNS = {"Iteration", "Window", "Threshold_Type", "Threshold"}


def read_csv_auto(path):
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to identify CSV encoding: {path}")


def find_result_dirs():
    found = []
    for current_dir, subdirs, filenames in os.walk(ROOT_DIR):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)
        if METRICS_FILENAME in filenames and RAW_BOOTSTRAP_FILENAME in filenames:
            found.append(current_dir)
    return sorted(found)


def format_with_ci(point, lower, upper, metric):
    point = float(point)
    if metric in COUNT_METRICS:
        return f"{int(point)} ({int(lower)}-{int(upper)})"
    return f"{point:.4f} ({float(lower):.4f}-{float(upper):.4f})"


def process_directory(base_dir):
    metrics_path = os.path.join(base_dir, METRICS_FILENAME)
    raw_path = os.path.join(base_dir, RAW_BOOTSTRAP_FILENAME)
    output_path = os.path.join(base_dir, OUTPUT_FILENAME)

    metrics_df = read_csv_auto(metrics_path)
    raw_df = read_csv_auto(raw_path)
    if "Evaluation_Level" not in metrics_df.columns:
        raise ValueError("metric.csv must contain Evaluation_Level.")
    missing_meta = META_COLUMNS - set(raw_df.columns)
    if missing_meta:
        raise ValueError(f"Bootstrap CSV is missing metadata columns: {sorted(missing_meta)}")

    output_df = metrics_df.copy().astype(object)
    ci_rows = 0
    for window in sorted(raw_df["Window"].dropna().astype(str).unique()):
        try:
            window_num = int(window.lower().replace("m", ""))
        except ValueError:
            print(f"  [Warning] Unrecognized Window value {window}; skipping it.")
            continue

        for threshold_type in ("Default", "OptTh"):
            group = raw_df[
                (raw_df["Window"].astype(str) == window)
                & (raw_df["Threshold_Type"].astype(str) == threshold_type)
            ].copy()
            if group.empty:
                continue
            if len(group) != N_BOOTSTRAP or group["Iteration"].nunique() != N_BOOTSTRAP:
                raise ValueError(
                    f"{window}/{threshold_type} has {len(group)} rows and "
                    f"{group['Iteration'].nunique()} unique iterations; expected {N_BOOTSTRAP}."
                )

            level = (
                f"Window_{window_num}m_DefaultTh(0.5)_Imbalanced"
                if threshold_type == "Default"
                else f"Window_{window_num}m_ValidatedOptTh_Imbalanced"
            )
            matches = output_df.index[output_df["Evaluation_Level"].astype(str) == level].tolist()
            if not matches:
                print(f"  [Warning] metric.csv has no row {level}; CI not attached.")
                continue
            row_idx = matches[0]

            for metric in group.columns:
                if metric in META_COLUMNS or metric not in output_df.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
                if len(values) != N_BOOTSTRAP:
                    raise ValueError(f"{window}/{threshold_type}/{metric} does not have 200 numeric values.")
                point = pd.to_numeric(pd.Series([output_df.at[row_idx, metric]]), errors="coerce").iloc[0]
                if pd.isna(point):
                    continue
                lower, upper = np.percentile(values, [2.5, 97.5])
                output_df.at[row_idx, metric] = format_with_ci(point, lower, upper, metric)
            ci_rows += 1

    output_df.to_csv(output_path, index=False, float_format="%.4f", encoding="utf-8-sig")
    print(f"  [Done] Saved CI results to {output_path}; window/threshold rows updated: {ci_rows}")
    print("  [Note] Patient-level rows are retained as point estimates; this script adds CI only to window-level rows.")


def main():
    directories = find_result_dirs()
    print(f"Root directory: {ROOT_DIR}")
    print(f"Found {len(directories)} directories containing both {METRICS_FILENAME} and {RAW_BOOTSTRAP_FILENAME}.")
    for i, directory in enumerate(directories, start=1):
        print(f"\n[{i}/{len(directories)}] {os.path.relpath(directory, ROOT_DIR)}")
        try:
            process_directory(directory)
        except Exception as exc:
            print(f"  [Error] {exc}")
    print("\nFinished confidence-interval generation from Bootstrap results.")


if __name__ == "__main__":
    main()
