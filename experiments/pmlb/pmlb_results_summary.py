import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_noise_strength(path: Path) -> str:
    if path.name == "pmlb_results.csv":
        return "0"
    prefix = "pmlb_batch_inference_noise_"
    suffix = ".csv"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"无法从文件名推断噪声强度: {path}")
    noise_text = path.name[len(prefix):-len(suffix)]
    return f"{float(noise_text):g}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize PMLB batch inference CSV results")
    parser.add_argument("--input_csvs", nargs="*", default=[])
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_results_summary.csv",
    )
    args = parser.parse_args()

    fixed_noise_strengths = ["0", "0.001", "0.01", "0.1"]
    input_paths_by_noise = {noise_strength: None for noise_strength in fixed_noise_strengths}

    if args.input_csvs:
        for input_csv in args.input_csvs:
            path = Path(input_csv)
            if not path.is_file():
                raise FileNotFoundError(path)
            noise_strength = parse_noise_strength(path)
            input_paths_by_noise[noise_strength] = path
    else:
        zero_noise_path = Path("experiments/pmlb/results/pmlb_batch_inference_noise_0.csv")
        if not zero_noise_path.is_file():
            zero_noise_path = Path("experiments/pmlb/results/pmlb_results.csv")
        if zero_noise_path.is_file():
            input_paths_by_noise["0"] = zero_noise_path

        for noise_strength in fixed_noise_strengths[1:]:
            path = Path(f"experiments/pmlb/results/pmlb_batch_inference_noise_{noise_strength}.csv")
            if path.is_file():
                input_paths_by_noise[noise_strength] = path

    rows = []
    for noise_strength in fixed_noise_strengths:
        path = input_paths_by_noise[noise_strength]
        if path is None:
            df = pd.DataFrame(columns=["dataset", "status", "r2", "complexity", "seconds"])
        else:
            df = pd.read_csv(path)
            required_columns = ["dataset", "status", "r2", "complexity", "seconds"]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"{path} 缺少列: {missing_columns}")

        dataset_names = df["dataset"].astype(str).str.lower() if "dataset" in df.columns else pd.Series(dtype=str)
        groups = pd.Series("Black-box", index=df.index, dtype=object)
        groups.loc[dataset_names.str.startswith("feynman_")] = "Feynman"
        groups.loc[dataset_names.str.startswith("strogatz_")] = "Strogatz"

        raw_r2 = pd.to_numeric(df["r2"], errors="coerce")
        finite_r2_mask = raw_r2.notna() & np.isfinite(raw_r2)
        clipped_r2 = raw_r2.copy()
        clipped_r2.loc[~finite_r2_mask] = 0.0
        clipped_r2.loc[clipped_r2 < 0] = 0.0
        r2_valid_mask = finite_r2_mask & (raw_r2 >= 0)

        status_series = df["status"].astype(str).str.lower()
        success_mask = status_series.isin(["ok", "success"])
        complexity_values = pd.to_numeric(df["complexity"], errors="coerce")
        complexity_mask = success_mask & complexity_values.notna() & np.isfinite(complexity_values)
        seconds_values = pd.to_numeric(df["seconds"], errors="coerce")
        seconds_mask = success_mask & seconds_values.notna() & np.isfinite(seconds_values)

        grouped_df = pd.DataFrame(
            {
                "group": groups,
                "raw_r2": raw_r2,
                "clipped_r2": clipped_r2,
                "r2_valid_mask": r2_valid_mask,
                "complexity_values": complexity_values,
                "complexity_mask": complexity_mask,
                "seconds_values": seconds_values,
                "seconds_mask": seconds_mask,
            }
        )

        for group_name in ["Feynman", "Strogatz", "Black-box"]:
            group_df = grouped_df[grouped_df["group"] == group_name]

            valid_complexity = group_df.loc[group_df["complexity_mask"], "complexity_values"]
            valid_seconds = group_df.loc[group_df["seconds_mask"], "seconds_values"]

            rows.append(
                {
                    "noise_strength": noise_strength,
                    "group": group_name,
                    "r2_mean": float(group_df["clipped_r2"].mean()) if len(group_df) else np.nan,
                    "r2_var": float(group_df["clipped_r2"].var(ddof=0)) if len(group_df) else np.nan,
                    "r2_valid_count": int(group_df["r2_valid_mask"].sum()),
                    "total_count": int(len(group_df)),
                    "recovery_rate": float((group_df["raw_r2"] > 0.9).sum() / len(group_df)) if len(group_df) else np.nan,
                    "complexity_mean": float(valid_complexity.mean()) if len(valid_complexity) else np.nan,
                    "complexity_var": float(valid_complexity.var(ddof=0)) if len(valid_complexity) else np.nan,
                    "complexity_count": int(len(valid_complexity)),
                    "seconds_mean": float(valid_seconds.mean()) if len(valid_seconds) else np.nan,
                    "seconds_var": float(valid_seconds.var(ddof=0)) if len(valid_seconds) else np.nan,
                    "seconds_count": int(len(valid_seconds)),
                }
            )

    summary_df = pd.DataFrame(
        rows,
        columns=[
            "noise_strength",
            "group",
            "r2_mean",
            "r2_var",
            "r2_valid_count",
            "total_count",
            "recovery_rate",
            "complexity_mean",
            "complexity_var",
            "complexity_count",
            "seconds_mean",
            "seconds_var",
            "seconds_count",
        ],
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    printable_df = summary_df.copy()
    for column in [
        "r2_mean",
        "r2_var",
        "recovery_rate",
        "complexity_mean",
        "complexity_var",
        "seconds_mean",
        "seconds_var",
    ]:
        printable_df[column] = printable_df[column].map(lambda value: "nan" if pd.isna(value) else f"{value:.6f}")

    widths = {}
    for column in printable_df.columns:
        widths[column] = max(len(column), printable_df[column].astype(str).map(len).max())

    header = " | ".join(column.ljust(widths[column]) for column in printable_df.columns)
    separator = "-+-".join("-" * widths[column] for column in printable_df.columns)
    print(header)
    print(separator)
    for _, row in printable_df.iterrows():
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in printable_df.columns))

    print(f"\n汇总 CSV 已保存到: {output_csv}")
