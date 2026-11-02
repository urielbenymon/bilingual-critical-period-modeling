#!/usr/bin/env python

"""
Quantify sentence-level representational trajectory contrasts.

Inputs:
    1. English L1 trajectory sentence-level CSV
    2. L2 trajectory sentence-level CSV

Outputs:
    1. Epoch-1 -> Final trajectory changes
    2. EWC -> Sequential contrasts
    3. German-vs-Finnish contrasts for English L1 retention
    4. German-vs-Finnish contrasts for L2 representation change

Important statistical interpretation:
    - Comparisons within the same language/probe set are paired.
    - German-vs-Finnish L1 comparisons are paired because both use the
      same English sentences.
    - German-vs-Finnish L2 comparisons are independent because German
      and Finnish use different probe sentences.
    - These are exploratory sentence-level statistics from one model seed.
      They do NOT quantify uncertainty across independently trained models.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def paired_effect_size(diff):
    """
    Cohen's dz for paired observations.

    Positive values mean the first quantity in the subtraction is larger.
    """
    diff = np.asarray(diff, dtype=float)

    if len(diff) < 2:
        return np.nan

    sd = diff.std(ddof=1)

    if sd == 0:
        return np.nan

    return diff.mean() / sd


def hedges_g(x, y):
    """
    Hedges' g for two independent groups.

    Positive values mean x > y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n1 = len(x)
    n2 = len(y)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = x.std(ddof=1)
    s2 = y.std(ddof=1)

    pooled_variance = (
        ((n1 - 1) * (s1 ** 2))
        + ((n2 - 1) * (s2 ** 2))
    ) / (n1 + n2 - 2)

    if pooled_variance <= 0:
        return np.nan

    pooled_sd = math.sqrt(pooled_variance)

    d = (x.mean() - y.mean()) / pooled_sd

    df = n1 + n2 - 2

    # Small-sample correction.
    correction = 1 - (3 / (4 * df - 1))

    return correction * d


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    """
    Bootstrap 95% confidence interval for a sample mean.
    """
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)

    n = len(values)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        boot_means[i] = sample.mean()

    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)

    return lower, upper


def trajectory_change_table(df):
    """
    Compare Final - Epoch 1 within each language/training/layer.

    Because the same probe sentences are represented at both stages,
    this is a paired contrast.
    """
    rows = []

    group_cols = [
        "language_condition",
        "training_condition",
        "layer_index",
        "layer_name",
    ]

    for keys, group in df.groupby(group_cols):

        language, training, layer_index, layer_name = keys

        pivot = group.pivot(
            index="sentence_id",
            columns="stage",
            values="cosine_similarity",
        )

        if "Epoch 1" not in pivot or "Final" not in pivot:
            continue

        pivot = pivot.dropna(
            subset=["Epoch 1", "Final"]
        )

        diff = (
            pivot["Final"]
            - pivot["Epoch 1"]
        )

        ci_low, ci_high = bootstrap_mean_ci(diff)

        rows.append(
            {
                "language_condition": language,
                "training_condition": training,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "contrast": "Final - Epoch 1",
                "mean_epoch1": pivot["Epoch 1"].mean(),
                "mean_final": pivot["Final"].mean(),
                "mean_change": diff.mean(),
                "change_ci95_low": ci_low,
                "change_ci95_high": ci_high,
                "paired_cohens_dz": paired_effect_size(diff),
                "n_sentences": len(diff),
            }
        )

    return pd.DataFrame(rows)


def ewc_contrast_table(df):
    """
    Compare EWC - Sequential within the same language, stage, and layer.

    This is paired because the same probe sentences are passed through
    both models.
    """
    rows = []

    group_cols = [
        "language_condition",
        "stage",
        "layer_index",
        "layer_name",
    ]

    for keys, group in df.groupby(group_cols):

        language, stage, layer_index, layer_name = keys

        pivot = group.pivot(
            index="sentence_id",
            columns="training_condition",
            values="cosine_similarity",
        )

        if "Sequential" not in pivot or "EWC" not in pivot:
            continue

        pivot = pivot.dropna(
            subset=["Sequential", "EWC"]
        )

        diff = (
            pivot["EWC"]
            - pivot["Sequential"]
        )

        ci_low, ci_high = bootstrap_mean_ci(diff)

        rows.append(
            {
                "language_condition": language,
                "stage": stage,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "contrast": "EWC - Sequential",
                "mean_sequential": pivot["Sequential"].mean(),
                "mean_ewc": pivot["EWC"].mean(),
                "mean_difference": diff.mean(),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "paired_cohens_dz": paired_effect_size(diff),
                "n_sentences": len(diff),
            }
        )

    return pd.DataFrame(rows)


def l1_language_contrast_table(df):
    """
    Compare Finnish - German for English L1 retention.

    This is paired because German and Finnish conditions use the same
    English probe sentences.
    """
    rows = []

    group_cols = [
        "training_condition",
        "stage",
        "layer_index",
        "layer_name",
    ]

    for keys, group in df.groupby(group_cols):

        training, stage, layer_index, layer_name = keys

        pivot = group.pivot(
            index="sentence_id",
            columns="language_condition",
            values="cosine_similarity",
        )

        if "German" not in pivot or "Finnish" not in pivot:
            continue

        pivot = pivot.dropna(
            subset=["German", "Finnish"]
        )

        diff = (
            pivot["Finnish"]
            - pivot["German"]
        )

        ci_low, ci_high = bootstrap_mean_ci(diff)

        rows.append(
            {
                "training_condition": training,
                "stage": stage,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "contrast": "Finnish - German",
                "mean_german": pivot["German"].mean(),
                "mean_finnish": pivot["Finnish"].mean(),
                "mean_difference": diff.mean(),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "paired_cohens_dz": paired_effect_size(diff),
                "n_sentences": len(diff),
            }
        )

    return pd.DataFrame(rows)


def l2_language_contrast_table(df):
    """
    Compare Finnish vs German L2 similarities.

    This comparison is NOT paired because Finnish and German use different
    probe sentences.

    We report:
        mean Finnish - mean German
        bootstrap CI for that independent mean difference
        Hedges' g
    """
    rows = []

    group_cols = [
        "training_condition",
        "stage",
        "layer_index",
        "layer_name",
    ]

    rng = np.random.default_rng(42)

    for keys, group in df.groupby(group_cols):

        training, stage, layer_index, layer_name = keys

        german = group[
            group["language_condition"] == "German"
        ]["cosine_similarity"].to_numpy()

        finnish = group[
            group["language_condition"] == "Finnish"
        ]["cosine_similarity"].to_numpy()

        if len(german) == 0 or len(finnish) == 0:
            continue

        observed_difference = (
            finnish.mean()
            - german.mean()
        )

        boot_differences = []

        for _ in range(10000):

            german_boot = rng.choice(
                german,
                size=len(german),
                replace=True,
            )

            finnish_boot = rng.choice(
                finnish,
                size=len(finnish),
                replace=True,
            )

            boot_differences.append(
                finnish_boot.mean()
                - german_boot.mean()
            )

        ci_low = np.percentile(
            boot_differences,
            2.5,
        )

        ci_high = np.percentile(
            boot_differences,
            97.5,
        )

        rows.append(
            {
                "training_condition": training,
                "stage": stage,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "contrast": "Finnish - German",
                "mean_german": german.mean(),
                "mean_finnish": finnish.mean(),
                "mean_difference": observed_difference,
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "hedges_g_finnish_vs_german": hedges_g(
                    finnish,
                    german,
                ),
                "n_german": len(german),
                "n_finnish": len(finnish),
            }
        )

    return pd.DataFrame(rows)


def print_compact_table(title, df, columns):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--l1_csv",
        required=True,
        help="English L1 sentence-level trajectory CSV.",
    )

    parser.add_argument(
        "--l2_csv",
        required=True,
        help="L2 sentence-level trajectory CSV.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for contrast CSV outputs.",
    )

    args = parser.parse_args()

    l1_df = pd.read_csv(args.l1_csv)
    l2_df = pd.read_csv(args.l2_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Training-stage changes.
    l1_changes = trajectory_change_table(l1_df)
    l2_changes = trajectory_change_table(l2_df)

    # 2. EWC effects.
    l1_ewc = ewc_contrast_table(l1_df)
    l2_ewc = ewc_contrast_table(l2_df)

    # 3. German-vs-Finnish comparisons.
    l1_language = l1_language_contrast_table(l1_df)
    l2_language = l2_language_contrast_table(l2_df)

    outputs = {
        "l1_stage_changes.csv": l1_changes,
        "l2_stage_changes.csv": l2_changes,
        "l1_ewc_contrasts.csv": l1_ewc,
        "l2_ewc_contrasts.csv": l2_ewc,
        "l1_language_contrasts.csv": l1_language,
        "l2_language_contrasts.csv": l2_language,
    }

    for filename, df in outputs.items():

        path = output_dir / filename

        df.to_csv(
            path,
            index=False,
        )

        print("Saved:", path)

    print_compact_table(
        "L1: Epoch 1 -> Final changes",
        l1_changes,
        [
            "language_condition",
            "training_condition",
            "layer_name",
            "mean_change",
            "change_ci95_low",
            "change_ci95_high",
            "paired_cohens_dz",
        ],
    )

    print_compact_table(
        "L2: Epoch 1 -> Final changes",
        l2_changes,
        [
            "language_condition",
            "training_condition",
            "layer_name",
            "mean_change",
            "change_ci95_low",
            "change_ci95_high",
            "paired_cohens_dz",
        ],
    )

    print_compact_table(
        "L1: EWC - Sequential",
        l1_ewc,
        [
            "language_condition",
            "stage",
            "layer_name",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "paired_cohens_dz",
        ],
    )

    print_compact_table(
        "L2: EWC - Sequential",
        l2_ewc,
        [
            "language_condition",
            "stage",
            "layer_name",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "paired_cohens_dz",
        ],
    )

    print_compact_table(
        "L1 retention: Finnish - German",
        l1_language,
        [
            "training_condition",
            "stage",
            "layer_name",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "paired_cohens_dz",
        ],
    )

    print_compact_table(
        "L2 representation: Finnish - German",
        l2_language,
        [
            "training_condition",
            "stage",
            "layer_name",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "hedges_g_finnish_vs_german",
        ],
    )

    print(
        "\nImportant: these confidence intervals and effect sizes describe "
        "variation across probe sentences within this single trained-model "
        "run. They are exploratory and should not be interpreted as "
        "uncertainty across independently trained model seeds."
    )


if __name__ == "__main__":
    main()