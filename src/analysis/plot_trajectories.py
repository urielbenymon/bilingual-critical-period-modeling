#!/usr/bin/env python

"""
Plot representational trajectories from the class experiment.

Creates:
    1. English L1 retention trajectory plot
    2. L2 representational trajectory plot

The plots focus on transformer layers 1 and 2.
The embedding layer is excluded because it behaves differently from
contextual transformer representations and is better analyzed separately.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("analysis_outputs/class")
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_l1_retention():
    """
    Plot similarity to the original English L1 representation during
    German or Finnish L2 training.
    """

    path = OUTPUT_DIR / "en_l1_trajectory_summary.csv"
    df = pd.read_csv(path)

    # Focus on contextual transformer representations.
    df = df[df["layer_index"].isin([1, 2])].copy()

    stage_order = ["Epoch 1", "Final"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for (
        language,
        training,
        layer_name,
    ), group in df.groupby(
        [
            "language_condition",
            "training_condition",
            "layer_name",
        ]
    ):

        group = group.copy()

        group["stage"] = pd.Categorical(
            group["stage"],
            categories=stage_order,
            ordered=True,
        )

        group = group.sort_values("stage")

        label = f"{language} | {training} | {layer_name}"

        ax.plot(
            group["stage"],
            group["mean_cosine_similarity"],
            marker="o",
            label=label,
        )

    ax.set_title(
        "English L1 Representational Retention During L2 Training"
    )

    ax.set_xlabel("L2 Training Stage")
    ax.set_ylabel("Mean Cosine Similarity to English L1 Baseline")

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "english_l1_retention_trajectory.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved L1 retention plot to: {output_path}")


def plot_l2_trajectory():
    """
    Plot German and Finnish representational trajectories relative
    to their respective pre-L2 English-only baselines.
    """

    path = OUTPUT_DIR / "l2_trajectory_summary.csv"
    df = pd.read_csv(path)

    # Again focus on contextual transformer representations.
    df = df[df["layer_index"].isin([1, 2])].copy()

    stage_order = ["Epoch 1", "Final"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for (
        language,
        training,
        layer_name,
    ), group in df.groupby(
        [
            "language_condition",
            "training_condition",
            "layer_name",
        ]
    ):

        group = group.copy()

        group["stage"] = pd.Categorical(
            group["stage"],
            categories=stage_order,
            ordered=True,
        )

        group = group.sort_values("stage")

        label = f"{language} | {training} | {layer_name}"

        ax.plot(
            group["stage"],
            group["mean_cosine_similarity"],
            marker="o",
            label=label,
        )

    ax.set_title(
        "L2 Representational Trajectories During Acquisition"
    )

    ax.set_xlabel("L2 Training Stage")

    ax.set_ylabel(
        "Mean Cosine Similarity to Pre-L2 Baseline"
    )

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "l2_acquisition_trajectory.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved L2 trajectory plot to: {output_path}")


def main():

    print("Generating trajectory plots...")

    plot_l1_retention()
    plot_l2_trajectory()

    print("Done.")


if __name__ == "__main__":
    main()