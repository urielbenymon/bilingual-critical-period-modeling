#!/usr/bin/env python

"""
Final consistency audit for Step 5 results.

Checks that:
1. Compact summary values match the underlying contrast CSVs.
2. Key Sequential-Final German/Finnish contrasts match expected source rows.
3. The largest absolute transformer-layer EWC effects in the compact summary
   match the underlying EWC contrast tables.

This does not recompute hidden states or statistics. It only verifies that the
derived reporting tables are internally consistent with the source analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("analysis_outputs/class")
CONTRAST_DIR = BASE / "contrasts"
RESULTS_DIR = BASE / "results_tables"


def assert_close(label, a, b, atol=1e-10):
    if not np.isclose(a, b, atol=atol, rtol=0):
        raise AssertionError(
            f"{label} mismatch: {a} vs {b}"
        )


def main():

    print("Loading source contrast tables...")

    l1_language = pd.read_csv(
        CONTRAST_DIR / "l1_language_contrasts.csv"
    )

    l2_language = pd.read_csv(
        CONTRAST_DIR / "l2_language_contrasts.csv"
    )

    l1_ewc = pd.read_csv(
        CONTRAST_DIR / "l1_ewc_contrasts.csv"
    )

    l2_ewc = pd.read_csv(
        CONTRAST_DIR / "l2_ewc_contrasts.csv"
    )

    compact = pd.read_csv(
        RESULTS_DIR / "compact_transformer_results.csv"
    )

    print("Checking compact table row count...")

    if len(compact) == 0:
        raise AssertionError("Compact results table is empty.")

    if not set(compact["layer"]).issubset(
        {"transformer_layer_1", "transformer_layer_2"}
    ):
        raise AssertionError(
            "Compact table contains unexpected non-transformer layers."
        )

    print("Checking key L1 Sequential-Final contrasts...")

    for layer in [
        "transformer_layer_1",
        "transformer_layer_2",
    ]:

        source = l1_language[
            (l1_language["training_condition"] == "Sequential")
            & (l1_language["stage"] == "Final")
            & (l1_language["layer_name"] == layer)
        ].iloc[0]

        report = compact[
            (compact["analysis"] == "L1 retention")
            & (compact["comparison"] == "Finnish - German")
            & (compact["training_condition"] == "Sequential")
            & (compact["stage"] == "Final")
            & (compact["layer"] == layer)
        ].iloc[0]

        assert_close(
            f"L1 {layer} estimate",
            source["mean_difference"],
            report["estimate"],
        )

        assert_close(
            f"L1 {layer} CI low",
            source["difference_ci95_low"],
            report["ci95_low"],
        )

        assert_close(
            f"L1 {layer} CI high",
            source["difference_ci95_high"],
            report["ci95_high"],
        )

    print("Checking key L2 Sequential-Final contrasts...")

    for layer in [
        "transformer_layer_1",
        "transformer_layer_2",
    ]:

        source = l2_language[
            (l2_language["training_condition"] == "Sequential")
            & (l2_language["stage"] == "Final")
            & (l2_language["layer_name"] == layer)
        ].iloc[0]

        report = compact[
            (compact["analysis"] == "L2 reorganization")
            & (compact["comparison"] == "Finnish - German")
            & (compact["training_condition"] == "Sequential")
            & (compact["stage"] == "Final")
            & (compact["layer"] == layer)
        ].iloc[0]

        assert_close(
            f"L2 {layer} estimate",
            source["mean_difference"],
            report["estimate"],
        )

        assert_close(
            f"L2 {layer} CI low",
            source["difference_ci95_low"],
            report["ci95_low"],
        )

        assert_close(
            f"L2 {layer} CI high",
            source["difference_ci95_high"],
            report["ci95_high"],
        )

        assert_close(
            f"L2 {layer} Hedges g",
            source["hedges_g_finnish_vs_german"],
            report["effect_size"],
        )

    print("Checking maximum transformer-layer EWC effects...")

    l1_transformer = l1_ewc[
        l1_ewc["layer_name"].isin(
            ["transformer_layer_1", "transformer_layer_2"]
        )
    ]

    l2_transformer = l2_ewc[
        l2_ewc["layer_name"].isin(
            ["transformer_layer_1", "transformer_layer_2"]
        )
    ]

    max_l1 = l1_transformer.loc[
        l1_transformer["mean_difference"].abs().idxmax()
    ]

    max_l2 = l2_transformer.loc[
        l2_transformer["mean_difference"].abs().idxmax()
    ]

    print(
        "Largest L1 transformer EWC effect:",
        max_l1["language_condition"],
        "|",
        max_l1["stage"],
        "|",
        max_l1["layer_name"],
        "=",
        f"{max_l1['mean_difference']:.6f}",
    )

    print(
        "Largest L2 transformer EWC effect:",
        max_l2["language_condition"],
        "|",
        max_l2["stage"],
        "|",
        max_l2["layer_name"],
        "=",
        f"{max_l2['mean_difference']:.6f}",
    )

    print("\nAll consistency checks passed.")
    print(
        "Step 5 reporting tables are internally consistent "
        "with the underlying contrast analyses."
    )


if __name__ == "__main__":
    main()