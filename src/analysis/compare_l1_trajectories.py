#!/usr/bin/env python
"""
Purpose:
    Compare English L1 hidden-state representations across L2 training
    trajectories for German and Finnish, with and without EWC.

What this script does:
    - Loads the English L1 baseline hidden states.
    - Loads epoch-1 and final hidden states for:
        * German Sequential
        * German EWC
        * Finnish Sequential
        * Finnish EWC
    - Computes cosine similarity and cosine distance to the English L1 baseline.
    - Aggregates results by language condition, training condition, stage, and layer.
    - Saves both:
        1. per-sentence comparison results
        2. aggregated mean trajectory results

Interpretation:
    Higher cosine similarity means the representation remained closer to
    the original English L1 representation.

    Lower cosine distance means less representational drift from English L1.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


LAYER_NAMES = {
    0: "embedding",
    1: "transformer_layer_1",
    2: "transformer_layer_2",
}


def load_hidden_states(path):
    """
    Load one hidden-state file created by extract_hidden_states.py
    and return its per-sentence results list.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Hidden-state file not found: {path}")

    data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, but found {type(data)}."
        )

    if "results" not in data:
        raise KeyError(
            f"Expected key 'results' in hidden-state file: {path}"
        )

    return data["results"]

def validate_against_baseline(baseline, candidate, candidate_name):
    """
    Ensure that the candidate file is directly comparable with the baseline.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"{candidate_name}: sentence count differs from baseline "
            f"({len(candidate)} vs {len(baseline)})."
        )

    for i, (base_item, cand_item) in enumerate(zip(baseline, candidate)):

        if base_item["sentence"] != cand_item["sentence"]:
            raise ValueError(
                f"{candidate_name}: sentence mismatch at index {i}."
            )

        if base_item["layer_vectors"].shape != cand_item["layer_vectors"].shape:
            raise ValueError(
                f"{candidate_name}: hidden-state shape mismatch at sentence {i}: "
                f"{cand_item['layer_vectors'].shape} vs "
                f"{base_item['layer_vectors'].shape}"
            )

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1-D representation vectors.
    """
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=1,
    ).item()


def compare_condition(
    baseline,
    candidate,
    language_condition,
    training_condition,
    stage,
):
    """
    Compare one candidate checkpoint against the English L1 baseline.

    Returns one row per:
        sentence x hidden-state layer
    """
    rows = []

    for sentence_id, (base_item, cand_item) in enumerate(
        zip(baseline, candidate)
    ):

        sentence = base_item["sentence"]

        base_vectors = base_item["layer_vectors"]
        cand_vectors = cand_item["layer_vectors"]

        num_layers = base_vectors.shape[0]

        for layer_index in range(num_layers):

            baseline_vector = base_vectors[layer_index]
            candidate_vector = cand_vectors[layer_index]

            similarity = cosine_similarity(
                baseline_vector,
                candidate_vector,
            )

            distance = 1.0 - similarity

            rows.append(
                {
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "language_condition": language_condition,
                    "training_condition": training_condition,
                    "stage": stage,
                    "layer_index": layer_index,
                    "layer_name": LAYER_NAMES.get(
                        layer_index,
                        f"layer_{layer_index}",
                    ),
                    "cosine_similarity": similarity,
                    "cosine_distance": distance,
                }
            )

    return rows


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare English L1 representation trajectories during "
            "German and Finnish L2 training."
        )
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help="English L1 baseline hidden-state file.",
    )

    parser.add_argument(
        "--de_seq_epoch1",
        required=True,
        help="German Sequential epoch-1 hidden states.",
    )

    parser.add_argument(
        "--de_seq_final",
        required=True,
        help="German Sequential final hidden states.",
    )

    parser.add_argument(
        "--de_ewc_epoch1",
        required=True,
        help="German EWC epoch-1 hidden states.",
    )

    parser.add_argument(
        "--de_ewc_final",
        required=True,
        help="German EWC final hidden states.",
    )

    parser.add_argument(
        "--fi_seq_epoch1",
        required=True,
        help="Finnish Sequential epoch-1 hidden states.",
    )

    parser.add_argument(
        "--fi_seq_final",
        required=True,
        help="Finnish Sequential final hidden states.",
    )

    parser.add_argument(
        "--fi_ewc_epoch1",
        required=True,
        help="Finnish EWC epoch-1 hidden states.",
    )

    parser.add_argument(
        "--fi_ewc_final",
        required=True,
        help="Finnish EWC final hidden states.",
    )

    parser.add_argument(
        "--output_sentence_csv",
        required=True,
        help="Output CSV for sentence-level results.",
    )

    parser.add_argument(
        "--output_summary_csv",
        required=True,
        help="Output CSV for aggregated trajectory results.",
    )

    args = parser.parse_args()

    print("Loading English L1 baseline...")
    baseline = load_hidden_states(args.baseline)

    condition_specs = [
        (
            "German",
            "Sequential",
            "Epoch 1",
            args.de_seq_epoch1,
        ),
        (
            "German",
            "Sequential",
            "Final",
            args.de_seq_final,
        ),
        (
            "German",
            "EWC",
            "Epoch 1",
            args.de_ewc_epoch1,
        ),
        (
            "German",
            "EWC",
            "Final",
            args.de_ewc_final,
        ),
        (
            "Finnish",
            "Sequential",
            "Epoch 1",
            args.fi_seq_epoch1,
        ),
        (
            "Finnish",
            "Sequential",
            "Final",
            args.fi_seq_final,
        ),
        (
            "Finnish",
            "EWC",
            "Epoch 1",
            args.fi_ewc_epoch1,
        ),
        (
            "Finnish",
            "EWC",
            "Final",
            args.fi_ewc_final,
        ),
    ]

    all_rows = []

    print(f"Baseline probe sentences: {len(baseline)}")

    for (
        language_condition,
        training_condition,
        stage,
        path,
    ) in condition_specs:

        label = (
            f"{language_condition} | "
            f"{training_condition} | "
            f"{stage}"
        )

        print(f"Processing: {label}")

        candidate = load_hidden_states(path)

        validate_against_baseline(
            baseline,
            candidate,
            label,
        )

        rows = compare_condition(
            baseline,
            candidate,
            language_condition,
            training_condition,
            stage,
        )

        all_rows.extend(rows)

    sentence_df = pd.DataFrame(all_rows)

    summary_df = (
        sentence_df
        .groupby(
            [
                "language_condition",
                "training_condition",
                "stage",
                "layer_index",
                "layer_name",
            ],
            as_index=False,
        )
        .agg(
            mean_cosine_similarity=(
                "cosine_similarity",
                "mean",
            ),
            std_cosine_similarity=(
                "cosine_similarity",
                "std",
            ),
            mean_cosine_distance=(
                "cosine_distance",
                "mean",
            ),
            std_cosine_distance=(
                "cosine_distance",
                "std",
            ),
            n_sentences=(
                "sentence_id",
                "count",
            ),
        )
    )

    # Add explicit ordering so CSVs are easier to inspect.
    language_order = {
        "German": 0,
        "Finnish": 1,
    }

    training_order = {
        "Sequential": 0,
        "EWC": 1,
    }

    stage_order = {
        "Epoch 1": 0,
        "Final": 1,
    }

    summary_df["_language_order"] = (
        summary_df["language_condition"]
        .map(language_order)
    )

    summary_df["_training_order"] = (
        summary_df["training_condition"]
        .map(training_order)
    )

    summary_df["_stage_order"] = (
        summary_df["stage"]
        .map(stage_order)
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "_language_order",
                "_training_order",
                "_stage_order",
                "layer_index",
            ]
        )
        .drop(
            columns=[
                "_language_order",
                "_training_order",
                "_stage_order",
            ]
        )
        .reset_index(drop=True)
    )

    sentence_output = Path(args.output_sentence_csv)
    summary_output = Path(args.output_summary_csv)

    sentence_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sentence_df.to_csv(
        sentence_output,
        index=False,
    )

    summary_df.to_csv(
        summary_output,
        index=False,
    )

    print("\n## Mean English L1 similarity to baseline\n")

    display_columns = [
        "language_condition",
        "training_condition",
        "stage",
        "layer_name",
        "mean_cosine_similarity",
        "mean_cosine_distance",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\nInterpretation: higher cosine similarity and lower cosine "
        "distance indicate that English L1 representations remained "
        "closer to their pre-L2 baseline."
    )

    print(
        "\nSaved sentence-level results to:",
        sentence_output,
    )

    print(
        "Saved aggregated trajectory results to:",
        summary_output,
    )


if __name__ == "__main__":
    main()