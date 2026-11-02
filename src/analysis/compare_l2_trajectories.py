#!/usr/bin/env python
"""
Purpose:
    Compare German and Finnish L2 hidden-state trajectories from the
    English-only L1 baseline through L2 training, with and without EWC.

Interpretation:
    Lower cosine similarity to the pre-L2 baseline means that the model's
    representation of the L2 probe sentence has changed more during L2 learning.

    This is not automatically "better" or "worse" learning; it measures
    representational change from the English-only starting state.
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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Hidden-state file not found: {path}")

    data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dictionary in {path}, found {type(data)}."
        )

    if "results" not in data:
        raise KeyError(f"'results' missing from {path}")

    return data["results"]


def validate_against_baseline(baseline, candidate, label):
    if len(baseline) != len(candidate):
        raise ValueError(
            f"{label}: sentence count mismatch "
            f"({len(candidate)} vs {len(baseline)})."
        )

    for i, (base_item, cand_item) in enumerate(zip(baseline, candidate)):
        if base_item["sentence"] != cand_item["sentence"]:
            raise ValueError(
                f"{label}: sentence mismatch at index {i}."
            )

        if (
            base_item["layer_vectors"].shape
            != cand_item["layer_vectors"].shape
        ):
            raise ValueError(
                f"{label}: shape mismatch at sentence {i}: "
                f"{cand_item['layer_vectors'].shape} vs "
                f"{base_item['layer_vectors'].shape}"
            )


def cosine_similarity(a, b):
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
    rows = []

    for sentence_id, (base_item, cand_item) in enumerate(
        zip(baseline, candidate)
    ):
        sentence = base_item["sentence"]

        base_vectors = base_item["layer_vectors"]
        cand_vectors = cand_item["layer_vectors"]

        for layer_index in range(base_vectors.shape[0]):
            similarity = cosine_similarity(
                base_vectors[layer_index],
                cand_vectors[layer_index],
            )

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
                    "cosine_distance": 1.0 - similarity,
                }
            )

    return rows


def process_language(
    language_name,
    baseline_path,
    seq_epoch1_path,
    seq_final_path,
    ewc_epoch1_path,
    ewc_final_path,
):
    print(f"\nLoading {language_name} pre-L2 baseline...")
    baseline = load_hidden_states(baseline_path)

    print(
        f"{language_name} baseline probe sentences:",
        len(baseline),
    )

    specs = [
        ("Sequential", "Epoch 1", seq_epoch1_path),
        ("Sequential", "Final", seq_final_path),
        ("EWC", "Epoch 1", ewc_epoch1_path),
        ("EWC", "Final", ewc_final_path),
    ]

    rows = []

    for training_condition, stage, path in specs:
        label = f"{language_name} | {training_condition} | {stage}"

        print("Processing:", label)

        candidate = load_hidden_states(path)

        validate_against_baseline(
            baseline,
            candidate,
            label,
        )

        rows.extend(
            compare_condition(
                baseline,
                candidate,
                language_name,
                training_condition,
                stage,
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--de_baseline", required=True)
    parser.add_argument("--de_seq_epoch1", required=True)
    parser.add_argument("--de_seq_final", required=True)
    parser.add_argument("--de_ewc_epoch1", required=True)
    parser.add_argument("--de_ewc_final", required=True)

    parser.add_argument("--fi_baseline", required=True)
    parser.add_argument("--fi_seq_epoch1", required=True)
    parser.add_argument("--fi_seq_final", required=True)
    parser.add_argument("--fi_ewc_epoch1", required=True)
    parser.add_argument("--fi_ewc_final", required=True)

    parser.add_argument(
        "--output_sentence_csv",
        required=True,
    )

    parser.add_argument(
        "--output_summary_csv",
        required=True,
    )

    args = parser.parse_args()

    all_rows = []

    all_rows.extend(
        process_language(
            "German",
            args.de_baseline,
            args.de_seq_epoch1,
            args.de_seq_final,
            args.de_ewc_epoch1,
            args.de_ewc_final,
        )
    )

    all_rows.extend(
        process_language(
            "Finnish",
            args.fi_baseline,
            args.fi_seq_epoch1,
            args.fi_seq_final,
            args.fi_ewc_epoch1,
            args.fi_ewc_final,
        )
    )

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

    print("\n## Mean L2 similarity to pre-L2 baseline\n")

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
        "\nInterpretation: lower cosine similarity means the L2 "
        "representation changed more from the English-only baseline."
    )

    print(
        "\nSaved sentence-level results to:",
        sentence_output,
    )

    print(
        "Saved summary results to:",
        summary_output,
    )


if __name__ == "__main__":
    main()