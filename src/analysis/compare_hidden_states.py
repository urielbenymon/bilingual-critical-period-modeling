#!/usr/bin/env python
#newly added for the custom project
"""
Purpose:
    Compare hidden-state representations from an L1 baseline checkpoint with
    ordinary sequential L2 training and EWC-constrained L2 training.

Project role:
    This script quantifies representation change using cosine similarity and
    cosine distance while verifying that the compared files contain the same
    probe sentences, model dimensions, and hidden-state levels.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    """Read representation-file paths and the requested CSV output path."""
    parser = argparse.ArgumentParser(
        description="Compare GPT-2 hidden-state representations across training conditions."
    )

    # Hidden states extracted from the model after L1 training.
    parser.add_argument(
        "--l1_file",
        type=str,
        required=True,
        help="Hidden-state .pt file from the L1 baseline model.",
    )

    # Hidden states extracted after ordinary sequential L2 training.
    parser.add_argument(
        "--sequential_file",
        type=str,
        required=True,
        help="Hidden-state .pt file from the ordinary sequential L2 model.",
    )

    # Hidden states extracted after EWC-constrained L2 training.
    parser.add_argument(
        "--ewc_file",
        type=str,
        required=True,
        help="Hidden-state .pt file from the EWC L2 model.",
    )

    # CSV file in which per-sentence/per-layer comparison results will be stored.
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path for the CSV containing representation-comparison results.",
    )

    return parser.parse_args()


def load_representation_file(path):
    """Load one saved hidden-state extraction file and check that it exists."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Representation file not found: {file_path}")

    return torch.load(file_path, map_location="cpu")


def validate_representation_sets(l1_data, sequential_data, ewc_data):
    """
    Verify that all three files contain representations that can be compared
    directly: same sentences, same ordering, same number of layers, and same
    vector dimensions.
    """
    datasets = {
        "L1": l1_data,
        "Sequential L2": sequential_data,
        "EWC L2": ewc_data,
    }

    # Confirm model-level dimensions match.
    reference_hidden_size = l1_data["hidden_size"]
    reference_num_layers = l1_data["num_transformer_layers"]

    for name, data in datasets.items():
        if data["hidden_size"] != reference_hidden_size:
            raise ValueError(
                f"{name} hidden size ({data['hidden_size']}) does not match "
                f"L1 hidden size ({reference_hidden_size})."
            )

        if data["num_transformer_layers"] != reference_num_layers:
            raise ValueError(
                f"{name} transformer-layer count ({data['num_transformer_layers']}) "
                f"does not match L1 ({reference_num_layers})."
            )

    # Confirm all files contain the same number of probe sentences.
    num_sentences = len(l1_data["results"])

    if len(sequential_data["results"]) != num_sentences:
        raise ValueError("Sequential L2 file has a different number of probe sentences.")

    if len(ewc_data["results"]) != num_sentences:
        raise ValueError("EWC L2 file has a different number of probe sentences.")

    # Confirm sentence identity, ordering, and tensor dimensions match.
    for index in range(num_sentences):
        l1_result = l1_data["results"][index]
        sequential_result = sequential_data["results"][index]
        ewc_result = ewc_data["results"][index]

        if not (
            l1_result["sentence_id"]
            == sequential_result["sentence_id"]
            == ewc_result["sentence_id"]
        ):
            raise ValueError(f"Sentence ID mismatch at index {index}.")

        if not (
            l1_result["sentence"]
            == sequential_result["sentence"]
            == ewc_result["sentence"]
        ):
            raise ValueError(f"Sentence text mismatch at index {index}.")

        l1_shape = tuple(l1_result["layer_vectors"].shape)
        sequential_shape = tuple(sequential_result["layer_vectors"].shape)
        ewc_shape = tuple(ewc_result["layer_vectors"].shape)

        if not (l1_shape == sequential_shape == ewc_shape):
            raise ValueError(
                f"Representation shape mismatch at sentence {index}: "
                f"L1={l1_shape}, Sequential={sequential_shape}, EWC={ewc_shape}"
            )

    return num_sentences


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two one-dimensional vectors."""
    return F.cosine_similarity(
        vector_a.unsqueeze(0),
        vector_b.unsqueeze(0),
        dim=1,
    ).item()


def compare_representations(l1_data, sequential_data, ewc_data):
    """
    Compare each sentence representation in the two L2 conditions against the
    same sentence's L1 baseline representation at every hidden-state level.
    """
    rows = []

    num_sentences = len(l1_data["results"])

    for sentence_index in range(num_sentences):
        l1_result = l1_data["results"][sentence_index]
        sequential_result = sequential_data["results"][sentence_index]
        ewc_result = ewc_data["results"][sentence_index]

        l1_vectors = l1_result["layer_vectors"]
        sequential_vectors = sequential_result["layer_vectors"]
        ewc_vectors = ewc_result["layer_vectors"]

        num_hidden_levels = l1_vectors.shape[0]

        for layer_index in range(num_hidden_levels):
            l1_vector = l1_vectors[layer_index]
            sequential_vector = sequential_vectors[layer_index]
            ewc_vector = ewc_vectors[layer_index]

            # Similarity to the original L1 representation after ordinary L2 training.
            sequential_similarity = cosine_similarity(
                l1_vector,
                sequential_vector,
            )

            # Similarity to the original L1 representation after EWC-constrained L2 training.
            ewc_similarity = cosine_similarity(
                l1_vector,
                ewc_vector,
            )

            # Cosine distance is 1 - cosine similarity; larger values indicate greater movement.
            sequential_distance = 1.0 - sequential_similarity
            ewc_distance = 1.0 - ewc_similarity

            rows.append(
                {
                    "sentence_id": l1_result["sentence_id"],
                    "sentence": l1_result["sentence"],
                    "layer_index": layer_index,
                    "layer_name": (
                        "embedding"
                        if layer_index == 0
                        else f"transformer_layer_{layer_index}"
                    ),
                    "sequential_cosine_similarity": sequential_similarity,
                    "sequential_cosine_distance": sequential_distance,
                    "ewc_cosine_similarity": ewc_similarity,
                    "ewc_cosine_distance": ewc_distance,
                    # Positive values mean EWC stayed more similar to the L1 baseline.
                    "ewc_minus_sequential_similarity": (
                        ewc_similarity - sequential_similarity
                    ),
                }
            )

    return rows


def summarize_by_layer(rows):
    """Average cosine similarities/distances across all probe sentences per layer."""
    layer_indices = sorted({row["layer_index"] for row in rows})

    summary = []

    for layer_index in layer_indices:
        layer_rows = [
            row for row in rows
            if row["layer_index"] == layer_index
        ]

        count = len(layer_rows)

        sequential_similarity = sum(
            row["sequential_cosine_similarity"] for row in layer_rows
        ) / count

        ewc_similarity = sum(
            row["ewc_cosine_similarity"] for row in layer_rows
        ) / count

        sequential_distance = sum(
            row["sequential_cosine_distance"] for row in layer_rows
        ) / count

        ewc_distance = sum(
            row["ewc_cosine_distance"] for row in layer_rows
        ) / count

        summary.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_rows[0]["layer_name"],
                "sequential_similarity": sequential_similarity,
                "ewc_similarity": ewc_similarity,
                "sequential_distance": sequential_distance,
                "ewc_distance": ewc_distance,
                "ewc_minus_sequential_similarity": (
                    ewc_similarity - sequential_similarity
                ),
            }
        )

    return summary


def save_csv(rows, output_csv):
    """Save all sentence-level and layer-level comparison measurements."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sentence_id",
        "sentence",
        "layer_index",
        "layer_name",
        "sequential_cosine_similarity",
        "sequential_cosine_distance",
        "ewc_cosine_similarity",
        "ewc_cosine_distance",
        "ewc_minus_sequential_similarity",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-sentence comparison results to: {output_path}")


def print_summary(summary):
    """Print an easy-to-read layer-level comparison table."""
    print("\nMean representation similarity to the L1 baseline")
    print("-" * 94)
    print(
        f"{'Layer':<24}"
        f"{'Sequential sim.':>18}"
        f"{'EWC sim.':>15}"
        f"{'Sequential dist.':>19}"
        f"{'EWC dist.':>15}"
    )
    print("-" * 94)

    for row in summary:
        print(
            f"{row['layer_name']:<24}"
            f"{row['sequential_similarity']:>18.6f}"
            f"{row['ewc_similarity']:>15.6f}"
            f"{row['sequential_distance']:>19.6f}"
            f"{row['ewc_distance']:>15.6f}"
        )

    print("-" * 94)

    print(
        "\nInterpretation: higher cosine similarity (or lower cosine distance) "
        "means the representation remained closer to its L1 baseline."
    )

    print(
        "A positive EWC-minus-Sequential similarity difference would mean that "
        "the EWC representation remained more similar to L1 for that layer."
    )

    print("\nEWC minus Sequential cosine similarity")
    print("-" * 55)

    for row in summary:
        print(
            f"{row['layer_name']:<24}"
            f"{row['ewc_minus_sequential_similarity']:>12.6f}"
        )


def main():
    """
    Load the L1, ordinary-L2, and EWC-L2 representation files; validate them;
    calculate cosine comparisons; print summaries; and save detailed results.
    """
    args = parse_args()

    # Load the three representation datasets.
    l1_data = load_representation_file(args.l1_file)
    sequential_data = load_representation_file(args.sequential_file)
    ewc_data = load_representation_file(args.ewc_file)

    # Ensure the comparison is methodologically valid before calculating anything.
    num_sentences = validate_representation_sets(
        l1_data,
        sequential_data,
        ewc_data,
    )

    print(f"Validated {num_sentences} matching probe sentences.")
    print(f"Hidden size: {l1_data['hidden_size']}")
    print(
        "Returned hidden-state levels per sentence: "
        f"{l1_data['results'][0]['layer_vectors'].shape[0]}"
    )

    # Calculate per-sentence/per-layer representation change.
    rows = compare_representations(
        l1_data,
        sequential_data,
        ewc_data,
    )

    # Average the comparisons across probe sentences for each hidden-state level.
    summary = summarize_by_layer(rows)

    # Display the first compact representation-change result.
    print_summary(summary)

    # Preserve detailed measurements so they can later be plotted or analyzed statistically.
    save_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()