#!/usr/bin/env python
#newly added for the custom project

"""
Purpose:
    Extract hidden-state representations from a saved GPT-2 checkpoint for a
    fixed set of probe sentences.

Project role:
    This script is separate from the training pipeline. It lets us compare how
    the same sentences are represented by the L1 model, the ordinary L2 model,
    and the EWC-constrained L2 model without modifying run_clm.py.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """
    Read the checkpoint, probe-file, and output locations supplied in the terminal.
    """
    parser = argparse.ArgumentParser(
        description="Extract mean-pooled GPT-2 hidden states from probe sentences."
    )

    # Saved GPT-2 model/checkpoint whose internal representations will be examined.
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved GPT-2 checkpoint or model directory.",
    )

    # Plain-text file containing one probe sentence per line.
    parser.add_argument(
        "--probe_file",
        type=str,
        required=True,
        help="Text file containing one probe sentence per line.",
    )

    # File in which the extracted hidden-state vectors and metadata will be saved.
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the saved .pt representation file.",
    )

    # Prevent unexpectedly long probe sentences from consuming unnecessary memory.
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokens per probe sentence.",
    )

    return parser.parse_args()


def load_probe_sentences(probe_file):
    """
    Load non-empty probe sentences while preserving their original order.
    """
    probe_path = Path(probe_file)

    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    with probe_path.open("r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    if not sentences:
        raise ValueError(f"No non-empty probe sentences found in: {probe_path}")

    return sentences


def mean_pool_hidden_state(hidden_state, attention_mask):
    """
    Convert token-level hidden states into one sentence-level representation.

    hidden_state shape:
        [batch_size, sequence_length, hidden_size]

    The attention mask ensures that only real input tokens contribute to the
    sentence representation if padding is ever present.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)

    summed_hidden_state = (hidden_state * mask).sum(dim=1)
    valid_token_count = mask.sum(dim=1).clamp(min=1)

    return summed_hidden_state / valid_token_count


def extract_hidden_states(model, tokenizer, sentences, device, max_length):
    """
    Run every probe sentence through GPT-2 and collect a mean-pooled vector
    from every returned hidden-state level.

    Hugging Face GPT-2 returns:
        hidden_states[0] = embedding output
        hidden_states[1] = transformer layer 1 output
        ...
        hidden_states[-1] = final transformer layer output
    """
    # Evaluation mode disables training-specific behavior such as dropout.
    model.eval()

    sentence_results = []

    # Gradients are unnecessary because we are measuring representations, not training.
    with torch.no_grad():
        for sentence_id, sentence in enumerate(sentences):

            # Tokenize one sentence at a time to avoid introducing unnecessary padding.
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            # Move every model input tensor onto the same device as GPT-2.
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            # Ask GPT-2 to return the internal hidden states from every layer.
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            # Convert every token-level hidden-state tensor into one vector for the sentence.
            pooled_layers = []

            for layer_index, hidden_state in enumerate(hidden_states):
                pooled = mean_pool_hidden_state(
                    hidden_state,
                    encoded["attention_mask"],
                )

                # Move the stored vectors back to CPU so saved results remain portable.
                pooled_layers.append(pooled.squeeze(0).cpu())

            sentence_results.append(
                {
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "num_tokens": int(encoded["attention_mask"].sum().item()),
                    "layer_vectors": torch.stack(pooled_layers),
                }
            )

    return sentence_results


def main():
    """
    Load a trained GPT-2 checkpoint, extract representations for the probe
    sentences, verify their dimensions, and save them for later comparison.
    """
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    output_file = Path(args.output_file)

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    # Use the CPU explicitly because the current local PyTorch setup has no CUDA GPU.
    device = torch.device("cpu")

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    # Load the tokenizer associated with this checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    # Load the trained causal-language model weights from the checkpoint.
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint))
    model.to(device)

    # Load the fixed probe sentences that will later be reused for every condition.
    sentences = load_probe_sentences(args.probe_file)

    print(f"Loaded {len(sentences)} probe sentences.")

    # Extract sentence-level representations from every hidden-state level.
    results = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        device=device,
        max_length=args.max_length,
    )

    # Examine the first result to verify the representation dimensions.
    first_vectors = results[0]["layer_vectors"]

    print(f"Number of returned hidden-state levels: {first_vectors.shape[0]}")
    print(f"Hidden-state vector size: {first_vectors.shape[1]}")

    # For GPT-2, the first returned state is the embedding output; the rest
    # correspond to transformer-block outputs.
    print(
        "Interpretation: hidden-state level 0 is the embedding output; "
        "levels 1..N are transformer-layer outputs."
    )

    # Package useful metadata with the actual representations.
    save_object = {
        "checkpoint": str(checkpoint),
        "probe_file": args.probe_file,
        "model_type": model.config.model_type,
        "num_transformer_layers": model.config.n_layer,
        "hidden_size": model.config.n_embd,
        "results": results,
    }

    # Create the output directory automatically if necessary.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save representations in PyTorch format for later cosine-similarity analysis.
    torch.save(save_object, output_file)

    print(f"Saved hidden-state representations to: {output_file}")

    # Print a compact summary of each probe to confirm extraction succeeded.
    for result in results:
        print(
            f"Sentence {result['sentence_id']}: "
            f"{result['num_tokens']} tokens | "
            f"vectors shape = {tuple(result['layer_vectors'].shape)}"
        )


if __name__ == "__main__":
    main()