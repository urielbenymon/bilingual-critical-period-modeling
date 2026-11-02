#!/usr/bin/env python

"""
Build matched English (L1), German (L2), and Finnish (L2) corpora.

Experimental controls:
- 3,000 sentences per language.
- Same broad source/domain: OPUS Books.
- Same train/validation/test proportions.
- English and German are sampled from disjoint rows of de-en.
- Finnish is sampled independently from en-fi.
- Therefore no selected English L1 sentence is the aligned translation
  counterpart of a selected German or Finnish L2 sentence.
"""

from pathlib import Path
import random
import re

import pandas as pd


SEED = 42
TARGET_SENTENCES = 3000

TRAIN_RATIO = 0.83
VALID_RATIO = 0.085

DE_URL = (
    "https://huggingface.co/datasets/Helsinki-NLP/opus_books/"
    "resolve/main/de-en/train-00000-of-00001.parquet"
)

FI_URL = (
    "https://huggingface.co/datasets/Helsinki-NLP/opus_books/"
    "resolve/main/en-fi/train-00000-of-00001.parquet"
)

OUTPUT_ROOT = Path("data/class_experiment")


def clean_sentence(text):
    """Conservatively clean an OPUS Books sentence."""

    if not isinstance(text, str):
        return None

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 10:
        return None

    lowered = text.lower()

    metadata_markers = [
        "source:",
        "project gutenberg",
        "manybooks.net",
        "audiobook available",
        "http://",
        "https://",
    ]

    if any(marker in lowered for marker in metadata_markers):
        return None

    return text


def get_clean_parallel_rows(df, lang1, lang2):
    """
    Keep rows where both sides of the parallel pair contain usable text.

    Keeping the row association lets us explicitly prevent English and
    German from being selected from the same aligned translation rows.
    """

    rows = []

    for row_id, translation in zip(df["id"], df["translation"]):
        if not isinstance(translation, dict):
            continue

        text1 = clean_sentence(translation.get(lang1))
        text2 = clean_sentence(translation.get(lang2))

        if text1 is None or text2 is None:
            continue

        rows.append(
            {
                "row_id": str(row_id),
                lang1: text1,
                lang2: text2,
            }
        )

    return rows


def unique_language_rows(rows, language):
    """
    Remove duplicate sentences while retaining their source-row identity.
    """

    seen = set()
    unique_rows = []

    for row in rows:
        sentence = row[language]

        if sentence in seen:
            continue

        seen.add(sentence)
        unique_rows.append(row)

    return unique_rows


def select_disjoint_english_german(rows, n, seed):
    """
    Select English and German from different de-en translation rows.
    """

    rng = random.Random(seed)

    # Require unique sentences on each side.
    english_candidates = unique_language_rows(rows, "en")
    german_candidates = unique_language_rows(rows, "de")

    rng.shuffle(english_candidates)

    english_rows = english_candidates[:n]

    if len(english_rows) < n:
        raise ValueError(
            f"Only {len(english_rows)} usable English rows available."
        )

    english_row_ids = {
        row["row_id"]
        for row in english_rows
    }

    # Exclude every row already used for English.
    german_candidates = [
        row
        for row in german_candidates
        if row["row_id"] not in english_row_ids
    ]

    rng.shuffle(german_candidates)

    german_rows = german_candidates[:n]

    if len(german_rows) < n:
        raise ValueError(
            f"Only {len(german_rows)} disjoint German rows available."
        )

    english = [row["en"] for row in english_rows]
    german = [row["de"] for row in german_rows]

    return english, german, english_row_ids, {
        row["row_id"] for row in german_rows
    }


def select_finnish(rows, n, seed):
    """Select n unique Finnish sentences reproducibly."""

    candidates = unique_language_rows(rows, "fi")

    rng = random.Random(seed + 1)
    rng.shuffle(candidates)

    selected = candidates[:n]

    if len(selected) < n:
        raise ValueError(
            f"Only {len(selected)} usable Finnish rows available."
        )

    return [row["fi"] for row in selected]


def split_sentences(sentences):
    """Apply the repository's 83% / 8.5% / 8.5% split."""

    n = len(sentences)

    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train = sentences[:train_end]
    validation = sentences[train_end:valid_end]
    test = sentences[valid_end:]

    return train, validation, test


def save_language(language, sentences):

    output_dir = OUTPUT_ROOT / language / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, validation, test = split_sentences(sentences)

    files = {
        "full.txt": sentences,
        "train.txt": train,
        "validation.txt": validation,
        "test.txt": test,
    }

    for filename, data in files.items():
        path = output_dir / filename
        path.write_text(
            "\n".join(data) + "\n",
            encoding="utf-8",
        )

    print(
        f"{language}: "
        f"full={len(sentences)}, "
        f"train={len(train)}, "
        f"validation={len(validation)}, "
        f"test={len(test)}"
    )


def main():

    print("Loading German-English OPUS Books...")
    de_en = pd.read_parquet(DE_URL)

    print("Loading English-Finnish OPUS Books...")
    en_fi = pd.read_parquet(FI_URL)

    print("\nCleaning parallel rows...")

    de_en_rows = get_clean_parallel_rows(
        de_en,
        "de",
        "en",
    )

    en_fi_rows = get_clean_parallel_rows(
        en_fi,
        "en",
        "fi",
    )

    print("Usable German-English aligned rows:", len(de_en_rows))
    print("Usable English-Finnish aligned rows:", len(en_fi_rows))

    print("\nSelecting English and German from disjoint rows...")

    english, german, english_ids, german_ids = (
        select_disjoint_english_german(
            de_en_rows,
            TARGET_SENTENCES,
            SEED,
        )
    )

    # Explicit sanity check.
    overlap = english_ids & german_ids

    print(
        "Shared de-en source rows between selected "
        f"English and German: {len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "English/German source-row overlap detected."
        )

    print("Selecting Finnish independently...")

    finnish = select_finnish(
        en_fi_rows,
        TARGET_SENTENCES,
        SEED,
    )

    print("\nSaving matched datasets...")

    save_language("en", english)
    save_language("de", german)
    save_language("fi", finnish)

    print("\nFinal corpus sizes:")
    print("English:", len(english))
    print("German:", len(german))
    print("Finnish:", len(finnish))

    print(
        "\nFinished. "
        f"Data saved under: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()