#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add row-wise indicator columns to the DHS main dataset.

This script reads:
    DATA/DHS/women_all_answers_gps.csv

and adds:
    1) sexual_violence_any
       D108 == 1 OR D124 == 1 OR D125 == 1

    2) sexual_violence_recent
       D105H in {1,2} OR D105I in {1,2} OR D105K in {1,2}

The implementation is intentionally row-wise and configuration-driven so it is
easy to read, debug, and extend with additional indicators later.
"""

from pathlib import Path
import sys
from typing import Any

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_CSV = PROJECT_ROOT / "DATA" / "DHS" / "women_all_answers_gps.csv"
OUTPUT_CSV = INPUT_CSV  # Update dataset in place by default.

# Rules are expressed with the original DHS-style variable names for readability.
# The script resolves these names case-insensitively against CSV column names.
INDICATOR_RULES = {
    "sexual_violence_any": {
        "description": "D108 == 1 OR D124 == 1 OR D125 == 1",
        "type": "equals_any",
        "allowed_values": {"1"},
        "columns": ["D108", "D124", "D125"],
    },
    "sexual_violence_recent": {
        "description": "D105H in {1,2} OR D105I in {1,2} OR D105K in {1,2}",
        "type": "in_set_any",
        "allowed_values": {"1", "2"},
        "columns": ["D105H", "D105I", "D105K"],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_value(value: Any) -> str | None:
    """
    Convert a cell value to a normalized comparable string.

    Why this exists:
    - DHS values can appear as ints/floats/strings (e.g., 1, 1.0, "1").
    - We want robust rule matching with minimal surprise.

    Returns:
        - A canonical string such as "1", "2", "6", ...
        - None for missing/empty values.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Treat whole-number floats like "1.0" as "1" for easier comparisons.
    if text.endswith(".0"):
        integer_part = text[:-2]
        if integer_part.lstrip("-").isdigit():
            text = integer_part

    return text


def build_column_lookup(columns: list[str]) -> dict[str, str]:
    """
    Build a case-insensitive map from uppercase name -> real column name.

    Example:
        {"D108": "d108", "D124": "d124", ...}
    """
    return {col.upper(): col for col in columns}


def validate_required_columns(
    column_lookup: dict[str, str],
    indicator_rules: dict[str, dict[str, Any]],
) -> None:
    """
    Ensure all source columns referenced in indicator rules exist in the dataset.
    """
    missing_columns: list[str] = []

    for indicator_name, rule in indicator_rules.items():
        for col in rule["columns"]:
            if col.upper() not in column_lookup:
                missing_columns.append(f"{indicator_name}: {col}")

    if missing_columns:
        print("ERROR: Missing required source columns for indicator creation:")
        for item in missing_columns:
            print(f"  - {item}")
        sys.exit(1)


def row_matches_any_rule(
    row: pd.Series,
    source_columns: list[str],
    allowed_values: set[str],
) -> int:
    """
    Return 1 if any source column matches allowed_values, otherwise 0.

    This is intentionally explicit and row-wise for readability and easy edits.
    """
    for col in source_columns:
        normalized = normalize_value(row[col])
        if normalized in allowed_values:
            return 1
    return 0


def add_indicator_column(
    df: pd.DataFrame,
    indicator_name: str,
    rule: dict[str, Any],
    column_lookup: dict[str, str],
) -> None:
    """
    Add one indicator column to the DataFrame in place.
    """
    source_columns = [column_lookup[col.upper()] for col in rule["columns"]]
    allowed_values = rule["allowed_values"]

    # Row-wise on purpose per project request: very easy to read/modify.
    df[indicator_name] = df.apply(
        lambda row: row_matches_any_rule(
            row=row,
            source_columns=source_columns,
            allowed_values=allowed_values,
        ),
        axis=1,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 72)
    print("ADD ROWS OF INTEREST - DHS MAIN DATASET")
    print("=" * 72)

    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at: {INPUT_CSV}")
        sys.exit(1)

    print(f"\nReading input dataset:\n  {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

    column_lookup = build_column_lookup(df.columns.tolist())
    validate_required_columns(column_lookup=column_lookup, indicator_rules=INDICATOR_RULES)

    print("\nCreating indicator columns...")
    for indicator_name, rule in INDICATOR_RULES.items():
        print(f"  - {indicator_name}: {rule['description']}")
        add_indicator_column(
            df=df,
            indicator_name=indicator_name,
            rule=rule,
            column_lookup=column_lookup,
        )

        yes_count = int((df[indicator_name] == 1).sum())
        print(f"    -> yes=1 in {yes_count:,} rows ({yes_count / max(len(df), 1):.2%})")

    print(f"\nWriting updated dataset:\n  {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
