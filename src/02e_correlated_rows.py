#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_CSV = PROJECT_ROOT / "DATA" / "DHS" / "women_all_answers_gps.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT = OUTPUT_DIR / "02e_correlations.txt"

IPV_INDICATORS = ["d111", "d104", "d106", "d108"]

EXCLUDED_VARIABLES = [
    "d111", "d104", "d106", "d108",
    "d107",
    "v001", "latnum", "longnum", "LATNUM", "LONGNUM",
    "caseid", "hhid",
    "v000", "v005", "v006", "v007", "v008", "v021", "v022", "v023",
]

MIN_NON_MISSING = 5_000  # only consider columns with >10k raw non-missing responses


# =============================================================================
# HELPERS
# =============================================================================

def create_extreme_disadvantage_indicator(df: pd.DataFrame, ipv_vars: list[str]) -> pd.Series:
    ipv_count = pd.Series(0, index=df.index, dtype=int)
    for v in ipv_vars:
        if v in df.columns:
            # treat literal "1" / "1.0" as yes too
            yes = df[v].astype(str).str.strip().isin({"1", "1.0"})
            ipv_count += yes.astype(int)
    return (ipv_count >= 2).astype(int)


def safe_pearson(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    a = a[mask].astype(float)
    b = b[mask].astype(float)
    if a.size < 3:
        return np.nan
    if a.nunique(dropna=True) < 2:
        return np.nan
    if b.nunique(dropna=True) < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cramers_v_binary_outcome(x_cat: pd.Series, y_bin: pd.Series) -> float:
    """
    Cramér's V for association between categorical X and binary Y.
    Returns value in [0,1]. (No sign.)
    """
    mask = x_cat.notna() & y_bin.notna()
    x = x_cat[mask].astype(str)
    y = y_bin[mask].astype(int)

    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    ct = pd.crosstab(y, x)  # 2 x K
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan

    observed = ct.to_numpy()
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    # chi-square
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    # For 2xK, V = sqrt(chi2 / (n * (min(r-1, c-1))))
    r, c = observed.shape
    denom = n * min(r - 1, c - 1)
    if denom <= 0:
        return np.nan
    v = np.sqrt(chi2 / denom)
    return float(v)


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """
    More forgiving numeric parsing:
    - strips whitespace
    - turns commas into dots if needed
    - parses to float where possible
    """
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    # basic decimal comma handling (only if it looks like a number)
    s2 = s2.str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at: {INPUT_CSV}")
        sys.exit(1)

    # Read everything as string to avoid mixed-type issues
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    missing_ipv = [v for v in IPV_INDICATORS if v not in df.columns]
    if missing_ipv:
        print(f"ERROR: missing IPV indicators: {missing_ipv}")
        sys.exit(1)

    # outcome
    y = create_extreme_disadvantage_indicator(df, IPV_INDICATORS)

    # valid-outcome filter: answered >=2 IPV questions (non-missing raw)
    ipv_answered = df[IPV_INDICATORS].notna().sum(axis=1)
    valid = ipv_answered >= 2
    df = df.loc[valid].copy()
    y = y.loc[valid].copy()

    n = len(df)
    prev = 100.0 * float(y.mean()) if n else 0.0
    print(f"Valid samples: {n:,} | outcome prevalence: {prev:.2f}%")

    excluded = set(v.lower() for v in EXCLUDED_VARIABLES)
    candidates = [c for c in df.columns if c.lower() not in excluded]

    results = []
    skipped = []

    for col in candidates:
        s_raw = df[col]

        raw_non_missing = int(s_raw.notna().sum())
        if raw_non_missing <= MIN_NON_MISSING:
            skipped.append((col, f"<= {MIN_NON_MISSING} raw non-missing ({raw_non_missing})"))
            continue

        # decide if numeric enough
        s_num = to_numeric_clean(s_raw)
        numeric_non_missing = int(s_num.notna().sum())

        # Also look at categorical cardinality
        s_cat = s_raw.dropna().astype(str).str.strip()
        nunique_cat = int(s_cat.nunique())

        # Binary categorical -> phi (signed depends on arbitrary mapping; still useful, but note it)
        if nunique_cat == 2:
            cats = sorted(s_cat.unique())
            mapping = {cats[0]: 0, cats[1]: 1}
            x_bin = s_raw.astype(str).str.strip().map(mapping)
            corr = safe_pearson(x_bin, y)
            if np.isnan(corr):
                skipped.append((col, "binary but undefined correlation"))
                continue
            metric = abs(float(corr))
            results.append({
                "column": col,
                "score": metric,
                "value": float(corr),
                "method": f"phi (binary; mapped '{cats[0]}'->0, '{cats[1]}'->1)",
                "type": "binary_categorical",
                "raw_non_missing": raw_non_missing,
                "numeric_non_missing": numeric_non_missing,
                "n_used": int((x_bin.notna() & y.notna()).sum()),
            })
            continue

        # Numeric (has enough numeric values AND enough variation)
        if numeric_non_missing > MIN_NON_MISSING and s_num.nunique(dropna=True) >= 3:
            corr = safe_pearson(s_num, y.astype(float))  # point-biserial == Pearson(x, y)
            if np.isnan(corr):
                skipped.append((col, "numeric but undefined correlation"))
                continue
            results.append({
                "column": col,
                "score": abs(float(corr)),
                "value": float(corr),
                "method": "point-biserial (numeric vs binary outcome)",
                "type": "numeric",
                "raw_non_missing": raw_non_missing,
                "numeric_non_missing": numeric_non_missing,
                "n_used": int((s_num.notna() & y.notna()).sum()),
            })
            continue

        # Non-binary categorical -> Cramér's V (0..1, unsigned)
        if nunique_cat >= 3:
            v = cramers_v_binary_outcome(
                s_raw.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}),
                y
            )
            if np.isnan(v):
                skipped.append((col, "categorical but undefined Cramér's V"))
                continue
            results.append({
                "column": col,
                "score": float(v),
                "value": float(v),
                "method": "Cramér's V (categorical vs binary outcome; unsigned)",
                "type": "categorical",
                "raw_non_missing": raw_non_missing,
                "numeric_non_missing": numeric_non_missing,
                "n_used": int((s_raw.notna() & y.notna()).sum()),
            })
            continue

        skipped.append((col, f"unhandled: nunique={nunique_cat}, numeric_non_missing={numeric_non_missing}"))

    if not results:
        print("\nERROR: still no usable columns after applying >10k rule.")
        print("This would usually mean your valid-outcome subset is small or most columns are empty there.")
        sys.exit(1)

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("2e - ASSOCIATION WITH EXTREME DISADVANTAGE (ranked)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Outcome: extremely_disadvantaged = 1 if 2+ IPV 'Yes' among: ")
        f.write(", ".join(IPV_INDICATORS) + "\n")
        f.write(f"Valid samples (answered >=2 IPV questions): {n:,}\n")
        f.write(f"Outcome prevalence: {prev:.2f}%\n")
        f.write(f"Inclusion rule: column must have > {MIN_NON_MISSING:,} raw (non-NaN) responses\n\n")

        f.write("Methods:\n")
        f.write("- binary categorical (2 categories): phi via Pearson on 0/1 mapping\n")
        f.write("- numeric: point-biserial (Pearson(x, y))\n")
        f.write("- non-binary categorical: Cramér's V (unsigned)\n\n")

        f.write("RANKING (highest score first):\n")
        f.write("-" * 70 + "\n")
        for i, row in enumerate(res_df.itertuples(index=False), start=1):
            f.write(
                f"{i:4d}. {row.column:35s}  "
                f"score={row.score:.6f}  value={row.value:+.6f}  "
                f"type={row.type:18s}  n_used={row.n_used:6d}  "
                f"raw_non_missing={row.raw_non_missing:6d}  "
                f"numeric_non_missing={row.numeric_non_missing:6d}  "
                f"method={row.method}\n"
            )

        if skipped:
            f.write("\nSKIPPED COLUMNS (reason):\n")
            f.write("-" * 70 + "\n")
            for col, reason in skipped:
                f.write(f"- {col}: {reason}\n")

    print(f"\n✓ Saved: {OUTPUT_TXT}")
    print("Top 10:")
    for i, row in enumerate(res_df.head(10).itertuples(index=False), start=1):
        print(f"{i:2d}. {row.column:35s} score={row.score:.4f} ({row.type})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
