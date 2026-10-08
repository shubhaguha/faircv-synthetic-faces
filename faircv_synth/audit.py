"""Check that nuisance factors in a manifest are balanced across demographic groups.

    python -m faircv_synth.audit out/faircv_faces/manifest.csv

For each nuisance column it prints the share of each value per
gender x ethnicity group, plus the largest gap between groups. A large gap means
the images carry a non-demographic cue that correlates with a group.
"""

from __future__ import annotations

import argparse

import pandas as pd

NUISANCE_COLUMNS = ("age", "attire", "background", "lighting", "clothing_colour")


def balance_table(manifest: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab([manifest.gender, manifest.ethnicity], manifest[column], normalize="index")


def max_gap(manifest: pd.DataFrame, column: str) -> float:
    table = balance_table(manifest, column)
    return float((table.max() - table.min()).max())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    args = ap.parse_args(argv)
    manifest = pd.read_csv(args.manifest)

    print(manifest.groupby(["gender", "ethnicity"]).size().rename("n").to_string(), "\n")
    for col in NUISANCE_COLUMNS:
        print(f"== {col} (max between-group gap: {max_gap(manifest, col):.3f})")
        print(balance_table(manifest, col).round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
