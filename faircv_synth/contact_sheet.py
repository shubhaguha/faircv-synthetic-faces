"""Tile generated images into one grid: one row per gender x ethnicity group.

    python -m faircv_synth.contact_sheet out/faces_preview/manifest.csv --per-row 10

If ``fidelity.csv`` sits next to the manifest, images that failed the CLIP
check get a red border.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

LABEL_W = 170


def contact_sheet(manifest: pd.DataFrame, root: str | Path, *, per_row: int = 10,
                  thumb: int = 128, failed: set[str] = frozenset()) -> Image.Image:
    root = Path(root)
    groups = sorted(manifest.groupby(["gender", "ethnicity"]))
    sheet = Image.new("RGB", (LABEL_W + per_row * thumb, len(groups) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for r, ((gender, ethnicity), rows) in enumerate(groups):
        draw.text((8, r * thumb + thumb // 2 - 6), f"{gender} / {ethnicity}", fill="black")
        for c, row in enumerate(rows.head(per_row).itertuples()):
            img = Image.open(root / row.image_path).convert("RGB").resize((thumb, thumb))
            x, y = LABEL_W + c * thumb, r * thumb
            sheet.paste(img, (x, y))
            if row.cv_id in failed:
                draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline="red", width=4)
    return sheet


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--per-row", type=int, default=10)
    ap.add_argument("--thumb", type=int, default=128)
    ap.add_argument("--out", help="output PNG (default: contact_sheet.png next to the manifest)")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    fidelity_path = root / "fidelity.csv"
    failed = set()
    if fidelity_path.exists():
        fid = pd.read_csv(fidelity_path)
        failed = set(fid.cv_id[~fid.passes])
    sheet = contact_sheet(pd.read_csv(manifest_path), root, per_row=args.per_row,
                          thumb=args.thumb, failed=failed)
    out = Path(args.out) if args.out else root / "contact_sheet.png"
    sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
