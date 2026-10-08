"""Generate one synthetic headshot per FairCV profile.

Example:
    python -m faircv_synth.generate --faircv data/FairCVdb.npy --out out/faircv_faces \
        --backend sdxl --split test --per-group 10

Outputs ``<out>/images/<split>_<index>.png`` and ``<out>/manifest.csv`` linking
each CV to its image, prompt, seed and nuisance factors. Existing images are
skipped, so an interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .backends import PRESETS, make_backend
from .faircv import SPLITS, CVProfile, load_faircv
from .prompts import build_prompt


def select_profiles(profiles: list[CVProfile], *, limit: int | None = None,
                    per_group: int | None = None, shard: tuple[int, int] = (0, 1),
                    seed: int = 0) -> list[CVProfile]:
    if per_group is not None:
        groups = defaultdict(list)
        for p in profiles:
            groups[(p.gender, p.ethnicity)].append(p)
        rng = random.Random(seed)
        picked = []
        for key in sorted(groups):
            members = groups[key]
            picked.extend(rng.sample(members, min(per_group, len(members))))
        profiles = sorted(picked, key=lambda p: (p.split, p.index))
    if limit is not None:
        profiles = profiles[:limit]
    k, n = shard
    return profiles[k::n]


def generate_images(profiles: list[CVProfile], backend, out_dir: str | Path, *,
                    batch_size: int = 4, global_seed: int = 0,
                    occupation_attire: bool = False, overwrite: bool = False) -> pd.DataFrame:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, todo = [], []
    for p in profiles:
        spec = build_prompt(p, global_seed=global_seed, occupation_attire=occupation_attire)
        path = img_dir / f"{p.cv_id}.png"
        rows.append({
            "cv_id": p.cv_id, "split": p.split, "index": p.index, "name": p.name,
            "gender": p.gender, "ethnicity": p.ethnicity, "occupation": p.occupation,
            "image_path": str(path.relative_to(out_dir)),
            "backend": backend.name, "model_id": getattr(backend, "model_id", ""),
            **spec.as_dict(),
        })
        if overwrite or not path.exists():
            todo.append((spec, path))

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        images = backend.generate([spec for spec, _ in batch])
        for img, (_, path) in zip(images, batch):
            img.save(path)
        print(f"generated {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        # Keep rows from earlier runs (e.g. other shards or splits).
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old[~old.cv_id.isin(manifest.cv_id)], manifest], ignore_index=True)
    manifest.sort_values("cv_id").to_csv(manifest_path, index=False)
    return manifest


def _shard(value: str) -> tuple[int, int]:
    k, n = (int(x) for x in value.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError("shard must be k/n with 0 <= k < n")
    return k, n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faircv", required=True, help="path to FairCVdb.npy")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--backend", default="sdxl", choices=["placeholder", *PRESETS])
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--limit", type=int, help="only the first N selected profiles")
    ap.add_argument("--per-group", type=int,
                    help="sample N profiles from each gender x ethnicity group")
    ap.add_argument("--shard", type=_shard, default=(0, 1),
                    help="k/n: process every n-th profile starting at k (for parallel GPUs)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="global seed mixed into every per-CV seed")
    ap.add_argument("--occupation-attire", action="store_true",
                    help="dress subjects for their occupation instead of neutral business attire")
    ap.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--steps", type=int, help="override the preset's inference steps")
    ap.add_argument("--size", type=int, help="override the preset's image size")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    splits = SPLITS if args.split == "all" else (args.split,)
    profiles = select_profiles(load_faircv(args.faircv, splits), limit=args.limit,
                               per_group=args.per_group, shard=args.shard, seed=args.seed)
    print(f"{len(profiles)} profiles selected")

    if args.backend == "placeholder":
        backend = make_backend("placeholder", size=args.size)
    else:
        backend = make_backend(args.backend, device=args.device, steps=args.steps, size=args.size)
    generate_images(profiles, backend, args.out, batch_size=args.batch_size,
                    global_seed=args.seed, occupation_attire=args.occupation_attire,
                    overwrite=args.overwrite)
    print(f"done; manifest at {Path(args.out) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
