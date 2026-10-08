"""Check that generated faces show the gender and ethnicity in their CV label.

    python -m faircv_synth.verify out/faces_preview/manifest.csv

Each image is scored zero-shot with CLIP against a small set of text prompts per
class. The prompt embeddings for a class are averaged, and the scores are
softmaxed separately over gender and over ethnicity. The script writes
``fidelity.csv`` next to the manifest, with one row per image: the predicted
class, the probability assigned to the labelled class, and ``passes``. It then
prints the pass rate for each gender x ethnicity group.

CLIP is itself a biased perceiver of demographics. Use this as a filter for
obvious generation failures (wrong gender, ignored ethnicity), not as ground
truth, and look at the failures by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MODEL = "openai/clip-vit-large-patch14"

TEMPLATES = (
    "a photo of {}.",
    "a professional headshot of {}.",
    "a close-up portrait photo of {}.",
)
CLASS_PHRASES = {
    "gender": {
        "male": "a man",
        "female": "a woman",
    },
    "ethnicity": {
        "east_asian": "an East Asian person",
        "white": "a white person",
        "black": "a Black person",
    },
}


class ClipScorer:
    """Wraps a Hugging Face CLIP model; returns L2-normalised embeddings."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.logit_scale = float(self.model.logit_scale.exp())
        self.model_id = model_id

    def _norm(self, x):
        # transformers >= 5 returns a ModelOutput whose pooler_output holds the
        # projected embedding; older versions return the tensor directly.
        if not isinstance(x, self._torch.Tensor):
            x = x.pooler_output
        x = x.float().cpu().numpy()
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            return self._norm(self.model.get_text_features(**inputs))

    def image_embeddings(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            return self._norm(self.model.get_image_features(**inputs))


def class_embeddings(scorer, attribute: str) -> tuple[list[str], np.ndarray]:
    """One prompt-ensembled, unit-norm text embedding per class."""
    labels, rows = [], []
    for label, phrase in CLASS_PHRASES[attribute].items():
        emb = scorer.text_embeddings([t.format(phrase) for t in TEMPLATES]).mean(axis=0)
        labels.append(label)
        rows.append(emb / np.linalg.norm(emb))
    return labels, np.stack(rows)


def classify(image_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float) -> np.ndarray:
    """Softmax class probabilities, shape (n_images, n_classes)."""
    logits = logit_scale * image_emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def score_manifest(manifest: pd.DataFrame, root: str | Path, scorer, *,
                   min_prob: float = 0.5, batch_size: int = 32) -> pd.DataFrame:
    root = Path(root)
    attrs = {a: class_embeddings(scorer, a) for a in CLASS_PHRASES}
    probs = {a: [] for a in attrs}
    for start in range(0, len(manifest), batch_size):
        paths = manifest.image_path.iloc[start:start + batch_size]
        images = [Image.open(root / p).convert("RGB") for p in paths]
        emb = scorer.image_embeddings(images)
        for a, (_, text_emb) in attrs.items():
            probs[a].append(classify(emb, text_emb, scorer.logit_scale))

    out = manifest[["cv_id", "gender", "ethnicity", "image_path"]].copy()
    passes = np.ones(len(out), dtype=bool)
    for a, (labels, _) in attrs.items():
        p = np.concatenate(probs[a]) if probs[a] else np.zeros((0, len(labels)))
        true_idx = out[a].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        out[f"pred_{a}"] = [labels[i] for i in p.argmax(axis=1)]
        out[f"p_{a}"] = p[np.arange(len(out)), true_idx].round(4)
        ok = (out[f"pred_{a}"] == out[a]).to_numpy() & (out[f"p_{a}"].to_numpy() >= min_prob)
        out[f"{a}_ok"] = ok
        passes &= ok
    out["passes"] = passes
    return out


def summarise(fidelity: pd.DataFrame) -> pd.DataFrame:
    return (fidelity.groupby(["gender", "ethnicity"])
            .agg(n=("cv_id", "size"), gender_ok=("gender_ok", "mean"),
                 ethnicity_ok=("ethnicity_ok", "mean"), passes=("passes", "mean"))
            .round(3))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-prob", type=float, default=0.5,
                    help="minimum probability on the labelled class to pass")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    scorer = ClipScorer(args.model, device=args.device)
    fidelity = score_manifest(manifest, manifest_path.parent, scorer,
                              min_prob=args.min_prob, batch_size=args.batch_size)
    out = manifest_path.parent / "fidelity.csv"
    fidelity.to_csv(out, index=False)
    print(summarise(fidelity).to_string())
    print(f"\n{int(fidelity.passes.sum())}/{len(fidelity)} images pass; details in {out}")


if __name__ == "__main__":
    main()
