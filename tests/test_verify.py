import numpy as np
import pandas as pd
import pytest
from PIL import Image

from faircv_synth.verify import classify, score_manifest, summarise

KEYWORDS = ("a man", "a woman", "East Asian", "white person", "Black person")
GENDER_DIM = {"male": 0, "female": 1}
ETH_DIM = {"east_asian": 2, "white": 3, "black": 4}


class FakeScorer:
    """Text -> one-hot on its keyword; image -> one-hot on the classes painted into its pixels."""

    logit_scale = 100.0

    def text_embeddings(self, texts):
        return np.stack([np.eye(5)[next(i for i, k in enumerate(KEYWORDS) if k in t)] for t in texts])

    def image_embeddings(self, images):
        rows = []
        for img in images:
            g, e, _ = img.getpixel((0, 0))
            v = np.zeros(5)
            v[g] = v[e] = 1.0
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


def _image(path, gender, ethnicity):
    Image.new("RGB", (4, 4), (GENDER_DIM[gender], ETH_DIM[ethnicity], 0)).save(path)


def test_classify_is_a_softmax():
    p = classify(np.array([[1.0, 0.0]]), np.eye(2), logit_scale=10.0)
    assert np.allclose(p.sum(axis=1), 1) and p[0, 0] > 0.99


def test_score_manifest_flags_mismatches(tmp_path):
    (tmp_path / "images").mkdir()
    rows = [
        # cv label               what the "generator" actually drew
        ("a", "male", "white", "male", "white"),
        ("b", "female", "black", "male", "black"),          # wrong gender
        ("c", "female", "east_asian", "female", "white"),   # wrong ethnicity
    ]
    for cv_id, g, e, drawn_g, drawn_e in rows:
        _image(tmp_path / "images" / f"{cv_id}.png", drawn_g, drawn_e)
    manifest = pd.DataFrame([{"cv_id": r[0], "gender": r[1], "ethnicity": r[2],
                              "image_path": f"images/{r[0]}.png"} for r in rows])

    fid = score_manifest(manifest, tmp_path, FakeScorer(), batch_size=2)
    assert fid.passes.tolist() == [True, False, False]
    assert fid.gender_ok.tolist() == [True, False, True]
    assert fid.ethnicity_ok.tolist() == [True, True, False]
    assert fid.pred_gender[1] == "male" and fid.pred_ethnicity[2] == "white"
    assert fid.p_gender[0] > 0.99

    summary = summarise(fid)
    assert summary.loc[("male", "white"), "passes"] == 1.0
    assert summary["n"].sum() == 3


def test_clip_scorer_with_tiny_random_model():
    """Runs the real ClipScorer code paths against a tiny, randomly initialised CLIP."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from faircv_synth.verify import ClipScorer

    cfg = transformers.CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, vocab_size=100, bos_token_id=0, eos_token_id=1),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, image_size=32, patch_size=8),
        projection_dim=16)

    class FakeProcessor:
        def __call__(self, text=None, images=None, **_):
            if text is not None:
                ids = torch.tensor([[0, 2 + len(t) % 90, 1] for t in text])
                return transformers.BatchEncoding({"input_ids": ids})
            return transformers.BatchEncoding({"pixel_values": torch.randn(len(images), 3, 32, 32)})

    scorer = ClipScorer.__new__(ClipScorer)
    scorer._torch, scorer.device = torch, "cpu"
    scorer.model = transformers.CLIPModel(cfg).eval()
    scorer.processor = FakeProcessor()

    t = scorer.text_embeddings(["a photo of a man.", "a photo of a woman."])
    i = scorer.image_embeddings([Image.new("RGB", (32, 32))] * 3)
    assert t.shape == (2, 16) and i.shape == (3, 16)
    assert np.allclose(np.linalg.norm(i, axis=1), 1, atol=1e-5)
