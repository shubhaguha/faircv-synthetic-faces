import pickle

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from faircv_synth.audit import max_gap
from faircv_synth.backends import PlaceholderBackend
from faircv_synth.faircv import load_faircv, read_faircv_dict
from faircv_synth.generate import generate_images, main, select_profiles
from faircv_synth.prompts import build_prompt

FOLDER = {(0, 0): "HA4K", (0, 1): "MA4K", (1, 0): "HB4K", (1, 1): "MB4K", (2, 0): "HN4K", (2, 1): "MN4K"}


def _split(rng, n):
    profiles = np.zeros((n, 51))
    profiles[:, 0] = np.arange(n) % 3
    profiles[:, 1] = (np.arange(n) // 3) % 2
    profiles[:, 2] = rng.integers(0, 10, n)
    profiles[:, 3] = rng.choice([0.25, 0.5, 0.75, 1.0], n)
    profiles[:, 5] = rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0], n)
    images = np.array([f"{FOLDER[(int(e), int(g))]}\\id_{i}\\img.jpg"
                       for i, (e, g) in enumerate(profiles[:, :2])])
    bios = np.array([[f"bio {i}", f"blind bio {i}"] for i in range(n)])
    names = np.array([f"Name{i}" for i in range(n)])
    return profiles, images, bios, names


@pytest.fixture
def faircv_path(tmp_path):
    rng = np.random.default_rng(0)
    data = {}
    for key, n in (("Train", 60), ("Test", 24)):
        p, img, bios, names = _split(rng, n)
        data |= {f"Profiles {key}": p, f"Image List {key}": img,
                 f"Bios {key}": bios, f"Names {key}": names}
    path = tmp_path / "FairCVdb.npy"
    np.save(path, data, allow_pickle=True)
    return path


def test_load_decodes_profiles(faircv_path):
    profiles = load_faircv(faircv_path)
    assert len(profiles) == 84
    p = profiles[0]
    assert (p.cv_id, p.gender, p.ethnicity, p.bio) == ("train_00000", "male", "east_asian", "bio 0")
    assert {q.ethnicity for q in profiles} == {"east_asian", "white", "black"}


def test_load_rejects_code_mismatch(tmp_path, faircv_path):
    data = read_faircv_dict(faircv_path)
    data["Image List Train"][0] = "MN4K\\x\\y.jpg"  # row 0 is a male, ethnicity 0
    bad = tmp_path / "bad.npy"
    np.save(bad, data, allow_pickle=True)
    with pytest.raises(ValueError, match="mapping mismatch"):
        load_faircv(bad)


def test_reader_refuses_arbitrary_pickle_globals(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.npy"
    np.save(path, np.array(Evil(), dtype=object), allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError):
        read_faircv_dict(path)


def test_prompt_is_deterministic_and_only_demographics_differ(faircv_path):
    p = load_faircv(faircv_path)[0]
    a, b = build_prompt(p), build_prompt(p)
    assert a == b
    assert "of an East Asian man" in a.prompt and "business attire" in a.prompt
    assert build_prompt(p, global_seed=1).seed != a.seed

    # Flipping only gender changes only the gender word.
    from dataclasses import replace
    flipped = build_prompt(replace(p, gender="female"))
    assert flipped.prompt == a.prompt.replace(" man ", " woman ")


def test_occupation_attire_option(faircv_path):
    p = next(q for q in load_faircv(faircv_path) if q.occupation == "surgeon")
    assert "surgical scrubs" in build_prompt(p, occupation_attire=True).prompt


def test_select_per_group_is_balanced(faircv_path):
    picked = select_profiles(load_faircv(faircv_path), per_group=3)
    counts = pd.Series([(p.gender, p.ethnicity) for p in picked]).value_counts()
    assert len(counts) == 6 and (counts == 3).all()


def test_generate_writes_images_and_manifest_and_resumes(tmp_path, faircv_path):
    profiles = load_faircv(faircv_path, splits=("test",))
    out = tmp_path / "out"
    manifest = generate_images(profiles, PlaceholderBackend(size=64), out, batch_size=5)
    assert len(manifest) == 24
    first = out / manifest.image_path[0]
    assert Image.open(first).size == (64, 64)

    class Exploding:
        name = "placeholder"

        def generate(self, specs):
            raise AssertionError("should not regenerate existing images")

    again = generate_images(profiles, Exploding(), out)
    assert again.cv_id.tolist() == manifest.cv_id.tolist()


def test_cli_end_to_end(tmp_path, faircv_path):
    out = tmp_path / "cli"
    main(["--faircv", str(faircv_path), "--out", str(out), "--backend", "placeholder",
          "--split", "train", "--per-group", "2", "--size", "32"])
    manifest = pd.read_csv(out / "manifest.csv")
    assert len(manifest) == 12
    assert all((out / p).exists() for p in manifest.image_path)
    assert 0.0 <= max_gap(manifest, "background") <= 1.0


def test_diffusers_backend_passes_seeds_and_settings():
    torch = pytest.importorskip("torch")
    from faircv_synth.backends import DiffusersBackend

    calls = {}

    class FakePipe:
        def __call__(self, **kw):
            calls.update(kw)
            return type("R", (), {"images": [Image.new("RGB", (8, 8))] * len(kw["prompt"])})()

    from faircv_synth.prompts import PromptSpec
    specs = [PromptSpec("p", "n", s, "", "", "", "", "") for s in (1, 2)]
    backend = DiffusersBackend(preset="sdxl", device="cpu", pipe=FakePipe())
    assert len(backend.generate(specs)) == 2
    assert calls["negative_prompt"] == ["n", "n"] and calls["num_inference_steps"] == 30
    assert [g.initial_seed() for g in calls["generator"]] == [1, 2]


def test_contact_sheet_has_one_row_per_group(tmp_path, faircv_path):
    from faircv_synth.contact_sheet import contact_sheet

    profiles = select_profiles(load_faircv(faircv_path), per_group=2)
    manifest = generate_images(profiles, PlaceholderBackend(size=32), tmp_path)
    sheet = contact_sheet(manifest, tmp_path, per_row=2, thumb=16, failed={manifest.cv_id[0]})
    assert sheet.size == (170 + 2 * 16, 6 * 16)
