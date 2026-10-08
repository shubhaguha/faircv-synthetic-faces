# FairCV synthetic headshots (prototype)

Generates one photorealistic, synthetic face image for each resume in
[FairCVdb](https://github.com/BiDAlab/FairCVtest) (24,000 profiles: 19,200 train,
4,800 test). The image matches the profile's gender and ethnicity. FairCV's own
faces come from DiveFace (real Flickr photos); these synthetic headshots can
replace them where real faces are a problem, or be used as a controlled
second image modality.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhaguha/faircv-synthetic-faces/blob/main/notebooks/preview_colab.ipynb)
— preview run on a free T4: generate, CLIP-check, contact sheet.

## How it works

1. **Load** `FairCVdb.npy` with a restricted unpickler that only rebuilds numpy
   arrays, so nothing in the file can run code (`faircv.py`). Ethnicity codes
   are checked against the DiveFace folder prefix of each profile's original
   image (`HA4K`, `MB4K`, ... : H/M = man/woman, A/B/N = East Asian/white/Black).
   On the released file this gives `0 → East Asian, 1 → white, 2 → Black`.
2. **Prompt** each profile with a fixed template (`prompts.py`). Only these slots
   change:
   * gender and ethnicity, from the profile;
   * apparent age, from `prev_experience` (independent of demographics in FairCV);
   * attire: neutral business attire by default, or occupation-specific with
     `--occupation-attire`;
   * background, lighting and clothing colour, drawn from a per-CV seeded RNG
     that never sees gender or ethnicity.
3. **Generate** with Stable Diffusion XL through `diffusers` (`backends.py`). The
   seed comes from a hash of the CV id, so every face can be reproduced.
4. **Record** everything in `manifest.csv`: CV id, attributes, image path, prompt,
   seed and nuisance factors.
5. **Verify** (`verify.py`): zero-shot CLIP (`openai/clip-vit-large-patch14`)
   scores each image against prompt-ensembled text classes for gender and for
   ethnicity. Writes `fidelity.csv` with the predicted class, the probability of
   the labelled class and `passes`, and prints pass rates per group.
6. **Audit** (`audit.py`): checks that nuisance factors are balanced across the
   six gender × ethnicity groups.
7. **Contact sheet** (`contact_sheet.py`): one row per group; CLIP failures are
   outlined in red.

## Usage

```bash
pip install -e ".[gen,test]"
# FairCVdb.npy is stored with Git LFS:
curl -L -o data/FairCVdb.npy \
  https://media.githubusercontent.com/media/BiDAlab/FairCVtest/master/data/FairCVdb.npy

# Quick look: 10 CVs per gender x ethnicity group with the fast 4-step model
python -m faircv_synth.generate --faircv data/FairCVdb.npy --out out/faces_preview \
    --backend sdxl-turbo --per-group 10

# Full dataset with SDXL base. Use one GPU per shard and give each shard its own --out.
python -m faircv_synth.generate --faircv data/FairCVdb.npy --out out/faces_s0 \
    --backend sdxl --shard 0/4 --batch-size 8

# CLIP attribute-fidelity check, nuisance balance, and a visual grid
python -m faircv_synth.verify out/faces_preview/manifest.csv
python -m faircv_synth.audit out/faces_preview/manifest.csv
python -m faircv_synth.contact_sheet out/faces_preview/manifest.csv

# Offline dry run with no model download (draws labelled silhouettes)
python -m faircv_synth.generate --faircv data/FairCVdb.npy --out out/dry --backend placeholder --limit 50
```

Runs can be resumed: images that already exist are skipped unless you pass
`--overwrite`.

| Backend | Model | Steps | Size | Notes |
|---|---|---|---|---|
| `sdxl` (default) | `stabilityai/stable-diffusion-xl-base-1.0` | 30 | 1024 | Most realistic; uses the negative prompt |
| `sdxl-turbo` | `stabilityai/sdxl-turbo` | 4 | 512 | Fast previews; no negative prompt |
| `placeholder` | none | – | 256 | For tests and dry runs |

To use any other `diffusers` text-to-image checkpoint, such as a photorealism
fine-tune, construct `DiffusersBackend(model_id=...)` in Python.

Rough cost: about 24k images × ~4 s/image with SDXL on an A100 ≈ 27 GPU-hours.
`sdxl-turbo` is roughly 10× faster.

## Tests

```bash
python -m pytest tests
```

The tests build a small FairCV-shaped fixture. They cover decoding, the code
mapping check, refusal of malicious pickles, prompt determinism, the property
that flipping gender changes only the gender word, balanced sampling, resuming,
the CLI, the arguments passed to the diffusers pipeline (using a fake
pipeline), the CLIP scoring logic (using a fake scorer), the `ClipScorer`
wrapper on a tiny randomly initialised CLIP, and the contact sheet.

## Status and caveats

* **Not yet run against real model weights.** The development container
  couldn't reach Hugging Face, so SDXL and CLIP are tested only with fakes and a
  tiny random CLIP. The rest of the pipeline was run on the real FairCVdb with
  the placeholder backend. Run the Colab notebook to get the first real preview.
* **CLIP is a filter, not ground truth.** CLIP has its own demographic biases,
  and the three ethnicity classes are coarse. Use `fidelity.csv` to catch obvious
  failures (wrong gender, ignored ethnicity), and look at failures by eye. If
  pass rates differ a lot between groups, that is itself a finding about the
  generator, the verifier, or both. Failed images are flagged but not yet
  regenerated automatically.
* **Generator bias is part of the data.** SDXL has its own stereotypes
  (expressions, attractiveness, skin tones, attire). For fairness studies,
  report which model and prompt produced the images. The manifest stores both.
* **Nuisance balance is statistical, not exact.** Background, lighting and
  colour are i.i.d. per CV. At 4,000 profiles per group the gaps are small, but
  in small previews they can reach about 0.2 (see `audit.py`).
* Ethnicity is described with three coarse labels inherited from DiveFace.
  These are a property of the benchmark, not a claim about how people should be
  categorised.
* FairCVdb has its own license terms and citation requests (see its README).
  The generated faces are synthetic and don't depict the DiveFace identities.
