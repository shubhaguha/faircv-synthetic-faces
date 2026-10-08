"""Turn a FairCV profile into a controlled text-to-image prompt.

The images are meant for bias experiments, so the prompt is a fixed template in
which only a few slots change:

* gender and ethnicity, from the profile (the attributes under study);
* apparent age, from ``prev_experience`` (independent of gender and ethnicity
  in FairCV, so it doesn't add a demographic correlation);
* attire, which is neutral business wear by default and can optionally follow
  the occupation;
* background, lighting and clothing colour. These are drawn from a per-CV
  seeded RNG that never sees the demographic attributes, so they stay balanced
  across groups instead of becoming a shortcut a model could learn.

Every choice is returned in ``PromptSpec`` and written to the manifest, so the
balance of nuisance factors per group can be audited afterwards.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

from .faircv import CVProfile

GENDER_TERMS = {"male": "man", "female": "woman"}

# Neutral, appearance-level descriptors for the three DiveFace groups.
ETHNICITY_TERMS = {
    "east_asian": "East Asian",
    "white": "white European",
    "black": "Black African",
}

AGE_BY_EXPERIENCE = {
    0.0: "in their mid-20s",
    0.2: "in their late 20s",
    0.4: "in their early 30s",
    0.6: "in their late 30s",
    0.8: "in their mid-40s",
    1.0: "in their early 50s",
}

OCCUPATION_ATTIRE = {
    "nurse": "wearing nurse's scrubs",
    "surgeon": "wearing surgical scrubs",
    "physician": "wearing a white doctor's coat",
    "journalist": "wearing a smart casual blazer",
    "photographer": "wearing smart casual clothes",
    "filmmaker": "wearing smart casual clothes",
    "teacher": "wearing smart casual clothes",
    "professor": "wearing a blazer",
    "attorney": "wearing a formal suit",
    "accountant": "wearing a formal suit",
}
NEUTRAL_ATTIRE = "wearing business attire"

BACKGROUNDS = ("plain light grey studio background", "plain off-white studio background",
               "softly blurred office background", "plain light blue studio background")
LIGHTING = ("soft even studio lighting", "soft natural window light", "diffuse daylight")
CLOTHING_COLOURS = ("navy", "charcoal", "black", "white", "light blue", "beige")

TEMPLATE = (
    "professional corporate headshot photograph of {article} {ethnicity} {gender} {age}, "
    "{attire} in {colour} tones, facing the camera, neutral friendly expression, "
    "head and shoulders, {background}, {lighting}, sharp focus, "
    "85mm lens, high detail, realistic skin texture"
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, 3d render, anime, deformed, "
    "distorted face, extra limbs, text, watermark, logo, multiple people, "
    "sunglasses, hat, blurry, low quality"
)


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    negative_prompt: str
    seed: int
    age: str
    attire: str
    background: str
    lighting: str
    clothing_colour: str

    def as_dict(self) -> dict:
        return asdict(self)


def stable_seed(cv_id: str, global_seed: int = 0) -> int:
    """Seed that depends only on the CV id, so re-runs reproduce the same face."""
    digest = hashlib.sha256(f"{global_seed}:{cv_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _age_for(experience: float) -> str:
    nearest = min(AGE_BY_EXPERIENCE, key=lambda k: abs(k - experience))
    return AGE_BY_EXPERIENCE[nearest]


def build_prompt(profile: CVProfile, *, global_seed: int = 0,
                 occupation_attire: bool = False) -> PromptSpec:
    seed = stable_seed(profile.cv_id, global_seed)
    # Nuisance RNG is keyed on the seed only, never on the demographic attributes.
    rng = random.Random(seed)
    background = rng.choice(BACKGROUNDS)
    lighting = rng.choice(LIGHTING)
    colour = rng.choice(CLOTHING_COLOURS)

    age = _age_for(profile.prev_experience)
    attire = OCCUPATION_ATTIRE[profile.occupation] if occupation_attire else NEUTRAL_ATTIRE
    ethnicity = ETHNICITY_TERMS[profile.ethnicity]
    prompt = TEMPLATE.format(
        article="an" if ethnicity[0] in "AEIOU" else "a",
        ethnicity=ethnicity,
        gender=GENDER_TERMS[profile.gender],
        age=age,
        attire=attire,
        colour=colour,
        background=background,
        lighting=lighting,
    )
    return PromptSpec(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, seed=seed, age=age,
                      attire=attire, background=background, lighting=lighting,
                      clothing_colour=colour)
