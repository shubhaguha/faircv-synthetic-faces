"""Loading and decoding of FairCVdb (https://github.com/BiDAlab/FairCVtest).

FairCVdb.npy is a pickled dict of numpy arrays. Each row of ``Profiles <Split>``
has 51 columns:

    0      ethnicity          {0, 1, 2}
    1      gender             0 = male, 1 = female
    2      occupation         [0-9], see OCCUPATIONS
    3      suitability        {0.25, 0.5, 0.75, 1}
    4      educ_attainment    {0.2 .. 1}
    5      prev_experience    {0 .. 1}
    6      recommendation     {0, 1}
    7      availability       {0.2 .. 1}
    8:11   language_prof      3 x {0 .. 1}
    11:31  face_embedding     20-d, unit norm
    31:51  blind_embedding    20-d, unit norm

The ethnicity code is not named in the FairCV README, but every profile's
``Image List`` entry points at a DiveFace folder whose prefix encodes both
attributes (Spanish: H/M = hombre/mujer, A/B/N = asiático/blanco/negro).
On the released file the mapping is 0 -> A, 1 -> B, 2 -> N for every row, and
``load_faircv`` re-checks it on load rather than trusting a hard-coded table.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OCCUPATIONS = (
    "nurse", "surgeon", "physician", "journalist", "photographer",
    "filmmaker", "teacher", "professor", "attorney", "accountant",
)
GENDERS = ("male", "female")
# DiveFace folder letter -> group key used by the prompt builder.
DIVEFACE_GROUPS = {"A": "east_asian", "B": "white", "N": "black"}
ETHNICITY_CODES = {0: "east_asian", 1: "white", 2: "black"}

SPLITS = ("train", "test")

# Only the globals needed to rebuild plain numpy arrays. FairCVdb.npy is a
# pickle, and np.load(allow_pickle=True) would run arbitrary code from it.
_SAFE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from FairCVdb")


def read_faircv_dict(path: str | Path) -> dict:
    """Read FairCVdb.npy without executing arbitrary pickle code."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(f)
        else:
            np.lib.format.read_array_header_2_0(f)
        obj = _NumpyOnlyUnpickler(f).load()
    return obj.item() if isinstance(obj, np.ndarray) else obj


@dataclass(frozen=True)
class CVProfile:
    split: str
    index: int
    name: str
    gender: str
    ethnicity: str
    occupation: str
    suitability: float
    educ_attainment: float
    prev_experience: float
    recommendation: int
    availability: float
    language_prof: tuple[float, float, float]
    source_image: str
    bio: str = field(repr=False, default="")

    @property
    def cv_id(self) -> str:
        return f"{self.split}_{self.index:05d}"


def _check_codes(profiles: np.ndarray, image_list: np.ndarray) -> None:
    """Fail loudly if the code->group table disagrees with the DiveFace folders."""
    for row, img in zip(profiles, image_list):
        prefix = str(img).replace("\\", "/").split("/", 1)[0]
        if len(prefix) < 2:
            continue
        g_letter, e_letter = prefix[0], prefix[1]
        expected_gender = {"H": "male", "M": "female"}.get(g_letter)
        expected_eth = DIVEFACE_GROUPS.get(e_letter)
        if expected_gender != GENDERS[int(row[1])] or expected_eth != ETHNICITY_CODES[int(row[0])]:
            raise ValueError(
                f"FairCV code mapping mismatch for {img!r}: "
                f"gender={int(row[1])}, ethnicity={int(row[0])}"
            )


def decode_profiles(data: dict, split: str) -> list[CVProfile]:
    key = split.capitalize()
    profiles = np.asarray(data[f"Profiles {key}"])
    names = data.get(f"Names {key}", [""] * len(profiles))
    images = data.get(f"Image List {key}", [""] * len(profiles))
    bios = data.get(f"Bios {key}")
    _check_codes(profiles, images)

    out = []
    for i, row in enumerate(profiles):
        out.append(CVProfile(
            split=split,
            index=i,
            name=str(names[i]),
            gender=GENDERS[int(row[1])],
            ethnicity=ETHNICITY_CODES[int(row[0])],
            occupation=OCCUPATIONS[int(row[2])],
            suitability=float(row[3]),
            educ_attainment=float(row[4]),
            prev_experience=float(row[5]),
            recommendation=int(row[6]),
            availability=float(row[7]),
            language_prof=tuple(float(x) for x in row[8:11]),
            source_image=str(images[i]),
            bio=str(bios[i][0]).strip() if bios is not None else "",
        ))
    return out


def load_faircv(path: str | Path, splits: tuple[str, ...] = SPLITS) -> list[CVProfile]:
    data = read_faircv_dict(path)
    return [p for split in splits for p in decode_profiles(data, split)]
