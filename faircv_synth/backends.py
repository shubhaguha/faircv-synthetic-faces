"""Image generation backends.

``DiffusersBackend`` produces the photorealistic headshots (Stable Diffusion XL
through Hugging Face ``diffusers``). ``PlaceholderBackend`` draws a labelled
silhouette with no model at all; it exists so the pipeline, manifest and resume
logic can be exercised offline and in tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .prompts import PromptSpec


@dataclass(frozen=True)
class ModelPreset:
    model_id: str
    steps: int
    guidance: float
    size: int


PRESETS = {
    # Best realism; needs a GPU with ~10 GB for reasonable speed.
    "sdxl": ModelPreset("stabilityai/stable-diffusion-xl-base-1.0", steps=30, guidance=6.0, size=1024),
    # Distilled, 4 steps; good for quick looks, ignores the negative prompt.
    "sdxl-turbo": ModelPreset("stabilityai/sdxl-turbo", steps=4, guidance=0.0, size=512),
}


class PlaceholderBackend:
    name = "placeholder"

    def __init__(self, size: int = 256):
        self.size = size

    def generate(self, specs: list[PromptSpec]) -> list[Image.Image]:
        return [self._draw(spec) for spec in specs]

    def _draw(self, spec: PromptSpec) -> Image.Image:
        s = self.size
        shade = hashlib.sha256(spec.background.encode()).digest()
        bg = (180 + shade[0] % 60, 180 + shade[1] % 60, 180 + shade[2] % 60)
        img = Image.new("RGB", (s, s), bg)
        d = ImageDraw.Draw(img)
        d.ellipse([s * 0.32, s * 0.14, s * 0.68, s * 0.56], fill=(120, 120, 120))
        d.rounded_rectangle([s * 0.18, s * 0.58, s * 0.82, s * 1.1], radius=s // 6, fill=(70, 70, 90))
        d.text((6, 6), f"PLACEHOLDER seed={spec.seed}", fill=(0, 0, 0))
        return img


class DiffusersBackend:
    name = "diffusers"

    def __init__(self, preset: str = "sdxl", device: str | None = None,
                 model_id: str | None = None, steps: int | None = None,
                 guidance: float | None = None, size: int | None = None, pipe=None):
        cfg = PRESETS[preset]
        self.model_id = model_id or cfg.model_id
        self.steps = steps or cfg.steps
        self.guidance = cfg.guidance if guidance is None else guidance
        self.size = size or cfg.size

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self._torch = torch

        if pipe is None:
            from diffusers import AutoPipelineForText2Image

            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def generate(self, specs: list[PromptSpec]) -> list[Image.Image]:
        # One generator per image keeps each CV's face independent of batch size.
        generators = [self._torch.Generator(device="cpu").manual_seed(s.seed) for s in specs]
        kwargs = dict(
            prompt=[s.prompt for s in specs],
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.size,
            width=self.size,
            generator=generators,
        )
        if self.guidance > 1.0:
            kwargs["negative_prompt"] = [s.negative_prompt for s in specs]
        return self.pipe(**kwargs).images


def make_backend(name: str, **kwargs):
    if name == "placeholder":
        return PlaceholderBackend(size=kwargs.get("size") or 256)
    if name in PRESETS:
        return DiffusersBackend(preset=name, **kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from placeholder, {', '.join(PRESETS)}")
