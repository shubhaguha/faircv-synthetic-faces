"""Synthetic face images for FairCV resumes."""

from .faircv import CVProfile, load_faircv
from .prompts import PromptSpec, build_prompt

__all__ = ["CVProfile", "PromptSpec", "build_prompt", "load_faircv"]
