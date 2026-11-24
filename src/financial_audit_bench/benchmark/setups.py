"""Shared contract: a *setup* is one harness + model combination to benchmark.

Workstream 1 (harness integration) adds harnesses in ``harnesses.py``;
Workstream 2 (parallel runs) loads many setups from a file and runs each one;
Workstream 3 (comparison) reads results back, keyed by ``Setup.name``.

Setup files are TOML, one ``[[setup]]`` table per setup::

    [[setup]]
    name = "fab-opus"                  # optional; derived from harness + model
    harness = "fab"                    # see harnesses.HARNESSES
    model = "claude-opus-5-5"
    reasoning_effort = "medium"

    [[setup]]
    harness = "terminus-2"
    model = "openrouter/deepseek/deepseek-v4.1-flash"
    harness_kwargs = { some_option = 1 }
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from financial_audit_bench.defaults import DEFAULT_MAX_MODEL_CALLS

DEFAULT_HARNESS = "fab"
_SETUP_KEYS = {"name", "harness", "model", "reasoning_effort", "temperature",
               "max_model_calls", "harness_kwargs"}


def slug(text: str) -> str:
    """Filesystem- and job-name-safe form of ``text``."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")
    if not value:
        raise ValueError(f"Name must contain a letter or digit: {text!r}")
    return value


@dataclass(frozen=True)
class Setup:
    """One harness + model combination. ``name`` identifies it across runs."""

    model: str
    harness: str = DEFAULT_HARNESS
    reasoning_effort: str = "medium"
    temperature: float | None = None
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    harness_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Setup needs a model")
        object.__setattr__(self, "name", slug(self.name or f"{self.harness}-{self.model}"))

    @property
    def model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"reasoning_effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Setup:
        unknown = set(data) - _SETUP_KEYS
        if unknown:
            raise ValueError(f"Unknown setup keys: {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "harness": self.harness, "model": self.model,
                "reasoning_effort": self.reasoning_effort, "temperature": self.temperature,
                "max_model_calls": self.max_model_calls,
                "harness_kwargs": dict(self.harness_kwargs)}


def load_setups(path: Path) -> list[Setup]:
    """Read and validate a TOML setup file."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {"setup"}
    if unknown:
        raise ValueError(f"Unknown top-level keys in {path}: {', '.join(sorted(unknown))}")
    setups = [Setup.from_dict(item) for item in data.get("setup", [])]
    if not setups:
        raise ValueError(f"No [[setup]] entries in {path}")
    names = [setup.name for setup in setups]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate setup names: {', '.join(duplicates)}")
    from financial_audit_bench.benchmark.harnesses import require_harness
    for setup in setups:
        require_harness(setup.harness)
    return setups
