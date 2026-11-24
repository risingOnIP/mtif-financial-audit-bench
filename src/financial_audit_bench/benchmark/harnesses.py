"""Shared contract: how a harness name becomes a Harbor agent config.

A harness builder takes a ``Setup`` and returns the dict that goes into
Harbor's ``JobConfig.agents`` list (``{"name": ...}`` for a built-in Harbor
agent, or ``{"import_path": ...}`` for one of ours). Everything else about a
run (tasks, grading, job layout) is shared, so harnesses compare fairly.

Workstream 1 owns this file: implement ``terminus_2_agent`` (and any task
image or instruction changes it needs) without changing the builder signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from financial_audit_bench.benchmark.setups import Setup

AgentConfig = dict[str, Any]
HarnessBuilder = Callable[[Setup], AgentConfig]

HARNESSES: dict[str, HarnessBuilder] = {}


def register_harness(name: str) -> Callable[[HarnessBuilder], HarnessBuilder]:
    def decorator(builder: HarnessBuilder) -> HarnessBuilder:
        if name in HARNESSES:
            raise ValueError(f"Harness already registered: {name}")
        HARNESSES[name] = builder
        return builder
    return decorator


def require_harness(name: str) -> HarnessBuilder:
    try:
        return HARNESSES[name]
    except KeyError:
        raise ValueError(f"Unknown harness {name!r}; choose from: {', '.join(sorted(HARNESSES))}") from None


def agent_config(setup: Setup) -> AgentConfig:
    """Harbor agent config for ``setup``."""
    return require_harness(setup.harness)(setup)


@register_harness("fab")
def fab_agent(setup: Setup) -> AgentConfig:
    """FAB's built-in audit agent (benchmark/agent.py + prompts.txt)."""
    return {
        "import_path": "financial_audit_bench.benchmark.agent:AuditAgent",
        "model_name": setup.model,
        "kwargs": {"max_model_calls": setup.max_model_calls,
                   "model_kwargs": setup.model_kwargs, **setup.harness_kwargs},
    }


@register_harness("terminus-2")
def terminus_2_agent(setup: Setup) -> AgentConfig:
    """Harbor's general-purpose Terminus-2 agent. Owned by Workstream 1."""
    raise NotImplementedError(
        "The terminus-2 harness is not wired up yet (Workstream 1). It needs tmux in the "
        "task image and the same audit notes FAB's agent gets from prompts.txt.")
