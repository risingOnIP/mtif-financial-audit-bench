"""Shared contract: one graded task result, read back from a finished job.

``run_setup`` (cli.py) writes ``jobs/<job>/fab_setup.json`` next to Harbor's
``result.json``, so every job says which setup produced it. Workstream 3 builds
its comparison and before/after reports on ``load_job_outcomes``; it may add
fields (e.g. audit area) but should keep the existing ones stable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from financial_audit_bench.benchmark.setups import Setup

SETUP_FILE = "fab_setup.json"


@dataclass(frozen=True)
class TaskOutcome:
    setup: str               # Setup.name
    job: str                 # jobs/<job> folder name
    task_id: str             # short task name, e.g. as passed to --task
    trial: str               # Harbor trial folder name (one per attempt)
    score: float | None      # grader reward in [0, 1]; None if ungraded or errored
    passed: bool | None      # score == 1.0; None if ungraded or errored
    cost_usd: float | None
    runtime_sec: float | None  # agent execution time
    error: str | None        # exception type if the trial failed

    def to_dict(self) -> dict:
        return asdict(self)


def write_setup(job_dir: Path, setup: Setup) -> None:
    (job_dir / SETUP_FILE).write_text(json.dumps(setup.to_dict(), indent=2) + "\n")


def read_setup(job_dir: Path) -> Setup:
    """Setup that produced ``job_dir``; older jobs fall back to their Harbor config."""
    path = job_dir / SETUP_FILE
    if path.is_file():
        return Setup.from_dict(json.loads(path.read_text()))
    config = json.loads((job_dir / "config.json").read_text())
    agent = config["agents"][0]
    harness = "fab" if "financial_audit_bench" in (agent.get("import_path") or "") else agent.get("name")
    return Setup(harness=harness or "unknown", model=agent.get("model_name") or harness or "unknown")


def load_job_outcomes(job_dir: Path) -> list[TaskOutcome]:
    """One ``TaskOutcome`` per trial in a finished (ideally graded) Harbor job."""
    from harbor.models.trial.result import TrialResult

    job_dir = Path(job_dir)
    setup = read_setup(job_dir)
    outcomes = []
    for path in sorted(job_dir.glob("*/result.json")):
        trial = TrialResult.model_validate_json(path.read_text())
        rewards = trial.verifier_result.rewards if trial.verifier_result else None
        score = rewards.get("reward") if rewards else None
        error = trial.exception_info.exception_type if trial.exception_info else None
        if error:
            score = None
        timing = trial.agent_execution
        runtime = ((timing.finished_at - timing.started_at).total_seconds()
                   if timing and timing.started_at and timing.finished_at else None)
        outcomes.append(TaskOutcome(
            setup=setup.name, job=job_dir.name,
            task_id=trial.task_name.rsplit("/", 1)[-1], trial=trial.trial_name,
            score=score, passed=None if score is None else score == 1.0,
            cost_usd=trial.compute_token_cost_totals()[3],
            runtime_sec=runtime, error=error,
        ))
    return outcomes
