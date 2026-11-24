from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from financial_audit_bench.benchmark.build import MARKER, prepare
from financial_audit_bench.benchmark.dataset import DATASET_REVISION, resolve_dataset
from financial_audit_bench.benchmark.harnesses import require_harness
from financial_audit_bench.benchmark.outcomes import write_setup
from financial_audit_bench.benchmark.setups import DEFAULT_HARNESS, Setup, slug
from financial_audit_bench.benchmark.grading.local import (
    copy_job,
    grade_job,
    require_libreoffice,
)
from financial_audit_bench.benchmark.grading.recalculation_cache import (
    RecalculationCache,
)
from financial_audit_bench.defaults import DEFAULT_MAX_MODEL_CALLS
from financial_audit_bench.synthetic_binders.models import (
    SUPPORTED_BUSINESS_TYPES,
    SyntheticBinderConfig,
)

DEFAULT_BINDERS_DIR = "generated_binders"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_generate_synthetic_binder_parser(subparsers)
    add_run_parser(subparsers)
    add_regrade_parser(subparsers)
    return parser


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tasks-dir", type=Path,
                        help="Use a local dataset containing tasks/, review_tasks/, and inputs/ instead of Hugging Face.")
    source.add_argument("--dataset-revision", default=DATASET_REVISION,
                        help=f"Hugging Face task revision (default: {DATASET_REVISION}).")


def add_run_parser(subparsers) -> None:
    parser = subparsers.add_parser("run", help="Rebuild tasks and run the benchmark with Harbor.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model",
                           help="Model name, e.g. gpt-6-astra or openrouter/google/gemini-3.8-flash.")
    selection.add_argument("--agent", choices=("oracle", "nop"), help="Run a Harbor diagnostic control instead of a model.")
    parser.add_argument("--harness", default=DEFAULT_HARNESS,
                        help=f"Agent harness to run the model in (default: {DEFAULT_HARNESS}).")
    parser.add_argument("--reasoning-effort", default="medium", help="Model reasoning effort (default: medium).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; omitted by default.")
    parser.add_argument("--review", action="store_true", help="Use the review task catalog.")
    parser.add_argument("--task", action="append", dest="task_ids", help="Task ID; repeat to select more.")
    parser.add_argument("--binder", help="Input binder name, e.g. manufacturing_v1_01.")
    parser.add_argument("--limit", type=int, help="Maximum number of tasks.")
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--grade", action="store_true", help="Grade saved workbooks locally after execution.")
    add_dataset_arguments(parser)
    parser.set_defaults(func=run_benchmark)


@dataclass(frozen=True)
class RunOptions:
    """Run settings shared by every setup in one run (same tasks, same grading)."""

    num_workers: int = 4
    n_attempts: int = 1
    limit: int | None = None
    grade: bool = False


@dataclass(frozen=True)
class RunContext:
    """Where one run's inputs and outputs live; one per ``fab run`` invocation."""

    root: Path
    run_id: str
    dataset_root: Path
    tasks: list

    @property
    def experiment(self) -> Path:
        return self.root / "experiments" / self.run_id

    def job_name(self, suffix: str) -> str:
        return f"{self.run_id}-{slug(suffix)}"

    def job_entry(self, suffix: str, **extra) -> dict:
        name = self.job_name(suffix)
        return {**extra, "path": f"jobs/{name}", "config": name + ".json"}

    @classmethod
    def start(cls, root: Path, dataset_root: Path, tasks: list) -> RunContext:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        context = cls(root=root, run_id=run_id, dataset_root=dataset_root, tasks=tasks)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        context.experiment.mkdir(parents=True)
        return context

    def write_metadata(self, jobs: list[dict], *, grade: bool) -> None:
        (self.experiment / "run.json").write_text(json.dumps({
            "kind": "benchmark_run", "run_id": self.run_id, "local_grading": grade,
            "dataset_path": str(self.dataset_root),
            "task_ids": [task.short_name for task in self.tasks],
            "jobs": jobs,
        }, indent=2) + "\n")


def select_tasks(args: argparse.Namespace, dataset_root: Path) -> list:
    from financial_audit_bench.benchmark.tasks import load_tasks

    tasks = load_tasks(dataset_root / ("review_tasks" if args.review else "tasks"))
    if args.task_ids:
        tasks = [task for task in tasks if task.short_name in args.task_ids]
    if args.binder:
        tasks = [task for task in tasks if task.config.metadata["binder"] == args.binder]
    if not tasks:
        raise ValueError("No tasks match the selected binder and task IDs")
    pending = [task.short_name for task in tasks if task.config.metadata.get("grading") == "pending"]
    if pending and (args.grade or args.agent == "oracle"):
        raise ValueError("Grading and oracle controls are not ready for: " + ", ".join(pending)
                         + ". Run these tasks without --grade and with a model or nop agent.")
    return tasks


def run_job(context: RunContext, agent: dict, job_suffix: str, options: RunOptions) -> Path:
    """Run one Harbor job (one agent over the run's tasks), grade it, and return its folder.

    Each job builds its tasks into its own ``artifacts/harbor/<job>`` folder, so several
    jobs from the same checkout can run at the same time.
    """
    from harbor.models.job.config import JobConfig

    from financial_audit_bench.benchmark.tasks import load_tasks

    root = context.root
    job_name = context.job_name(job_suffix)
    prepared_rel = f"artifacts/harbor/{job_name}"
    config = JobConfig.model_validate({
        "job_name": job_name, "jobs_dir": "jobs",
        "n_concurrent_trials": options.num_workers, "n_attempts": options.n_attempts,
        "retry": {"max_retries": 0}, "verifier": {"disable": True},
        "datasets": [{"path": prepared_rel, "n_tasks": options.limit}],
        "agents": [agent],
    })
    prepared = prepare(context.tasks, root, dataset_root=context.dataset_root,
                       output=root / prepared_rel)
    shutil.copy2(prepared / MARKER, context.experiment / MARKER)
    print(f"Prepared {len(context.tasks)} tasks in {prepared}", flush=True)
    config_path = context.experiment / (job_name + ".json")
    config_path.write_text(config.model_dump_json(indent=2, exclude_none=True) + "\n")
    job_dir = root / "jobs" / job_name

    with ExitStack() as stack:
        recalculation_cache = None
        if options.grade:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="fab-grade-cache-"))
            recalculation_cache = RecalculationCache(Path(directory))
        subprocess.run([sys.executable, "-m", "harbor.cli.main", "run", "-c", str(config_path), "-y"],
                       cwd=root, check=True)
        if options.grade:
            grade_job(job_dir, load_tasks(prepared), options.num_workers,
                      recalculation_cache=recalculation_cache)
    return job_dir


def run_setup(context: RunContext, setup: Setup, options: RunOptions) -> Path:
    """Run and grade one harness + model setup. Workstream 2 calls this once per setup."""
    from financial_audit_bench.benchmark.harnesses import agent_config

    job_dir = run_job(context, agent_config(setup), setup.name, options)
    write_setup(job_dir, setup)
    return job_dir


def run_benchmark(args: argparse.Namespace) -> None:
    root = Path.cwd()
    if args.agent == "oracle" and args.tasks_dir is None:
        raise ValueError("The published dataset has no reference solutions. "
                         "Oracle controls require --tasks-dir with local solution files.")
    setup = None if args.agent else Setup(
        harness=args.harness, model=args.model, reasoning_effort=args.reasoning_effort,
        temperature=args.temperature, max_model_calls=args.max_model_calls)
    if setup:
        require_harness(setup.harness)

    dataset_root = resolve_dataset(args.tasks_dir, revision=args.dataset_revision)
    tasks = select_tasks(args, dataset_root)
    if args.grade:
        require_libreoffice()
    options = RunOptions(num_workers=args.num_workers, n_attempts=args.n_attempts,
                         limit=args.limit, grade=args.grade)
    context = RunContext.start(root, dataset_root, tasks)
    print(f"Saving jobs to {root / 'jobs'}; experiment metadata to {context.experiment}", flush=True)
    if setup:
        context.write_metadata([context.job_entry(setup.name, model_id=setup.model,
                                                  setup=setup.to_dict())], grade=args.grade)
        job_dir = run_setup(context, setup, options)
    else:
        context.write_metadata([context.job_entry(args.agent, model_id=args.agent)], grade=args.grade)
        job_dir = run_job(context, {"name": args.agent}, args.agent, options)
    require_completed_job(job_dir)


def require_completed_job(path: Path) -> None:
    result = json.loads((path / "result.json").read_text())
    stats = result["stats"]
    if stats["n_errored_trials"] or stats["n_completed_trials"] != result["n_total_trials"]:
        raise RuntimeError(f"Incomplete Harbor job: {path}")


def add_regrade_parser(subparsers) -> None:
    parser = subparsers.add_parser("regrade", help="Regrade a saved Harbor job locally using current rubrics.")
    parser.add_argument("source", type=Path, help="Original Harbor job directory containing config.json and result.json.")
    parser.add_argument("--review", action="store_true", help="Use the review task catalog.")
    parser.add_argument("--num-workers", type=int, default=4)
    add_dataset_arguments(parser)
    parser.set_defaults(func=regrade_benchmark)


def regrade_benchmark(args: argparse.Namespace) -> None:
    from financial_audit_bench.benchmark.tasks import load_tasks

    root = Path.cwd()
    source = args.source.resolve()
    require_libreoffice()
    source_id = json.loads((source / "result.json").read_text())["id"]
    dataset_root = resolve_dataset(args.tasks_dir, revision=args.dataset_revision)
    tasks = load_tasks(dataset_root / ("review_tasks" if args.review else "tasks"))
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    name = f"{run_id}-regrade-{source_id[:8]}"
    output = root / "jobs" / name
    experiment = root / "experiments" / run_id
    experiment.mkdir(parents=True)
    (experiment / "run.json").write_text(json.dumps({
        "kind": "regrade", "run_id": run_id,
        "dataset_path": str(dataset_root),
        "source_job": {"path": str(source), "id": source_id},
        "jobs": [{"path": f"jobs/{name}"}],
        "task_ids": [task.short_name for task in tasks], "grading": "local",
    }, indent=2) + "\n")
    print(f"Regrading {source}; saving native Harbor results to {output}", flush=True)
    copy_job(source, output)
    grade_job(output, tasks, args.num_workers)
    require_completed_job(output)


def add_generate_synthetic_binder_parser(subparsers):
    synthetic_parser = subparsers.add_parser(
        "generate-synthetic-binder",
    )
    synthetic_parser.add_argument(
        "business_type", choices=sorted(SUPPORTED_BUSINESS_TYPES)
    )
    synthetic_parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Output directory.",
    )
    synthetic_parser.add_argument("--seed", type=int, default=42)
    synthetic_parser.add_argument("--run-id")
    synthetic_parser.set_defaults(func=generate_synthetic_binder_command)


def generate_synthetic_binder_command(args: argparse.Namespace) -> None:
    from financial_audit_bench.synthetic_binders.fsm import run_synthetic_binder

    output_dir = args.output_dir or Path(DEFAULT_BINDERS_DIR) / args.business_type
    manifest = run_synthetic_binder(
        SyntheticBinderConfig(
            business_type=args.business_type,
            seed=args.seed,
            output_dir=output_dir,
            run_id=args.run_id,
        )
    )
    print(f"Generated synthetic binder {manifest['run_id']} at {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    try:
        args.func(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
