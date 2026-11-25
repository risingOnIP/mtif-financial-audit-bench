# Harness comparison: working on the three workstreams

This fork (`mtif-financial-audit-bench`) extends FAB so the same tasks and grading
run across many harness + model setups, and the results are compared. The
`serialize` branch holds the shared interface all three workstreams build on.
Each person creates their own branch off `serialize` and works in parallel.

## Get started

```bash
git clone -b serialize https://github.com/risingOnIP/mtif-financial-audit-bench.git
cd mtif-financial-audit-bench
git remote add upstream https://github.com/modus-audit/financial-audit-bench.git
git checkout -b ws1-harness     # your own branch, e.g. ws2-parallel-runs / ws3-results
git push -u origin ws1-harness
uv sync
cp .env.example .env            # add your API keys
uv run pytest tests/test_setups_interface.py
```

To push your branch you need to be a collaborator on the fork (ask the
fork owner). Without that, fork this repo yourself and open pull requests from your fork.

## The shared interface (do not change without telling the other workstreams)

| Piece | File | What it is |
|---|---|---|
| `Setup` | `benchmark/setups.py` | One harness + model combination: `name`, `harness`, `model`, `reasoning_effort`, `temperature`, `max_model_calls`, `harness_kwargs`. `name` labels it everywhere. |
| `load_setups(path)` | `benchmark/setups.py` | Reads a TOML file of `[[setup]]` entries (see `setups.example.toml`) and validates it. |
| Harness registry | `benchmark/harnesses.py` | `@register_harness("name")` builders that turn a `Setup` into a Harbor agent config. `fab` is implemented; `terminus-2` is a placeholder. |
| `RunContext`, `RunOptions` | `cli.py` | One run's shared tasks, dataset and run id; and the settings every setup in it shares (workers, attempts, limit, grading). |
| `run_setup(context, setup, options)` | `cli.py` | Runs and grades one setup and returns its `jobs/<job>` folder. Each job builds its tasks in its own `artifacts/harbor/<job>` folder, so several can run at once from one checkout. |
| `TaskOutcome`, `load_job_outcomes(job_dir)` | `benchmark/outcomes.py` | One graded result per trial: `setup`, `job`, `task_id`, `trial`, `score`, `passed`, `cost_usd`, `runtime_sec`, `error`. Every job records its setup in `jobs/<job>/fab_setup.json`. |

`fab run --model X` still works as before and now takes `--harness` (default `fab`).
Job folders are named `<run id>-<setup name>`, e.g. `...-fab-gpt-6-astra`.

## Who owns what

| Suggested branch | Workstream | Owns | Done when |
|---|---|---|---|
| `ws1-harness` | 1. Harness integration | `benchmark/harnesses.py` (`terminus_2_agent`), task image changes in `benchmark/build.py` (e.g. adding `tmux`), passing the same audit notes from `benchmark/prompts.txt` to every harness, a short "adding a Harbor agent" guide | `fab run --harness terminus-2 --model ... --grade` runs and grades a task |
| `ws2-parallel-runs` | 2. Models and parallel runs | A new command (e.g. `fab run-setups setups.toml`) in `cli.py` that builds one `RunContext` and calls `run_setup` for every setup in parallel, then writes `run.json` listing every job | One setup file with several setups produces graded results for all of them from one command |
| `ws3-results` | 3. Results and comparison | A new module (e.g. `benchmark/compare.py`) and `fab report` / `fab compare` commands built on `load_job_outcomes`; may add fields such as audit area to `TaskOutcome` | The report works on any finished runs; compare flags tasks that changed between two runs |

Notes for each workstream:

- **WS1:** keep the builder signature `Setup -> dict`. Anything harness-specific goes in `harness_kwargs`.
- **WS2:** use only `harness = "fab"` setups until WS1 merges; `terminus-2` setups then work with no change on your side. Parallel jobs use Docker and API budget, so keep the total worker count modest on a laptop.
- **WS3:** start from one ordinary graded run (`uv run fab run --model ... --limit 3 --grade`). Jobs made before this branch have no `fab_setup.json`; `read_setup` falls back to their Harbor config.

## Keeping in sync and merging

- Small, frequent commits on your own branch; open a pull request into `serialize` when a piece works.
- When `serialize` changes, update your branch: `git fetch origin && git rebase origin/serialize`.
- To pick up upstream FAB changes: `git fetch upstream && git rebase upstream/main` on `serialize` (the fork owner does this and pushes).
- When all three are in `serialize`, open one pull request from the fork's `serialize` into `modus-audit/financial-audit-bench` `main`.
