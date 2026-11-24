"""Contract tests for the shared setup / harness / outcome interface (serialize branch)."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from financial_audit_bench.benchmark import harnesses
from financial_audit_bench.benchmark.outcomes import load_job_outcomes, read_setup, write_setup
from financial_audit_bench.benchmark.setups import Setup, load_setups


def test_setup_file_round_trips_and_names_are_job_safe(tmp_path):
    path = tmp_path / "setups.toml"
    path.write_text('''
[[setup]]
model = "claude-opus-5-5"

[[setup]]
name = "terminus deepseek"
harness = "terminus-2"
model = "openrouter/deepseek/deepseek-v4.1-flash"
reasoning_effort = "high"
temperature = 0.5
harness_kwargs = { max_turns = 50 }
''')
    first, second = load_setups(path)
    assert first.name == "fab-claude-opus-5-5" and first.harness == "fab"
    assert second.name == "terminus-deepseek"
    assert second.model_kwargs == {"reasoning_effort": "high", "temperature": 0.5}
    assert Setup.from_dict(second.to_dict()) == second


@pytest.mark.parametrize("body,message", [
    ("", "No [[setup]]"),
    ('[[setup]]\nmodel = "m"\nharness = "nope"\n', "Unknown harness"),
    ('[[setup]]\nmodel = "m"\ncolour = "red"\n', "Unknown setup keys"),
    ('[[setup]]\nmodel = "m"\n[[setup]]\nmodel = "m"\n', "Duplicate setup names"),
])
def test_bad_setup_files_are_rejected(tmp_path, body, message):
    path = tmp_path / "setups.toml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_setups(path)


def test_fab_harness_matches_the_original_agent_config():
    setup = Setup(model="gpt-test", reasoning_effort="high", temperature=1.0, max_model_calls=7)
    assert harnesses.agent_config(setup) == {
        "import_path": "financial_audit_bench.benchmark.agent:AuditAgent",
        "model_name": "gpt-test",
        "kwargs": {"max_model_calls": 7,
                   "model_kwargs": {"reasoning_effort": "high", "temperature": 1.0}},
    }


def test_terminus_2_is_registered_but_left_to_workstream_1():
    with pytest.raises(NotImplementedError, match="Workstream 1"):
        harnesses.agent_config(Setup(harness="terminus-2", model="m"))


def test_outcomes_read_back_score_cost_runtime_and_setup(tmp_path):
    from harbor.models.trial.result import TrialResult

    job = tmp_path / "20261123-run-fab-gpt-test"
    start = datetime(2026, 11, 23, tzinfo=UTC)
    for name, reward, cost in [("cash__a", 1.0, 0.25), ("debt__b", 0.5, None)]:
        trial = TrialResult.model_validate({
            "task_name": f"local/{name.split('__')[0]}", "trial_name": name,
            "trial_uri": "file:///x", "task_id": {"path": "x"}, "task_checksum": "0",
            "config": {"task": {"path": "x"}},
            "agent_info": {"name": "financial-audit", "version": "1.0.0"},
            "agent_result": {"cost_usd": cost},
            "verifier_result": {"rewards": {"reward": reward}},
            "agent_execution": {"started_at": start.isoformat(),
                                "finished_at": (start + timedelta(seconds=90)).isoformat()},
        })
        (job / name).mkdir(parents=True)
        (job / name / "result.json").write_text(trial.model_dump_json())
    setup = Setup(model="gpt-test")
    write_setup(job, setup)
    assert read_setup(job) == setup
    cash, debt = load_job_outcomes(job)
    assert (cash.setup, cash.task_id, cash.passed, cash.cost_usd, cash.runtime_sec) == \
        ("fab-gpt-test", "cash", True, 0.25, 90.0)
    assert (debt.task_id, debt.score, debt.passed, debt.cost_usd) == ("debt", 0.5, False, None)
    assert json.loads(json.dumps(cash.to_dict()))["job"] == job.name
