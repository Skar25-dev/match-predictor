import re
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "record.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _commands(workflow):
    for step in workflow["jobs"]["record"]["steps"]:
        for line in step.get("run", "").splitlines():
            match = re.match(r"\s*python -m (\S+)(.*)", line)
            if match:
                yield match.group(1), re.findall(r"--[\w-]+", match.group(2))


def test_runs_daily_and_can_be_started_by_hand(workflow):
    triggers = workflow[True] if True in workflow else workflow["on"]

    assert triggers["schedule"][0]["cron"].count(" ") == 4
    assert "workflow_dispatch" in triggers


def test_job_may_commit_the_log(workflow):
    assert workflow["permissions"] == {"contents": "write"}
    commit = next(s for s in workflow["jobs"]["record"]["steps"] if "git push" in s.get("run", ""))
    assert "data/live_predictions.csv" in commit["run"]


def test_every_command_uses_flags_that_really_exist(workflow):
    commands = list(_commands(workflow))

    assert {module for module, _ in commands} >= {
        "matchpredictor.data.download",
        "matchpredictor.evaluation.predictions_log",
    }
    for module, flags in commands:
        help_text = subprocess.run(
            [sys.executable, "-m", module, "--help"], capture_output=True, text=True, check=True
        ).stdout
        for flag in flags:
            assert flag in help_text, f"{module} no tiene {flag}"


def test_record_and_report_use_the_same_log_file(workflow):
    logs = {
        re.search(r"--log (\S+)", line).group(1)
        for step in workflow["jobs"]["record"]["steps"]
        for line in step.get("run", "").splitlines()
        if "predictions_log" in line
    }

    assert logs == {"data/live_predictions.csv"}
