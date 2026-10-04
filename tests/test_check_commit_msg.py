# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""scripts/check_commit_msg.py: the commit message format of .github/CONTRIBUTING.md."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_msg.py"
SPEC = importlib.util.spec_from_file_location("check_commit_msg", SCRIPT)
check_commit_msg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_commit_msg)


@pytest.mark.parametrize(
    "subject",
    [
        "[fix] deepeyes: give no accuracy to answers of 1,000+ characters",
        "[align] vepo: update on 128 prompts per step, as the paper",
        "[feat] eval: add the tallyqa_relabeled benchmark",
        "[docs] fix typos in the README",  # no scope
        "[list] add VEPO (2606.03937)",
        "[config] qwen3-vl.8b: save a checkpoint every 25 steps",
        "Merge pull request #12 from someone/branch",
        "fixup! [fix] eval: read the boxes",
    ],
)
def test_subjects_that_follow_the_format(subject):
    assert check_commit_msg.check_subject(subject) == []


@pytest.mark.parametrize(
    "subject,problem",
    [
        ("Fix the reward", "'[type] scope: summary'"),
        ("fix: the reward", "'[type] scope: summary'"),
        ("[Fix] reward: zero long answers", "'[type] scope: summary'"),
        ("[bugfix] reward: zero long answers", "unknown type [bugfix]"),
        ("[fix] reward: Zero long answers", "lowercase verb"),
        ("[fix] reward: zero long answers.", "period"),
        ("[fix]reward: zero long answers", "'[type] scope: summary'"),
        ("[fix] reward: " + "a" * 70, "characters"),
    ],
)
def test_subjects_that_do_not(subject, problem):
    problems = check_commit_msg.check_subject(subject)
    assert any(problem in item for item in problems), problems


def test_full_messages():
    good = "[fix] eval: read boxes in parentheses\n\nGRIT writes (x1, y1), (x2, y2).\n\nImpact: grit_iou changes.\n"
    assert check_commit_msg.check_message(good) == []
    assert check_commit_msg.check_message("# Please enter the commit message\n" + good) == []
    assert "second line" in check_commit_msg.check_message("[fix] eval: read boxes\nbody right away\n")[0]
    assert check_commit_msg.check_message("\n# only comments\n") == ["the commit message is empty"]


def test_command_line(tmp_path):
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("[test] eval: cover the summary columns\n")
    assert check_commit_msg.main([str(message)]) == 0
    message.write_text("update stuff\n")
    assert check_commit_msg.main([str(message)]) == 1
    assert check_commit_msg.main(["--title", "[feat] eval: add a benchmark"]) == 0
    assert check_commit_msg.main(["--title", "Add a benchmark"]) == 1


def test_range_of_commits(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "base")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "[test] ok")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "Not ok")
    result = subprocess.run(
        ["python3", str(SCRIPT), "--range", "HEAD~2..HEAD"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1 and "Not ok" in result.stderr and "[test] ok" not in result.stderr
