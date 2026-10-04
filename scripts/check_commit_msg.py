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
"""Check commit messages against the format in .github/CONTRIBUTING.md.

    [type] scope: summary

    Body: why, and the behavior before and after.

    Impact: what changes for existing runs, results, caches or data

Usage:
    python3 scripts/check_commit_msg.py .git/COMMIT_EDITMSG    # a message file (pre-commit commit-msg hook)
    python3 scripts/check_commit_msg.py --title "[fix] eval: ..."  # a pull request title
    python3 scripts/check_commit_msg.py --range origin/main..HEAD   # the subjects of a range of commits
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


TYPES = {
    "feat": "new capability: an option, method, benchmark, script or tool",
    "fix": "code that did not do what it should",
    "align": "match a paper or its official code (settings, rewards, prompts, evaluation protocol)",
    "config": "a default or script setting changed for another reason",
    "data": "data preparation, converters, released datasets",
    "list": "the paper list in README.md and README_zh.md",
    "results": "result tables",
    "docs": "documentation only",
    "test": "tests only",
    "refactor": "restructuring without a change of behavior",
    "perf": "speed or memory, same results",
    "build": "dependencies, CI, hooks, packaging",
    "revert": "revert a commit (the body names the commit and why)",
}
MAX_SUBJECT = 72
SUBJECT = re.compile(r"^\[(?P<type>[a-z]+)\] (?:(?P<scope>[a-z0-9][a-z0-9_.-]*): )?(?P<summary>\S.*)$")
# messages that git or a pull request workflow writes and that a squash merge replaces
EXEMPT = re.compile(r"^(Merge (branch|pull request|remote-tracking branch|tag) |fixup! |squash! |amend! )")


def check_subject(subject: str) -> list[str]:
    """Problems of one subject line (empty when it follows the format)."""
    if EXEMPT.match(subject):
        return []
    problems = []
    match = SUBJECT.match(subject)
    if match is None:
        problems.append("the subject must read '[type] scope: summary' (the scope is optional)")
    else:
        if match.group("type") not in TYPES:
            problems.append(f"unknown type [{match.group('type')}]; use one of: {', '.join(TYPES)}")
        summary = match.group("summary")
        if summary[0].isupper():
            problems.append("start the summary with a lowercase verb (imperative mood), e.g. 'add', 'fix', 'use'")
        if summary.endswith("."):
            problems.append("do not end the subject with a period")
    if len(subject) > MAX_SUBJECT:
        problems.append(f"the subject has {len(subject)} characters; keep it to {MAX_SUBJECT}")
    return problems


def check_message(message: str) -> list[str]:
    """Problems of a full commit message; lines starting with '#' are git comments."""
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["the commit message is empty"]
    problems = check_subject(lines[0])
    if len(lines) > 1 and lines[1].strip():
        problems.append("leave the second line empty, between the subject and the body")
    return problems


def _report(where: str, text: str, problems: list[str]) -> None:
    print(f"{where}: {text!r}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check commit messages against .github/CONTRIBUTING.md.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("message_file", nargs="?", help="file with the commit message (commit-msg hook)")
    group.add_argument("--title", help="a pull request title")
    group.add_argument("--range", dest="revision_range", help="check the subjects of the commits in A..B")
    args = parser.parse_args(argv)

    failures = []
    if args.title is not None:
        problems = check_subject(args.title.strip())
        if problems:
            failures.append(("pull request title", args.title.strip(), problems))
    elif args.revision_range is not None:
        log = subprocess.run(
            ["git", "log", "--no-merges", "--format=%h %s", args.revision_range],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in log.splitlines():
            commit, _, subject = line.partition(" ")
            problems = check_subject(subject)
            if problems:
                failures.append((f"commit {commit}", subject, problems))
    else:
        message = Path(args.message_file).read_text(encoding="utf-8")
        problems = check_message(message)
        if problems:
            failures.append(("commit message", message.splitlines()[0] if message.strip() else "", problems))

    for where, text, problems in failures:
        _report(where, text, problems)
    if failures:
        print(
            "\nFormat: '[type] scope: summary', e.g. '[fix] deepeyes: give no accuracy to answers of 1,000+ "
            "characters'.\nTypes: " + ", ".join(TYPES) + ". See 'Commit messages' in .github/CONTRIBUTING.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
