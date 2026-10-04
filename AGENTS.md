# AGENTS.md

Guidance for coding agents (Codex, Claude Code and others) working in this repository. It follows
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md), which has the details for human contributors.

## Layout

- `verl/`: the training framework (a fork of EasyR1 / veRL): trainer, actor and rollout workers, the
  agent loop, the reward manager.
- `examples/reproduction/<method>/`: launchers and a README per reproduced paper;
  `examples/comparison/`: the controlled comparison of the methods.
- `examples/reward_function/`, `examples/format_prompt/`, `examples/system_prompt/`: rewards and prompts.
- `eval/`: the evaluation harness (benchmarks in `eval/config/benchmarks.yaml`, suites in
  `eval/config/suites.yaml`).
- `scripts/`: data preparation, environment setup and the repository checks.
- `docs/`: implementation notes and algorithm parameters; `tests/`: unit tests (CPU).

## Before you finish a change

```bash
pre-commit run --all-files                     # ruff, license headers, shell syntax, docs, file checks
python -m pytest -q tests/                     # or the test files of the code you changed
DRY_RUN=1 bash examples/reproduction/<method>/<script>.sh   # for a changed launcher
```

## Conventions

- **Commits.** `[type] scope: summary`, as in "Commit messages" of CONTRIBUTING.md (checked by
  `scripts/check_commit_msg.py`). One change per commit, with its tests and documentation; add an
  `Impact:` footer when results, cached evaluations or prepared data change.
- **Compatibility.** New options default to the existing behavior. Evaluation options enter the
  result fingerprint only when they are set, so earlier results stay valid.
- **Reproductions.** Follow the paper and its official code. Where they differ, or where we keep a
  different behavior, the method README says so, describing other authors' code factually.
- **Documentation.** Keep `README.md` and `README_zh.md` in sync (`python scripts/check_docs.py --fix`).
  Code, comments, documentation and commit messages are in English.
- **Files.** Do not commit data, checkpoints, evaluation results or machine-specific paths
  (`data/`, `checkpoints/`, `eval/results/` and `tmp/` are ignored).
