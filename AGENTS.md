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

## Design principles

- **Opt-in only.** New options default to the existing behavior. With a feature off, no new forward
  pass, data column, reward term or random draw runs, existing configurations reproduce bit for bit,
  and cached results stay valid (evaluation options enter the fingerprint only when set). Never change
  the meaning of an existing option, loss or metric.
- **Evidence order.** The paper defines the method, the official code settles what the paper leaves
  open, the official scripts give the settings. Report a conflict instead of choosing silently; correct
  only slips that contradict the authors' own paper or comments, and document them. Without official
  code, state what is inferred.
- **Mechanisms, not methods.** A method is a launcher that combines orthogonal options: auxiliary view,
  sensitivity signal and the model that scores it, token selection, advantage shaping, auxiliary loss,
  schedule, aggregation. First check what existing options already express; reuse one only when its
  meaning is the same. Core code is organized and named by mechanism; a paper name may appear as an
  option value (`advantage_scaling_method=pepo`) or prefix a hyper-parameter only that paper has.
- **One stage per operation.** Keep signal computation, token or response weighting, advantage shaping,
  loss weighting, auxiliary losses and aggregation apart, and put each operation where the paper does.
  No loss or advantage may depend on how a batch is split into micro-batches, dynamic batches or ranks.
- **One contract per output format.** Prompt, training reward, validation and offline evaluation share
  one parser. Missing tags or fields fail closed; add no fallbacks, partial credit or reward terms
  beyond the paper.
- **Fail early.** Reject unknown kwargs, unsupported models, incompatible combinations, missing data
  columns and incompatible extra models (teacher, judge or detector: tokenizer, processor) when the
  configuration is built, with a message that says how to fix it.
- **Two kinds of launchers.** `examples/reproduction/<method>/` follows the paper; `examples/comparison/`
  changes only the method-specific options of the shared recipe. Add a variant as a sibling script; the
  method README lists every difference, describing other authors' code factually.
- **Reproducible inputs.** Data comes from a prepare script that records its source, never from
  hand-edited local files; check new training data against the evaluation sets. New randomness uses
  its own generator seeded from the config.
- **Few metrics.** Log what shows that a mechanism is active and healthy; keep existing metric names
  and meanings.

## Before you finish a change

```bash
pre-commit run --all-files                     # ruff, license headers, shell syntax, docs, file checks
python -m pytest -q tests/                     # or the test files of the code you changed
DRY_RUN=1 bash examples/reproduction/<method>/<script>.sh   # for a changed launcher
```

Tests cover the default path unchanged, the new path active, the old parsers and rewards, and invalid
configurations. A change to shared code also gets a fixed-batch check that losses, gradients and
metrics of existing configurations are unchanged.

## Conventions

- **Commits.** `[type] scope: summary`, as in "Commit messages" of CONTRIBUTING.md (checked by
  `scripts/check_commit_msg.py`). One change per commit, with its tests and documentation; add an
  `Impact:` footer when results, cached evaluations or prepared data change.
- **Documentation.** Keep `README.md` and `README_zh.md` in sync (`python scripts/check_docs.py --fix`).
  Code, comments, documentation and commit messages are in English.
- **Files.** Do not commit data, checkpoints, evaluation results or machine-specific paths
  (`data/`, `checkpoints/`, `eval/results/` and `tmp/` are ignored).
