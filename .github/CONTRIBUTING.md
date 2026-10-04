# Contributing

Everyone is welcome to contribute: new papers for the list, new methods, new benchmarks,
reproduction results, bug reports and documentation fixes.

## Ways to contribute

- **Add a paper to the list.** Edit the tables in `README.md` and `README_zh.md` (newest first): arXiv
  link, first author, venue only if it is confirmed by the arXiv comments, OpenReview or the
  official repository, code link and a one-line key idea.
- **Add a method.** Follow [docs/add_method.md](../docs/add_method.md). A new method needs config
  fields with validation, unit tests, a `examples/reproduction/<method>/` directory with baseline and method
  scripts, and a README that states the differences from the official recipe.
- **Add a benchmark.** Follow the tutorial in [eval/README.md](../eval/README.md#adding-a-new-benchmark).
- **Report results.** Open an issue with the exact command, the commit, the training logs
  (`experiment_log.jsonl`) and the evaluation summary.

## Checks with pre-commit

The repository's checks run with [pre-commit](https://pre-commit.com): the same checks run on your
machine before every commit and in CI on every pull request, so a pull request that passes them
locally passes the lint job.

**Set up once** (pre-commit needs git 2.31 or newer; the first run downloads the hook environments
from GitHub):

```bash
bash scripts/install_env.sh   # the training environment, with pytest, ruff and pre-commit
# or, for documentation and paper-list changes only:
pip install pre-commit
pre-commit install            # in your clone: installs the pre-commit and commit-msg hooks
```

**On every commit** the hooks check the staged files and the commit message. A hook that fixes files
(whitespace, end of file, ruff) changes them and stops the commit: look at the change with
`git diff`, `git add` it and commit again. A hook that only checks prints what to fix.

**Before opening a pull request** run every check on the whole repository, and the unit tests:

```bash
pre-commit run --all-files    # or: make commit (installs the hooks and runs them)
make test                     # unit tests on CPU (python -m pytest -q tests/)
DRY_RUN=1 bash examples/reproduction/<method>/<script>.sh   # validates a launcher without GPUs
```

| Hook | Checks | When it fails |
|---|---|---|
| ruff check, ruff format | lint and formatting (the ruff version of `scripts/constraints.txt`) | most problems are fixed in place; fix the rest by hand (`make style` runs both) |
| Apache license headers | every Python file starts with the license header | copy the header of any Python file in the repository |
| shell syntax | `bash -n` on the shell scripts | fix the reported line |
| paper list, badges and Markdown links | `scripts/check_docs.py`: the two READMEs list the same papers, newest first, the badges count them, links resolve | `python scripts/check_docs.py --fix` updates the badges; fix the rest as reported |
| file checks | valid Python and YAML, no merge markers, files under 25 MB, sorted `requirements.txt`, no trailing whitespace | fixed in place, or as reported |
| don't commit to branch | no commits on `main` | commit on a branch of your fork |
| commit message format | the message follows [Commit messages](#commit-messages) | `git commit --amend` with a corrected message |

`git commit --no-verify` skips the hooks; CI runs the same checks, so use it only to save work in
progress on your own branch.

## Commit messages

```
[type] scope: summary

Body: why the change is needed, and the behavior before and after.

Impact: what changes for existing runs, results, caches or prepared data
Refs: #123
```

**Subject.** In English, `[type]` and an optional `scope:`, then a summary in the imperative mood that
starts with a lowercase verb, without a final period, at most 72 characters in all. Describe the
behavior, not the file: "give no accuracy to answers of 1,000+ characters", not "update deepeyes.py".

**Type.** Exactly one:

| Type | For |
|---|---|
| `feat` | a new capability: an option, method, benchmark, script or tool |
| `fix` | code that did not do what it should |
| `align` | matching a paper or its official code (settings, rewards, prompts, evaluation protocol) |
| `config` | a default or a script setting changed for another reason |
| `data` | data preparation, converters, released datasets |
| `list` | the paper list in `README.md` and `README_zh.md` |
| `results` | result tables |
| `docs` | documentation only |
| `test` | tests only |
| `refactor` | restructuring without a change of behavior |
| `perf` | speed or memory, same results |
| `build` | dependencies, CI, hooks, packaging |
| `revert` | reverting a commit (the body names the commit and why) |

**Scope.** Optional, one, lowercase: a method (`cgpo`, `papo`, `vppo`, `tor`, `dvrp`, `pgpo`, `pepo`,
`cfpo`, `vepo`, `grit`, `deepeyes`), `comparison`, or a component (`trainer`, `actor`, `rollout`,
`agent`, `reward`, `data`, `eval`, `docs`, `ci`). Leave it out when a change spans several.

**Body.** Say why and what changes, wrapped at 72 columns; it may be left out only when the subject
says everything (a typo). An `align` commit names its source: the section or table of the paper, or
the file and function (or commit) of the official code. A `fix` commit describes the case that went
wrong. Describe other authors' code factually ("the released code reads ...", "differs from").

**Impact.** Required when the change alters the results of an existing script, invalidates cached
evaluation results, requires preparing data again, or rejects a configuration that used to pass.

**One change per commit.** Keep the tests and the documentation of a change in its commit, and
unrelated changes in separate commits.

Examples:

```
[fix] deepeyes: give no accuracy to answers of 1,000+ characters
[align] vepo: update on 128 prompts per step, as the paper
[feat] eval: add the tallyqa_relabeled benchmark
[config] pepo: save a checkpoint every 25 steps
[list] add VEPO (2606.03937)
```

Pull requests are merged with **squash and merge**, so the **pull request title** becomes the subject
of the commit on `main`: write it in the same format. CI checks the title (and again when you edit
it), not the commits inside the pull request. Locally the commit-msg hook checks every commit;
`fixup!` and `squash!` commits are exempt.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Set up the environment and the hooks ([Checks with pre-commit](#checks-with-pre-commit)).
3. Make your change with tests, and run the checks, the unit tests and `DRY_RUN` for launchers.
4. Open a pull request with a title in the [commit message format](#commit-messages) and fill in the
   template. GitHub Actions runs the pre-commit checks, the CPU unit tests and the title check on
   every pull request; GPU training and evaluation are not run in CI, so describe how you tested
   them.

Please be respectful and follow the [code of conduct](CODE_OF_CONDUCT.md).
