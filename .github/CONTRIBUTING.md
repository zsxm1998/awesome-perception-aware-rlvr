# Contributing

Everyone is welcome to contribute: new papers for the list, new methods, new benchmarks,
reproduction results, bug reports and documentation fixes.

## Ways to contribute

- **Add a paper to the list.** Edit the tables in `README.md` and `README_zh.md` (newest first): arXiv
  link, first author, venue only if it is confirmed by the arXiv comments, OpenReview or the
  official repository, code link and a one-line key idea.
- **Add a method.** Follow [docs/add_method.md](../docs/add_method.md). A new method needs config
  fields with validation, unit tests, a `reproduction/<method>/` directory with baseline and method
  scripts, and a README that states the differences from the official recipe.
- **Add a benchmark.** Follow the tutorial in [eval/README.md](../eval/README.md#adding-a-new-benchmark).
- **Report results.** Open an issue with the exact command, the commit, the training logs
  (`experiment_log.jsonl`) and the evaluation summary.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Install the environment (`bash scripts/install_env.sh`) and the dev tools (`pip install ruff pytest`).
3. Make your change, add tests, and run:

   ```bash
   make quality   # ruff
   make license   # Apache license headers
   make test      # unit tests (CPU)
   DRY_RUN=1 bash reproduction/<method>/<script>.sh   # validates a launcher without GPUs
   ```

4. Open a pull request describing the change and how it was tested.

Please be respectful and follow the [code of conduct](CODE_OF_CONDUCT.md).
