## What does this PR do?

<!-- One or two sentences. Link related issues with "Closes #123". -->

## Type of change

- [ ] Add or update a paper in the list
- [ ] Add a method (`examples/reproduction/<method>/`)
- [ ] Add a benchmark to the evaluation harness
- [ ] Report reproduction results
- [ ] Bug fix
- [ ] Documentation

## Checklist

<!-- Keep the items that apply and tick them. -->

**Paper list**

- [ ] Added to both `README.md` and `README_zh.md`, newest first: arXiv link, first author, venue (only if confirmed by the arXiv comments, OpenReview or the official repository), code link and a one-line key idea
- [ ] `python scripts/check_docs.py --fix` passes (it also updates the paper-count badges)

**Code**

- [ ] `make quality`, `make license` and `make test` pass
- [ ] New or changed training scripts pass `DRY_RUN=1 bash <script>`
- [ ] A new method or benchmark comes with unit tests and a README that states the setting and the differences from the official code

**Results**

- [ ] The command, commit, hardware, key package versions and the evaluation summary are included, and the numbers come from this repository's evaluation harness
