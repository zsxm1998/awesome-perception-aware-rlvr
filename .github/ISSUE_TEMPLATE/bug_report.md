---
name: Bug report
about: Something does not work as documented
title: ''
labels: 'bug'
assignees: ''

---

**Describe the bug**
What happened and what you expected instead.

**To reproduce**
The exact command (script + overrides) and the commit hash.

**Logs**
The relevant part of the console output or `experiment_log.jsonl`.

**Environment**
 - OS / GPU / CUDA driver:
 - Installed via `scripts/install_env.sh`, Docker, or manually:
 - `python -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)"`:
