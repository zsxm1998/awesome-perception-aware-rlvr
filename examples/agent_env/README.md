# Agent environment examples

Environments served over HTTP for multi-turn agentic RL; the protocol and the options are in
[docs/agent_environments.md](../../docs/agent_environments.md).

| File | What it is |
| --- | --- |
| [`server.py`](server.py) | the server side of the protocol on the Python standard library: episode ids, `step_index` deduplication, idempotent `/finalize` and `/close`, dropping idle episodes. Write an `Episode` class and call `serve` |
| [`toy_search.py`](toy_search.py) | a toy visual-search environment: find the one digit among small letters on a 1024x1024 canvas, with a `zoom_in` tool; the environment gives the reward (1 for the right digit) and a `state_key` per zoom (zoom level and grid cell) |
| [`qwen3_vl_4b_toy_search.sh`](qwen3_vl_4b_toy_search.sh) | GRPO of Qwen3-VL-4B on it, with `examples/reward_function/environment.py` |
| [`deepeyes_http.py`](deepeyes_http.py) | DeepEyes' zoom-in task over HTTP: the same prompts, tool and observations as the in-process `deepeyes` environment, token for token (native prompt, 0-1000 boxes); the reward stays with DeepEyes' reward function |
| [`qwen3_vl_8b_deepeyes_http.sh`](qwen3_vl_8b_deepeyes_http.sh) | DeepEyes on Qwen3-VL-8B with the task served over HTTP |

```bash
# toy visual search
python examples/agent_env/toy_search.py tasks --out data/agent_env/toy_search
python examples/agent_env/toy_search.py serve --port 8000 &
bash examples/agent_env/qwen3_vl_4b_toy_search.sh

# DeepEyes over HTTP (after bash scripts/prepare_data.sh deepeyes)
python examples/agent_env/deepeyes_http.py tasks --data-dir data/deepeyes --out data/agent_env/deepeyes_http
python examples/agent_env/deepeyes_http.py serve --data-dir data/deepeyes --port 8001 &
bash examples/agent_env/qwen3_vl_8b_deepeyes_http.sh
```

`AGENT_ENV_URL` points a training script at another server. In a 2-step test of each script on 4 GPUs (8
prompts, 4 rollouts each), Qwen3-VL-4B answered about 90% of the toy tasks with 1.4 zoom-ins on average, and
DeepEyes over HTTP gave the same first-step accuracy as the in-process environment.
