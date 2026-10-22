# Agent environment examples

Environments served over HTTP for multi-turn agentic RL; the protocol and the options are in
[docs/agent_environments.md](../../docs/agent_environments.md).

| File | What it is |
| --- | --- |
| [`server.py`](server.py) | the server side of the protocol on the Python standard library: episode ids, `step_index` deduplication, idempotent `/finalize` and `/close`, dropping idle episodes. Write an `Episode` class and call `serve` |
| [`toy_search.py`](toy_search.py) | a toy visual-search environment: find the one digit among small letters on a 1024x1024 canvas, with a `zoom_in` tool; the environment gives the reward (1 for the right digit) and a `state_key` per zoom (zoom level and grid cell) |
| [`qwen3_vl_4b_toy_search.sh`](qwen3_vl_4b_toy_search.sh) | GRPO of Qwen3-VL-4B on it, with `examples/reward_function/environment.py` |
| [`toy_search_policy.py`](toy_search_policy.py) | a scripted policy for it, to make SFT demonstrations: zoom into the four quarters in order and answer once a view shows the digit |
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

Evaluation in the same environment, with a local checkpoint or a model behind an OpenAI-compatible API
([eval/README.md](../../eval/README.md#agent-environments)); here Qwen3-VL-4B served by vLLM with its tool
parser stands in for an API model:

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8002 --served-model-name qwen3-vl-4b \
    --enable-auto-tool-choice --tool-call-parser hermes &
python eval/run_all_benchmarks.py --model qwen3-vl-4b --backend api --api-base-url http://127.0.0.1:8002/v1 \
    --interaction-mode agentic --agent-profile environment \
    --agent-environment-kwargs '{"url": "http://127.0.0.1:8000"}' --config my_benchmarks.yaml --benchmarks toy_search \
    --agent-max-tool-calls 4 --agent-max-response-tokens 8192 --agent-max-tokens-per-turn 2048
```

SFT demonstrations for a warm-up before RL ([docs/agent_environments.md](../../docs/agent_environments.md#sft-data)):
the scripted policy runs in the environment through the rollout's own loop and the result is written for
ms-swift, tokenized exactly as RL training tokenizes the same turns.

```bash
python scripts/agent/export_sft.py --model Qwen/Qwen3-VL-4B-Instruct \
    --environment remote_http --environment-kwargs '{"url": "http://127.0.0.1:8000"}' \
    --policy examples/agent_env/toy_search_policy.py:ToySearchPolicy --tasks data/agent_env/toy_search/train.parquet \
    --max-tool-calls 4 --max-response-tokens 8192 --max-tokens-per-turn 2048 \
    --min-pixels 200704 --max-pixels 1048576 --out data/agent_env/toy_search_sft
# in an environment with ms-swift:
python scripts/agent/check_swift_export.py --model Qwen/Qwen3-VL-4B-Instruct --export data/agent_env/toy_search_sft
# or demonstrations of a model behind an API, keeping its successes (here the vLLM server above):
python scripts/agent/export_sft.py --model Qwen/Qwen3-VL-4B-Instruct \
    --environment remote_http --environment-kwargs '{"url": "http://127.0.0.1:8000"}' \
    --api-model qwen3-vl-4b --api-base-url http://127.0.0.1:8002/v1 \
    --tasks data/agent_env/toy_search/train.parquet --samples-per-task 2 --temperature 1.0 --min-reward 1.0 \
    --max-tool-calls 4 --max-response-tokens 8192 --max-tokens-per-turn 2048 \
    --min-pixels 200704 --max-pixels 1048576 --out data/agent_env/toy_search_sft_api
```

`AGENT_ENV_URL` points a training script at another server. In a 2-step test of each script on 4 GPUs (8
prompts, 4 rollouts each), Qwen3-VL-4B answered about 90% of the toy tasks with 1.4 zoom-ins on average, and
DeepEyes over HTTP gave the same first-step accuracy as the in-process environment.

The scripted policy answered all 32 training tasks it was given, and ms-swift 4.6.0 (also development
versions of 4.3 and 3.12) encoded every exported sample with the rollout's tokens.

On 16 validation tasks, Qwen3-VL-4B solved all of them both as a local checkpoint (`--backend vllm`) and behind
vLLM's OpenAI-compatible server (`--backend api`, native tool calls, images resized as for local runs). Sent at
the environment's own size (`--api-images original`), the 1024x1024 canvas is larger than in local runs, and the
same model solved 13 of 16.
With Qwen3-VL-4B behind the same server as the demonstrating model (2 samples per task, temperature 1.0), the
16 episodes of 8 training tasks were all correct and written, and ms-swift 4.6.0 encoded every sample with the
rollout's tokens.
