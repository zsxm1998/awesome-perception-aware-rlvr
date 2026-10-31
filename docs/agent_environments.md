# Agent environments

Multi-turn agentic training (`worker.rollout.interaction_mode=agentic`) runs every trajectory in an
**agent environment**: an object that starts a task, executes the model's tool calls and reports how the
episode ended. The same environments serve training and evaluation
([eval/README.md](../eval/README.md#agent-environments)). Built in:

| Name | What it is |
| --- | --- |
| `deepeyes` (also when `agent_environment` is null) | DeepEyes' zoom-in task, in process: the data row is the task (its prompt and images), `image_zoom_in_tool` crops a source image ([DeepEyes](../examples/reproduction/deepeyes/README.md)) |
| `remote_http` | any environment served over HTTP with the protocol below, e.g. one written in another project |
| `path/to/file.py:ClassName` | your own in-process class |

Examples in [examples/agent_env/](../examples/agent_env/README.md): a toy visual-search environment
(find the one digit among small letters with zoom-ins; its reward comes from the environment) and
DeepEyes served over HTTP (the same prompts and observations as the in-process environment, token for
token, with DeepEyes' reward function on the trainer).

## How a trajectory runs

1. `reset(config, seed)` gives the task: the chat messages (system and user, with image placeholders), the
   images and the tool schemas. `config` is the row's environment configuration; `seed` is the same for all
   rollouts of a prompt and for their retries, so a group starts from one task state.
2. The model writes a turn. A well-formed `<tool_call>` goes to `step(call)`, whose observation (text and
   images) is appended as a tool message; a malformed call gets an error observation from the agent loop and
   never reaches the environment (it is counted as `tool_call_errors`).
3. The episode ends when the model answers in `<answer>...</answer>`, when it uses up the tool-call limit
   (`worker.rollout.agent_max_tool_calls`; after it, one more turn may only answer), when a length budget
   runs out, or when a step returns `done`.
4. `finalize(final_answer, status)` reports the episode reward with its components and metrics, and any
   details (or nothing); `close()` releases the episode.

Rendering. Tool schemas are rendered by the model's chat template (`tools=`), as in DeepEyes' native prompt.
The environment's messages and observation texts are rendered verbatim, so they must not contain the
tokenizer's special tokens (such as `<|im_start|>` or `<|image_pad|>`); such text is rejected. The model's
own tokens are never re-encoded.

## Training configuration

| Option (`worker.rollout.*`) | Default | Meaning |
| --- | --- | --- |
| `agent_environment` | `null` (= `deepeyes`) | registered name or `path/to/file.py:ClassName` |
| `agent_environment_kwargs` | `{}` | options of the environment's constructor, e.g. `agent_environment_kwargs.url=http://127.0.0.1:8000` |
| `agent_environment_config_key` | `env_config` | data column with each row's configuration, a JSON object (or its string) given to `reset`; not used by `deepeyes` |
| `agent_max_tool_calls` | `6` | tool calls per trajectory; the environment receives it in `/reset` (`max_tool_calls`) |

Data rows of an environment task need the configuration column, a prompt column (`data.prompt_key`; only
used for the length filter and logs, as the model's prompt comes from `reset`) and an answer column
(`data.answer_key`, required by the dataset class; the environment's own reward does not read it). They do
not need images.

Rewards: `worker.reward.reward_function=examples/reward_function/environment.py:compute_score` takes the
episode reward that `finalize` reports. A trajectory without it, or with `reward: null`, fails the step: null
means that the environment does not score its episodes, and a reward function that reads the answer has to
(DeepEyes over HTTP does this with DeepEyes' reward function). A step's `reward`, what a single tool call
earned, adds to the episode's return as in the usual RL definition (the return is the sum of the rewards): the
trainer puts it on the last model token of its turn, and the reward function's score is the reward of the last
turn. Every advantage estimator, the logs (`reward/overall`) and online filtering see the sum; the step rewards of
an episode are also logged as `reward/step_rewards`. Do not count them again in the `finalize` reward. The
environment's components and metrics are logged under `env/`, apart from the trainer's own names (`overall`,
`tool_call_errors`, ...):

| Where | Name |
| --- | --- |
| training step | `reward/env/<name>`, and `reward_count/env/<name>` |
| validation | `val/env/<name>_reward`, and `val/env/<name>_reward_count` |
| generation records (`trainer.train_generations_to_log`, `val_generations_to_log`) | `reward/env/<name>` per sample |

An episode may report only the names that apply to it. Each name is averaged over the samples that reported
it; the `*_count` entries give their number and appear only when some samples of the batch (or of the
validation set) did not report the name. The same holds for any reward function whose samples return
different keys. A name used per sample, `algorithm.filter_key` of online filtering or `accuracy`, must come
from every sample. A name given both as a component and as a metric is an error.

Trajectory records keep what the environment reported, as given: the generation records have an
`environment` field, and evaluation outputs (`agent_diagnostics`) and SFT exports the same entries:
`initial_state_key` (when `reset` named the initial state), `env_steps` (per tool call: `turn_index`, `tool`,
`state_key`, `cost`, `reward`, `done`, `info`, for the calls where the environment gave any of them) and
`env_outcome` (`reward`, `components`, `metrics`, `details`, `episode_id`). Reward functions receive the reward,
components and metrics only.

`remote_http` options: `url`; `timeout_s` (60) per request; `retries` (3) after a network error, a timeout,
HTTP 408, 429 or 5xx, with exponential backoff from `backoff_s` (0.5); `max_concurrency` (64) requests at a
time from one rollout worker; `max_images_per_step` (1), the most images one step returns, which sizes
`worker.rollout.limit_images` (it must hold the initial images plus `agent_max_tool_calls` steps of them);
`trust_env` (false): proxy variables such as `HTTP_PROXY` or `ALL_PROXY` are ignored unless it is true,
since an environment server usually runs on the same machine or cluster. After the retries a failing
request fails the training step, which can resume from the last checkpoint; the error names the episode and
the step.

The batched rollout runs every trajectory on each rank of a tensor-parallel group, so `remote_http` needs
`worker.rollout.tensor_parallel_size=1`. While a trajectory waits for the server, the other trajectories'
turns go to vLLM in smaller batches; DeepEyes over HTTP took more generation time than the
in-process environment (1.3 to 1.5 times) in a 2-step test.

## HTTP protocol

JSON bodies, `POST`, HTTP 200 on success. Images are PNG, base64-encoded.

| Endpoint | Request | Response |
| --- | --- | --- |
| `/reset` | `{"config": {...}, "seed": int, "max_tool_calls": int}` | `{"episode_id": str, "system_prompt": str or null, "user_content": [part, ...], "tool_schemas": [schema, ...], "state_key": str or null}` (`state_key` optional) |
| `/step` | `{"episode_id": str, "step_index": int, "tool_call": {"name": str, "arguments": {...}}}` | `{"content": [part, ...], "success": bool, "error_code": str (when not success), "done": bool, "reward": float or null, "state_key": str or null, "cost": float, "info": {...}}` |
| `/finalize` | `{"episode_id": str, "final_answer": str or null, "status": str}` | `{"reward": float or null, "components": {name: float}, "metrics": {name: float}, "details": {...}}` (`details` optional) |
| `/close` | `{"episode_id": str}` | `{}` |

A part is `{"type": "text", "text": str}` or `{"type": "image", "png_base64": str}`, in the order the model
sees them. `tool_schemas` are OpenAI function schemas (`{"type": "function", "function": {"name": ..., ...}}`)
and must be present; an empty list means a single turn without tools.

- `seed` is shared by the rollouts of a prompt and by their retries; the same `(config, seed)` must give the
  same initial messages, images, tools and task state, including anything sampled per episode.
- `max_tool_calls` is the trainer's limit; write it into the prompt and the remaining-calls notes instead of
  keeping a second limit.
- `step_index` counts the calls the episode received, from 0. A repeated `(episode_id, step_index)` must
  return the first response without executing the call again (the client retries requests whose response
  was lost); an index that skips ahead or goes back is an error. When a call fails inside the server after
  it may have changed the episode, the episode should take no more calls: the reference server answers the
  retry with HTTP 409 instead of running the call again, which fails the training step.
- `done: true` ends the episode after this call; its content is not shown to the model, and `finalize` is
  still called with status `environment_done`.
- `final_answer` is the text inside the last `<answer>...</answer>` outside the reasoning, stripped; it is
  null unless `status` is `answered`. The other statuses: `invalid_final_answer`, `invalid_turn_end`,
  `no_action`, `tool_call_limit`, `length_limit`, `environment_done`. The environment parses the answer and
  scores the episode.
- `reward`, `components` and `metrics` are finite numbers; `reward: null` means that the environment does
  not score the episode (see Rewards above). `details` is any JSON object (text included), kept verbatim in the
  trajectory records with the `episode_id`, for matching them with the server's own logs; it is not checked
  or aggregated.
- `/finalize` called again returns the same result; `/close` of an unknown or closed episode returns `{}`.
  A server should drop episodes that stay idle for long: a crashed trainer never closes them, and a retried
  `/reset` leaves one behind.
- A step's `reward` is what the call earned; it counts in the episode's return (see Rewards above), so the
  `finalize` reward should not include it again. `cost` and `info` are recorded for diagnostics.
- `state_key` names the state after the call, the one the next action is taken in, and `/reset` may name the
  initial state: GiGPO with `algorithm.step_state_key=environment` groups the actions of a prompt's rollouts
  taken in the same state (below). A call the environment rejects without changing its state should return
  the unchanged key; `null` leaves the state unnamed, and the next action is then compared with no other. A
  malformed call does not reach the environment and leaves the state as it was.

[examples/agent_env/server.py](../examples/agent_env/server.py) implements the server side of these rules on
the Python standard library: write an `Episode` class with `reset`, `step` and `finalize`, and call `serve`. It
drops idle episodes when a later request arrives, never one with a request in progress.

## Step-level advantages (GiGPO)

`algorithm.adv_estimator=gigpo` gives each turn (one model reply) its own advantage, as GiGPO
([Feng et al., NeurIPS 2025](https://arxiv.org/abs/2505.10978)): A = A_E + w * A_S. A_E normalizes the episode's
return within the rollouts of a prompt, as GRPO does; A_S normalizes the turn's discounted return
(`algorithm.gamma`) within the turns of that prompt's rollouts that acted in the same state. The arithmetic is
that of verl-agent's `gigpo/core_gigpo.py` (commit 20bd331): it matched value for value on random inputs. The
options are listed in [algorithm_parameters.md](algorithm_parameters.md#step-level-advantages-gigpo).

Which turns act in the same state (`algorithm.step_state_key`):

- `observation` (default, as in verl-agent): the observation before the turn is the same, tokens and image pixels
  as the model receives them, after resizing to the pixel bounds (the last tool result; the prompt for the first
  turn). With `algorithm.step_similarity_threshold`, text-only
  observations whose `difflib.SequenceMatcher` ratio reaches the threshold are also grouped (verl-agent's
  `enable_similarity`, 0.9 in its search example; OR-GiGPO, [arXiv 2606.12830](https://arxiv.org/abs/2606.12830),
  uses it for tools whose outputs vary slightly between equal calls). Images are matched exactly.
- `environment`: the environment names the states (`state_key` above). This suits tasks whose observations rarely
  repeat exactly, crops of a large image for instance, while the environment knows which states are equivalent
  (a quantized view, the regions selected so far).

A turn's action is invalid when it is malformed, as verl-agent judges the format only: a malformed tool call or
answer, a wrong turn end, a turn cut by a length limit, a turn without an action, a call after the tool-call
limit. `algorithm.invalid_action_penalty` is taken from its episode score and its step return. A call that the
environment rejects is a valid action. The logs report `gigpo/step_group_size_mean`,
`gigpo/singleton_step_group_fraction` (a group of one gives no step advantage), `gigpo/turns_in_shared_groups`,
`gigpo/turns_per_trajectory`, `gigpo/invalid_turn_fraction`, `gigpo/episode_advantage_abs` and
`gigpo/step_advantage_abs`.

verl-agent trains one row per step whose prompt holds the task, the current observation and the last few steps as
text (two in its ALFWorld and WebShop scripts, without past reasoning or images). Here a trajectory is one row
with every turn, as the model sees it at inference. For the same steps the advantages are the same, and the
training differs in:

1. the context of the policy: the whole history here, a window of past steps in verl-agent;
2. the mini-batch unit: trajectories here, steps in verl-agent (GiGPO requires `worker.actor.loss_avg_mode=token`,
   under which every model token of a mini-batch weighs the same in both);
3. length: a whole trajectory must fit `data.max_response_length`, so tasks of tens of steps (verl-agent's
   ALFWorld runs up to 50) do not fit;
4. states: a state is grouped by its last observation while the policy sees the whole history, so turns with
   different histories are compared (HGPO, [arXiv 2602.22817](https://arxiv.org/abs/2602.22817), discusses this
   for verl-agent's window as well).

verl-agent also copies random rows to pad the batch before computing the statistics; nothing is copied here. Its
similarity grouping follows the length-balanced batch order; here it follows each rollout's index among the
rollouts of its prompt, then the turn, whatever the batch order or its split across workers. verl-agent's results
are not reproduced here.

## Evaluation and API models

`--agent-profile environment` evaluates a checkpoint in an environment, and `--backend api` a model behind an
OpenAI-compatible API (a commercial model, or a large open model served by vLLM) on the same tasks, tools,
limits and scorer: [eval/README.md](../eval/README.md#agent-environments).

## SFT data

A warm-up by supervised fine-tuning before RL needs demonstrations in the format RL will train on.
[scripts/agent/export_sft.py](../scripts/agent/export_sft.py) replays scripted demonstrations in an environment
through the rollout's own agent loop, chat adapter, budgets and image handling
(`verl/workers/agent/replay.py`), and writes them as [ms-swift](https://github.com/modelscope/ms-swift) data.
The turns come from a scripted policy over a task file (`--policy path/to/file.py:ClassName --tasks`), which sees
each step's result and may decide on it, from a model behind an API (`--api-model`, below), or from recorded
episodes (`--episodes`):

```python
class MyPolicy:
    def start(self, env_config, seed, reset):  # optional; reset holds the messages, images and tool schemas
        ...

    def act(self, step):  # step: the EnvStep of the previous tool call, None on the first turn
        return {
            "content": "The tumor border is unclear; I will zoom in.",
            "tool_call": {"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}},
        }
        # or {"content": "... <answer>...</answer>"}, or the raw assistant text as a string
```

Demonstrations of another model (distillation, or rejection sampling with several samples per task): with
`--api-model`, a model behind any OpenAI-compatible endpoint writes the turns, a local vLLM server or a cloud
API (GLM, Gemini, Qwen on Model Studio, GPT, ...), with the `--api-*` options of `--backend api` evaluation
([eval/README.md](../eval/README.md#api-models-in-agent-environments)). Its native tool calls are rendered in the
trained model's own format, and each episode still runs once, live, through the rollout's loop:

```bash
export GLM_API_KEY=...
python scripts/agent/export_sft.py --model Qwen/Qwen3-VL-4B-Instruct \
    --environment remote_http --environment-kwargs '{"url": "http://127.0.0.1:8000"}' \
    --api-model glm-4.5v --api-base-url https://open.bigmodel.cn/api/paas/v4 --api-key-env GLM_API_KEY \
    --tasks data/agent_env/toy_search/train.parquet --samples-per-task 4 --temperature 1.0 --min-reward 1.0 \
    --max-tool-calls 4 --max-response-tokens 8192 --max-tokens-per-turn 2048 \
    --min-pixels 200704 --max-pixels 1048576 --out data/agent_env/toy_search_sft_glm
```

The samples of a task share its environment seed (one task state) and differ in the API's sampling;
`--min-reward` keeps the successful ones. The model is sent the images resized as the trained model will see
them, so that its replies refer only to what the trained model can see. A reply the rollout could not take as it
is (cut off at `--max-tokens-per-turn`, a malformed or second tool call, more tokens than the budget in the trained
model's tokenizer) leaves its episode out and is counted in `export_summary.json`, with the API's token usage.
Reasoning that an API returns apart from its reply is not kept. Whether a provider's outputs may be used to train
other models is set by its terms of service; check them first.

A structured turn is rendered into the model's native tool-call text by its chat template, then tokenized as
the rollout would tokenize a model that wrote it. A turn that the rollout could not have produced (a malformed
call, more tokens than the per-turn or trajectory budget) is reported and its episode left out; by default only
answered episodes are written (`--keep all`, `--min-reward`). Give the model, chat template, budgets and pixel
bounds of the RL run.

The output (`--out`): `train.jsonl` for `swift sft --dataset`, with system, user, assistant and tool messages,
the tool definitions (`tools`), `<image>` tags and absolute image paths; `images/`; `rollout_tokens.jsonl`, the rollout's tokens, loss mask and
image grids with what the environment reported; `export_summary.json`. Three details make ms-swift encode a
sample exactly as the rollout did:

- the tool definitions are ms-swift's `tools` field (a JSON string, kept as written by its dataset loader) and the
  system message is the environment's own; ms-swift renders the tool descriptions after it as the chat template
  does. Without a system message, ms-swift would put two extra newlines before them, so such a sample carries the
  system block the chat template rendered, tool descriptions included, instead of a `tools` field;
- tool results use the `tool` role, which ms-swift's hermes agent template (the default for Qwen models) wraps in
  the same `<tool_response>` user turn as the chat template;
- images are saved at the size the model's image processor resizes them to, resized the same way, so that
  neither side resizes them again (the export checks that the saved image gives the processor's input).

[scripts/agent/check_swift_export.py](../scripts/agent/check_swift_export.py) proves it for a model: it runs in
an environment with ms-swift (it does not import this repository; the export runs it itself when ms-swift is
importable) and encodes each sample with ms-swift's own template in training mode, then compares the token ids,
the trained tokens and each image's grid and pixel values with the rollout. The one difference it allows is
ms-swift's own: it also trains the newline after each `<|im_end|>` that ends an assistant turn, which the model
never generates, since generation stops at `<|im_end|>`. For Qwen3-VL-4B-Instruct, every sample of the toy
example matched with ms-swift 4.6.0, as did samples with images that both sides had to resize. One setting to
keep the same on both sides: when training tokenizes `<think>` as plain text
(`worker.actor.model.plain_think_tokens`, applied by default to Qwen3-VL Instruct tokenizers that have it as an
added token) and ms-swift loads the released tokenizer, demonstrations with `<think>` tags are tokenized
differently, and the check reports it.

## In-process environments

A class with `async reset / step / finalize / close` and `chat_adapter` (see `AgentEnvironment` in
`verl/workers/agent/protocol.py`), built as `cls(context, **agent_environment_kwargs)` once per trajectory.
`context` holds the processor, the batch's model-side image resize, the tool-call limit and objects shared by
the batch's environments. Constructors must not acquire resources (one instance is built when the rollout
starts, to check the options). `LocalToolEnvironment` runs stateless `AgentTool`s; `DeepEyesEnvironment`
builds on it. Methods that never suspend (no awaiting I/O, threads or sleeps) leave the batched rollout's vLLM
batches as they were without environments; DeepEyes relies on this to give the same results as before the
environment interface. `EnvironmentChatAdapter` renders an environment's own prompt and observations;
DeepEyes keeps its own adapters.
