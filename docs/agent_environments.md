# Agent environments

Multi-turn agentic training (`worker.rollout.interaction_mode=agentic`) runs every trajectory in an
**agent environment**: an object that starts a task, executes the model's tool calls and reports how the
episode ended. The same environments serve training and evaluation
([eval/README.md](../eval/README.md#agent-environments)). Built in:

| Name | What it is |
| --- | --- |
| `deepeyes` (also when `agent_environment` is null) | DeepEyes' zoom-in task, in process: the data row is the task (its prompt and images), `image_zoom_in_tool` crops a source image ([DeepEyes](../examples/reproduction/deepeyes/README.md)) |
| `remote_http` | any environment served over HTTP with the protocol below, e.g. one written in another project |
| `openreward` | an environment of the Open Reward Standard, on openreward.ai or self-hosted ([below](#openreward-environments)) |
| `path/to/file.py:ClassName` | your own in-process class |

Examples in [examples/agent_env/](../examples/agent_env/README.md): a toy visual-search environment
(find the one digit among small letters with zoom-ins; its reward comes from the environment), the same task as
an OpenReward environment, and DeepEyes served over HTTP (the same prompts and observations as the in-process environment, token for
token, with DeepEyes' reward function on the trainer).

## How a trajectory runs

1. `reset(config, seed)` gives the task: the chat messages (system and user, with image placeholders), the
   images and the tool schemas. `config` is the row's environment configuration; `seed` is the same for all
   rollouts of a prompt and for their retries, so a group starts from one task state.
2. The model writes a turn. A well-formed `<tool_call>` goes to `step(call)`, whose observation (text and
   images) is appended as a tool message; a malformed call gets an error observation from the agent loop and
   never reaches the environment. A turn may hold several calls (below). `tool_call_errors` counts every error
   result: malformed calls that count towards the limit (`malformed_tool_calls`; the environment receives
   `tool_call_attempts` − `malformed_tool_calls` calls), calls the loop did not run because of the per-turn limit
   or the remaining calls (counted in `unrun_tool_calls`; a malformed one among them is unrun, not malformed),
   and calls the environment answers with `success: false`. `unrun_tool_calls` also counts the calls within the
   per-turn limit after one that ended the episode, which get no result and are not in `tool_call_errors`. Calls
   written in the answer-only turn after the limit are in neither.
3. The episode ends when the model answers in `<answer>...</answer>`, when it uses up the tool-call limit
   (`worker.rollout.agent_max_tool_calls`; after it, one more turn may only answer), when a length budget
   runs out, or when a step returns `done`.
4. `finalize(final_answer, status)` reports the episode reward with its components and metrics, and any
   details (or nothing); `close()` releases the episode.

Several calls per turn. The first `worker.rollout.agent_max_tool_calls_per_turn` (default 1) calls of a turn run,
one after another in the order written; each counts towards `agent_max_tool_calls`, a malformed one too. Every
other call gets an error result and does not count: one beyond the per-turn limit ("not run: at most N tool calls
run per turn"), or one beyond the trajectory's remaining calls ("... used up", after which one more turn may only
answer). The turn's results come back together, as consecutive tool messages, which Qwen's chat template joins
into one turn. When a call returns `done`, the calls after it do not run. Like verl's
`multi_turn.max_parallel_calls` (default 1), the limit keeps the first calls; verl runs them concurrently and drops
the others without a result, while here they run in order (environments have state, and `/step` carries an index)
and every call the model reads about gets a result, as OpenAI and Anthropic require of a conversation with tool
calls (calls within the per-turn limit after one that ended the episode get none, since no turn follows). The model is not
told the limit by the API or the chat template (Qwen's tool preamble reads "You may call one or more functions"):
the environment receives it in `reset` (`EnvironmentContext.max_tool_calls_per_turn`, `/reset`
`max_tool_calls_per_turn`) and writes it into its prompt. DeepEyes supports 1 only.

Calls inside the reasoning. When a turn has calls after its reasoning (`<think>...</think>`), the blocks inside the
reasoning are part of it, a draft the model wrote while thinking: only the calls after it are the turn's calls, as
an OpenAI-compatible vLLM server with a reasoning parser reads tool calls from the content only. When every call is
inside the reasoning, as when the model has not closed it, they are the turn's calls (DeepEyes' environment runs
such a call too). A `<think>` or `</think>` inside a call's JSON (in an argument) is not a reasoning tag. The reward
function logs the blocks inside the reasoning per trajectory as `tool_calls_inside_reasoning`.

Rendering. Tool schemas are rendered by the model's chat template (`tools=`), as in DeepEyes' native prompt.
The environment's messages and observation texts are rendered verbatim, so they must not contain the
tokenizer's special tokens (such as `<|im_start|>` or `<|image_pad|>`); such text is rejected. The model's
own tokens are never re-encoded.

## Rules of this repository

Some rules of the agent loop and the environment interface are this repository's own, where frameworks differ or
no common practice was found. For comparison, verl v0.9.1 (`verl/experimental/agent_loop/`) and verl-agent (commit
20bd331):

| Rule here | Elsewhere |
| --- | --- |
| The limit counts tool calls (`agent_max_tool_calls`), a malformed one too; after it, one more turn may only answer, and a call in it ends the episode (`tool_call_limit`). | verl ends a trajectory at `multi_turn.max_assistant_turns` or `max_user_turns`, counted in turns, with no turn reserved for the answer. |
| Tool calls inside the reasoning are not run when calls follow it; when all are inside, they run. | verl's Hermes parser reads every `<tool_call>` block of the text; vLLM's server with a reasoning parser reads the content after the reasoning only. |
| `/step` tells the environment how many calls counted before this one (`tool_calls_used`), since malformed calls count but never reach it. | In gym-style interfaces (`step(action)`) the environment counts its own steps. |
| A malformed call gets an error result from the loop and never reaches the environment. | verl's Hermes parser drops a block it cannot decode, and a turn without a decodable call ends the trajectory; verl-agent gives the text to the environment's projection, which marks the step invalid. |
| Several calls of a turn run one after another, and every call gets a result, except those within the per-turn limit after one that ended the episode (above). | verl runs the first `max_parallel_calls` concurrently and drops the others. |
| `reward: null` from `finalize` means that the environment does not score the episode; the environment reward function and scorer then fail rather than count 0. | verl's reward functions return a number, or a dict with its `score`. |
| The environment's components and metrics are logged as `env/<name>`; a name reported by some samples only is averaged over them, with a `*_count` entry. | verl's reward managers collect the extra values of each key in a list (`reward_manager/naive.py`), whose lengths differ when samples return different keys. |
| The `openreward` environment adds a system message with the loop's rules (the tool-call limits, the answer inside `<answer></answer>`), since the environment writes only the task. | ROLL's OpenReward environment adds no system message (the chat template's tool preamble only); SkyRL writes the tool descriptions into one; TRL adds none and does not tell the model the limit. |
| An OpenReward session that ends with an error (a tool that raised, the network after the SDK's retries) fails the training step; a call the environment refuses (unknown tool, invalid input) is a result the model reads. | TRL, ROLL and SkyRL give the model the error text as the tool's result and go on (TRL then leaves a rollout without any reward out of the group's baseline; ROLL and SkyRL count 0). |
| The rewards of an OpenReward episode add up (each tool's, then the terminal tool's); an episode that ends unfinished gets 0. | ROLL sums them by default (mean, max or min optional), SkyRL returns each step's; TRL takes the last non-null one and gives `None` (no score) when there is none. |

## Training configuration

| Option (`worker.rollout.*`) | Default | Meaning |
| --- | --- | --- |
| `agent_environment` | `null` (= `deepeyes`) | registered name or `path/to/file.py:ClassName` |
| `agent_environment_kwargs` | `{}` | options of the environment's constructor, e.g. `agent_environment_kwargs.url=http://127.0.0.1:8000` |
| `agent_environment_config_key` | `env_config` | data column with each row's configuration, a JSON object (or its string) given to `reset`; not used by `deepeyes` |
| `agent_max_tool_calls` | `6` | tool calls per trajectory; the environment receives it in `/reset` (`max_tool_calls`) |
| `agent_max_tool_calls_per_turn` | `1` | tool calls run per model turn, in the order written; the environment receives it in `/reset` (`max_tool_calls_per_turn`) |
| `record_rollout_log_probs` | `null` (off) | keep vLLM's log-probability of each sampled token (after temperature, `logprobs_mode=processed_logprobs`) as `rollout_log_probs` and log how far it is from the trainer's recomputed `old_log_probs` under `rollout_corr/` (`kl`, `k3_kl`, `training_ppl`, `rollout_ppl`, `log_ppl_diff`, `ppl_ratio`, `chi2_token`, `chi2_seq`, as verl defines them); diagnostics only, the loss does not use them |

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
reward function also logs the agent loop's counts per trajectory: `tool_call_attempts`, `tool_call_errors`,
`malformed_tool_calls`, `unrun_tool_calls`, `tool_calls_inside_reasoning` and `trajectory_retries`. The
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
`initial_state_key` (when `reset` named the initial state), `env_steps` (per tool call: `turn_index`,
`call_index`, `tool`, `state_key`, `cost`, `reward`, `done`, `info`, for the calls where the environment gave any of
them; `call_index` counts the turn's tool-call blocks in the order written, malformed and unrun ones included, so it
is not the environment's `step_index`; blocks inside the reasoning that gave way to calls after it are not counted)
and `env_outcome` (`reward`, `components`, `metrics`, `details`, `episode_id`). Reward functions receive the reward,
components and metrics only.

`remote_http` options: `url`; `timeout_s` (60) per request; `retries` (3) after a network error, a timeout,
HTTP 408, 429, 500, 502, 503 or 504, with exponential backoff from `backoff_s` (0.5); `max_concurrency` (64) requests at a
time from one rollout worker; `max_images_per_step` (1), the most images one step returns, which sizes
`worker.rollout.limit_images` (it must hold the initial images plus `agent_max_tool_calls` steps of them);
`trust_env` (false): proxy variables such as `HTTP_PROXY` or `ALL_PROXY` are ignored unless it is true,
since an environment server usually runs on the same machine or cluster. After the retries a failing
request fails the training step, which can resume from the last checkpoint; the error names the episode and
the step.

The batched rollout runs every trajectory on each rank of a tensor-parallel group, so the environments whose
episodes live on a server (`remote_http`, `openreward`; a class of your own sets `remote = True`) need
`worker.rollout.tensor_parallel_size=1` (the server mode below runs every trajectory once). While a trajectory waits for the server, the other trajectories'
turns go to vLLM in smaller batches; DeepEyes over HTTP took more generation time than the
in-process environment (1.3 to 1.5 times) in a 2-step test.

## Server mode

`worker.rollout.agent_engine_mode=server` (default `batch`) serves agentic rollouts from a vLLM server on each
training GPU, as verl's colocated agent loop (v0.9.1, `verl/experimental/agent_loop/` and
`verl/workers/rollout/vllm_rollout/vllm_async_server.py`), instead of the batch mode's engine inside each training
process:

- **Servers.** Each training process gets a server on its GPU: a Ray actor holding vLLM's `AsyncLLM`, whose engine
  and GPU worker run in their own processes. vLLM schedules each model turn as it arrives (continuous batching),
  where the batch mode sends a rank's waiting turns to vLLM together and the next ones wait for that batch to finish.
- **Agent loops.** CPU workers (`agent_num_workers`, one per GPU by default) run the trajectories as coroutines. A
  router sends every turn of a trajectory to the server of its first turn, which holds its context in the prefix
  cache, and a new trajectory to the server with the fewest requests in flight (ties at random), as verl's router
  does. A worker takes whole groups, so the rollouts of a row share their environment context and images as in the
  batch mode; options such as `remote_http`'s `max_concurrency` hold per worker as they hold per GPU there.
- **Weights.** Training stays synchronous. Before each rollout the training process wakes its server's weights, sends
  its full weights over CUDA IPC in buckets (`server_weight_bucket_mb`, verl's bucketed transfer), and wakes the KV
  cache once every rank is done; after the rollout the server sleeps (`server_sleep_level`; 2, verl's level, discards
  the weights as each step sends them anyway) and training gets the memory back. The first sync checks that every
  parameter of the rollout model was loaded. A server that fails while receiving, or stops answering (10 minutes),
  fails the training step with its own error.
- **Retries and failures.** A trajectory that materialization rejects and may be retried is generated again at once
  in its own coroutine, with the batch mode's seeds; the batch mode retries after the whole batch. When a trajectory
  fails, the others are cancelled (their requests are aborted on the servers) and close their environments before
  the error is raised, as in the batch mode. A failed cleanup (closing the environments, releasing a turn on the
  router) fails the rollout too, unless another error is already being raised: that one is raised, and the cleanup's
  is logged.

The two modes build everything around the generation with the same code: the requests (sampling parameters, the
per-turn seeds, the images and their multimodal ids), the environment calls, the reward inputs and the training
batch. With a scripted engine that answers by the request seed they give the same batch. With vLLM they sample
differently, as vLLM's kernels do not give bit-identical logits for different batch compositions (two batch-mode
runs with different batch sizes differ too). The server mode records the engine's log-probability of each sampled
token by default (`record_rollout_log_probs`) and logs how far it is from the trainer's under `rollout_corr/`.

| Option (`worker.rollout.*`) | Default | Meaning |
| --- | --- | --- |
| `agent_engine_mode` | `batch` | `server` for this mode |
| `server_sleep_level` | `2` | vLLM's sleep level between rollouts (1 keeps a copy of the weights in host memory) |
| `server_weight_bucket_mb` | `2048` | size of the weight-sync buckets |
| `server_actor_max_concurrency` | `null` | calls a server actor takes at once; null: max(1024, trajectories of a batch + 64) |
| `max_num_seqs` | `null` | vLLM's limit of sequences scheduled at once, in both modes; null: vLLM's default |
| `agent_num_workers` | `null` | CPU workers that run the agent loops; null: one per GPU |
| `agent_group_affinity` | `false` | send a group's new trajectories to the server of its first one while that server has at most `max_num_seqs // 4` (256 when null) more requests in flight than the least busy one, for the prompt's prefix cache; no framework found pins a group's multi-turn trajectories to one server (ROLL pins a prompt's samples in its single-turn pipeline) |

The server mode needs `interaction_mode=agentic`, `tensor_parallel_size=1` and `agent_max_batch_images=0` (a limit
of the batch mode's batches), and does not support LoRA, `adv_estimator=remax` or Qwen3.5 yet.
`scripts/agent/gpu_utilization.py` measures the GPU utilization of each step's generation, to compare the modes on
a workload.

## HTTP protocol

JSON bodies, `POST`, HTTP 200 on success. Images are PNG, base64-encoded.

| Endpoint | Request | Response |
| --- | --- | --- |
| `/reset` | `{"config": {...}, "seed": int, "max_tool_calls": int, "max_tool_calls_per_turn": int}` | `{"episode_id": str, "system_prompt": str or null, "user_content": [part, ...], "tool_schemas": [schema, ...], "state_key": str or null}` (`state_key` optional) |
| `/step` | `{"episode_id": str, "step_index": int, "tool_calls_used": int, "tool_call": {"name": str, "arguments": {...}}}` | `{"content": [part, ...], "success": bool, "error_code": str (when not success), "done": bool, "reward": finite float or null, "state_key": str or null, "cost": float, "info": {...}}` |
| `/finalize` | `{"episode_id": str, "final_answer": str or null, "status": str}` | `{"reward": float or null, "components": {name: float}, "metrics": {name: float}, "details": {...}}` (`details` optional) |
| `/close` | `{"episode_id": str}` | `{}` |

A part is `{"type": "text", "text": str}` or `{"type": "image", "png_base64": str}`, in the order the model
sees them. `tool_schemas` are OpenAI function schemas (`{"type": "function", "function": {"name": ..., ...}}`)
and must be present; an empty list means a single turn without tools.

- `seed` is shared by the rollouts of a prompt and by their retries; the same `(config, seed)` must give the
  same initial messages, images, tools and task state, including anything sampled per episode.
- `max_tool_calls` is the trainer's limit; write it into the prompt and the remaining-calls notes instead of
  keeping a second limit (the calls left come from `tool_calls_used` below). `max_tool_calls_per_turn` is the
  number of calls run per turn; write it into the prompt too. A server should ignore request fields it does not
  know, so that later fields do not break it.
- `step_index` counts the calls the episode received, from 0. A repeated `(episode_id, step_index)` must
  return the first response without executing the call again (the client retries requests whose response
  was lost); an index that skips ahead or goes back is an error. When a call fails inside the server after
  it may have changed the episode, the episode should take no more calls: the reference server answers the
  retry with HTTP 409 instead of running the call again, which fails the training step.
- `tool_calls_used` counts the trajectory's calls before this one that counted towards `max_tool_calls`,
  malformed ones included, which never reach the server; so it can be above `step_index`. After this call,
  `max_tool_calls − tool_calls_used − 1` calls are left: use this, not `step_index`, for a remaining-calls note.
- A call the environment refuses (an unknown tool, invalid arguments, a budget used up) is an ordinary answer:
  HTTP 200 with `success: false` and the reason in `content`, which the model reads. An error status is for the
  server's own failures and fails the training step (after the client's retries, for the statuses above): the
  reference server answers an exception raised by `Episode.step` with 400 (`KeyError`, `TypeError`, `ValueError`)
  or 500, so an environment should not signal a bad call by raising.
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
- A step's `reward` is what the call earned, a finite number or null (anything else fails the training step);
  it counts in the episode's return (see Rewards above), so the `finalize` reward should not include it again.
  `cost` and `info` are recorded for diagnostics.
- `state_key` names the state after the call, the one the next action is taken in (after a turn with several
  calls, the state after its last call that ran), and `/reset` may name the initial state: GiGPO with
  `algorithm.step_state_key=environment` groups the actions of a prompt's rollouts taken in the same state
  (below). The same state must get the same key in every rollout of the prompt: a key holding something that
  differs between episodes, such as the `episode_id` or a call counter, matches no other, and its turns get no
  step advantage. Whether a count such as the remaining calls belongs in the key is the environment's choice
  (verl-agent's anchors hold the observation only, without the step). A call the environment rejects without
  changing its state should return the unchanged key; `null` from `/step` leaves the state unnamed, and the next
  action is then compared with no other. When `/reset` names no initial state, the first actions of a prompt's
  rollouts are still compared with each other, since they start from the same task. A malformed call does not reach the environment and leaves the state as it was.

[examples/agent_env/server.py](../examples/agent_env/server.py) implements the server side of these rules on
the Python standard library: write an `Episode` class with `reset`, `step` and `finalize`, and call `serve`. It
drops idle episodes when a later request arrives, never one with a request in progress. An `Episode.reset` that
takes a `max_tool_calls_per_turn` keyword gets the limit; for one that does not, the server refuses a limit other
than 1. An `Episode.step` that takes a `tool_calls_used` keyword gets it.

## OpenReward environments

[OpenReward](https://openreward.ai) environments follow the [Open Reward Standard](https://openrewardstandard.io) (ORS),
an HTTP and server-sent-events protocol: an environment has splits of tasks, and a session of one task gives the task's
prompt (text and images), its tools (names, descriptions and JSON schemas) and, for each call, a result with an
optional reward and whether the task is finished. `agent_environment=openreward` runs every trajectory in its own
session through the official SDK, as TRL (`trl.experimental.openreward`), ROLL and SkyRL do; the SDK takes care of
creating sessions, keep-alive pings, resumed event streams and secrets. It is an optional dependency, imported when
an episode starts: `pip install openreward` in the training environment (tested with 0.1.167; it pins `sse-starlette==2.3.6`, a
downgrade of a package that only `mcp` uses in this repository's environment).

| Option (`agent_environment_kwargs.*`) | Default | Meaning |
| --- | --- | --- |
| `environment` | `null` | the environment: `owner/Name` on openreward.ai, or its name on a self-hosted server (a row may name another) |
| `split` | `train` | the split of the rows that give only an `index` |
| `url` | `null` | a self-hosted ORS server (the SDK's `Server`); otherwise openreward.ai, with `OPENREWARD_API_KEY` from the environment variables |
| `max_sessions` | `64` | sessions open at a time per rollout worker, each from `reset` to `close` (in server mode, per agent worker) |
| `max_images_per_step` | `1` | as for `remote_http`: the most images one call returns, which sizes `worker.rollout.limit_images` |
| `system_prompt` | `null` | the system message with the loop's rules (below); a template with `{max_tool_calls}` and `{max_tool_calls_per_turn}`, or `""` for none |
| `secrets_from_env` | `[]` | names of environment variables whose values the session gets as secrets (an environment that calls a judge model, say); the values never enter the configuration or the logs |

Each data row's configuration names the task: `{"index": 17}`, with `environment`, `split` or `variant` (one
environment of a server that serves several) when they differ from the options, or `{"task_spec": {...}}` for a task
that the environment's lists do not hold. `scripts/agent/prepare_openreward_tasks.py` writes such rows for the tasks
of a split (asking the server only for their number). An episode:

- The session's prompt is the user message, in order (images included); the tools go to the chat template as OpenAI
  function schemas. The environment writes only the task, so a system message adds the loop's rules: the tool-call
  limit and the per-turn limit, then "When you are done, give your final answer inside `<answer></answer>`, for
  example `<answer>42</answer>`" for an environment with a terminal tool (without the example, Qwen3-VL-4B left the tags
  out on 16 of the toy search's 64 validation tasks, with it on 1), else "The task ends when a tool says that it is
  finished". The reset `seed` is
  not used: an ORS task is fixed by its index, and the rollouts of a group open sessions of the same task.
- A call's result is shown in order (text and images); its reward is the step's reward, which adds to the return
  (Rewards above); `finished` ends the episode (`environment_done`); its metadata is the step's `info` in the
  trajectory records. A call the environment refuses (`ToolCallError`: unknown tool, invalid input) is a result with
  `success: false` that the model reads, written as escaped JSON like the loop's own errors since it may echo the
  model's input. Anything else that goes wrong (a tool that raised, which ends the session, a network error after the
  SDK's retries, a result outside the standard) fails the training step, which can resume from the last checkpoint;
  the error names the task and the call. A call is never sent twice: the SDK resumes a call whose task the server
  has named, and a connection lost before that could only be retried by sending the call again, which may run the
  action twice, so it fails the step too (the adapter turns on the SDK's `resumable_only` for `/call`, which
  `call_tool` leaves off in 0.1.167).
- An environment's terminal tool (`@terminal`) is hidden from the model: the text the model answers inside
  `<answer></answer>` is passed to it, and its reward is the episode reward that `finalize` reports. Otherwise that
  reward is 0, since the tools' rewards are in the return already: an episode that ends unfinished (the call limit,
  the length budget, no action, or an answer to an environment without a terminal tool) gets 0. `finalize` reports
  the metric `finished` (logged as `env/finished`) and keeps the task, the session id and the terminal tool's text
  and metadata in `details`.

A self-hosted server: run the environment with the SDK's `Server` and give its address as `url`. The SDK follows the
proxy variables (`HTTP_PROXY`, ...), so a server on the same machine needs `NO_PROXY=127.0.0.1,localhost` when they
are set. Not supported yet: group scoring (`score_group`), uploading rollouts to OpenReward, conversation-shaped
prompts (`get_messages`) and built-in toolsets.

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
answer, more calls than a turn runs (as verl-agent's search environment flags several search tags), a wrong turn
end, a turn cut by a length limit, a turn without an action, a call in the answer-only turn after the tool-call
limit. Well-formed calls left unrun because the trajectory's remaining calls ran out do not make a turn invalid (a
malformed one still does). A turn's step
reward is the sum of its calls' rewards. `algorithm.invalid_action_penalty` is taken from its episode score and its
step return. A call that the environment rejects is a valid action. The logs report `gigpo/step_group_size_mean`,
`gigpo/singleton_step_group_fraction` (a group of one gives no step advantage), `gigpo/turns_in_shared_groups`
(the share of turns in groups of two or more; groups go by the state alone, so turns of one rollout in the same
state count too, for instance after a call that left the state as it was), `gigpo/turns_in_cross_rollout_groups`
(the share of turns in groups that hold turns of two or more rollouts, the ones compared with other rollouts),
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
are not reproduced here. A GiGPO run on the toy environment:
[examples/agent_env/qwen3_vl_4b_gigpo_toy_search.sh](../examples/agent_env/qwen3_vl_4b_gigpo_toy_search.sh).

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
`context` holds the processor, the batch's model-side image resize, the tool-call limits (`max_tool_calls`,
`max_tool_calls_per_turn`) and objects shared by the batch's environments. A `step` that takes a keyword argument
`tool_calls_used` gets the calls counted before this one, as in `/step`. Constructors must not acquire resources
(one instance is built when the rollout starts, to check the options). `LocalToolEnvironment` runs stateless `AgentTool`s; `DeepEyesEnvironment`
builds on it. Methods that never suspend (no awaiting I/O, threads or sleeps) leave the batched rollout's vLLM
batches as they were without environments; DeepEyes relies on this to give the same results as before the
environment interface. `EnvironmentChatAdapter` renders an environment's own prompt and observations;
DeepEyes keeps its own adapters. An observation encoder renders `ObservationEncodingRequest.results`, the results of
a turn's calls in order: a turn with several calls has several, also with a per-turn limit of 1.
