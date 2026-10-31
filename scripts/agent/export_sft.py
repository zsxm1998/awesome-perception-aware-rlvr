#!/usr/bin/env python3
# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay scripted demonstrations in an agent environment and write them as ms-swift SFT data.

Each episode runs through the same agent loop, chat adapter, budgets and image handling as an RL rollout
(verl/workers/agent/replay.py), so the SFT data is tokenized exactly as RL training would tokenize the same
actions. Give the turns either as a scripted policy over a task file, or recorded in an episodes file:

    python scripts/agent/export_sft.py --model Qwen/Qwen3-VL-4B-Instruct \\
        --environment remote_http --environment-kwargs '{"url": "http://127.0.0.1:8000"}' \\
        --policy examples/agent_env/toy_search_policy.py:ToySearchPolicy \\
        --tasks data/agent_env/toy_search/train.parquet \\
        --max-tool-calls 4 --max-response-tokens 8192 --max-tokens-per-turn 2048 \\
        --min-pixels 200704 --max-pixels 1048576 --out data/agent_env/toy_search_sft

    python scripts/agent/export_sft.py ... --episodes demos.jsonl --out ...
        # one line per episode: {"env_config": {...}, "turns": [...], "id": optional, "seed": optional}

    python scripts/agent/export_sft.py ... --api-model glm-4.5v --api-base-url https://open.bigmodel.cn/api/paas/v4 \
        --tasks ... --samples-per-task 4 --temperature 1.0 --min-reward 1.0 --out ...
        # a model behind any OpenAI-compatible API writes the turns (as eval --backend api); keep its successes

Use the model, chat template, budgets and pixel bounds of the RL run the data is for. Output in --out:
train.jsonl (ms-swift: messages, tools and images), images/, rollout_tokens.jsonl (the rollout's tokens, for the
check) and export_summary.json. When ms-swift is importable, scripts/agent/check_swift_export.py runs at the
end; otherwise run it in an environment with ms-swift (docs/agent_environments.md, "SFT data").
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verl.utils.dataset import process_image  # noqa: E402
from verl.utils.tokenizer import get_processor  # noqa: E402
from verl.workers.agent.api_loop import APIError, APIPolicy, APISampling, ChatCompletionsClient  # noqa: E402
from verl.workers.agent.environments import (  # noqa: E402
    EnvironmentContext,
    RemoteEnvironmentError,
    class_from_file,
    environment_factory,
)
from verl.workers.agent.protocol import (  # noqa: E402
    AgentImageConfig,
    AgentLoopConfig,
    AgentSeedContext,
    derive_environment_seed,
    environment_report,
)
from verl.workers.agent.replay import RecordedPolicy, ReplayError, load_episodes, replay_episode  # noqa: E402
from verl.workers.agent.sft_export import ExportError, keep_episode, swift_sample  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the model of the RL run (its processor and chat template)")
    parser.add_argument("--chat-template", help="a chat template file, as data.override_chat_template in training")
    parser.add_argument(
        "--plain-think-tokens", default="auto", help="as worker.actor.model.plain_think_tokens (default auto)"
    )
    parser.add_argument("--environment", required=True, help="a registered environment or path/to/file.py:Class")
    parser.add_argument("--environment-kwargs", type=json.loads, default={}, help="its options as JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", help="a scripted policy, path/to/file.py:ClassName (with --tasks)")
    source.add_argument("--episodes", help="recorded episodes (JSONL)")
    source.add_argument("--api-model", help="a model behind an OpenAI-compatible API writes the turns (with --tasks)")
    parser.add_argument(
        "--tasks", help="tasks for --policy or --api-model: parquet or JSONL with an env_config column"
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="episodes per task for --policy or --api-model, all from the task's environment seed (with "
        "--temperature > 0 and --min-reward: rejection sampling)",
    )
    parser.add_argument("--policy-kwargs", type=json.loads, default={}, help="options of the policy class as JSON")
    parser.add_argument("--env-config-key", default="env_config", help="the task column with each env_config")
    parser.add_argument("--limit", type=int, help="replay the first N episodes only")
    parser.add_argument("--seed", type=int, default=1, help="base seed of the episodes (as data.seed)")
    parser.add_argument("--max-tool-calls", type=int, default=6, help="worker.rollout.agent_max_tool_calls")
    parser.add_argument("--max-response-tokens", type=int, default=8192, help="data.max_response_length")
    parser.add_argument(
        "--max-tokens-per-turn", type=int, default=10240, help="worker.rollout.agent_max_tokens_per_turn"
    )
    parser.add_argument("--max-model-len", type=int, help="worker.rollout.max_model_len (default: no limit)")
    parser.add_argument("--min-pixels", type=int, default=262144, help="data.min_pixels")
    parser.add_argument("--max-pixels", type=int, default=4194304, help="data.max_pixels")
    parser.add_argument("--observation-min-pixels", type=int, help="worker.rollout.agent_observation_min_pixels")
    parser.add_argument("--keep", choices=["answered", "all"], default="answered", help="which episodes to write")
    parser.add_argument(
        "--min-reward",
        type=float,
        help="also require at least this return (the episode reward plus the environment's rewards for single calls)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="episodes replayed at once")
    api = parser.add_argument_group("--api-model options (as eval --backend api; docs/agent_environments.md)")
    api.add_argument("--api-base-url", help="the endpoint, e.g. http://127.0.0.1:8001/v1 (default $OPENAI_BASE_URL)")
    api.add_argument("--api-key-env", default="OPENAI_API_KEY", help="the environment variable with the API key")
    api.add_argument("--api-concurrency", type=int, default=8, help="requests at a time")
    api.add_argument("--api-request-retries", type=int, default=5)
    api.add_argument("--api-timeout", type=float, default=300.0, help="seconds per request")
    api.add_argument("--api-extra-body", type=json.loads, default=None, help="JSON merged into each request")
    api.add_argument("--api-max-tokens-field", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    api.add_argument(
        "--api-max-total-tokens", type=int, help="no new request once the API has reported this many tokens in total"
    )
    api.add_argument("--api-tool-images", choices=["user", "tool"], default="user", help="as eval --api-tool-images")
    api.add_argument("--temperature", type=float, default=0.0)
    api.add_argument("--top-p", type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-check", action="store_true", help="do not run the ms-swift check at the end")
    args = parser.parse_args(argv)
    if (args.policy or args.api_model) and not args.tasks:
        parser.error("--policy and --api-model need --tasks")
    if args.episodes and (args.tasks or args.policy_kwargs or args.samples_per_task != 1):
        parser.error("--tasks, --policy-kwargs and --samples-per-task do not go with --episodes")
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be positive")
    if args.api_model:
        args.api_base_url = args.api_base_url or os.environ.get("OPENAI_BASE_URL")
        if not args.api_base_url:
            parser.error("--api-model needs --api-base-url (or OPENAI_BASE_URL)")
    return args


def load_tasks(path: str, key: str) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pandas as pd

        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    tasks = []
    for index, row in enumerate(rows):
        config = row.get(key)
        if isinstance(config, str):
            config = json.loads(config)
        if not isinstance(config, dict):
            raise SystemExit(f"{path}: row {index} has no {key} object")
        tasks.append({"env_config": config, **({"id": str(row["id"])} if "id" in row else {})})
    return tasks


async def export(args: argparse.Namespace) -> dict[str, Any]:
    processor = get_processor(
        args.model,
        override_chat_template=args.chat_template,
        plain_think_tokens=args.plain_think_tokens,
        trust_remote_code=True,
        use_fast=True,
    )
    image_config = AgentImageConfig(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        observation_min_pixels=args.observation_min_pixels,
    )
    config = AgentLoopConfig(
        max_tool_calls=args.max_tool_calls,
        max_response_tokens=args.max_response_tokens,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )

    def prepare(image: Any) -> Any:  # the rollout's image preparation (the vLLM image cache's resize)
        return process_image(image, args.min_pixels, args.max_pixels)

    context = EnvironmentContext(processor=processor, image_preprocessor=prepare, max_tool_calls=args.max_tool_calls)
    make_environment = environment_factory(args.environment, context, args.environment_kwargs)
    client = None
    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        if args.policy:
            policy_class = class_from_file(args.policy, what="policy")
        else:
            client = ChatCompletionsClient(
                args.api_base_url,
                args.api_model,
                os.environ.get(args.api_key_env) or None,
                timeout_s=args.api_timeout,
                retries=args.api_request_retries,
                max_concurrency=args.api_concurrency,
                extra_body=args.api_extra_body,
                max_tokens_field=args.api_max_tokens_field,
                max_total_tokens=args.api_max_total_tokens,
            )
        tasks = load_tasks(args.tasks, args.env_config_key)
        tasks = tasks[: args.limit] if args.limit is not None else tasks
        episodes = []
        for task_index, task in enumerate(tasks):
            for sample in range(args.samples_per_task):
                task_id = task.get("id", str(task_index))
                episodes.append(
                    {
                        **task,
                        "id": task_id if args.samples_per_task == 1 else f"{task_id}#{sample}",
                        "seed": derive_environment_seed(args.seed, task_index),  # one task state per task
                        "task_index": task_index,
                        "sample": sample,
                    }
                )
    if args.episodes and args.limit is not None:
        episodes = episodes[: args.limit]
    # what the API model sees: the images the trained model will see
    pixels = (args.min_pixels, args.max_pixels)
    observation_pixels = (args.observation_min_pixels or args.min_pixels, args.max_pixels)

    def make_policy(episode: dict[str, Any]) -> Any:
        if args.episodes:
            return RecordedPolicy(episode["turns"])
        if args.policy:
            return policy_class(**args.policy_kwargs)
        return APIPolicy(
            client,
            sampling=APISampling(temperature=args.temperature, top_p=args.top_p),
            max_tokens=args.max_tokens_per_turn,
            seed_context=AgentSeedContext(args.seed, episode["task_index"], episode["sample"]),
            pixels=pixels,
            observation_pixels=observation_pixels,
            tool_images=args.api_tool_images,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    skipped: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    samples: list[Any] = [None] * len(episodes)

    async def run_one(index: int, episode: dict[str, Any]) -> None:
        episode_id = str(episode.get("id", index))
        policy = make_policy(episode)
        seed = int(episode["seed"]) if "seed" in episode else derive_environment_seed(args.seed, index)
        async with semaphore:
            try:
                run = await replay_episode(
                    make_environment(),
                    episode["env_config"],
                    policy,
                    processor,
                    env_seed=seed,
                    config=config,
                    image_config=image_config,
                    image_preprocessor=prepare,
                    max_model_len=args.max_model_len or 2**31 - 1,
                )
                reason = keep_episode(run, keep=args.keep, min_reward=args.min_reward)
                if reason is not None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    return
                samples[index] = (index, episode_id, run, swift_sample(run, processor, prepare))
            except (ReplayError, ExportError, APIError, RemoteEnvironmentError) as exc:
                # this episode is left out (an environment error names the episode and the step)
                failures.append({"id": episode_id, "error": f"{type(exc).__name__}: {exc}"})

    tasks = [asyncio.ensure_future(run_one(index, episode)) for index, episode in enumerate(episodes)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # an unexpected error: stop the other episodes and let them close before the shared resources close below
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        await context.aclose()
        if client is not None:
            await client.aclose()

    written = 0
    with (
        (args.out / "train.jsonl").open("w", encoding="utf-8") as train,
        (args.out / "rollout_tokens.jsonl").open("w", encoding="utf-8") as rollout,
    ):
        for item in samples:
            if item is None:
                continue
            index, episode_id, run, sample = item
            episode_sample = episodes[index].get("sample", 0)
            paths = []
            for position, image in enumerate(sample.images):
                path = image_dir / f"{index:06d}" / f"{position}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths.append(str(path.resolve()))
            train.write(json.dumps({**sample.record, "images": paths}, ensure_ascii=False) + "\n")
            trajectory = run.trajectory
            rollout.write(
                json.dumps(
                    {
                        "id": episode_id,
                        **({"source": f"api:{args.api_model}", "sample": episode_sample} if args.api_model else {}),
                        **sample.rollout,
                        "status": trajectory.status.value,
                        **environment_report(trajectory.steps, trajectory.env_outcome, trajectory.initial_state_key),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    summary = {
        "episodes": len(episodes),
        "written": written,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures[:50],
        "model": args.model,
        "environment": args.environment,
        "plain_think_tokens": args.plain_think_tokens,
        "budgets": config.__dict__,
        "pixels": {"min": args.min_pixels, "max": args.max_pixels, "observation_min": args.observation_min_pixels},
    }
    if client is not None:
        summary["api"] = {
            "model": args.api_model,
            "base_url": args.api_base_url,
            "samples_per_task": args.samples_per_task,
            "temperature": args.temperature,
            "requests": client.requests,
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
        }
    (args.out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = asyncio.run(export(args))
    print(
        f"[export] {summary['written']} of {summary['episodes']} episodes written to {args.out / 'train.jsonl'}; "
        f"skipped {summary['skipped'] or 0}, failed {summary['failed']}",
        flush=True,
    )
    for failure in summary["failures"][:5]:
        print(f"[export] failed {failure['id']}: {failure['error']}", flush=True)
    check = [str(Path(__file__).with_name("check_swift_export.py")), "--model", args.model, "--export", str(args.out)]
    if args.no_check or summary["written"] == 0:
        return 0
    if importlib.util.find_spec("swift") is None:
        print(
            "[export] ms-swift is not installed here; check the data in an environment with ms-swift:\n  python "
            + " ".join(check),
            flush=True,
        )
        return 0
    return subprocess.run([sys.executable, *check], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
