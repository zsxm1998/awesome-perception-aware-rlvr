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
"""Merge an EasyR1 actor checkpoint and safely finalize its directory layout."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from safetensors import safe_open


MODEL_RANK_ZERO_RE = re.compile(r"model_world_size_\d+_rank_0\.pt")
MODEL_INDEX = "model.safetensors.index.json"
SINGLE_MODEL = "model.safetensors"
TOTAL_STEPS = 8


class FinalizeError(RuntimeError):
    """A safe-to-report checkpoint finalization error."""


@dataclass(frozen=True)
class CheckpointValidation:
    ok: bool
    detail: str
    weight_files: tuple[Path, ...] = ()


def log(message: str) -> None:
    print(f"[checkpoint-finalizer] {message}", flush=True)


def step_log(step: int, message: str) -> None:
    log(f"[步骤 {step}/{TOTAL_STEPS}] {message}")


def existing_path(path: Path) -> bool:
    """Return True for normal paths and broken symlinks."""
    return os.path.lexists(path)


def locate_unique(relative_name: str, roots: Sequence[Path]) -> Path:
    relative_path = PurePosixPath(relative_name)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise FinalizeError(f"权重索引包含不安全路径: {relative_name!r}")

    matches = [
        root.joinpath(*relative_path.parts) for root in roots if existing_path(root.joinpath(*relative_path.parts))
    ]
    if not matches:
        raise FinalizeError(f"缺少文件: {relative_name}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise FinalizeError(f"同一文件同时出现在多个位置，拒绝猜测版本: {locations}")
    return matches[0]


def locate_optional(relative_name: str, roots: Sequence[Path]) -> Path | None:
    try:
        return locate_unique(relative_name, roots)
    except FinalizeError as error:
        if str(error).startswith("缺少文件:"):
            return None
        raise


def validate_regular_nonempty_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"不是普通文件: {path}")
    if path.stat().st_size <= 0:
        raise FinalizeError(f"文件为空: {path}")


def validate_safetensors(path: Path, expected_keys: Iterable[str] | None = None) -> None:
    validate_regular_nonempty_file(path)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
    except Exception as error:
        raise FinalizeError(f"无法读取 safetensors 文件 {path}: {error}") from error

    if not actual_keys:
        raise FinalizeError(f"safetensors 文件不含任何 tensor: {path}")
    if expected_keys is not None:
        missing_keys = set(expected_keys) - actual_keys
        if missing_keys:
            examples = ", ".join(sorted(missing_keys)[:3])
            raise FinalizeError(
                f"权重索引与 shard 不一致，{path} 缺少 {len(missing_keys)} 个 tensor（例如: {examples}）"
            )


def inspect_checkpoint(roots: Sequence[Path]) -> CheckpointValidation:
    """Validate a checkpoint whose top-level files may be split across roots."""
    roots = tuple(root for root in roots if root.is_dir() and not root.is_symlink())
    if not roots:
        return CheckpointValidation(False, "checkpoint 目录不存在")

    try:
        config_path = locate_unique("config.json", roots)
        validate_regular_nonempty_file(config_path)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FinalizeError(f"config.json 无法解析: {config_path}: {error}") from error
        if not isinstance(config, dict):
            raise FinalizeError(f"config.json 顶层不是 JSON object: {config_path}")

        index_path = locate_optional(MODEL_INDEX, roots)
        single_model_path = locate_optional(SINGLE_MODEL, roots)
        if index_path is not None and single_model_path is not None:
            raise FinalizeError(f"同时存在 {MODEL_INDEX} 和 {SINGLE_MODEL}，无法确定有效权重布局")

        if index_path is None:
            if single_model_path is None:
                raise FinalizeError(f"未找到 {SINGLE_MODEL} 或 {MODEL_INDEX}")
            validate_safetensors(single_model_path)
            return CheckpointValidation(True, f"单文件权重有效: {single_model_path}", (single_model_path,))

        validate_regular_nonempty_file(index_path)
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FinalizeError(f"权重索引无法解析: {index_path}: {error}") from error

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise FinalizeError(f"权重索引缺少非空 weight_map: {index_path}")

        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise FinalizeError(f"权重索引 weight_map 格式错误: {index_path}")
            keys_by_shard[shard_name].append(tensor_name)

        shard_paths = []
        for shard_name, expected_keys in sorted(keys_by_shard.items()):
            shard_path = locate_unique(shard_name, roots)
            validate_safetensors(shard_path, expected_keys)
            shard_paths.append(shard_path)

        return CheckpointValidation(
            True,
            f"分片权重有效: {len(shard_paths)} 个 shard，索引为 {index_path}",
            tuple(shard_paths),
        )
    except FinalizeError as error:
        return CheckpointValidation(False, str(error))


def actor_pt_files(actor_dir: Path) -> list[Path]:
    files = []
    for path in actor_dir.iterdir():
        if path.name.endswith(".pt"):
            if path.is_dir() and not path.is_symlink():
                raise FinalizeError(f"发现以 .pt 结尾的目录，拒绝递归删除: {path}")
            files.append(path)
    return sorted(files)


def raw_model_rank_zero_exists(actor_dir: Path) -> bool:
    return any(MODEL_RANK_ZERO_RE.fullmatch(path.name) for path in actor_dir.iterdir() if path.is_file())


def ensure_no_move_collisions(actor_dir: Path, hf_dir: Path) -> None:
    if not hf_dir.exists():
        return
    if hf_dir.is_symlink() or not hf_dir.is_dir():
        raise FinalizeError(f"huggingface 路径不是普通目录: {hf_dir}")

    collisions = [actor_dir / source.name for source in hf_dir.iterdir() if existing_path(actor_dir / source.name)]
    if collisions:
        examples = ", ".join(str(path) for path in collisions[:3])
        raise FinalizeError(f"huggingface 内容与 actor 中的现有路径冲突，尚未删除任何 checkpoint 文件: {examples}")


def delete_cleanup_files(actor_dir: Path) -> None:
    pt_files = actor_pt_files(actor_dir)
    dataloader_path = actor_dir.parent / "dataloader.pt"
    dataloader_exists = existing_path(dataloader_path)

    if pt_files:
        log(f"检测到 actor 目录下 {len(pt_files)} 个待删除的 .pt 文件")
    else:
        log("actor 目录下没有 .pt 文件，跳过对应删除操作")

    for path in pt_files:
        try:
            path.unlink()
            log(f"已删除 .pt 文件：{path}")
        except FileNotFoundError:
            log(f"文件已不存在，视为删除完成：{path}")
        except OSError as error:
            raise FinalizeError(f"删除失败 {path}: {error}") from error

    if dataloader_exists:
        log(f"检测到待删除的 dataloader 文件：{dataloader_path}")
        if dataloader_path.is_dir() and not dataloader_path.is_symlink():
            raise FinalizeError(f"dataloader.pt 是目录，拒绝递归删除: {dataloader_path}")
        try:
            dataloader_path.unlink()
            log(f"已删除 dataloader 文件：{dataloader_path}")
        except FileNotFoundError:
            log(f"文件已不存在，视为删除完成：{dataloader_path}")
        except OSError as error:
            raise FinalizeError(f"删除失败 {dataloader_path}: {error}") from error
    else:
        log("actor 同级没有 dataloader.pt，跳过对应删除操作")

    remaining = actor_pt_files(actor_dir)
    if remaining or existing_path(dataloader_path):
        raise FinalizeError("清理校验失败，仍有应删除的 .pt 文件")
    log("清理结果校验通过：目标 .pt 文件均已移除")


def move_huggingface_contents(actor_dir: Path, hf_dir: Path) -> None:
    if not hf_dir.exists():
        log("huggingface 目录不存在，说明搬移步骤已经完成，跳过搬移")
        return
    if hf_dir.is_symlink() or not hf_dir.is_dir():
        raise FinalizeError(f"huggingface 路径不是普通目录: {hf_dir}")

    # Put the index last so a directly inspected actor directory never advertises
    # a sharded checkpoint before all of its shards have arrived.
    sources = sorted(
        hf_dir.iterdir(),
        key=lambda path: (path.name == MODEL_INDEX, path.name),
    )
    if not sources:
        log("huggingface 目录已经为空，无内容需要搬移")
        return

    log(f"检测到 {len(sources)} 个待搬移项目，目标目录：{actor_dir}")
    for source in sources:
        destination = actor_dir / source.name
        if existing_path(destination):
            raise FinalizeError(f"移动目标已存在，拒绝覆盖: {destination}（源: {source}）")
        try:
            source.rename(destination)
            log(f"已移动：{source} -> {destination}")
        except OSError as error:
            raise FinalizeError(f"移动失败 {source} -> {destination}: {error}") from error
    log(f"搬移完成：共移动 {len(sources)} 个项目")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "调用 model_merger.py 融合 actor checkpoint，校验后清理训练状态，"
            "并将 huggingface 内容提升到 actor 目录。脚本可从已完成步骤继续。"
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="actor checkpoint 目录，例如 checkpoints/.../global_step_100/actor",
    )
    return parser.parse_args()


def resolve_actor_dir(path: Path) -> Path:
    try:
        actor_dir = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalizeError(f"checkpoint 目录不存在或无法访问: {path}: {error}") from error
    if not actor_dir.is_dir():
        raise FinalizeError(f"checkpoint 路径不是目录: {actor_dir}")
    if actor_dir.name != "actor":
        raise FinalizeError(f"为防止误删，只接受目录名为 actor 的路径；收到: {actor_dir}")
    return actor_dir


def run_merger(actor_dir: Path) -> None:
    merger = Path(__file__).resolve().with_name("model_merger.py")
    if not merger.is_file():
        raise FinalizeError(f"找不到 model merger: {merger}")

    log(f"准备调用模型融合脚本：{merger}")
    log(f"模型融合输入目录：{actor_dir}")
    try:
        subprocess.run(
            [sys.executable, str(merger), "--local_dir", str(actor_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise FinalizeError(
            f"model_merger.py 执行失败（退出码 {error.returncode}）；未执行清理和移动，可修复问题后重跑本脚本"
        ) from error
    log("model_merger.py 已正常退出，模型融合命令执行成功")


def main() -> int:
    args = parse_args()
    try:
        step_log(1, f"开始检查输入路径：{args.checkpoint_dir}")
        actor_dir = resolve_actor_dir(args.checkpoint_dir)
        hf_dir = actor_dir / "huggingface"
        step_log(1, f"输入路径检查完成，actor 目录为：{actor_dir}")

        step_log(2, "开始检测当前 checkpoint 的完成状态")
        actor_checkpoint = inspect_checkpoint((actor_dir,))
        hf_checkpoint = inspect_checkpoint((hf_dir,))
        combined_checkpoint = inspect_checkpoint((actor_dir, hf_dir))
        log(f"actor 目录融合产物检查：{'通过' if actor_checkpoint.ok else '未完成'}；{actor_checkpoint.detail}")
        log(f"huggingface 目录融合产物检查：{'通过' if hf_checkpoint.ok else '未完成'}；{hf_checkpoint.detail}")
        log(f"可恢复联合布局检查：{'通过' if combined_checkpoint.ok else '未完成'}；{combined_checkpoint.detail}")
        step_log(2, "checkpoint 状态检测完成")

        if actor_checkpoint.ok:
            step_log(3, "actor 中已有完整融合产物，跳过 model_merger.py")
        elif hf_checkpoint.ok:
            step_log(3, "huggingface 中已有完整融合产物，跳过 model_merger.py")
        elif combined_checkpoint.ok:
            step_log(3, "检测到搬移中断后的完整联合产物，跳过 model_merger.py")
        else:
            if not raw_model_rank_zero_exists(actor_dir):
                raise FinalizeError(
                    "没有找到完整的融合产物，也没有可供 model_merger.py 使用的 "
                    f"rank-0 模型分片。当前校验错误: {combined_checkpoint.detail}"
                )
            step_log(3, "未发现完整融合产物，开始执行 model_merger.py")
            run_merger(actor_dir)
            log("开始校验 model_merger.py 生成的 huggingface 产物")
            hf_checkpoint = inspect_checkpoint((hf_dir,))
            if not hf_checkpoint.ok:
                raise FinalizeError(
                    "model_merger.py 已正常退出，但 huggingface 产物校验失败；"
                    f"未执行清理和移动。原因: {hf_checkpoint.detail}"
                )
            combined_checkpoint = hf_checkpoint
            log(f"融合产物校验通过：{hf_checkpoint.detail}")
            step_log(3, "模型融合与产物校验完成")

        # Check every destination before deleting training-state files. A collision
        # is therefore non-destructive and can be resolved manually.
        step_log(4, "开始执行删除前安全检查")
        log("正在检查 huggingface 内容与 actor 目标路径是否冲突")
        ensure_no_move_collisions(actor_dir, hf_dir)
        log("目标路径冲突检查通过，不会覆盖 actor 中的现有文件")

        # Revalidate immediately before the destructive step, including the
        # partially-moved layout used when resuming an interrupted run.
        log("正在重新校验融合权重，确认满足删除原始训练状态的条件")
        safe_checkpoint = inspect_checkpoint((actor_dir, hf_dir))
        if not safe_checkpoint.ok:
            raise FinalizeError(f"删除前的融合产物校验失败，未删除任何文件。原因: {safe_checkpoint.detail}")
        log(f"删除前融合产物校验通过：{safe_checkpoint.detail}")
        step_log(4, "删除前安全检查全部完成")

        step_log(5, "开始清理 actor/*.pt 和 actor 同级 dataloader.pt")
        delete_cleanup_files(actor_dir)
        step_log(5, "训练状态文件清理完成")

        step_log(6, "开始将 huggingface 目录内容搬移到 actor")
        move_huggingface_contents(actor_dir, hf_dir)
        step_log(6, "huggingface 内容搬移步骤完成")

        step_log(7, "开始校验 actor 目录中的最终 Hugging Face checkpoint")
        final_checkpoint = inspect_checkpoint((actor_dir,))
        if not final_checkpoint.ok:
            raise FinalizeError(
                f"移动后 actor checkpoint 校验失败；保留 huggingface 目录以便重试。原因: {final_checkpoint.detail}"
            )
        log(f"最终 checkpoint 校验通过：{final_checkpoint.detail}")
        step_log(7, "最终 checkpoint 校验完成")

        step_log(8, "开始清理空的 huggingface 目录")
        if hf_dir.exists():
            try:
                hf_dir.rmdir()
                log(f"已删除空目录：{hf_dir}")
            except OSError as error:
                raise FinalizeError(f"融合权重已安全位于 actor，但 huggingface 目录无法删除: {error}") from error
        else:
            log("huggingface 目录已不存在，跳过目录删除")
        step_log(8, "huggingface 目录清理完成")

        log(f"[执行成功] checkpoint 融合与清理全部完成：{actor_dir}")
        return 0
    except FinalizeError as error:
        print(
            f"[checkpoint-finalizer] [执行失败] {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
