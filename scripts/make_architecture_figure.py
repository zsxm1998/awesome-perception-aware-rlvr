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
"""Draw the architecture figure of the README (docs/assets/architecture.svg and architecture_zh.svg).

    python scripts/make_architecture_figure.py --font-regular NotoSansSC-Regular.otf --font-bold NotoSansSC-Bold.otf

Text is stored as vector paths, so the SVG looks the same on every machine. Any font with Latin and
CJK glyphs works; the committed figures use Noto Sans SC (https://github.com/notofonts/noto-cjk).
Edit CONTENT below when a method or a block is added, then re-run and check the PNG preview (--png).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WIDTH, MARGIN = 1200, 18
# block kinds: (edge, fill, title, text)
KINDS = {
    "base": ("#D1D5DB", "#F3F4F6", "#4B5563", "#6B7280"),  # EasyR1 / veRL
    "perception": ("#3B82F6", "#EFF6FF", "#1D4ED8", "#1F2937"),
    "reward": ("#F59E0B", "#FFFBEB", "#B45309", "#1F2937"),
    "distill": ("#14B8A6", "#F0FDFA", "#0F766E", "#1F2937"),
}
TAG = ("#FFFFFF", "#CBD5E1", "#334155")  # method pill: fill, edge, text
INK, MUTED, ARROW = "#111827", "#6B7280", "#9CA3AF"
# wrap units: a CJK character or a Latin word, with opening / closing punctuation kept on its side of a line break
_OPEN, _CLOSE = "（《「『【“", "，。；：、！？）》」』】”"
_TOKEN = re.compile(rf"[{_OPEN}]*(?:[　-鿿＀-￯]|[^\s　-鿿＀-￯]+)[{_CLOSE}]*|\s+")

# Each method's blocks follow its scripts in examples/; keep the tags in sync when a script changes.
CONTENT = {
    "en": {
        "title": "Framework architecture: where the perception-aware blocks enter one RL step",
        "legend": [
            ("base", "EasyR1 / veRL"),
            ("perception", "perception-aware addition"),
            ("reward", "reward"),
            ("distill", "distillation"),
            ("tag", "methods using the block"),
        ],
        "loop": "new weights → vLLM",
        "lanes": [
            ("Rollout", "vLLM engine"),
            ("Before the update", "No-gradient forwards; reward and advantages on the driver"),
            ("Actor update", "FSDP, per micro-batch"),
        ],
        "rollout": [
            dict(
                kind="perception",
                title="Noised rollout images",
                desc="Half of each group is sampled from a noised image; the update reads the clean one.",
                tags=["NoisyRollout"],
            ),
            dict(
                kind="base",
                title="n responses per prompt",
                desc="Sampled in a single turn.",
                addon=("zoom-in tool", ["DeepEyes"]),
            ),
            dict(
                kind="base",
                title="Online filtering",
                desc="Keep the groups whose rewards differ; sample again until the batch is full.",
                tags=["DAPO"],
            ),
        ],
        # independent of each other: each is one method's extra reward input
        "rollout_rewards": (
            "Reward inputs the policy must generate, before vLLM sleeps",
            [
                dict(
                    kind="reward",
                    title="Grounding consistency",
                    desc="The policy re-detects every entity it grounded; only groups with a correct answer.",
                    tags=["CGPO"],
                ),
                dict(
                    kind="reward",
                    title="Visual-claim probes",
                    desc="At cuts of a correct response, the policy judges claims about the image (Yes / No).",
                    tags=["VAPO"],
                ),
            ],
        ),
        "scoring": {
            "roles": [
                dict(
                    kind="base",
                    title="Rollout policy",
                    desc="sampled-token log-probs",
                    addon=("+ hidden states", ["VGPO"]),
                ),
                dict(kind="base", title="Reference model", desc="log-probs for the KL term"),
                dict(
                    kind="distill",
                    title="Teacher (frozen)",
                    desc="sampled-token log-probs",
                    tags=["OPD", "VA-OPD"],
                ),
            ],
            "views": dict(
                kind="perception",
                title="Auxiliary forward on a changed input",
                desc="The same responses scored again without gradient, by the rollout policy (VA-OPD: by the teacher). "
                "Against the original log-probs or distributions this gives each token's visual dependence.",
                grid=[
                    (
                        "On the image",
                        [
                            ("patches masked", ["PAPO", "VPPO", "DVRP"]),
                            ("noise added", ["VEPO", "DVRP"]),
                            ("pixelated", ["VA-OPD"]),
                            ("image removed", ["ToR"]),
                            ("own boxes masked", ["CGPO"]),
                        ],
                    ),
                    (
                        "Inside the model",
                        [
                            ("image attention masked", ["PGPO"]),
                            ("attention values averaged", ["CFPO"]),
                        ],
                    ),
                ],
            ),
            "reward": dict(
                kind="reward",
                title="Reward function",
                grid=[
                    (
                        "",
                        [
                            ("accuracy + format (all)", []),
                            ("+ γ × grounding consistency *", ["CGPO"]),
                            ("+ w × probe score *", ["VAPO"]),
                            ("+ grounded-format terms", ["GRIT"]),
                            ("+ tool-use bonus *", ["DeepEyes"]),
                        ],
                    )
                ],
                note="* only for a correct answer",
            ),
            "advantage": dict(
                kind="base",
                title="Advantages",
                desc=["Group-normalized", "(GRPO, DAPO, …)", "or the teacher log-ratio"],
                tags=["OPD"],
            ),
        },
        "loss_title": ("compute_perception_reasoning_policy_loss", "five steps on each micro-batch"),
        "steps": [
            dict(
                kind="perception",
                title="Control signals",
                desc="Token sensitivity from the auxiliary forward or from hidden states; top-ρ entropy mask; "
                "top-ρ sensitivity mask ∪ region tokens.",
                tags=["ToR", "VPPO", "VEPO", "CGPO"],
            ),
            dict(
                kind="perception",
                title="Advantage modulation",
                desc="Scale the advantage per response, per token or per prompt group.",
                tags=["VPPO", "CGPO", "PGPO", "PEPO", "VGPO"],
            ),
            dict(
                kind="base",
                title="Token-wise PPO terms",
                desc=["Clipped ratio × modulated advantage, with dual-clip.", "As in plain GRPO."],
                tags=[],
            ),
            dict(
                kind="perception",
                title="Token selection",
                desc="Keep the union of the masks, or weight reasoning and perception tokens.",
                tags=["VPPO", "VEPO", "CGPO", "ToR"],
            ),
            dict(
                kind="perception",
                title="Auxiliary losses",
                desc="KL to the auxiliary view (k3), entropy terms, robustness KL.",
                tags=["PAPO", "CFPO", "DVRP", "VPPO", "VEPO", "NoisyRollout"],
            ),
        ],
        "distill": dict(
            kind="distill",
            title="Or a distillation loss",
            desc="Per micro-batch, a student forward and a no-grad teacher forward (frozen or EMA); divergence on the full next-token distributions.",
            tags=["OPD", "VA-OPD", "VGS", "VCSD", "Vision-OPD"],
        ),
        "optimizer": dict(
            kind="base",
            title="Total loss and optimizer step",
            desc=[
                "+ KL to the reference; token / sequence averaging",
                "Backward and step; new weights synced to vLLM",
            ],
        ),
        "footer": "A method is a launcher that switches these blocks on: examples/reproduction (each paper's "
        "setting)  ·  examples/comparison (one shared setting)  ·  eval/ (30+ benchmarks)",
    },
    "zh": {
        "title": "框架架构：感知导向模块在一个 RL 训练步中的位置",
        "legend": [
            ("base", "EasyR1 / veRL 原有"),
            ("perception", "感知导向的新增模块"),
            ("reward", "奖励"),
            ("distill", "蒸馏"),
            ("tag", "使用该模块的方法"),
        ],
        "loop": "新权重 → vLLM",
        "lanes": [
            ("采样", "vLLM 推理引擎"),
            ("更新前计算", "无梯度前向；奖励与优势在 driver 上算"),
            ("策略更新", "FSDP，逐 micro-batch"),
        ],
        "rollout": [
            dict(
                kind="perception",
                title="加噪图像采样",
                desc="每组一半回答从加噪图像采样；更新时读原图。",
                tags=["NoisyRollout"],
            ),
            dict(
                kind="base",
                title="每题采样 n 条回答",
                desc="策略单轮采样。",
                addon=("多轮放大工具调用", ["DeepEyes"]),
            ),
            dict(
                kind="base",
                title="在线过滤",
                desc="保留奖励有差异的组；不够一批就继续采样。",
                tags=["DAPO"],
            ),
        ],
        "rollout_rewards": (
            "需要策略生成的奖励输入，在 vLLM 休眠前算完",
            [
                dict(
                    kind="reward",
                    title="定位一致性",
                    desc="策略重新检测自己定位的每个实体；只对有答对回答的组。",
                    tags=["CGPO"],
                ),
                dict(
                    kind="reward",
                    title="视觉断言探针",
                    desc="在答对回答的推理中截断，让策略判断关于图像的断言（是 / 否）。",
                    tags=["VAPO"],
                ),
            ],
        ),
        "scoring": {
            "roles": [
                dict(kind="base", title="rollout 策略", desc="采样 token 的 log-prob", addon=("+ 隐状态", ["VGPO"])),
                dict(kind="base", title="参考模型", desc="KL 项用的 log-prob"),
                dict(
                    kind="distill",
                    title="教师（冻结）",
                    desc="采样 token 的 log-prob",
                    tags=["OPD", "VA-OPD"],
                ),
            ],
            "views": dict(
                kind="perception",
                title="改动输入的辅助前向",
                desc="同一批回答在改动后的输入上再无梯度前向一次（rollout 策略；VA-OPD 用教师）。"
                "与原输入的 log-prob 或分布相比，得到每个 token 的视觉依赖。",
                grid=[
                    (
                        "图像上",
                        [
                            ("随机遮挡图块", ["PAPO", "VPPO", "DVRP"]),
                            ("加高斯噪声", ["VEPO", "DVRP"]),
                            ("像素化", ["VA-OPD"]),
                            ("去掉图像", ["ToR"]),
                            ("遮挡自己的框", ["CGPO"]),
                        ],
                    ),
                    (
                        "模型内部",
                        [
                            ("遮挡图像注意力", ["PGPO"]),
                            ("注意力 value 取均值", ["CFPO"]),
                        ],
                    ),
                ],
            ),
            "reward": dict(
                kind="reward",
                title="奖励函数",
                grid=[
                    (
                        "",
                        [
                            ("准确率 + 格式（所有方法）", []),
                            ("+ γ × 定位一致性 *", ["CGPO"]),
                            ("+ w × 探针感知分 *", ["VAPO"]),
                            ("+ 定位格式项", ["GRIT"]),
                            ("+ 工具使用奖励 *", ["DeepEyes"]),
                        ],
                    )
                ],
                note="* 只给答对的回答",
            ),
            "advantage": dict(
                kind="base",
                title="优势",
                desc=["组内归一化", "（GRPO、DAPO 等）", "或教师对数比"],
                tags=["OPD"],
            ),
        },
        "loss_title": ("compute_perception_reasoning_policy_loss", "每个 micro-batch 五步"),
        "steps": [
            dict(
                kind="perception",
                title="控制信号",
                desc="由辅助前向或隐状态得到 token 敏感度；熵前 ρ 掩码；敏感度前 ρ 掩码 ∪ 区域 token。",
                tags=["ToR", "VPPO", "VEPO", "CGPO"],
            ),
            dict(
                kind="perception",
                title="优势调制",
                desc="按回答、按 token 或按问题组缩放优势。",
                tags=["VPPO", "CGPO", "PGPO", "PEPO", "VGPO"],
            ),
            dict(
                kind="base",
                title="逐 token PPO 项",
                desc=["裁剪后的比率 × 调制后的优势，带 dual-clip。", "与普通 GRPO 相同。"],
                tags=[],
            ),
            dict(
                kind="perception",
                title="token 筛选",
                desc="只保留掩码并集的 token，或分别加权推理与感知 token。",
                tags=["VPPO", "VEPO", "CGPO", "ToR"],
            ),
            dict(
                kind="perception",
                title="辅助损失",
                desc="与辅助前向的 KL（k3）、熵项、鲁棒性 KL。",
                tags=["PAPO", "CFPO", "DVRP", "VPPO", "VEPO", "NoisyRollout"],
            ),
        ],
        "distill": dict(
            kind="distill",
            title="或换成蒸馏损失",
            desc="逐 micro-batch 做学生前向和无梯度的教师前向（冻结或 EMA），在完整下一 token 分布上计算散度。",
            tags=["OPD", "VA-OPD", "VGS", "VCSD", "Vision-OPD"],
        ),
        "optimizer": dict(
            kind="base",
            title="总损失与优化器更新",
            desc=["+ 参考模型 KL；token / 序列平均", "反向传播并更新；新权重同步到 vLLM"],
        ),
        "footer": "每个方法就是一个打开相应模块的启动脚本：examples/reproduction（论文原设定）  ·  "
        "examples/comparison（统一设定）  ·  eval/（30+ 个基准一键评测）",
    },
}

# sizes in canvas units (1 unit = 1 px at 100 dpi)
TITLE, LANE, LANE_SUB = 20, 15, 11.5
BLOCK_TITLE, BLOCK_TEXT, LINE = 13, 11.5, 15.5
PILL, PILL_H, PILL_GAP = 10, 16, 4
PAD, GAP, GRID_GAP = 9, 16, 18  # inside blocks, between blocks of a lane, between grid columns


class Canvas:
    def __init__(self, fonts: dict[str, str]):
        self.fig = plt.figure(figsize=(WIDTH / 100, 10), dpi=100)
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(1000, 0)
        self.ax.axis("off")
        self.fonts = fonts
        self.renderer = self.fig.canvas.get_renderer()
        self.draw = True
        self.overflows: list[str] = []

    def prop(self, size: float, weight: str = "regular") -> FontProperties:
        return FontProperties(fname=self.fonts[weight], size=size * 0.72)  # canvas px -> pt at 100 dpi

    def width(self, text: str, size: float, weight: str = "regular") -> float:
        w, _, _ = self.renderer.get_text_width_height_descent(text, self.prop(size, weight), ismath=False)
        return w

    def wrap(self, text: str, size: float, max_width: float, weight: str = "regular") -> list[str]:
        lines, current = [], ""
        for token in _TOKEN.findall(text):
            candidate = current + token
            if current and self.width(candidate.rstrip(), size, weight) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current.strip():
            lines.append(current.rstrip())
        return lines

    def text(self, x, y, s, size, weight="regular", color=INK, ha="left", va="top", max_width=None, **kw):
        if self.draw:
            self.ax.text(x, y, s, fontproperties=self.prop(size, weight), color=color, ha=ha, va=va, **kw)
            if max_width is not None and self.width(s, size, weight) > max_width + 0.5:
                self.overflows.append(s)

    def box(self, x, y, w, h, edge, fill, radius=8, lw=1.2, ls="-"):
        if self.draw:
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle=f"round,pad=0,rounding_size={radius}",
                    linewidth=lw,
                    edgecolor=edge,
                    facecolor=fill,
                    linestyle=ls,
                )
            )

    def arrow(self, x0, y0, x1, y1, color=ARROW, lw=1.4):
        if self.draw:
            self.ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11, shrinkA=0, shrinkB=0),
            )

    def line(self, xs, ys, color=ARROW, lw=1.4, ls="-"):
        if self.draw:
            self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round")

    def pill_width(self, label: str) -> float:
        return self.width(label, PILL, "bold") + 11

    def pill(self, x, y, label) -> float:
        w = self.pill_width(label)
        fill, edge, color = TAG
        self.box(x, y, w, PILL_H, edge, fill, radius=PILL_H / 2, lw=1.0)
        self.text(x + w / 2, y + PILL_H / 2 + 0.5, label, PILL, "bold", color=color, ha="center", va="center")
        return w

    def pills(self, x, y, labels, max_width) -> float:
        """Pills wrapped into rows of ``max_width``; returns their height (0 without labels)."""
        if not labels:
            return 0.0
        cx, cy = x, y
        for label in labels:
            w = self.pill_width(label)
            if cx > x and cx + w > x + max_width:
                cx, cy = x, cy + PILL_H + PILL_GAP
            self.pill(cx, cy, label)
            cx += w + PILL_GAP
        return cy + PILL_H - y


def block(c: Canvas, x, y, w, spec, h=None, number=None) -> float:
    """A block: title, description, an optional addon chip and grid, method pills pinned to the bottom."""
    edge, fill, _, _ = KINDS[spec["kind"]]
    inner = w - 2 * PAD
    tags = spec.get("tags") or []
    draw, c.draw = c.draw, False
    body_h = _block_body(c, x, y, w, spec, number)
    tags_h = c.pills(0, 0, tags, inner)
    c.draw = draw
    natural = body_h + (6 + tags_h if tags else 0) + PAD
    height = natural if h is None else h
    c.box(x, y, w, height, edge, fill, lw=1.6 if spec["kind"] != "base" else 1.1)
    _block_body(c, x, y, w, spec, number)
    if tags:
        c.pills(x + PAD, y + height - PAD - tags_h, tags, inner)
    return natural


def _block_body(c: Canvas, x, y, w, spec, number) -> float:
    """Draw the title, description, addon and grid; return the height used (top padding included)."""
    _, _, title_color, text_color = KINDS[spec["kind"]]
    inner = w - 2 * PAD
    cy = y + PAD
    tx = x + PAD
    if number is not None:
        r = 9
        if c.draw:
            c.ax.add_patch(Circle((tx + r, cy + r), r, facecolor=title_color, edgecolor="none"))
        c.text(tx + r, cy + r + 0.5, str(number), 11, "bold", color="#FFFFFF", ha="center", va="center")
        tx += 2 * r + 6
    for line in c.wrap(spec["title"], BLOCK_TITLE, x + w - PAD - tx, "bold"):
        c.text(tx, cy, line, BLOCK_TITLE, "bold", color=title_color)
        cy += 18
    cy += 2
    desc = spec.get("desc", "")
    for paragraph in [desc] if isinstance(desc, str) else desc:
        for line in c.wrap(paragraph, BLOCK_TEXT, inner):
            c.text(x + PAD, cy, line, BLOCK_TEXT, color=text_color)
            cy += LINE
    if "addon" in spec:
        label, tags = spec["addon"]
        cy += 4
        tags_w = sum(c.pill_width(t) + PILL_GAP for t in tags)
        label_lines = c.wrap(label, BLOCK_TEXT, inner - 16 - tags_w)
        chip_h = max(PILL_H + 8, 8 + LINE * len(label_lines))
        edge, _, chip_title, _ = KINDS["perception"]
        c.box(x + PAD, cy, inner, chip_h, edge, "#FFFFFF", radius=6, lw=1.2)
        for i, line in enumerate(label_lines):
            c.text(x + PAD + 8, cy + 4 + LINE * i, line, BLOCK_TEXT, color=chip_title)
        px = x + PAD + inner - 6 - tags_w + PILL_GAP
        for t in tags:
            px += c.pill(px, cy + (chip_h - PILL_H) / 2, t) + PILL_GAP
        cy += chip_h
    if "grid" in spec:
        cy += 6
        top = cy
        bottom = cy
        gx = x + PAD
        for header, rows in spec["grid"]:
            label_w, col_w = grid_column_width(c, rows)
            gy = top
            if header:
                c.text(gx, gy, header, 11, "bold", color=title_color)
                gy += 17
            for label, tags in rows:
                c.text(gx, gy + 1, label, BLOCK_TEXT, color=text_color)
                px = gx + label_w
                for t in tags:
                    px += c.pill(px, gy, t) + PILL_GAP
                if c.draw and px - PILL_GAP > gx + col_w + 0.5:
                    c.overflows.append(f"grid row '{label}'")
                gy += PILL_H + 4
            bottom = max(bottom, gy)
            gx += col_w + GRID_GAP
        cy = bottom - 4
    if "note" in spec:
        cy += 4
        c.text(x + PAD, cy, spec["note"], 10.5, color=MUTED)
        cy += 14
    return cy - y


def grid_column_width(c: Canvas, rows) -> tuple[float, float]:
    """Width of the label column and of the whole column of a block's grid."""
    label_w = max(c.width(label, BLOCK_TEXT) for label, _ in rows) + 8
    pills_w = max(max(sum(c.pill_width(t) + PILL_GAP for t in tags) - PILL_GAP, 0) for _, tags in rows)
    return label_w, label_w + pills_w


def grid_width(c: Canvas, grid) -> float:
    return sum(grid_column_width(c, rows)[1] for _, rows in grid) + GRID_GAP * (len(grid) - 1)


def lane_frame(c: Canvas, x, y, w, h, number, name, sub, label_w):
    c.box(x, y, w, h, "#E5E7EB", "#F9FAFB", radius=12, lw=1.0)
    r = 13
    cx = x + 14
    if c.draw:
        c.ax.add_patch(Circle((cx + r, y + 14 + r), r, facecolor=INK, edgecolor="none"))
    c.text(cx + r, y + 14 + r + 0.5, str(number), 14, "bold", color="#FFFFFF", ha="center", va="center")
    ty = y + 14 + 2 * r + 8
    for line in c.wrap(name, LANE, label_w - 22, "bold"):
        c.text(cx, ty, line, LANE, "bold", color=INK)
        ty += 20
    ty += 2
    for line in c.wrap(sub, LANE_SUB, label_w - 22):
        c.text(cx, ty, line, LANE_SUB, color=MUTED)
        ty += 15


def legend(c: Canvas, right, y, items) -> None:
    sample = "PAPO"
    widths = []
    for key, label in items:
        widths.append((14 + 5 if key in KINDS else c.pill_width(sample) + 5) + c.width(label, 11.5) + 14)
    x = right - sum(widths) + 14
    for (key, label), w in zip(items, widths):
        if key in KINDS:
            edge, fill, _, _ = KINDS[key]
            c.box(x, y + 1, 14, 14, edge, fill, radius=3, lw=1.3)
            lx = x + 19
        else:
            lx = x + c.pill(x, y, sample) + 5
        c.text(lx, y + 0.5, label, 11.5, color=MUTED)
        x += w


def layout(c: Canvas, content: dict) -> float:
    label_w = 104
    lane_x = MARGIN + 32  # room for the loop arrow on the left
    lane_w = WIDTH - MARGIN - lane_x
    body_x = lane_x + label_w
    body_w = lane_x + lane_w - 12 - body_x
    y = MARGIN + 4
    c.text(MARGIN + 6, y, content["title"], TITLE, "bold", color=INK)
    y += 34
    legend(c, WIDTH - MARGIN - 6, y, content["legend"])
    y += 28
    lane_tops = []

    # 1. rollout: the sequential blocks, then a frame with the independent reward inputs
    specs = content["rollout"]
    frame_title, reward_specs = content["rollout_rewards"]
    fpad, fgap = 8, 10
    n = len(specs) + len(reward_specs)
    bw = (body_w - GAP * len(specs) - 2 * fpad - fgap * (len(reward_specs) - 1)) / n
    draw, c.draw = c.draw, False
    bh = max(block(c, 0, 0, bw, s) for s in specs + reward_specs)
    c.draw = draw
    caption_h = 22
    lane_h = 12 + bh + fpad + caption_h + 12
    lane_tops.append((y, lane_h))
    lane_frame(c, lane_x, y, lane_w, lane_h, 1, *content["lanes"][0], label_w)
    by = y + 12
    bx = body_x
    for i, s in enumerate(specs):
        block(c, bx, by, bw, s, h=bh)
        c.arrow(bx + bw + 2, by + bh / 2, bx + bw + GAP - 2, by + bh / 2)
        bx += bw + GAP
    edge, _, dark, _ = KINDS["reward"]
    rx0 = bx
    rx1 = body_x + body_w
    c.box(rx0, by - fpad, rx1 - rx0, bh + 2 * fpad + caption_h, edge, "#FFFFFF", radius=10, lw=1.4, ls=(0, (5, 3)))
    bx += fpad
    for s in reward_specs:
        block(c, bx, by, bw, s, h=bh)
        bx += bw + fgap
    c.text((rx0 + rx1) / 2, by + bh + fpad + 2, frame_title, 11, "bold", color=dark, ha="center")
    frame_bottom = by + bh + fpad + caption_h
    y += lane_h + 12

    # 2. before the update: the three scoring roles stacked, visual-dependence signal, reward, advantages
    sc = content["scoring"]
    roles = sc["roles"]
    order = ["views", "reward", "advantage"]
    role_w = 168
    widths = {k: grid_width(c, sc[k]["grid"]) + 2 * PAD for k in ("views", "reward")}
    widths["advantage"] = body_w - role_w - sum(widths.values()) - GAP * len(order)
    draw, c.draw = c.draw, False
    role_hs = [block(c, 0, 0, role_w, r) for r in roles]
    stack_gap = 8
    bh = max(max(block(c, 0, 0, widths[k], sc[k]) for k in order), sum(role_hs) + stack_gap * (len(roles) - 1))
    c.draw = draw
    lane_h = 12 + bh + 12
    lane_tops.append((y, lane_h))
    lane_frame(c, lane_x, y, lane_w, lane_h, 2, *content["lanes"][1], label_w)
    extra = (bh - sum(role_hs) - stack_gap * (len(roles) - 1)) / len(roles)
    ry = y + 12
    for r, rh in zip(roles, role_hs):
        block(c, body_x, ry, role_w, r, h=rh + extra)
        ry += rh + extra + stack_gap
    bx = body_x + role_w + GAP
    for k in order:
        c.arrow(bx - GAP + 2, y + 12 + bh / 2, bx - 2, y + 12 + bh / 2)
        block(c, bx, y + 12, widths[k], sc[k], h=bh)
        if k == "reward":
            # from the rollout's reward inputs down into the reward function
            ax = (max(rx0, bx) + min(rx1, bx + widths[k])) / 2
            c.arrow(ax, frame_bottom + 1, ax, y + 12 - 1, color=edge, lw=1.6)
        bx += widths[k] + GAP
    y += lane_h + 12

    # 3. actor update: the five steps of the policy loss, then distillation and the optimizer step
    steps = content["steps"]
    frame_pad = 10
    sw = (body_w - 2 * frame_pad - GAP * (len(steps) - 1)) / len(steps)
    draw, c.draw = c.draw, False
    sh = max(block(c, 0, 0, sw, s, number=i + 1) for i, s in enumerate(steps))
    c.draw = draw
    frame_h = 34 + sh + frame_pad
    draw, c.draw = c.draw, False
    tail_w = (body_w - GAP) * 0.6
    opt_w = body_w - GAP - tail_w
    th = max(block(c, 0, 0, tail_w, content["distill"]), block(c, 0, 0, opt_w, content["optimizer"]))
    c.draw = draw
    lane_h = 12 + frame_h + 10 + th + 12
    lane_tops.append((y, lane_h))
    lane_frame(c, lane_x, y, lane_w, lane_h, 3, *content["lanes"][2], label_w)
    fy = y + 12
    p_edge = KINDS["perception"][0]
    c.box(body_x, fy, body_w, frame_h, p_edge, "#FFFFFF", radius=10, lw=1.6, ls=(0, (5, 3)))
    fname, fsub = content["loss_title"]
    c.text(body_x + 12, fy + 9, fname, 13, "mono", color=KINDS["perception"][2])
    c.text(body_x + 12 + c.width(fname, 13, "mono") + 10, fy + 10, fsub, 12, color=MUTED)
    for i, s in enumerate(steps):
        sx = body_x + frame_pad + i * (sw + GAP)
        block(c, sx, fy + 34, sw, s, h=sh, number=i + 1)
        if i:
            c.arrow(sx - GAP + 2, fy + 34 + sh / 2, sx - 2, fy + 34 + sh / 2)
    ty = fy + frame_h + 10
    block(c, body_x, ty, tail_w, content["distill"], h=th)
    block(c, body_x + tail_w + GAP, ty, opt_w, content["optimizer"], h=th)
    y += lane_h

    # spine: down arrows between the lanes, and the loop back to the rollout
    for (top, h), (next_top, _) in zip(lane_tops, lane_tops[1:]):
        cx = lane_x + 14 + 13
        c.arrow(cx, top + h - 6, cx, next_top + 6, color=INK, lw=1.4)
    lx = lane_x - 12
    (first_top, _), (last_top, last_h) = lane_tops[0], lane_tops[-1]
    y_start, y_end = last_top + last_h - 22, first_top + 27
    c.line([lane_x, lx, lx], [y_start, y_start, y_end], color=MUTED, lw=1.4)
    c.arrow(lx, y_end, lane_x - 1, y_end, color=MUTED, lw=1.4)
    c.text(lx - 12, (y_start + y_end) / 2, content["loop"], 11, color=MUTED, ha="center", va="center", rotation=90)

    y += 12
    c.text(WIDTH / 2, y, content["footer"], 11.5, color=MUTED, ha="center", max_width=WIDTH - 2 * MARGIN)
    y += 18 + MARGIN
    return y


def render(lang: str, fonts: dict[str, str], out_dir: Path, png_dpi: int) -> None:
    canvas = Canvas(fonts)
    height = layout(canvas, CONTENT[lang])
    if canvas.overflows:
        raise SystemExit("text does not fit:\n  " + "\n  ".join(canvas.overflows))
    canvas.fig.set_size_inches(WIDTH / 100, height / 100)
    canvas.ax.set_ylim(height, 0)
    canvas.ax.add_patch(
        FancyBboxPatch(
            (4, 4),
            WIDTH - 8,
            height - 8,
            boxstyle="round,pad=0,rounding_size=16",
            linewidth=1.2,
            edgecolor="#D1D5DB",
            facecolor="#FFFFFF",
            zorder=-10,
        )
    )
    stem = "architecture" if lang == "en" else f"architecture_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "architecture"}):
        canvas.fig.savefig(out_dir / f"{stem}.svg", transparent=True, metadata={"Date": None})
    if png_dpi:
        canvas.fig.savefig(out_dir / f"{stem}.png", dpi=png_dpi, transparent=False, facecolor="#FFFFFF")
    plt.close(canvas.fig)
    print(f"wrote {out_dir / stem}.svg ({WIDTH}x{height:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--font-regular", required=True, help="regular font file with Latin and CJK glyphs")
    parser.add_argument("--font-bold", required=True, help="bold font file with Latin and CJK glyphs")
    parser.add_argument("--font-mono", default=None, help="monospace font for code names (default: DejaVu Sans Mono)")
    parser.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "assets"))
    parser.add_argument("--png", action="store_true", help="also write a PNG next to the SVG")
    parser.add_argument("--png-dpi", type=int, default=200, help="PNG resolution (the canvas is 12 inches wide)")
    args = parser.parse_args()
    fonts = {
        "regular": args.font_regular,
        "bold": args.font_bold,
        "mono": args.font_mono or findfont(FontProperties(family="DejaVu Sans Mono")),
    }
    for lang in ["en", "zh"] if args.lang == "all" else [args.lang]:
        render(lang, fonts, Path(args.out_dir), args.png_dpi if args.png else 0)


if __name__ == "__main__":
    main()
