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
"""Draw the overview figure of the README (docs/assets/taxonomy.svg and taxonomy_zh.svg).

    python scripts/make_taxonomy_figure.py --font-regular NotoSansSC-Regular.otf --font-bold NotoSansSC-Bold.otf

Text is stored as vector paths, so the SVG looks the same on every machine. Any font with Latin and
CJK glyphs works; the committed figures use Noto Sans SC (https://github.com/notofonts/noto-cjk).
Edit CONTENT below when a method is added, then re-run and check the PNG preview (--png).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "counterfactual": ("#2563EB", "#EFF6FF", "#1D4ED8"),
    "token": ("#059669", "#ECFDF5", "#047857"),
    "grounded": ("#EA580C", "#FFF7ED", "#C2410C"),
    "cgpo": ("#7C3AED", "#F5F3FF", "#6D28D9"),
}
GRAY, LIGHT_GRAY = "#374151", "#6B7280"

CONTENT = {
    "en": {
        "title": "Perception-aware RLVR: where the visual signal enters training",
        "problem": "Outcome-reward RLVR (GRPO / DAPO) only checks the final answer: "
        "a model can be right for the wrong visual reason.",
        "lead": "Perception-aware methods put visual perception back into the objective:",
        "columns": [
            (
                "grounded",
                "Grounded reasoning",
                "The model points to or zooms into the visual evidence while it reasons.",
                [
                    ("GRIT", "NeurIPS 2025", "Bounding boxes interleaved in a single-turn reasoning chain."),
                    ("DeepEyes", "ICLR 2026", "Multi-turn zoom-in tool calls on the original image."),
                ],
            ),
            (
                "counterfactual",
                "Counterfactual views",
                "An auxiliary objective contrasts the policy on the original and on a corrupted or "
                "counterfactual image.",
                [
                    ("PAPO", "ICLR 2026", "Maximize the KL to a patch-masked image, plus a double entropy loss."),
                    ("DVRP", "arXiv", "Masked view: push the KL up. Noised view: pull the KL down."),
                    ("CFPO", "ICML 2026", "Counterfactual inside the model (attention values); maximize the KL."),
                ],
            ),
            (
                "token",
                "Token-level visual credit",
                "Measure how much each token depends on the image, then reweight or select tokens.",
                [
                    ("VPPO", "ICLR 2026", "Gradients only on the most image-dependent tokens; trajectory scaling."),
                    ("ToR", "arXiv", "Separate weights for reasoning tokens and perception tokens."),
                    ("PGPO", "arXiv", "Threshold-gated token advantages from visual dependency."),
                    ("PEPO", "arXiv", "Hidden-state perception prior, no extra forward pass."),
                    ("VEPO", "arXiv", "Token selection by JS divergence and entropy gap."),
                ],
            ),
        ],
        "cgpo_name": "CGPO (ours)",
        "cgpo_venue": "ACM MM 2026 Oral",
        # (column index, step text): the steps of CGPO, in order, under the family each step belongs to
        "cgpo_steps": [
            (0, "1  Ground the evidence inline"),
            (1, "2  Mask it into a counterfactual image"),
            (2, "3  Dependence scales token advantages"),
        ],
        "cgpo_extra": "+ grounding-consistency reward",
        "footer": "Reproduced in one EasyR1 codebase  ·  controlled comparison on Qwen3-VL-4B  ·  "
        "one-click evaluation on 25+ benchmarks",
    },
    "zh": {
        "title": "Perception-Aware RLVR：视觉信号从哪里进入训练",
        "problem": "结果奖励 RLVR（GRPO / DAPO）只检查最终答案：模型可能答对了，但依据的视觉理由是错的。",
        "lead": "感知导向的方法把视觉感知重新放回优化目标：",
        "columns": [
            (
                "grounded",
                "基于定位的推理",
                "模型在推理过程中指出或放大所依据的视觉证据。",
                [
                    ("GRIT", "NeurIPS 2025", "在单轮推理链中穿插边界框。"),
                    ("DeepEyes", "ICLR 2026", "多轮调用放大工具查看原图。"),
                ],
            ),
            (
                "counterfactual",
                "反事实视图",
                "加入辅助目标，对比策略在原图与被遮挡或反事实图像上的输出。",
                [
                    ("PAPO", "ICLR 2026", "最大化与块遮挡图像之间的 KL，并加双熵损失。"),
                    ("DVRP", "arXiv", "对遮挡视图增大 KL，对加噪视图减小 KL。"),
                    ("CFPO", "ICML 2026", "在模型内部（注意力 value）构造反事实，最大化 KL。"),
                ],
            ),
            (
                "token",
                "token 级视觉信用分配",
                "衡量每个 token 对图像的依赖程度，据此重新加权或筛选 token。",
                [
                    ("VPPO", "ICLR 2026", "只更新最依赖图像的 token，并按轨迹缩放优势。"),
                    ("ToR", "arXiv", "推理 token 与感知 token 分别加权。"),
                    ("PGPO", "arXiv", "按视觉依赖对 token 优势做带阈值的门控。"),
                    ("PEPO", "arXiv", "隐状态感知先验，无需额外前向。"),
                    ("VEPO", "arXiv", "按 JS 散度与熵差筛选 token。"),
                ],
            ),
        ],
        "cgpo_name": "CGPO（本仓库）",
        "cgpo_venue": "ACM MM 2026 Oral",
        "cgpo_steps": [
            (0, "1  在推理中内联定位证据"),
            (1, "2  遮挡证据，得到反事实图像"),
            (2, "3  证据依赖决定 token 优势缩放"),
        ],
        "cgpo_extra": "+ 定位一致性奖励",
        "footer": "统一的 EasyR1 代码库复现  ·  Qwen3-VL-4B 统一设定对比  ·  25+ 个评测基准一键评测",
    },
}

WIDTH, MARGIN, GAP = 1200, 32, 22
_TOKEN = re.compile(r"[　-鿿＀-￯]|[^\s　-鿿＀-￯]+|\s+")


class Canvas:
    def __init__(self, height: float, regular: str, bold: str):
        self.fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100)
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.height = height
        self.fonts = {False: regular, True: bold}
        self.renderer = self.fig.canvas.get_renderer()
        self.overflows: list[str] = []

    def prop(self, size: float, bold: bool = False) -> FontProperties:
        return FontProperties(fname=self.fonts[bold], size=size * 0.72)  # canvas px -> pt at 100 dpi

    def width(self, text: str, size: float, bold: bool = False) -> float:
        w, _, _ = self.renderer.get_text_width_height_descent(text, self.prop(size, bold), ismath=False)
        return w  # display pixels at 100 dpi == canvas units

    def wrap(self, text: str, size: float, max_width: float, bold: bool = False) -> list[str]:
        lines, current = [], ""
        for token in _TOKEN.findall(text):
            candidate = current + token
            if current and self.width(candidate.rstrip(), size, bold) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current.strip():
            lines.append(current.rstrip())
        return lines

    def text(self, x, y, s, size, bold=False, color=GRAY, ha="left", max_width=None, where=""):
        self.ax.text(x, y, s, fontproperties=self.prop(size, bold), color=color, ha=ha, va="top")
        if max_width is not None and self.width(s, size, bold) > max_width + 0.5:
            self.overflows.append(f"{where}: '{s}'")

    def box(self, x, y, w, h, edge, fill, radius=12, lw=1.4, ls="-"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
            linestyle=ls,
        )
        self.ax.add_patch(patch)


def layout(canvas: Canvas | None, content: dict, regular: str, bold: str) -> float:
    """Draw the figure on ``canvas`` (or only measure it when ``canvas`` is None); return its height."""
    measure = canvas or Canvas(100, regular, bold)
    draw = canvas is not None
    y = MARGIN
    if draw:
        canvas.box(4, 4, WIDTH - 8, canvas.height - 8, "#D1D5DB", "#FFFFFF", radius=18, lw=1.2)
        canvas.text(
            MARGIN, y, content["title"], 26, bold=True, color="#111827", max_width=WIDTH - 2 * MARGIN, where="title"
        )
    y += 48

    problem_lines = measure.wrap(content["problem"], 15, WIDTH - 2 * MARGIN - 40)
    banner_h = 22 + 22 * len(problem_lines)
    if draw:
        canvas.box(MARGIN, y, WIDTH - 2 * MARGIN, banner_h, "#D1D5DB", "#F3F4F6", radius=10, lw=1.0)
        for i, line in enumerate(problem_lines):
            canvas.text(WIDTH / 2, y + 12 + 22 * i, line, 15, color="#1F2937", ha="center")
    y += banner_h + 16
    if draw:
        canvas.text(WIDTH / 2, y, content["lead"], 14, color=LIGHT_GRAY, ha="center")
        canvas.ax.annotate(
            "",
            xy=(WIDTH / 2, y + 40),
            xytext=(WIDTH / 2, y + 22),
            arrowprops=dict(arrowstyle="-|>", color="#9CA3AF", lw=1.4),
        )
    y += 48

    col_w = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
    inner = col_w - 36
    col_x = [MARGIN + i * (col_w + GAP) for i in range(3)]

    def column_height(column) -> float:
        _, header, sub, items = column
        h = 18 + 30 + 22 * len(measure.wrap(sub, 13, inner)) + 12
        for _, _, desc in items:
            h += 26 + 20 * len(measure.wrap(desc, 13, inner)) + 12
        return h + 6

    card_h = max(column_height(c) for c in content["columns"])
    if draw:
        for x, (key, header, sub, items) in zip(col_x, content["columns"]):
            edge, fill, dark = PALETTE[key]
            canvas.box(x, y, col_w, card_h, edge, fill)
            cy = y + 18
            canvas.text(x + 18, cy, header, 18, bold=True, color=dark, max_width=inner, where=header)
            cy += 30
            for line in canvas.wrap(sub, 13, inner):
                canvas.text(x + 18, cy, line, 13, color=LIGHT_GRAY)
                cy += 22
            cy += 4
            canvas.ax.plot([x + 18, x + col_w - 18], [cy, cy], color=edge, lw=0.8, alpha=0.35)
            cy += 8
            for name, venue, desc in items:
                canvas.text(x + 18, cy, name, 16, bold=True, color="#111827")
                name_w = canvas.width(name, 16, bold=True)
                canvas.text(
                    x + 18 + name_w + 10, cy + 3, venue, 12, color=dark, max_width=inner - name_w - 10, where=name
                )
                cy += 26
                for line in canvas.wrap(desc, 13, inner):
                    canvas.text(x + 18, cy, line, 13, color=GRAY)
                    cy += 20
                cy += 12
    y += card_h

    # CGPO: one bar under the three families, one step chip under the family each step belongs to
    connector_top = y
    y += 34
    edge, fill, dark = PALETTE["cgpo"]
    chip_h = 34
    bar_h = 20 + 32 + 12 + chip_h + 18
    if draw:
        canvas.box(MARGIN, y, WIDTH - 2 * MARGIN, bar_h, edge, fill, lw=1.8)
        canvas.text(MARGIN + 18, y + 16, content["cgpo_name"], 19, bold=True, color=dark)
        name_w = canvas.width(content["cgpo_name"], 19, bold=True)
        venue_w = canvas.width(content["cgpo_venue"], 13, bold=True) + 20
        vx = MARGIN + 18 + name_w + 14
        canvas.box(vx, y + 16, venue_w, 26, dark, dark, radius=13, lw=1.0)
        canvas.text(vx + venue_w / 2, y + 20, content["cgpo_venue"], 13, bold=True, color="#FFFFFF", ha="center")
        canvas.text(
            WIDTH - MARGIN - 18,
            y + 22,
            content["cgpo_extra"],
            14,
            color=dark,
            ha="right",
            max_width=WIDTH / 2,
            where="cgpo extra",
        )
        chip_y = y + 20 + 32 + 12
        for column, step in content["cgpo_steps"]:
            cx = col_x[column]
            canvas.ax.plot(
                [cx + col_w / 2] * 2, [connector_top + 2, y - 2], color=edge, lw=1.4, ls=(0, (4, 3)), alpha=0.8
            )
            canvas.box(cx + 10, chip_y, col_w - 20, chip_h, edge, "#FFFFFF", radius=17, lw=1.2)
            canvas.text(
                cx + col_w / 2,
                chip_y + 8,
                step,
                14,
                bold=True,
                color=dark,
                ha="center",
                max_width=col_w - 40,
                where=step,
            )
        columns = [column for column, _ in content["cgpo_steps"]]
        for left, right in zip(columns, columns[1:]):
            if right == left + 1:
                canvas.ax.annotate(
                    "",
                    xy=(col_x[right] + 8, chip_y + chip_h / 2),
                    xytext=(col_x[left] + col_w - 8, chip_y + chip_h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.6),
                )
    y += bar_h + 22

    if draw:
        canvas.text(
            WIDTH / 2,
            y,
            content["footer"],
            14,
            color=LIGHT_GRAY,
            ha="center",
            max_width=WIDTH - 2 * MARGIN,
            where="footer",
        )
    y += 22 + MARGIN
    if canvas is None:
        plt.close(measure.fig)
    return y


def render(lang: str, regular: str, bold: str, out_dir: Path, png: bool) -> None:
    content = CONTENT[lang]
    height = layout(None, content, regular, bold)
    canvas = Canvas(height, regular, bold)
    layout(canvas, content, regular, bold)
    if canvas.overflows:
        raise SystemExit("text does not fit:\n  " + "\n  ".join(canvas.overflows))
    stem = "taxonomy" if lang == "en" else f"taxonomy_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "taxonomy"}):
        canvas.fig.savefig(out_dir / f"{stem}.svg", transparent=True, metadata={"Date": None})
    if png:
        canvas.fig.savefig(out_dir / f"{stem}.png", dpi=150, transparent=False, facecolor="#FFFFFF")
    plt.close(canvas.fig)
    print(f"wrote {out_dir / stem}.svg ({WIDTH}x{height:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--font-regular", required=True, help="regular font file with Latin and CJK glyphs")
    parser.add_argument("--font-bold", required=True, help="bold font file with Latin and CJK glyphs")
    parser.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "assets"))
    parser.add_argument("--png", action="store_true", help="also write a PNG preview next to the SVG")
    args = parser.parse_args()
    for lang in ["en", "zh"] if args.lang == "all" else [args.lang]:
        render(lang, args.font_regular, args.font_bold, Path(args.out_dir), args.png)


if __name__ == "__main__":
    main()
