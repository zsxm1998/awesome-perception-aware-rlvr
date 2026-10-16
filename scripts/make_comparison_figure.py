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
"""Draw the RLVR comparison figure of the README: each method's difference from GRPO, overall and per group.

    python scripts/make_comparison_figure.py --font-regular NotoSansSC-Regular.otf --font-bold NotoSansSC-Bold.otf

The numbers come from the summary table of examples/comparison/rlvr_qwen3_vl_4b/README.md (re-run after it
changes). Writes docs/assets/comparison_rlvr{,_zh}{,_dark}.svg: English and Chinese, for light and dark pages.
Text is stored as vector paths, as in scripts/make_architecture_figure.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "comparison" / "rlvr_qwen3_vl_4b" / "README.md"
REFERENCE = "GRPO"
# rows of the table that are not perception-aware methods: the untrained model and the other base algorithm
LEFT_OUT = ("(baseline)", "DAPO")
COLUMNS = ("Mean of the 3 groups", "Math reasoning", "Vision-dependent reasoning", "Perception and hallucination")
BAND = 1.0  # the gray band: differences of about a point are within run-to-run variation

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "border": "#e1e0d9",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#898781",
        "band": "#e1e0d9",
        "above": "#4a7ab9",
        "below": "#c06b65",
    },
    "dark": {
        "surface": "#1a1a19",
        "border": "#2c2c2a",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#898781",
        "band": "#383835",
        "above": "#5789c8",
        "below": "#ca7370",
    },
}

TEXT = {
    "en": {
        "title": "Perception-aware RLVR methods against GRPO in one shared setting",
        "subtitle": "Qwen3-VL-4B-Instruct, ViRL39K, the same training budget: difference from GRPO in points at the "
        "last step, one run per method",
        "above": "Above GRPO",
        "below": "Below GRPO",
        "band": f"Within ±{BAND:g} point: about the variation between two runs of one configuration",
        "columns": COLUMNS,
        "reference": "GRPO {value:.1f}",
        "footer": "Data: examples/comparison/rlvr_qwen3_vl_4b (19 benchmarks in 3 groups; the mean of the 3 groups "
        "weighs each group equally).",
    },
    "zh": {
        "title": "统一设定下各感知导向 RLVR 方法与 GRPO 的分差",
        "subtitle": "Qwen3-VL-4B-Instruct，ViRL39K，相同训练量；最后一步相对 GRPO 的分差，每个方法训练一次",
        "above": "高于 GRPO",
        "below": "低于 GRPO",
        "band": f"±{BAND:g} 分以内：与同一配置两次训练之间的波动相当",
        "columns": ("三组平均", "数学推理", "依赖视觉的推理", "感知与幻觉"),
        "reference": "GRPO {value:.1f}",
        "footer": "数据：examples/comparison/rlvr_qwen3_vl_4b（19 项基准分三组；三组平均让每组权重相同）。",
    },
}

# layout in canvas pixels (100 px per inch)
WIDTH, MARGIN = 1000, 28
LABEL_WIDTH, PANEL_GAP = 112, 18
VALUE_ROOM = 34  # room at each side of a panel for the value at the tip of its longest bars
ROW, BAR = 22, 12
PLOT_TOP = 150


def read_table(path: Path) -> dict[str, dict[str, float]]:
    """The summary table of the comparison README: method -> {column: score}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        i for i, line in enumerate(lines) if line.startswith("| Method |") and all(c in line for c in COLUMNS)
    )
    header = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    table = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        row = dict(zip(header, cells))
        table[row["Method"]] = {column: float(row[column].strip("*")) for column in COLUMNS}
    return table


def differences(table: dict[str, dict[str, float]]) -> tuple[dict[str, float], list[tuple[str, dict[str, float]]]]:
    """GRPO's scores and the methods' differences from them, sorted by the difference of the overall mean."""
    reference = table[REFERENCE]
    rows = [
        (method, {column: round(scores[column] - reference[column], 1) for column in COLUMNS})
        for method, scores in table.items()
        if method != REFERENCE and not any(left_out in method for left_out in LEFT_OUT)
    ]
    rows.sort(key=lambda item: (-item[1][COLUMNS[0]], item[0]))
    return reference, rows


def signed(value: float, spec: str) -> str:
    """+3.5, −6.7 (a true minus sign) and 0."""
    if value == 0:
        return "0"
    return ("+" if value > 0 else "\u2212") + format(abs(value), spec)


class Canvas:
    def __init__(self, fonts: dict[str, str], height: float, theme: dict[str, str]):
        self.fonts, self.theme = fonts, theme
        self.fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size, weight="regular", color="primary", ha="left", va="baseline"):
        prop = FontProperties(fname=self.fonts[weight], size=size * 0.72)  # canvas px -> pt at 100 dpi
        return self.ax.text(x, y, s, fontproperties=prop, color=self.theme[color], ha=ha, va=va)

    def rect(self, x0, y0, x1, y1, color, z=1):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=self.theme[color], linewidth=0, zorder=z))

    def line(self, x0, y0, x1, y1, color, width=1.0, z=2):
        self.ax.plot(
            [x0, x1], [y0, y1], color=self.theme[color], linewidth=width * 0.72, zorder=z, solid_capstyle="butt"
        )

    def bar(self, x_base, x_end, y_top, color):
        """A bar from the baseline to x_end: square at the baseline, the data end rounded with a 4 px radius."""
        length = abs(x_end - x_base)
        if length < 0.5:
            return
        radius = min(4.0, length, BAR / 2)
        sign = 1 if x_end > x_base else -1
        y0, y1 = y_top, y_top + BAR
        tip = x_end - sign * radius
        vertices = [
            (x_base, y0),
            (tip, y0),
            (x_end, y0),
            (x_end, y0 + radius),
            (x_end, y1 - radius),
            (x_end, y1),
            (tip, y1),
            (x_base, y1),
            (x_base, y0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        self.ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=self.theme[color], linewidth=0, zorder=3))


def render(lang: str, mode: str, fonts: dict[str, str], out_dir: Path, png_dpi: int) -> None:
    text = TEXT[lang]
    reference, rows = differences(read_table(SOURCE))
    values = [row[column] for _, row in rows for column in COLUMNS]
    low, high = min(-BAND, min(values)), max(BAND, max(values))
    plot_bottom = PLOT_TOP + ROW * len(rows)
    height = plot_bottom + 92
    canvas = Canvas(fonts, height, THEMES[mode])
    canvas.ax.add_patch(
        FancyBboxPatch(
            (4, 4),
            WIDTH - 8,
            height - 8,
            boxstyle="round,pad=0,rounding_size=16",
            linewidth=1.2,
            edgecolor=canvas.theme["border"],
            facecolor=canvas.theme["surface"],
            zorder=-10,
        )
    )

    canvas.text(MARGIN, 40, text["title"], 18, "bold")
    canvas.text(MARGIN, 64, text["subtitle"], 12.5, color="secondary")
    # legend: a swatch beside each label, the labels in text ink
    x = MARGIN
    for key, color in (("above", "above"), ("below", "below"), ("band", "band")):
        canvas.rect(x, 82, x + 14, 92, color)
        label = canvas.text(x + 20, 91.5, text[key], 12, color="secondary")
        canvas.fig.canvas.draw()
        box = label.get_window_extent().transformed(canvas.ax.transData.inverted())
        x = box.x1 + 22

    panel_width = (WIDTH - 2 * MARGIN - LABEL_WIDTH - 3 * PANEL_GAP) / len(COLUMNS)
    for index, (method, _) in enumerate(rows):
        y = PLOT_TOP + ROW * index + ROW / 2
        canvas.text(MARGIN + LABEL_WIDTH - 10, y, method, 12.5, ha="right", va="center_baseline")
    for panel, (column, title) in enumerate(zip(COLUMNS, text["columns"])):
        left = MARGIN + LABEL_WIDTH + panel * (panel_width + PANEL_GAP)

        def scale(value: float) -> float:
            return left + VALUE_ROOM + (value - low) / (high - low) * (panel_width - 2 * VALUE_ROOM)

        canvas.text(left, 120, title, 13, "bold")
        canvas.text(left, 138, text["reference"].format(value=reference[column]), 11.5, color="muted")
        if panel == 0:
            canvas.rect(scale(-BAND), PLOT_TOP, scale(BAND), plot_bottom, "band", z=0)
        for tick in range(math.ceil(low / 5) * 5, math.floor(high) + 1, 5):
            canvas.line(scale(tick), PLOT_TOP, scale(tick), plot_bottom, "grid", z=1)
            canvas.text(scale(tick), plot_bottom + 18, signed(tick, "d"), 11, color="muted", ha="center")
        canvas.line(scale(0), PLOT_TOP - 4, scale(0), plot_bottom + 4, "baseline", width=1.2, z=4)
        for index, (_, row) in enumerate(rows):
            value = row[column]
            y_top = PLOT_TOP + ROW * index + (ROW - BAR) / 2
            canvas.bar(scale(0), scale(value), y_top, "above" if value > 0 else "below")
            end = scale(value)
            canvas.text(
                end + (5 if value >= 0 else -5),
                y_top + BAR / 2,
                signed(value, ".1f"),
                10.5,
                color="secondary",
                ha="left" if value >= 0 else "right",
                va="center_baseline",
            )
    canvas.text(MARGIN, height - 26, text["footer"], 11.5, color="muted")

    stem = "comparison_rlvr" + ("" if lang == "en" else f"_{lang}") + ("" if mode == "light" else "_dark")
    out_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "comparison"}):
        canvas.fig.savefig(out_dir / f"{stem}.svg", transparent=True, metadata={"Date": None})
    if png_dpi:
        canvas.fig.savefig(out_dir / f"{stem}.png", dpi=png_dpi, transparent=True)
    plt.close(canvas.fig)
    print(f"wrote {out_dir / stem}.svg ({WIDTH}x{height:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--font-regular", required=True, help="regular font file with Latin and CJK glyphs")
    parser.add_argument("--font-bold", required=True, help="bold font file with Latin and CJK glyphs")
    parser.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "assets"))
    parser.add_argument("--png", action="store_true", help="also write a PNG next to each SVG (for checking)")
    parser.add_argument("--png-dpi", type=int, default=200)
    args = parser.parse_args()
    fonts = {"regular": args.font_regular, "bold": args.font_bold}
    for lang in ["en", "zh"] if args.lang == "all" else [args.lang]:
        for mode in THEMES:
            render(lang, mode, fonts, Path(args.out_dir), args.png_dpi if args.png else 0)


if __name__ == "__main__":
    main()
