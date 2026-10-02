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
"""Consistency checks for the README paper list and the Markdown links (run by CI).

    python scripts/check_docs.py          # check
    python scripts/check_docs.py --fix    # also rewrite the paper-count badges

Checks: the "Papers" / "Reproduced" badges match the lists, README.md and README_zh.md list the
same papers, every table of "Other papers" is sorted newest first without duplicates, and every
relative Markdown link or image points to an existing file (and, with ``#anchor``, to an existing
heading).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READMES = {
    "README.md": ("Paper list", "Reproduced in this repository", "Other papers"),
    "README_zh.md": ("论文清单", "本仓库已复现的论文", "其他论文"),
}
ROW = re.compile(r"^\| (\d{4}(?:-\d{2})?) \|")
ARXIV_ID = re.compile(r"`(\d{4}\.\d{4,5})`|arxiv\.org/abs/(\d{4}\.\d{4,5})")
BADGE = re.compile(r"(img\.shields\.io/badge/{name}-)(\d+)(-)")
LINK = re.compile(r"\]\(([^)\s]*?)#([^)\s]+)\)")
FILE_LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)|<img[^>]*\ssrc=\"([^\"]+)\"")


def section(text: str, level: int, title: str) -> str:
    """Body of the first heading of ``level`` whose text ends with ``title`` (emoji prefixes allowed)."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6}) (.*)$", line)
        if match and start is None and len(match.group(1)) == level and match.group(2).endswith(title):
            start = index + 1
        elif match and start is not None and len(match.group(1)) <= level:
            return "\n".join(lines[start:index])
    if start is None:
        raise KeyError(f"heading '{'#' * level} ... {title}' not found")
    return "\n".join(lines[start:])


def github_slug(heading: str) -> str:
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    seen: Counter = Counter()
    anchors = set()
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        match = re.match(r"^#{1,6} (.*)$", line)
        if match and not in_code:
            slug = github_slug(match.group(1))
            anchors.add(slug if seen[slug] == 0 else f"{slug}-{seen[slug]}")
            seen[slug] += 1
    return anchors


def check_paper_list(fix: bool) -> list[str]:
    errors: list[str] = []
    ids_by_file = {}
    for name, (list_title, reproduced_title, other_title) in READMES.items():
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        paper_list = section(text, 2, list_title)
        reproduced = section(paper_list, 3, reproduced_title)
        others = section(paper_list, 3, other_title)
        n_reproduced = sum(1 for line in reproduced.splitlines() if line.startswith("- **"))
        n_others = 0
        for table in re.split(r"(?m)^#### ", others)[1:]:
            title = table.splitlines()[0]
            dates = [ROW.match(line).group(1) for line in table.splitlines() if ROW.match(line)]
            n_others += len(dates)
            for previous, current in zip(dates, dates[1:]):
                if current > previous:
                    errors.append(f"{name}: '{title}' is not sorted newest first ({previous} before {current})")
                    break
        # one id per entry (an entry repeats its id in the link and in the label)
        ids = [i for line in paper_list.splitlines() for i in {a or b for a, b in ARXIV_ID.findall(line)}]
        duplicates = sorted(i for i, count in Counter(ids).items() if count > 1)
        if duplicates:
            errors.append(f"{name}: arXiv ids listed more than once: {', '.join(duplicates)}")
        ids_by_file[name] = set(ids)
        expected = {"Papers": n_reproduced + n_others, "Reproduced": n_reproduced}
        for badge, value in expected.items():
            pattern = re.compile(BADGE.pattern.format(name=badge))
            match = pattern.search(text)
            if match is None:
                errors.append(f"{name}: '{badge}' badge not found")
            elif int(match.group(2)) != value:
                if fix:
                    text = pattern.sub(rf"\g<1>{value}\g<3>", text, count=1)
                    print(f"{name}: {badge} badge {match.group(2)} -> {value}")
                else:
                    errors.append(
                        f"{name}: '{badge}' badge says {match.group(2)}, the list has {value} (run with --fix)"
                    )
        if fix:
            path.write_text(text, encoding="utf-8")
    english, chinese = ids_by_file["README.md"], ids_by_file["README_zh.md"]
    for name, missing in (("README_zh.md", english - chinese), ("README.md", chinese - english)):
        if missing:
            errors.append(f"{name}: missing papers listed in the other README: {', '.join(sorted(missing))}")
    return errors


def check_links() -> list[str]:
    errors: list[str] = []
    tracked = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    cache: dict[Path, set[str]] = {}
    for relative in tracked.stdout.split():
        source = ROOT / relative
        text = source.read_text(encoding="utf-8")
        for link, image in FILE_LINK.findall(text):
            target = link or image
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("<"):
                continue
            if not (source.parent / target).exists():
                errors.append(f"{relative}: link to '{target}' points to a missing file")
        for target, anchor in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (source.parent / target).resolve() if target else source
            if path.suffix != ".md" or not path.is_file():
                continue
            if path not in cache:
                cache[path] = anchors_of(path)
            if anchor not in cache[path]:
                errors.append(f"{relative}: link to '{target}#{anchor}' has no matching heading")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="rewrite the paper-count badges")
    args = parser.parse_args()
    errors = check_paper_list(args.fix) + check_links()
    for error in errors:
        print(f"error: {error}")
    if errors:
        sys.exit(1)
    print("docs: ok")


if __name__ == "__main__":
    main()
