# 贡献指南

[English](CONTRIBUTING.md) | 中文

欢迎任何形式的贡献：补充论文、新增方法、新增评测基准、提交复现结果、报告问题、修改文档。

## 贡献方式

- **补充论文。** 在 `README.md` 和 `README_zh.md` 的清单中按时间倒序各加一行：arXiv 链接、第一作者、会议
  （仅当 arXiv 备注、OpenReview 或官方仓库能确认时才写）、代码链接和一句话核心思路。
- **新增方法。** 参考 [docs/add_method.md](../docs/add_method.md)。新方法需要带校验的配置项、单元测试、
  `examples/reproduction/<method>/` 目录（基线与方法脚本），以及说明与官方配方差异的 README。
- **新增评测基准。** 参考 [eval/README.md](../eval/README.md#adding-a-new-benchmark) 中的教程。
- **提交复现结果。** 用 "Reproduction results" 模板开 issue，附上完整命令、提交哈希、训练日志
  （`experiment_log.jsonl`）和评测汇总。

## 用 pre-commit 做检查

仓库的检查由 [pre-commit](https://pre-commit.com) 执行：同一套检查在你本机每次提交前运行，也在 CI 中对每个
PR 运行，所以本地通过的 PR，CI 的代码检查也会通过。

**一次性设置**（pre-commit 需要 git 2.31 或更新的版本；第一次运行会从 GitHub 下载各项检查的运行环境）：

```bash
bash scripts/install_env.sh   # 训练环境，同时安装 pytest、ruff 和 pre-commit
# 只改文档或论文清单时，也可以只装：
pip install pre-commit
pre-commit install            # 在你的仓库目录中执行：安装提交前检查和提交信息检查
```

**每次提交时**，检查会作用于暂存的文件和提交信息。会自动修复的检查（行尾空格、文件末尾换行、ruff）
改完文件后会中止这次提交：用 `git diff` 看一下改动，`git add` 之后再提交一次。只检查不修复的项会打印出要改的地方。

**开 PR 之前**，对整个仓库跑一遍全部检查和单元测试：

```bash
pre-commit run --all-files    # 或者 make commit（安装钩子并运行）
make test                     # CPU 上的单元测试（python -m pytest -q tests/）
DRY_RUN=1 bash examples/reproduction/<method>/<script>.sh   # 不用 GPU 校验启动脚本
```

| 检查 | 内容 | 不通过时 |
|---|---|---|
| ruff check、ruff format | 代码规范和格式（ruff 版本与 `scripts/constraints.txt` 一致） | 多数问题会被自动修复；其余手动修改（`make style` 两项都会执行） |
| Apache license headers | 每个 Python 文件开头有许可证声明 | 照抄仓库里任一 Python 文件的开头 |
| shell syntax | 用 `bash -n` 检查 shell 脚本语法 | 按报错修改对应行 |
| paper list, badges and Markdown links | `scripts/check_docs.py`：两份 README 的论文一致且按时间倒序、徽章计数正确、链接有效 | `python scripts/check_docs.py --fix` 会更新徽章；其余按报错修改 |
| 文件检查 | Python 和 YAML 语法、没有合并冲突标记、文件小于 25 MB、`requirements.txt` 有序、没有行尾空格 | 自动修复，或按报错修改 |
| don't commit to branch | 不在 `main` 上提交 | 在你 fork 的分支上提交 |
| commit message format | 提交信息符合下面的[提交信息](#提交信息)规范 | 用 `git commit --amend` 改正提交信息 |

`git commit --no-verify` 会跳过检查；CI 跑的是同一套检查，所以只在保存自己分支上未完成的工作时使用。

## 提交信息

```
[type] scope: summary

正文：为什么要改，改动前后的行为分别是什么。

Impact: 对已有运行、结果、缓存或已准备数据的影响
Refs: #123
```

**标题。** 用英文：`[type]` 加可选的 `scope:`，然后是祈使句的概括，以小写动词开头，不加句号，整行不超过
72 个字符。写改了什么行为，而不是改了哪个文件："give no accuracy to answers of 1,000+ characters"，
而不是 "update deepeyes.py"。

**type**（只选一个）：

| type | 用途 |
|---|---|
| `feat` | 新能力：选项、方法、基准、脚本或工具 |
| `fix` | 代码没做到它应该做的事 |
| `align` | 对齐论文或其官方代码（设置、奖励、提示、评测协议） |
| `config` | 出于其他原因修改默认值或脚本设置 |
| `data` | 数据准备、转换脚本、发布的数据集 |
| `list` | `README.md` 和 `README_zh.md` 中的论文清单 |
| `results` | 结果表 |
| `docs` | 只改文档 |
| `test` | 只改测试 |
| `refactor` | 重构，不改变行为 |
| `perf` | 速度或显存，结果不变 |
| `build` | 依赖、CI、钩子、打包 |
| `revert` | 回退某个提交（正文写明被回退的提交和原因） |

**scope**（可选，只写一个，小写）：方法名（`cgpo`、`papo`、`vppo`、`tor`、`dvrp`、`pgpo`、`pepo`、
`cfpo`、`vepo`、`grit`、`deepeyes`）、`comparison`，或组件名（`trainer`、`actor`、`rollout`、`agent`、
`reward`、`data`、`eval`、`docs`、`ci`）。改动涉及多处时省略。

**正文。** 写清为什么改、改了什么，每行不超过 72 列；只有标题已经说尽时（如修正错别字）才可以省略。
`align` 提交要写出处：论文的章节或表格，或官方代码的文件与函数（或提交）。`fix` 提交要写出错的具体情形。
描述其他作者的代码时就事论事（"the released code reads ..."、"differs from"）。

**Impact。** 以下任一情况都必须写：已有脚本的结果会变、已缓存的评测结果失效、数据需要重新准备、原来能通过的配置现在会被拒绝。

**一个改动一个提交。** 一个改动的测试和文档放在它的提交里，不相关的改动分开提交。

示例：

```
[fix] deepeyes: give no accuracy to answers of 1,000+ characters
[align] vepo: update on 128 prompts per step, as the paper
[feat] eval: add the tallyqa_relabeled benchmark
[config] pepo: save a checkpoint every 25 steps
[list] add VEPO (2606.03937)
```

PR 以 **squash and merge** 方式合并，**PR 标题**会成为 `main` 上那个提交的标题，所以 PR 标题也按同一格式写。
CI 会检查 PR 标题（修改标题后会重新检查），不检查 PR 内部的各个提交。本地的提交信息检查会作用于每一次提交；
以 `fixup!` 和 `squash!` 开头的提交不检查。

## 提交 PR

1. fork 本仓库，从 `main` 拉一个分支。
2. 配好环境和检查（见[用 pre-commit 做检查](#用-pre-commit-做检查)）。
3. 完成改动并补测试，运行检查、单元测试，改了启动脚本的还要跑 `DRY_RUN`。
4. 用符合[提交信息](#提交信息)格式的标题开 PR，并填写模板。GitHub Actions 会对每个 PR 运行 pre-commit 检查、
   CPU 单元测试和标题检查；CI 不跑 GPU 训练和评测，请在 PR 中说明你是怎么测试的。

请互相尊重，并遵守[行为准则](CODE_OF_CONDUCT.md)。
