# Qwen3-VL-2B-Instruct 上的 OPD 统一设定对比

[English](README.md) | [简体中文](README_zh.md)

在线策略蒸馏（On-Policy Distillation, OPD）让学生在自己采样的回答上向更强教师的下一 token 分布学习。本仓库复现的五篇 OPD 论文（[VA-OPD](../../reproduction/va_opd/README.md)、[VGS](../../reproduction/vgs/README.md)、[VCSD](../../reproduction/vcsd/README.md)、[Vision-OPD](../../reproduction/vision_opd/README.md)、[S-OPD](../../reproduction/s_opd/README.md)）使用不同的学生、教师、数据和评测。本目录在**同一套**设定下训练其中不需要区域标注的四个方法、两种基础 OPD 目标和一个 GRPO 参照：沿用 [RLVR 对比](../rlvr_qwen3_vl_4b/README_zh.md)的设定，学生换成 2B 模型，教师是在该对比中经 GRPO 训练的 Qwen3-VL-4B-Instruct 的验证最优步（见[教师](#教师)）。每个脚本只设置 `ALGO_ARGS`；其余配置都来自 [common.sh](common.sh)。其他对比见 [examples/comparison/](../README_zh.md)。

> [!IMPORTANT]
> **这些结果仅供参考，不作为对方法本身的评判。** 这些方法都是我们的复现，每个配置只训练一次（RLVR 对比中同一配置的两次运行，三组平均相差 0.6），统一设定也不是各方法设计和调参时所用的设定。阅读表格前请先看[关于结果的说明](../README_zh.md#关于结果的说明)。

## 统一设定

| | `examples/comparison/opd_qwen3_vl_2b/common.sh` | RLVR 对比（`../rlvr_qwen3_vl_4b/common.sh`） |
| --- | --- | --- |
| 学生 / 教师 | Qwen3-VL-2B-Instruct / 在 RLVR 对比中经 GRPO 训练的 Qwen3-VL-4B-Instruct，取验证最优步（VCSD：学生的 EMA） | Qwen3-VL-4B-Instruct / – |
| 数据、验证、提示词 | ViRL39K；每 15 步在 MMK12 test 上验证；`math_perception.jinja`，`<think>` 按普通文字分词 | 相同；每 5 步验证 |
| 采样 | 每步 128 道题 × 8 个回答，**每步只更新一次**，T=1.0，**top-p 1.0** | 每步 384 × 8，每次更新 128 道题（每步更新 3 次），top-p 0.99 |
| 损失 | 各方法自己的目标，token 级平均 | GRPO，token 级，clip 0.2 / 0.2 |
| 正则化 | **无**（不加参考模型 KL，不加熵项） | KL 损失 0.01，熵惩罚 0.005 |
| 奖励 | 0.9 准确率 + 0.1 格式，只记录（只有 `grpo.sh` 用它训练） | 相同，进入损失 |
| 优化 | AdamW（bf16），学习率 1e-6 恒定，视觉编码器参与训练，2 个 epoch（606 次更新） | 相同（606 次更新） |
| 长度 / 像素 | 提示词最长 4,096，回答最长 2,048；200,704-1,003,520 像素 | 相同 |
| GPU | 4 张 | 4 张 |

两套配置每次更新都用 1,024 个回答，更新次数也相同。OPD 每个采样批次只更新一次，使每次更新都在当前策略上进行：这样分布级损失不需要重要性权重，各 OPD 论文也是这样做的。采样不截断（top-p 1.0），因为这些目标是在学生自身分布下的期望；有教师时训练器会强制这一点。教师读取学生的 token id 和像素值，训练器在启动时检查两个模型是否一致。

## 教师

教师是 `../rlvr_qwen3_vl_4b/grpo.sh` 在验证奖励最高的一步（其 `checkpoint_tracker.json` 中的 `best_global_step`；我们的运行中是 202 步中的第 175 步），而不是某个 Instruct 模型：

- **长度。** 在这个提示词和 2,048 token 的上限下，Instruct 模型在很多题上会写超上限：在 128 道 MMK12 test 题上每题采样 4 次（T=1.0），Qwen3-VL-8B-Instruct 有 43% 的回答超过 2,048 token（不设上限时准确率 0.676，限制在上限内为 0.469），Qwen3-VL-2B-Instruct 为 44%，换成非思考的对话模板也不会变短。OPD 学生会保留教师冗长的推理（[COPD](https://arxiv.org/abs/2607.19046) 表 1：从 Qwen3-VL-8B 做标准 OPD 后为 2.23K token，2B 基座模型为 2.12K），而 GRPO 参照会学会在上限内结束。该 GRPO 运行的验证回答只有约 1-2% 被截断。[VGS](../../reproduction/vgs/README.md) 使用同一模型系列和同样的上限，也是先用 GRPO 训练教师再蒸馏。
- **规模。** 在 Qwen3-VL 技术报告（表 4）的大多数推理基准上，Qwen3-VL-8B-Instruct 只比 4B 高 1-3.5 分；在 VA-OPD 的表 2 中，标准 OPD 从 4B 和 8B 教师把同一个 2B 学生带到几乎相同的分数（数学平均 45.3 和 45.4）。4B 的 GRPO 运行不需要额外训练，还把教师的开销减半。
- **步数。** GRPO 的熵和 KL 在最后若干步会漂移（我们的运行中约从第 188 步开始）。在去掉 MMK12 的对比套件上，验证最优步的分数也高于最后一步（我们的运行中为 72.1 对 70.5，主要差在数学组）。MMK12 test 是验证集，因此这一选择可能让学生的 MMK12 分数略占便宜。

先运行 `../rlvr_qwen3_vl_4b/grpo.sh` 并保留它的最优步（`python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/grpo` 会保留最后一步和最优步，也可以用 `scripts/model_merger.py` 合并）；脚本从那里读取教师，如果该步没有完整权重，会在训练开始前停止。在命令行上设置 `TEACHER_PATH` 或 `worker.teacher.model.model_path=...` 可以换成别的教师，设置 `GRPO_TEACHER_RUN` 可以换成别的 GRPO 运行。

## 方法

| 脚本 | 方法 | `ALGO_ARGS`（节选） |
| --- | --- | --- |
| `grpo.sh` | GRPO 参照 | RLVR 对比的 `grpo.sh`，学生换成 2B |
| `opd_sampled.sh` | 基于采样 token 的 OPD | `adv_estimator=teacher_log_ratio`（逐 token 的 log q − log π_old，截断到 ±10），配合 PPO 损失 |
| `opd_full.sh` | 基于完整分布的 OPD（VA-OPD 和 VGS 中的 "Standard OPD"） | `distill_loss_coef=1`，`policy_loss_coef=0`，`distill_divergence=reverse_kl` |
| `va_opd.sh` | [VA-OPD](../../reproduction/va_opd/README.md) | `opd_full` + `distill_weighting=va_opd`（像素化 0.1，由教师打分的 `sampled_positive_log_ratio`，τ 1，p_v 0.2，λ 0.5） |
| `vgs.sh` | [VGS](../../reproduction/vgs/README.md) | `distill_target=visual_gain`，`distill_contrast_view=no_image`，γ 2，λ 0.01，在每个 micro-batch 内按教师的视觉依赖取前 30%，η 0.41 |
| `vcsd.sh` | [VCSD](../../reproduction/vcsd/README.md) | EMA 教师（0.05），`distill_target=contrast_sharpened`，`distill_contrast_view=black`，T=2 下的前向 KL（乘以 4），α 1，β 0.1 |
| `s_opd.sh` | [S-OPD](../../reproduction/s_opd/README.md) | `opd_sampled` + TPC（`random_patch` 16 px，p 0.6，`visual_sensitivity_loss_weighting=teacher`，0.02）+ PA（`rgb_gaussian_noise` σ 0.2，`visual_robustness_loss_coef` 0.02） |
| `opd_full_seq.sh` | 可选消融，不在表中 | `opd_full.sh` 改为先在每个回答内平均、再在回答间平均 |

各论文中属于其基础配置而非方法本身的设置（批大小、学习率、长度、数据和提示词）都换成统一设定；方法自身的组件和常数保持不变。说明：

- **与 GRPO 的比较并不对等。** OPD 方法向教师（4B 的 GRPO 运行）学习，而 GRPO 这一行看不到教师；`grpo.sh` 只是把它们和同一学生、同样数据和训练量下的强化学习放在一起。VCSD 不需要外部教师（它从学生的 EMA 蒸馏），因此是唯一不用这个教师的 OPD 方法。
- **损失平均方式。** 所有方法都在批次内的所有回答 token 上平均，与已公开的 OPD 代码（VCSD、Vision-OPD）以及 verl、TRL 和 NeMo-RL 一致；论文中的损失写成先在每个回答内平均、再在回答间平均。VA-OPD 的权重按定义在每道题内归一化（同一道题的权重之和为 1），因此它与 `opd_full.sh` 的差别也包含这一归一化；`opd_full_seq.sh`（按回答平均）可以把这部分分离出来。
- **VGS** 使用 η = 0.41，即论文给 2B 学生的值；有梯度裁剪和 AdamW 时，整个损失乘一个常数影响不大。它的门控在每个更新 micro-batch 内取 70% 分位数。
- **VCSD** 按论文所述对所有分布都施加蒸馏温度（其公开代码只乘了 T² 因子；见 VCSD README）。
- **S-OPD** 建立在基于采样 token 的 OPD 之上，所以它的对照是 `opd_sampled.sh`。两个系数取其论文主实验的值（0.02 和 0.02），遮挡块为 16 px、遮挡概率 0.6，噪声 σ 为 0.2；遮挡图和加噪图与本对比的其他视图一样在缩放后的图像上生成，每条回答各一份（其公开代码按样本各生成一份）。教师门控使用本对比的教师。
- **Vision-OPD** 没有包含在内：它需要每道题的区域标注来给教师裁剪图像，而 ViRL39K 没有这种标注。
- 学生用 2B 模型：在 Qwen3-VL 技术报告（表 4）的大多数数学和推理基准上，Qwen3-VL-2B-Instruct 比 4B 和 8B 低 10-22 分，因此有向教师学习的空间。

## 运行

```bash
bash scripts/prepare_data.sh opd_comparison
bash examples/comparison/rlvr_qwen3_vl_4b/grpo.sh                    # 教师（如果 RLVR 对比还没有运行它）
python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/grpo   # 保留其最后一步和最优步
for m in grpo opd_sampled opd_full va_opd vgs vcsd s_opd; do
    bash examples/comparison/opd_qwen3_vl_2b/$m.sh
done
bash scripts/prepare_eval_data.sh opd
bash scripts/eval.sh checkpoints/Comparison-OPD-Qwen3-VL-2B/va_opd --suite opd
python3 scripts/finalize_run.py checkpoints/Comparison-OPD-Qwen3-VL-2B/*   # 训练结束后：保留最后一步和最优步
```

检查点保存在 `checkpoints/Comparison-OPD-Qwen3-VL-2B/<method>`。`MODEL_PATH` 和 `TEACHER_PATH` 分别设置学生和教师；在脚本后追加 `key=value` 可以覆盖统一设定。教师会增加前向计算，在 4 张 GPU 上每张约多占 2 GB 的 bf16 权重：更新时，蒸馏损失在每个 micro-batch 做一次教师前向（VGS 和 VCSD 还要算对照视图，共两次）；更新前，基于采样 token 的 OPD、VA-OPD 的权重和 S-OPD 需要教师给出采样 token 的 log-prob（VA-OPD 还要在像素化的图像上算一次，S-OPD 还要在遮挡图上算一次，同时学生也要在遮挡图和加噪图上各算一次）。

## 结果

`opd` 套件：与 RLVR 对比相同的 19 项基准和三个分组，评测协议和两种均值（加粗的三组平均，以及 19 项基准平均）也相同；见 [RLVR 对比](../rlvr_qwen3_vl_4b/README_zh.md#结果)。所有方法都取最后一步评测。

| 方法 | 数学推理 | 依赖视觉的推理 | 感知与幻觉 | 三组平均 | 19 项基准平均 | 训练时长（h） |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct（基线） | 36.7 | 51.8 | 69.0 | 52.5 | 51.7 | – |
| 教师：经 GRPO 训练的 Qwen3-VL-4B-Instruct（验证最优步） | 66.4 | 72.4 | 78.0 | 72.3 | 72.0 | – |
| GRPO | 55.3 | 64.5 | 72.3 | **64.0** | 63.6 | 21.0 (1.00×) |
| OPD（基于采样 token） | 51.3 | 63.1 | 72.7 | **62.4** | 61.8 | 23.0 (1.10×) |
| OPD（基于完整分布） | 51.8 | 63.8 | 72.9 | **62.8** | 62.3 | 25.2 (1.20×) |
| VA-OPD | 46.7 | 60.7 | 73.2 | **60.2** | 59.5 | 34.4 (1.64×) |
| VGS | 49.5 | 61.3 | 72.4 | **61.0** | 60.4 | 35.3 (1.68×) |
| VCSD（无外部教师） | 2.7 | 16.3 | 35.5 | **18.2** | 17.3 | 40.3 (1.92×) |
| S-OPD | 51.1 | 62.7 | 72.8 | **62.2** | 61.6 | 35.7 (1.70×) |

**训练时长**是训练步在 4 张 A800 80GB 上的墙钟时间（小时）：OPD 为 606 步，GRPO 为 202 步、每步更新 3 次，两者都是 606 次更新、每次 1,024 个回答；包含采样、教师前向、奖励和更新，不含验证和检查点保存；括号中是相对 GRPO 的倍数。部分运行与另一个运行共用一个 8 卡节点，因此几个百分点以内的差异可能来自节点负载。VCSD 越来越多的回答写到 2,048 token 的上限，所以耗时更长。

### 数学推理

| 方法 | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVision | DynaMath | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct（基线） | 30.9 | 56.6 | 46.8 | 29.6 | 36.7 | 8.7 | 47.2 | 36.7 |
| 教师：经 GRPO 训练的 Qwen3-VL-4B-Instruct（验证最优步） | 58.4 | 74.1 | 75.2 | 69.3 | 73.9 | 39.9 | 73.7 | 66.4 |
| GRPO | 45.4 | 65.2 | 68.1 | 52.2 | 64.0 | 29.5 | 62.4 | 55.3 |
| OPD（基于采样 token） | 41.4 | 61.6 | 63.4 | 48.5 | 57.4 | 24.4 | 62.4 | 51.3 |
| OPD（基于完整分布） | 41.9 | 61.7 | 63.7 | 49.7 | 58.4 | 24.2 | 63.2 | 51.8 |
| VA-OPD | 37.8 | 59.3 | 58.1 | 43.6 | 50.2 | 17.8 | 60.2 | 46.7 |
| VGS | 38.8 | 60.4 | 60.8 | 45.7 | 56.0 | 22.8 | 61.7 | 49.5 |
| VCSD（无外部教师） | 0.0 | 15.1 | 0.0 | 0.1 | 0.1 | 0.0 | 3.4 | 2.7 |
| S-OPD | 40.9 | 61.7 | 63.3 | 48.2 | 56.9 | 24.1 | 62.4 | 51.1 |

### 依赖视觉的推理

| 方法 | MathVerse-V | MMMU-Pro | LogicVista | 计数 | AI2D | MME（认知） | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct（基线） | 33.8 | 20.2 | 31.2 | 69.3 | 68.0 | 88.0 | 51.8 |
| 教师：经 GRPO 训练的 Qwen3-VL-4B-Instruct（验证最优步） | 69.6 | 43.6 | 58.7 | 87.7 | 83.0 | 92.1 | 72.4 |
| GRPO | 60.4 | 31.8 | 47.3 | 83.2 | 74.6 | 89.6 | 64.5 |
| OPD（基于采样 token） | 53.2 | 29.0 | 46.6 | 85.7 | 74.1 | 90.0 | 63.1 |
| OPD（基于完整分布） | 54.6 | 29.1 | 47.0 | 86.9 | 74.7 | 90.5 | 63.8 |
| VA-OPD | 45.6 | 26.8 | 42.7 | 86.1 | 73.3 | 89.4 | 60.7 |
| VGS | 51.7 | 27.7 | 44.5 | 82.8 | 73.2 | 87.6 | 61.3 |
| VCSD（无外部教师） | 0.1 | 0.1 | 0.5 | 71.4 | 3.1 | 22.8 | 16.3 |
| S-OPD | 53.3 | 29.0 | 45.8 | 84.8 | 73.2 | 89.8 | 62.7 |

### 感知与幻觉

| 方法 | POPE | HallusionBench | MMStar | BLINK | MME（感知） | CV-Bench | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct（基线） | 88.2 | 65.2 | 53.9 | 52.6 | 79.3 | 75.1 | 69.0 |
| 教师：经 GRPO 训练的 Qwen3-VL-4B-Instruct（验证最优步） | 88.8 | 74.0 | 69.6 | 66.3 | 84.0 | 85.2 | 78.0 |
| GRPO | 88.3 | 67.5 | 58.6 | 57.8 | 81.6 | 80.0 | 72.3 |
| OPD（基于采样 token） | 88.7 | 67.9 | 62.1 | 57.3 | 81.8 | 78.5 | 72.7 |
| OPD（基于完整分布） | 88.4 | 70.0 | 60.7 | 55.9 | 83.9 | 78.7 | 72.9 |
| VA-OPD | 88.8 | 69.0 | 60.1 | 58.8 | 83.1 | 79.4 | 73.2 |
| VGS | 87.6 | 68.6 | 60.5 | 57.4 | 81.8 | 78.7 | 72.4 |
| VCSD（无外部教师） | 89.0 | 26.0 | 8.0 | 6.3 | 53.9 | 29.7 | 35.5 |
| S-OPD | 88.0 | 69.4 | 61.1 | 57.4 | 82.9 | 78.1 | 72.8 |

MME（认知）和 MME（感知）分别是 MME 的认知分（常识推理、数值计算、文本翻译和代码推理；260 题，满分 800）和感知分（另外 10 个子任务；2,114 题，满分 2,000），都换算到 0-100。

### 验证最优步

各方法在验证奖励最高的一步上的表现（运行的 `checkpoint_tracker.json` 中的 `best_global_step`；每 15 步在 MMK12 test 上验证），在 MMK12 之外的 18 项基准上计算，与 [RLVR 对比](../rlvr_qwen3_vl_4b/README_zh.md#验证最优步)相同。最后两列是最后一步（第 606 步；GRPO 为第 202 步）上同样的平均，以及两者之差（按表中保留一位小数的数值计算）。

| 方法 | 最优步 | 数学推理（不含 MMK12） | 依赖视觉的推理 | 感知与幻觉 | 三组平均 | 最后一步的三组平均 | 最优步 − 最后一步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | 202 | 55.8 | 64.5 | 72.3 | **64.2** | 64.2 | 0.0 |
| OPD（基于采样 token） | 606 | 51.8 | 63.1 | 72.7 | **62.5** | 62.5 | 0.0 |
| OPD（基于完整分布） | 600 | 52.1 | 63.6 | 72.7 | **62.8** | 63.0 | −0.2 |
| VA-OPD | 255 | 52.3 | 62.9 | 72.1 | **62.5** | 60.4 | +2.1 |
| VGS | 600 | 49.9 | 61.2 | 72.0 | **61.0** | 61.3 | −0.3 |
| VCSD（无外部教师） | 30 | 43.3 | 55.9 | 71.1 | **56.8** | 18.3 | +38.5 |
| S-OPD | 600 | 51.3 | 62.4 | 72.6 | **62.1** | 62.3 | −0.2 |

VCSD 的验证奖励在第 30 步最高，之后持续下降；到最后一步，它的大多数回答都写到了 2,048 token 的上限。它的论文训练 90 步、每步 32 道题 × 8 个回答，约为这里 606 次更新、每次 1,024 个回答的 4%，因此这里的训练量远长于 VCSD 设计时所用的训练量。
