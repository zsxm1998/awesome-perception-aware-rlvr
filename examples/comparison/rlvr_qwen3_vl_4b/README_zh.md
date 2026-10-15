# Qwen3-VL-4B-Instruct 上的 RLVR 统一设定对比

[English](README.md) | [简体中文](README_zh.md)

[examples/reproduction/](../../reproduction/) 中复现的感知导向 RLVR 论文使用不同的基座模型、训练集、基线定义和评测协议，其中有几篇还沿用了其他论文的基线数字。本目录用**同一套**配置训练除 DeepEyes 外的所有 RLVR 方法（[原因](#为什么不包含-deepeyes)）：相同的模型、相同的数据、相同的 GRPO 超参数和相同的评测。每个脚本只设置 `ALGO_ARGS`，即定义该方法的开关；其余配置都来自 [common.sh](common.sh)。其他对比见 [examples/comparison/](../README_zh.md)。

> [!IMPORTANT]
> **这些结果仅供参考，不作为对方法本身的评判。** 这些方法都是我们的复现，每个配置只训练一次（同一配置的两次运行，三组平均相差 0.6），统一设定也不是各方法设计和调参时所用的设定。阅读表格前请先看[关于结果的说明](../README_zh.md#关于结果的说明)。

## 统一设定

| | `examples/comparison/rlvr_qwen3_vl_4b/common.sh` |
| --- | --- |
| 模型 | Qwen3-VL-4B-Instruct；`<think>` / `</think>` 按普通文字分词，因为它们在 Instruct 检查点中是未训练的附加 token（`worker.actor.model.plain_think_tokens=auto`，见 [README](../../../README_zh.md#-配置)） |
| 训练数据 | PAPO 处理过的 ViRL39K（`PAPOGalaxy/PAPO_ViRL39K_train`，38,870 条）；`vapo` 按相同顺序读取同样的数据，并附加 VAPO 的视觉断言列 |
| 验证 | MMK12 test（`PAPOGalaxy/PAPO_MMK12_test`，2,000 条），每 5 步一次，每题 8 个样本，T=1.0，top-p 0.99 |
| 提示词 / 奖励 | `<think>` + `\boxed{}`（`examples/format_prompt/math_perception.jinja`）；0.9 准确率 + 0.1 格式（`examples/reward_function/math.py:compute_score`）。`grit` 和 `cgpo` 换成各自的定位格式（见下文） |
| 采样 | 每步 384 道题 × 8 个回答（`data.rollout_batch_size=384`，`data.mini_rollout_batch_size=128`），T=1.0，top-p 0.99 |
| 策略更新 | GRPO；每次更新 128 道题（每步更新 3 次）；clip 0.2 / 0.2；token 级损失 |
| 正则化 | 对参考模型的 low-var KL 损失，系数 0.01；全词表熵惩罚 0.005（`algorithm.invariant_entropy_coef=0.005`，`algorithm.entropy_loss_type=full`） |
| 优化 | AdamW（bf16），学习率 1e-6 恒定，视觉编码器参与训练，2 个 epoch |
| 长度 / 像素 | 提示词最长 4,096，回答最长 2,048；200,704-1,003,520 像素 |
| GPU | 4 张（A800 80GB） |

## 方法

所有方法都使用上面的 GRPO 配置，包括论文基于 DAPO 的方法，使得只有感知导向的部分不同；DAPO 作为第二个基线（更强的强化学习配置，不含任何感知导向的部分）。每个方法沿用其论文中的图像扰动（"辅助视图"一列），因为它的阈值和系数是针对这种扰动调的；块遮挡使用 Qwen3-VL 的 16 像素图块。参数名见 [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md)。

| 脚本 | 方法 | 辅助视图 | `ALGO_ARGS`（节选） |
| --- | --- | --- | --- |
| `grpo.sh` | GRPO 基线 | – | 无 |
| `dapo.sh` | DAPO 基线 | – | 不加 KL（`algorithm.disable_kl=true`），clip-higher 0.2 / 0.28，按准确率在 (0.01, 0.99) 内做动态采样；保留共用的熵惩罚 |
| `papo.sh` | [PAPO](../../reproduction/papo/README.md) | 随机 16 像素图块以 p=0.6 涂黑，每道题一个遮罩 | 以 `visual_sensitivity_loss_coef=0.01` 最大化 KL_prcp，`visual_sensitivity_reference=current`；除共用熵惩罚外不加 Double Entropy 项 |
| `vppo.sh` | [VPPO](../../reproduction/vppo/README.md) | 随机 16 像素图块以 p=0.5 涂黑，每个回答一个遮罩 | TGF：每个回答 `top_perception_quantile=0.4`；TAS：`response_advantage_scaling_method=vppo`，`vppo_response_scaling_min=0.9`；`visual_sensitivity_reference=old`；不加 0.06 的熵惩罚 |
| `tor.sh` | [ToR](../../reproduction/tor/README.md) | 去掉图像（`no_image`） | 在 top-p 0.95 集合的熵上取 `top_entropy_quantile=0.3`（`entropy_top_p=0.95`），在 \|log p − log q\| 上取 `top_perception_quantile=0.3`（`visual_sensitivity_metric=sampled_abs_log_ratio`），两者都在整个采样批次的所有 token 上取阈值（`*_thr_granularity=batch`），`tor_use_token_weighting=true`，`tor_rsn_weight=1.0`，`tor_prcp_weight=0.5`；`visual_sensitivity_reference=old` |
| `dvrp.sh` | [DVRP](../../reproduction/dvrp/README.md) | 随机图块（p=0.6）和 `vp_diffusion` 噪声 | `visual_sensitivity_loss_coef=0.01`（λ_nec），`visual_robustness_loss_coef=0.01`（λ_rob），`noise_t_init=500`，`noise_gamma=10`，`noise_t_max=1000`；不加视图熵项 |
| `pgpo.sh` | [PGPO](../../reproduction/pgpo/README.md) | 屏蔽对视觉 token 的注意力（`mask_visual_attention`） | `advantage_scaling_method=pgpo`，`pgpo_token_scaling_threshold=0.4`，`pgpo_token_scaling_boost=2.0`，按作者开发版代码（`pgpo_threshold_mode=quantile`，`pgpo_low_weight_floor=0.1`，`pgpo_mass_normalization=false`；见 PGPO README），`visual_sensitivity_metric=sampled_low_var_kl`，`visual_sensitivity_reference=old` |
| `pepo.sh` | [PEPO](../../reproduction/pepo/README.md) | 无（单次前向） | `visual_sensitivity_metric=hidden_state_similarity`（余弦），`advantage_scaling_method=pepo`，`advantage_scaling_schedule=linear`，`pepo_gate_alpha=0.05`，`pepo_gate_temperature=1.8` |
| `cfpo.sh` | [CFPO](../../reproduction/cfpo/README.md) | 模型内部的注意力 value 反事实 | `corrupt_image=cross_modal_attention_value_mean`（λ=2），`visual_sensitivity_loss_coef=0.02`，`visual_sensitivity_reference=current` |
| `vepo.sh` | [VEPO](../../reproduction/vepo/README.md) | 在归一化像素值上加高斯噪声，std 1.08（Qwen3-VL 的 image_std 为 0.5，约合像素单位 0.54，即 VEPO 代码注释和论文所述的强度），每道题一张加噪图 | `visual_sensitivity_metric=vepo`，`visual_sensitivity_jsd_weight=0.7`，`visual_sensitivity_entropy_gate=normal_entropy`，每个回答 `top_perception_quantile=0.2`，`normalize_pg_loss_by_selected_tokens=true` |
| `noisyrollout.sh` | [NoisyRollout](../../reproduction/noisyrollout/README.md) | 像素上的 VP 扩散噪声，每道题每步一张加噪图，只用于采样该题 8 个回答中的 4 个 | `rollout_image_transform=vp_diffusion`；α₀=450，λ=60，γ = t_max / 3（官方 MMK12-7B 脚本的形状按本次运行的步数换算），`pixel_rounding=floor`；8 个回答为一组，都在干净图像上训练 |
| `vgpo.sh` | [VGPO](../../reproduction/vgpo/README.md) | 无（单次前向） | 在最后一层上对图像 token 平均隐状态计算 `visual_sensitivity_metric=hidden_state_similarity`（`visual_sensitivity_hidden_layers=last`，`visual_sensitivity_hidden_pooling=prototype`），在更新前打分（`visual_sensitivity_reference=old`）；`advantage_scaling_method=vgpo`，β=0.3，γ=0.5，κ=0.2；系数按题目组计算 |
| `vapo.sh` | [VAPO](../../reproduction/vapo/README.md) | –（无图像扰动） | 每个答对的回答 `claim_probe_count=20` 个探针，每个探针在推理的随机截断处询问图像的一条 GPT-5 视觉断言是否正确，`claim_probe_late_emphasis=1.5`；探针问题去掉了官方版本末尾的空格，这样 Qwen3-VL-4B 用 "1" 或 "0" 回答（见 [VAPO README](../../reproduction/vapo/README.md#probe-question)）；奖励为 0.8 准确率 + 0.1 格式 + 0.1 感知（`perception_weight=0.1`）；在 `virl39k_claims`（附断言的 ViRL39K；2,289 道多图题没有断言，使用 GRPO 奖励）上训练 |
| `grit.sh` | [GRIT](../../reproduction/grit/README.md) | – | think / rethink / answer 格式，推理中内联 JSON 证据框（`examples/system_prompt/grit_GR.txt`，不用格式提示词），奖励为 `grit.py:compute_score`（格式、框格式、基于规则的答案、0.1·BLEU-1）；在 ViRL39K 上训练，而不是 GRIT 的 20 条样本 |
| `cgpo.sh` | [CGPO](../../reproduction/cgpo/README.md) | 证据区域被替换为其局部均值（`cgpo_flat`，每个回答一个） | `<region>` 证据格式（`xml_grounded_reasoning_v2.jinja` 及其奖励，格式权重 0.1）；`top_entropy_quantile=0.3`，`top_perception_quantile=0.3`（在每个更新 micro-batch 内取阈值，`*_thr_granularity=micro_batch`，与 CGPO 论文一致），`include_region_tokens_in_perception_mask=true`；`advantage_scaling_method=cgpo`，`cgpo_response_scaling_coef=0.1`；定位一致性奖励权重 0.1；`visual_sensitivity_reference=old` |

属于论文基础配置而非方法本身的设置都换成共用配置。论文基于 DAPO 的方法（VPPO、PGPO、VGPO；PAPO、ToR、DVRP 同时报告了 GRPO 和 DAPO）在这里都基于 GRPO，不用 DAPO 的动态采样和 clip-higher；DAPO 基线两者都保留。VPPO 的熵惩罚（论文对其基线也加了）、NoisyRollout 的熵奖励，以及 VEPO 和 NoisyRollout 冻结视觉编码器的做法也都不使用。共用的 0.005 熵惩罚让 Qwen3-VL-4B 的训练保持稳定，去掉后有几个方法会崩溃。PAPO 的 Double Entropy 损失中只有原图上的那一项有梯度，由共用熵惩罚代替。PAPO 遮挡视图上以及 DVRP 两个辅助视图上的熵项没有梯度，因为这些视图在更新前只打分一次（与 PAPO 官方默认一致），所以去掉它们不改变训练；PAPO 作者确认了这一点，并发现带梯度重算遮挡视图也没有帮助（[PAPO issue #20](https://github.com/MikeWangWZHL/PAPO/issues/20)）。各论文原设定见 [examples/reproduction/](../../reproduction/)。

### 为什么不包含 DeepEyes

DeepEyes 改的是采样而不是损失：模型调用放大工具，在多轮中读取原图的裁剪图。它不适合这个设定：

- **它的配置不会在这类数据上用工具。** DeepEyes 只在 V* 衍生的高分辨率图像和 ArxivQA 图表上启用工具；它的推理题（ThinkLite-VL 的 11,031 条）不用工具、单轮作答。ViRL39K 与 ThinkLite-VL 一样是多模态推理数据，会得到同样的处理，因此 DeepEyes 的配置在这里会像 GRPO 基线一样训练，只是奖励不同。
- **几乎没有可放大的内容。** 在 360 张 ViRL39K 图像的抽样中，尺寸中位数约 73,000 像素；72% 低于 200,704 像素的下限，模型看到的已经是放大后的图，只有 3% 超过 1,003,520 像素的上限。原图的裁剪图几乎不会提供模型还没看到的细节。
- **评测体现不出来。** 对比套件在训练分辨率下评测推理、感知和幻觉，而工具针对的是高分辨率图像中的细节，对应 V* 和 HR-Bench（不在套件中）；评测 DeepEyes 还需要多轮智能体循环。
- **开销。** 智能体采样（每条轨迹多轮、最多六次工具调用）比其他方法的单轮采样慢得多。

在每道 ViRL39K 题上都启用工具，就成了我们自己的改编，而不是 DeepEyes。GRIT 同样在推理中定位图像证据，但它保持单轮，只替换提示词和奖励，所以可以在统一设定下运行。[DeepEyes 复现](../../reproduction/deepeyes/README.md)在 DeepEyes 自己的数据和奖励上，把 DeepEyes 与纯文本的 GRPO 基线比较（`qwen*_grpo.sh` 对 `qwen*_grpo_deepeyes.sh`）。

## 运行

```bash
bash scripts/prepare_data.sh comparison
bash examples/comparison/rlvr_qwen3_vl_4b/papo.sh                      # 一个方法
for m in grpo papo vppo tor dvrp pgpo pepo cfpo vepo noisyrollout vgpo vapo grit cgpo; do
    bash examples/comparison/rlvr_qwen3_vl_4b/$m.sh                    # 所有方法，依次运行
done
bash scripts/prepare_eval_data.sh comparison
bash scripts/eval.sh checkpoints/Comparison-Qwen3-VL-4B/papo --suite comparison
python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/*   # 训练结束后：保留最后一步和最优步
                                                                   # （GRPO 的最优步是 OPD 对比的教师）
```

检查点保存在 `checkpoints/Comparison-Qwen3-VL-4B/<method>`。在脚本后追加 `key=value` 可以覆盖共用设定（例如 `bash examples/comparison/rlvr_qwen3_vl_4b/papo.sh trainer.total_epochs=1`）；`N_GPUS_PER_NODE`、`MODEL_PATH`、`DATA_ROOT`、`LOGGER` 和 `EXPERIMENT_NAME` 从环境变量读取。`grit` 和 `cgpo` 用各自的格式作答，评测时要用对应的提示词：`grit` 用 `--format-prompt none --system-prompt examples/system_prompt/grit_GR.txt`（在 GRIT 的数据集上用这组参数加 `--benchmarks grit_vsr,grit_tallyqa,grit_gqa,ovdeval_position`；`grit` 套件是 GRIT 自己的协议，用于复现模型），`cgpo` 用 `--format-prompt examples/format_prompt/xml_grounded_reasoning_v2.jinja`（有训练记录的运行默认使用训练时的提示词）。见 [eval/README.md](../../../eval/README.md)。

## 结果

`comparison` 套件：19 项基准，分为三组。

- **数学推理**：Geo3K、MathVista、We-Math、MMK12、MathVerse、MathVision、DynaMath。题目信息大多在文字中。
- **依赖视觉的推理**：MathVerse-V（MathVerse 中依赖视觉的子集，2,180 题中的 1,308 题）、MMMU-Pro（视觉设定：题目和选项只出现在图中）、LogicVista、计数（Counting）、AI2D、MME 认知（MME 的 4 个认知子任务：常识推理，以及图中给出的数字、代码和文字）。
- **感知与幻觉**：POPE、HallusionBench、MMStar、BLINK、MME 感知（MME 的另外 10 个子任务）、CV-Bench。

数学和依赖视觉的推理基准（AI2D 和 MME 认知除外）使用 PAPO-Eval 协议（按最后一个 `\boxed{}` 规则匹配，T=1.0、top-p 1.0 下的 avg@8；DynaMath 和 MathVision 取自 VPPO-Eval）；其余基准用贪心解码。POPE 取 random / popular / adversarial 三个划分的宏平均 F1，HallusionBench 取问题准确率，MME 认知和感知分别取满分 800 和 2,000 的分数并换算到 0-100，CV-Bench 取其数据集卡片中的总分（2D 与 3D 准确率的均值）。**三组平均**是主要分数（表中加粗），即三个组均值的平均，因此无论各组基准数多少，三组权重相同；**19 项基准平均**是 19 项基准的简单平均，仅供参考。`bash scripts/eval.sh <run> --suite comparison` 会把这些均值写入 `eval/results/summary.csv` 的 `comparison: ...` 列（见 [eval/README.md](../../../eval/README.md#suites)）。

汇总表和三个分组表都取各方法的最后一步（第 202 步），比较的是同样训练量结束时的结果，后期的漂移也计算在内。MMK12 test 同时是训练中监控的验证集（与 PAPO、VPPO 一致）；[验证最优步](#验证最优步)给出第二种视角：各方法在验证奖励最高的一步上，在其余 18 项基准上的表现。

| 方法 | 数学推理 | 依赖视觉的推理 | 感知与幻觉 | 三组平均 | 19 项基准平均 | 训练时长（h） |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct（基线） | 46.5 | 62.0 | 76.0 | 61.5 | 60.7 | – |
| GRPO | 62.8 | 70.7 | 78.4 | **70.6** | 70.2 | 31.0 (1.00×) |
| DAPO | 62.0 | 70.8 | 79.1 | **70.7** | 70.2 | 50.6 (1.63×) |
| PAPO | 69.1 | 75.1 | 78.2 | **74.1** | 73.9 | 35.6 (1.15×) |
| VPPO | 67.4 | 72.8 | 77.9 | **72.7** | 72.4 | 37.4 (1.21×) |
| ToR | 68.6 | 73.3 | 78.3 | **73.4** | 73.2 | 36.0 (1.16×) |
| DVRP | 68.9 | 74.6 | 77.8 | **73.8** | 73.5 | 40.5 (1.31×) |
| PGPO | 66.1 | 71.7 | 77.7 | **71.8** | 71.5 | 31.7 (1.02×) |
| PEPO | 67.4 | 73.4 | 78.1 | **73.0** | 72.7 | 32.0 (1.03×) |
| CFPO | 67.5 | 74.7 | 79.0 | **73.7** | 73.4 | 46.7 (1.51×) |
| VEPO | 68.6 | 73.6 | 78.7 | **73.6** | 73.4 | 34.7 (1.12×) |
| NoisyRollout | 59.0 | 70.5 | 79.3 | **69.6** | 69.1 | 31.0 (1.00×) |
| VGPO | 52.2 | 63.0 | 76.6 | **63.9** | 63.3 | 29.8 (0.96×) |
| VAPO | 63.0 | 70.1 | 78.2 | **70.4** | 70.0 | 34.2 (1.10×) |
| GRIT | 65.8 | 70.4 | 77.5 | **71.2** | 70.9 | 32.0 (1.03×) |
| CGPO | 67.0 | 70.5 | 77.8 | **71.8** | 71.5 | 41.1 (1.33×) |

**训练时长**是 202 个训练步在 4 张 A800 80GB 上的墙钟时间（单位：小时），包括采样、奖励、辅助视图和更新，不含各方法相同的验证和检查点保存；括号中是相对 GRPO 的倍数。这些运行两两共用 8 卡节点，因此几个百分点以内的差异可能来自节点负载。

### 数学推理

| 方法 | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVision | DynaMath | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct（基线） | 48.9 | 66.4 | 57.7 | 41.9 | 47.4 | 14.0 | 49.0 | 46.5 |
| GRPO | 54.3 | 73.6 | 72.5 | 64.7 | 69.6 | 34.0 | 70.8 | 62.8 |
| DAPO | 55.6 | 73.8 | 72.7 | 61.5 | 67.9 | 32.8 | 70.1 | 62.0 |
| PAPO | 63.9 | 74.4 | 76.7 | 71.8 | 77.9 | 45.1 | 73.8 | 69.1 |
| VPPO | 62.6 | 73.2 | 76.2 | 68.5 | 75.6 | 42.0 | 73.7 | 67.4 |
| ToR | 62.7 | 73.7 | 77.1 | 70.2 | 77.4 | 44.8 | 74.7 | 68.6 |
| DVRP | 63.2 | 73.9 | 76.8 | 71.5 | 77.8 | 44.4 | 74.8 | 68.9 |
| PGPO | 62.0 | 71.8 | 74.3 | 68.6 | 74.5 | 40.5 | 71.0 | 66.1 |
| PEPO | 60.3 | 72.8 | 76.9 | 70.2 | 75.4 | 42.7 | 73.4 | 67.4 |
| CFPO | 59.7 | 75.6 | 73.8 | 70.7 | 78.0 | 39.9 | 74.7 | 67.5 |
| VEPO | 60.9 | 75.9 | 77.4 | 71.1 | 77.0 | 44.0 | 74.0 | 68.6 |
| NoisyRollout | 54.5 | 69.0 | 71.7 | 55.4 | 64.5 | 30.1 | 67.9 | 59.0 |
| VGPO | 44.8 | 68.8 | 62.3 | 49.4 | 54.9 | 21.3 | 63.7 | 52.2 |
| VAPO | 56.0 | 73.4 | 72.5 | 64.1 | 69.8 | 33.8 | 71.1 | 63.0 |
| GRIT | 59.5 | 73.4 | 74.8 | 67.6 | 74.8 | 40.1 | 70.6 | 65.8 |
| CGPO | 60.2 | 74.2 | 77.0 | 68.8 | 75.0 | 41.8 | 72.3 | 67.0 |

### 依赖视觉的推理

| 方法 | MathVerse-V | MMMU-Pro | LogicVista | 计数 | AI2D | MME（认知） | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct（基线） | 44.1 | 27.1 | 41.7 | 87.9 | 78.5 | 92.5 | 62.0 |
| GRPO | 64.3 | 41.5 | 54.2 | 89.6 | 83.9 | 90.8 | 70.7 |
| DAPO | 63.8 | 39.7 | 54.9 | 90.1 | 83.9 | 92.5 | 70.8 |
| PAPO | 73.5 | 45.8 | 62.4 | 93.1 | 83.2 | 92.9 | 75.1 |
| VPPO | 72.2 | 43.5 | 58.5 | 84.9 | 83.7 | 94.1 | 72.8 |
| ToR | 73.5 | 45.0 | 59.7 | 86.9 | 82.7 | 92.0 | 73.3 |
| DVRP | 73.9 | 46.0 | 60.8 | 92.9 | 82.5 | 91.6 | 74.6 |
| PGPO | 71.1 | 42.3 | 58.2 | 81.3 | 83.2 | 94.1 | 71.7 |
| PEPO | 72.0 | 44.4 | 59.6 | 87.5 | 82.9 | 93.8 | 73.4 |
| CFPO | 74.4 | 44.7 | 59.1 | 97.1 | 81.9 | 90.9 | 74.7 |
| VEPO | 72.8 | 43.8 | 58.4 | 90.6 | 83.8 | 92.5 | 73.6 |
| NoisyRollout | 59.9 | 37.9 | 53.9 | 95.5 | 84.7 | 91.2 | 70.5 |
| VGPO | 50.4 | 33.8 | 42.7 | 80.1 | 79.4 | 91.7 | 63.0 |
| VAPO | 65.3 | 40.8 | 54.7 | 83.9 | 82.8 | 92.8 | 70.1 |
| GRIT | 70.8 | 39.3 | 52.6 | 82.9 | 81.8 | 94.7 | 70.4 |
| CGPO | 71.5 | 41.5 | 57.2 | 77.8 | 83.1 | 91.6 | 70.5 |

### 感知与幻觉

| 方法 | POPE | HallusionBench | MMStar | BLINK | MME（感知） | CV-Bench | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct（基线） | 88.4 | 72.8 | 65.0 | 62.3 | 83.0 | 84.4 | 76.0 |
| GRPO | 89.0 | 74.1 | 71.0 | 66.1 | 84.4 | 85.5 | 78.4 |
| DAPO | 89.0 | 73.8 | 71.9 | 67.3 | 86.5 | 86.0 | 79.1 |
| PAPO | 89.3 | 75.0 | 68.9 | 66.9 | 83.0 | 86.2 | 78.2 |
| VPPO | 88.4 | 75.0 | 69.7 | 66.0 | 82.9 | 85.7 | 77.9 |
| ToR | 88.3 | 75.3 | 69.9 | 67.7 | 82.3 | 86.1 | 78.3 |
| DVRP | 88.5 | 73.7 | 69.1 | 66.7 | 82.9 | 85.8 | 77.8 |
| PGPO | 88.4 | 73.6 | 68.7 | 65.1 | 84.7 | 85.8 | 77.7 |
| PEPO | 89.3 | 74.5 | 69.7 | 66.3 | 83.3 | 85.7 | 78.1 |
| CFPO | 88.9 | 74.3 | 69.3 | 69.0 | 85.8 | 86.4 | 79.0 |
| VEPO | 89.0 | 74.0 | 71.3 | 67.7 | 83.9 | 86.1 | 78.7 |
| NoisyRollout | 89.3 | 75.8 | 71.1 | 69.0 | 83.7 | 87.1 | 79.3 |
| VGPO | 88.9 | 71.4 | 67.2 | 64.8 | 81.8 | 85.6 | 76.6 |
| VAPO | 88.9 | 74.1 | 69.7 | 66.6 | 83.6 | 86.2 | 78.2 |
| GRIT | 89.2 | 71.5 | 68.5 | 65.0 | 86.2 | 84.4 | 77.5 |
| CGPO | 88.8 | 72.6 | 68.5 | 66.5 | 84.9 | 85.8 | 77.8 |

MME（认知）和 MME（感知）分别是 MME 的认知分（常识推理、数值计算、文本翻译和代码推理；260 题，满分 800）和感知分（另外 10 个子任务；2,114 题，满分 2,000），都换算到 0-100。

### 验证最优步

各方法在验证奖励最高的一步上的表现（运行的 `checkpoint_tracker.json` 中的 `best_global_step`；`scripts/finalize_run.py` 会保留这一步），评测命令为 `bash scripts/eval.sh checkpoints/Comparison-Qwen3-VL-4B/<method>/global_step_<best> --suite comparison`。由于最优步是按 MMK12 选的，这里去掉 MMK12，数学推理与另外两组一样各含 6 项基准，三组平均也等于 18 项基准的平均。该评测的 `comparison: ...` 列仍包含 19 项基准，这里的均值由去掉 MMK12 后的各基准分数计算。其余数学基准与 MMK12 相近，因此最优步明显早于最后一步的方法，其数学推理分数可能仍略偏高。最后两列是最后一步（第 202 步）上同样的平均，以及两者之差（按表中保留一位小数的数值计算）。

| 方法 | 最优步 | 数学推理（不含 MMK12） | 依赖视觉的推理 | 感知与幻觉 | 三组平均 | 最后一步的三组平均 | 最优步 − 最后一步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct（基线） | – | 47.2 | 62.0 | 76.0 | 61.7 | – | – |
| GRPO | 175 | 65.9 | 72.4 | 78.0 | **72.1** | 70.5 | +1.6 |
| DAPO | 105 | 70.2 | 74.6 | 78.6 | **74.5** | 70.7 | +3.8 |
| PAPO | 202 | 68.6 | 75.1 | 78.2 | **74.0** | 74.0 | 0.0 |
| VPPO | 200 | 67.4 | 73.4 | 77.6 | **72.8** | 72.6 | +0.2 |
| ToR | 202 | 68.4 | 73.3 | 78.3 | **73.3** | 73.3 | 0.0 |
| DVRP | 200 | 68.3 | 74.6 | 77.7 | **73.5** | 73.6 | −0.1 |
| PGPO | 202 | 65.7 | 71.7 | 77.7 | **71.7** | 71.7 | 0.0 |
| PEPO | 200 | 67.0 | 73.0 | 78.1 | **72.7** | 72.8 | −0.1 |
| CFPO | 190 | 67.1 | 74.8 | 78.9 | **73.6** | 73.5 | +0.1 |
| VEPO | 170 | 68.4 | 73.2 | 78.5 | **73.4** | 73.5 | −0.1 |
| NoisyRollout | 160 | 67.0 | 74.6 | 79.1 | **73.6** | 69.8 | +3.8 |
| VGPO | 165 | 65.9 | 71.1 | 77.1 | **71.3** | 64.1 | +7.2 |
| VAPO | 155 | 66.9 | 72.5 | 78.4 | **72.6** | 70.3 | +2.3 |
| GRIT | 202 | 65.5 | 70.4 | 77.5 | **71.1** | 71.1 | 0.0 |
| CGPO | 160 | 66.8 | 71.3 | 77.6 | **71.9** | 71.7 | +0.2 |
