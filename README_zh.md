<div align="center">

# Awesome Perception-Aware RLVR

**面向大视觉语言模型的感知导向可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）与在线策略蒸馏（On-Policy Distillation, OPD）：统一复现、统一设定对比与一键评测，并附论文清单**

RLVR：PAPO · VPPO · DVRP · ToR · PGPO · PEPO · CFPO · VEPO · NoisyRollout · VGPO · VAPO · GRIT · DeepEyes · CGPO<br>
OPD：VA-OPD · VGS · VCSD · Vision-OPD · S-OPD<br>
统一的 EasyR1 代码库，30+ 个评测基准

🌟 **[CGPO](#-cgpo-acm-mm-2026-oral)（ACM MM 2026 Oral）官方代码仓库** 🌟

[English](README.md) | [简体中文](README_zh.md)

[![Awesome](https://awesome.re/badge.svg)](https://github.com/sindresorhus/awesome) [![Papers](https://img.shields.io/badge/Papers-125-blue)](#-论文清单) [![Reproduced](https://img.shields.io/badge/Reproduced-19-brightgreen)](#-已复现方法) [![CGPO](https://img.shields.io/badge/CGPO-ACM%20MM%202026%20Oral-8A2BE2)](https://doi.org/10.1145/3767308.3835969) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Built on EasyR1](https://img.shields.io/badge/built%20on-EasyR1-orange)](https://github.com/hiyouga/EasyR1)

</div>

RLVR 能提升大视觉语言模型的推理能力，但它的奖励只检查最终答案：模型可能"答对但理由不对"，依赖语言先验而不是图像本身。**感知导向的 RLVR** 方法把视觉感知重新放回优化目标中，例如在图像的反事实视图上对比策略输出、把信用分配给真正依赖图像的 token，或者让模型在推理过程中定位、放大视觉证据。同样的思路也用于 **OPD**：由教师对学生自己采样的回答逐 token 打分，可以加权那些由图像决定的 token、只蒸馏教师从图像中得到的增益而非其语言先验，或者让模型在看到更多图像信息的条件下给自己当教师。

<p align="center">
  <img src="docs/assets/architecture_zh.svg" width="100%" alt="框架架构：一个 RL 训练步分三个阶段。vLLM 采样：加噪图像采样（NoisyRollout）、放大工具（DeepEyes）、在线过滤（DAPO），以及需要策略生成、在 vLLM 休眠前算完的奖励输入：定位一致性（CGPO）和视觉断言探针（VAPO）。更新前无梯度计算：rollout 策略和参考模型的 log-prob，以及基于采样 token 的 OPD 和 VA-OPD 所用冻结教师的 log-prob，改动输入（遮挡、加噪、像素化或去掉图像，或模型内部改动）上的辅助前向（由此得到每个 token 的视觉依赖），带各方法额外项的奖励函数，以及优势。策略更新：执行 compute_perception_reasoning_policy_loss 的五步（控制信号、优势调制、逐 token PPO 项、token 筛选、辅助损失），或 OPD 的蒸馏损失（逐 micro-batch 做学生前向和无梯度的教师前向），然后优化器更新。">
</p>

本仓库提供：

- **[CGPO](#-cgpo-acm-mm-2026-oral) 官方实现**（ACM MM 2026 Oral）；
- **[方法复现](#-已复现方法)**：18 个方法（13 个 RLVR、5 个 OPD），各自使用原论文的数据、模型和超参数，并附带论文中对比的基线；
- **[统一设定对比](#-统一设定对比)**：两个对比各用一套统一设定，RLVR 方法（除 DeepEyes 外）在 Qwen3-VL-4B 上对比；OPD 方法以 Qwen3-VL-2B 为学生，以经 GRPO 训练的 Qwen3-VL-4B 为教师进行对比；
- **[一键评测](eval/README.md)**：覆盖这些论文所用评测基准的并集，每个基准都有下载处理脚本，每篇论文都有对应的评测套件；
- **[论文清单](#-论文清单)**：感知导向的 RLVR（含基于定位的推理与看图思考，即 thinking with images）与 OPD，按训练环节分类，以及这一方向常用的评测基准。

所有方法共用同一个训练框架（[EasyR1](https://github.com/hiyouga/EasyR1) 的分支），通过少量配置开关即可组合、对比和扩展。

> [!NOTE]
> 除 CGPO 外，其余方法均为**非官方复现**。请参考并引用原论文和原仓库。与官方实现的差异记录在各方法目录的 README 中。我们的数字来自这些复现的单次运行，未必能复现论文报告的提升，见[关于结果的说明](examples/comparison/README_zh.md#关于结果的说明)。

## 🔍 主要发现

来自[统一设定对比](#-统一设定对比)：每个方法只训练一次，用的是同一套设定，而不是各方法设计时针对的设定；下结论前请先读[关于结果的说明](examples/comparison/README_zh.md#关于结果的说明)。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/comparison_rlvr_zh_dark.svg">
    <img src="docs/assets/comparison_rlvr_zh.svg" width="100%" alt="13 个感知导向 RLVR 方法在 Qwen3-VL-4B-Instruct 上相对 GRPO 的分差，分三组平均与各组。三组平均：PAPO +3.5、DVRP +3.2、CFPO +3.1、VEPO +3.0、ToR +2.8、PEPO +2.4、VPPO +2.1、CGPO +1.2、PGPO +1.2、GRIT +0.6、VAPO -0.2、NoisyRollout -1.0、VGPO -6.7。提升来自数学推理和依赖视觉的推理；感知与幻觉组中除一个方法外都在 GRPO 的一分以内。">
  </picture>
</p>

- **多数感知导向 RLVR 方法最终高于 GRPO。** 在 Qwen3-VL-4B-Instruct 上用相同数据和训练量，13 个方法中有 9 个的三组平均比 GRPO 高 1.2 到 3.5 分（GRPO 为 70.6），训练时长是 GRPO 的 1.0 到 1.5 倍。
- **提升在推理上，不在感知类基准上。** 这 9 个方法的数学推理提高 3.3 到 6.3 分；而在感知与幻觉组（POPE、HallusionBench、MMStar、BLINK、MME 感知、CV-Bench），13 个方法中有 12 个与 GRPO 相差不到一分。
- **一分左右的差距在多次训练的波动范围内。** 同一配置训练两次，三组平均相差 0.6 分。
- **OPD 对比中，直接对学生做 GRPO 的结果最高。** 以 Qwen3-VL-2B 为学生、经 GRPO 训练的 Qwen3-VL-4B 为教师时，直接对学生做 GRPO 的三组平均为 64.0，OPD 各变体中最高为 62.8。

## 📖 目录

- [更新](#-更新)
- [计划](#-计划)
- [CGPO (ACM MM 2026 Oral)](#-cgpo-acm-mm-2026-oral)
- [已复现方法](#-已复现方法)
- [快速开始](#-快速开始)
- [仓库结构](#-仓库结构)
- [统一设定对比](#-统一设定对比)
- [评测](#-评测)
- [配置](#-配置)
- [论文清单](#-论文清单)
- [贡献](#-贡献)
- [引用](#-引用)
- [致谢](#-致谢)

## 🔥 更新

- **2026-10**：在 agent 环境中，通过 OpenAI 兼容接口调用的模型可以和本地检查点在同样的任务、工具与评分下评测（[评测说明](eval/README.md#api-models-in-agent-environments)）；示范轨迹（脚本写好的，或由通过接口调用的模型生成的）经 rollout 回放后导出为 ms-swift 的 SFT 数据，分词与 rollout 完全一致（[文档](docs/agent_environments.md#sft-data)）。
- **2026-10**：多轮 agent 强化学习与评测的环境接口（进程内或通过 HTTP，[文档](docs/agent_environments.md)），回合奖励可由环境给出；DeepEyes 改为进程内环境，并提供[示例](examples/agent_env/README.md)（玩具视觉搜索、通过 HTTP 提供的 DeepEyes）。
- **2026-10**：复现 [S-OPD](examples/reproduction/s_opd/README.md)（给 OPD 加两项学生侧损失：经教师门控的原图与遮挡图对比，以及原图与加噪图的一致），S-OPD 加入 [OPD 统一对比](examples/comparison/opd_qwen3_vl_2b/README_zh.md)，评测新增 VisualPuzzles 和 ZeroBench（子问题）。
- **2026-10**：两个[统一设定对比](#-统一设定对比)的结果，每个方法给出最后一步和验证最优步；[论文清单](#-论文清单)新增 30 篇感知导向的 OPD 论文。
- **2026-10**：OPD：教师模型（冻结模型或策略的 EMA）、基于采样 token 和基于完整下一 token 分布的两种 OPD 目标，复现 [VA-OPD](examples/reproduction/va_opd/README.md)、[VGS](examples/reproduction/vgs/README.md)、[VCSD](examples/reproduction/vcsd/README.md)、[Vision-OPD](examples/reproduction/vision_opd/README.md)，新增 [OPD 统一对比](examples/comparison/opd_qwen3_vl_2b/README_zh.md)，评测新增 ZoomBench、BLINK、MMStar、AI2D、MMMU。
- **2026-10**：🎉 首次发布：[CGPO](#-cgpo-acm-mm-2026-oral)（ACM MM 2026 Oral）官方实现、13 个复现的 RLVR 方法（PAPO、VPPO、DVRP、ToR、PGPO、PEPO、CFPO、VEPO、NoisyRollout、VGPO、VAPO、GRIT、DeepEyes）、Qwen3-VL-4B 上的统一设定对比、一键评测。

## 🚧 计划

- [x] 两个统一设定对比的结果（RLVR 在 Qwen3-VL-4B 上，OPD 在 Qwen3-VL-2B 上）。
- [x] OPD：GRPO、DAPO 之外的基础训练方式（教师模型，基于采样 token 和基于完整分布的两种目标），及 VA-OPD、VGS、VCSD、Vision-OPD、S-OPD。
- [x] 统一设定对比加入更多单轮方法：NoisyRollout、VGPO、VAPO。
- [ ] 在带证据框的数据上新开定位推理对比：TreeVGR、DeFacto、iVGR，与 GRIT、CGPO 一起比较。
- [ ] 更多 DeepEyes 方向的看图思考方法：MGPO、Mini-o3、Pixel Reasoner，以及 MED 的"有工具/无工具"评测。

欢迎提建议：可以开 issue 提议要加入的方法，或参见[贡献](#-贡献)。

## 🌟 CGPO (ACM MM 2026 Oral)

本仓库是以下论文的官方代码：

> **CGPO: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning**<br>
> Shengxuming Zhang, Linyun Zhou, Hengrui Lou, Zhenyang Wang, Xiuming Zhang, Zunlei Feng<br>
> *Proceedings of the 34th ACM International Conference on Multimedia (MM '26)*, Rio de Janeiro, Brazil, 2026 · **Oral 报告**<br>
> [[论文]](https://doi.org/10.1145/3767308.3835969) · [[代码与脚本]](examples/reproduction/cgpo/README.md) · [[BibTeX]](#-引用)

CGPO 训练视觉语言模型进行*证据敏感推理*：关键推理步骤要定位它所依据的视觉证据，并且去掉这些证据后结论应当改变。策略在思维链中以内联方式定位证据；把定位到的区域遮挡得到反事实图像，用策略在原图与反事实图像上输出分布的变化衡量每个 token 对证据的依赖程度。证据依赖越强的回答，其感知关键 token 的优势被放大；定位一致性奖励（由策略对每个定位实体重新检测）防止证据框被刻意放大。整个方法只需要答案级监督。

```bash
bash scripts/prepare_data.sh cgpo
bash examples/reproduction/cgpo/qwen3_vl_8b_cgpo.sh
bash scripts/eval.sh checkpoints/CGPO-Reproduce/qwen3_vl_8b_cgpo --suite cgpo
```

## 🧪 已复现方法

每个方法都可以用 `examples/reproduction/<method>` 中的脚本训练，这些脚本沿用其论文的数据、模型和超参数（各目录的 README 说明了设定、基线、与官方代码的差异以及论文报告的结果）；除 DeepEyes 和 Vision-OPD 外，也都可以用[统一设定对比](#-统一设定对比)的脚本训练。除 CGPO 外均为非官方复现。

方法（以及[论文清单](#-论文清单)中的论文）按其感知导向改动所在的训练环节分类，环节按一个训练步的顺序排列：

| 环节 | RLVR | OPD |
| --- | --- | --- |
| 输入 | **训练数据与输入**：用哪些任务和图像训练，采样时怎样改动图像 | **输入与特权信息**：教师和学生分别看到什么（教师看裁剪图、更高分辨率或符号状态，学生看降质视图） |
| 回答 | **回答结构**：策略在回答中写什么、做什么（感知描述、定位、视觉操作与工具） | **回答结构**：学生生成什么（先描述再推理、视觉操作） |
| 监督 | **奖励**：在答案正确之外检查感知的逐回答打分 | **蒸馏目标**：学生被拉向的分布（对比锐化、视觉增益、组合教师） |
| 信用 | **信用分配**：更新如何分到各个 token 和回答上（加权、筛选、token 级惩罚） | **信用分配**：蒸馏哪些 token 和回答、强度多大 |
| 辅助 | **辅助目标**：策略梯度之外的损失项（对改动视图的 KL、一致性、注意力监督） | **辅助目标**：蒸馏损失之外的学生侧损失项 |

一个方法改动多个环节时，“感知导向改动所在的训练环节”一列把它们都列出来，主要环节加粗放在最前；表格按主要环节排序。

### 🎯 RLVR

策略在自己采样的回答上，从可验证奖励中学习。

| 方法 | 感知导向改动所在的训练环节 | 论文 | 会议 | 代码 | 核心思路 | 复现设定 |
| --- | --- | --- | --- | --- | --- | --- |
| [**NoisyRollout**](examples/reproduction/noisyrollout/README.md) | **训练数据与输入** | [NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](https://arxiv.org/abs/2504.13055)<br>Xiangyan Liu et al. · `2504.13055` | NeurIPS 2025 | [GitHub](https://github.com/real-absolute-AI/NoisyRollout) | 每组一半的 rollout 从加了扩散噪声的图像采样，噪声退火到接近零；全部 rollout 都在干净图像上更新。 | Qwen2.5-VL-7B/32B，Geometry3K / K12 |
| [**GRIT**](examples/reproduction/grit/README.md) | **回答结构** · 奖励 | [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)<br>Yue Fan et al. · `2505.15879` | NeurIPS 2025 | [GitHub](https://github.com/UCSB-AI/GRIT) | 单轮推理链中穿插边界框；GRPO-GR 奖励输出结构、输出框（含计数奖励）和答案，不监督框本身。 | Qwen2.5-VL-3B / InternVL3-2B，GRIT 的 20 条样本 |
| [**DeepEyes**](examples/reproduction/deepeyes/README.md) | **回答结构** · 奖励 | [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362)<br>Ziwei Zheng et al. · `2505.14362` | ICLR 2026 | [GitHub](https://github.com/Visual-Agent/DeepEyes) | 带放大工具的多轮智能体强化学习：原图裁剪结果作为新观测返回，工具奖励只给用了工具且答对的回答。 | Qwen2.5-VL-7B / Qwen3-VL-8B，DeepEyes-47k，多轮放大工具 |
| [**VAPO**](examples/reproduction/vapo/README.md) | **奖励** | [More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models](https://arxiv.org/abs/2509.25848)<br>Xinyu Tian et al. · `2509.25848` | ICLR 2026 | [GitHub](https://github.com/xytian1008/VAPO) | 在答对回答推理的随机位置让策略判断一条图像断言是否正确（是 / 否），加权答对率加到奖励上。 | Qwen2.5-VL-3B/7B / Qwen3-VL-8B，ViRL39K（单图题）及视觉断言 |
| [**VEPO**](examples/reproduction/vepo/README.md) | **信用分配** | [Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection](https://arxiv.org/abs/2606.03937)<br>Senjie Jin et al. · `2606.03937` | arXiv | [GitHub](https://github.com/Leonnnnnn929/VEPO) | 按原图与扰动图像下预测的 JS 散度、熵差及 token 熵给 token 打分，每个回答只优化得分最高的 20%。 | Qwen2.5-VL-7B，Geometry3K |
| [**VGPO**](examples/reproduction/vgpo/README.md) | **信用分配** | [Visually-Guided Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2604.09349)<br>Zengbin Wang et al. · `2604.09349` | ACL 2026 | [GitHub](https://github.com/wzb-bupt/VGPO) | 以最后一层隐状态与图像 token 均值的相似度，逐 token、在组内逐回答重加权优势，并放大回答后段的视觉 token。 | Qwen2.5-VL-3B/7B/32B，ViRL39K |
| [**PGPO**](examples/reproduction/pgpo/README.md) | **信用分配** | [Not All Tokens See Equally: Perception-Grounded Policy Optimization for Large Vision-Language Models](https://arxiv.org/abs/2604.01840)<br>Zekai Ye et al. · `2604.01840` | arXiv | - | 额外做一次屏蔽视觉 token 注意力的前向得到视觉依赖；带阈值门控、保持回答总权重的 token 权重乘到优势上。 | Qwen2.5-VL-3B/7B，ViRL39K |
| [**ToR**](examples/reproduction/tor/README.md) | **信用分配** | [Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs](https://arxiv.org/abs/2603.25077)<br>Jinda Lu et al. · `2603.25077` | arXiv | - | 熵最高的 30% token（推理）与去掉图像后对数概率变化最大的 30% token（感知）分别加权，其余 token 不参与优化。 | Qwen2.5-VL-7B，Geometry3K |
| [**PEPO**](examples/reproduction/pepo/README.md) | **信用分配** | [Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought](https://arxiv.org/abs/2603.22847)<br>Yunheng Li et al. · `2603.22847` | arXiv | [GitHub](https://github.com/xzxxntxdy/PEPO) | 无需额外前向的感知先验：token 隐状态与视觉 token 的相似度经熵门控后，以逐渐增大的系数混入 token 优势。 | Qwen2.5-VL-3B / InternVL3-2B，Geometry3K |
| [**VPPO**](examples/reproduction/vppo/README.md) | **信用分配** | [Spotlight on Token Perception for Multimodal Reinforcement Learning](https://arxiv.org/abs/2510.09285)<br>Siyuan Huang et al. · `2510.09285` | ICLR 2026 | [GitHub](https://github.com/huaixuheqing/VPPO-RL) | 每个 token 的视觉依赖（与扰动图像的 KL）：每个回答只有依赖最高的 40% token 接收梯度，并按其平均依赖缩放优势。 | Qwen2.5-VL-7B / Qwen3-VL-8B，ViRL39K |
| [**CGPO**](examples/reproduction/cgpo/README.md)（本仓库） | **信用分配** · 回答结构 · 奖励 | [CGPO: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning](https://doi.org/10.1145/3767308.3835969)<br>Shengxuming Zhang et al. | ACM MM 2026 (Oral) | 本仓库 | 在推理中内联定位证据；遮挡定位区域得到反事实图像，用其 KL 缩放感知关键 token 的优势；定位一致性奖励让框保持紧凑。 | Qwen2.5-VL-7B / Qwen3-VL-8B，ViRL39K（自然图像） |
| [**CFPO**](examples/reproduction/cfpo/README.md) | **辅助目标** | [CFPO: Counterfactual Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2606.23206)<br>Zhangyuan Yu et al. · `2606.23206` | ICML 2026 | [GitHub](https://github.com/Raven-July/CFPO) | 在模型内部构造反事实：每一层中文本到图像注意力最高的图像 token 的 value 被替换为均值，并最大化与该反事实的 KL。 | Qwen2.5-VL-3B，ViRL39K |
| [**DVRP**](examples/reproduction/dvrp/README.md) | **辅助目标** | [Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy](https://arxiv.org/abs/2601.06801)<br>Shujian Gao et al. · `2601.06801` | arXiv | - | 每张图像三个视图：最大化与块遮挡视图的 KL，最小化与扩散加噪视图的 KL，并对两个辅助视图加熵惩罚。 | Qwen2.5-VL-3B/7B，ViRL39K |
| [**PAPO**](examples/reproduction/papo/README.md) | **辅助目标** | [Perception-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2507.06448)<br>Zhenhailong Wang et al. · `2507.06448` | ICLR 2026 | [GitHub](https://github.com/MikeWangWZHL/PAPO) | 隐式感知损失：最大化策略在原图与随机块遮挡图像上输出分布的 KL；双熵损失防止该 KL 以退化方式增大。 | Qwen2.5-VL-3B/7B，ViRL39K |

### 🧑‍🏫 OPD

教师（另一个模型或学生自己）对学生自己采样的回答逐 token 打分。

| 方法 | 感知导向改动所在的训练环节 | 论文 | 会议 | 代码 | 核心思路 | 复现设定 |
| --- | --- | --- | --- | --- | --- | --- |
| [**Vision-OPD**](examples/reproduction/vision_opd/README.md) | **输入与特权信息** | [Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation](https://arxiv.org/abs/2605.18740)<br>Qianhao Yuan et al. · `2605.18740` | NeurIPS 2026 | [GitHub](https://github.com/VisionOPD/Vision-OPD) | 学生看整图，EMA 教师看问题相关区域的放大裁剪图；在学生的 top-100 token 上用 JS 散度拟合。 | Qwen3.5-4B / 9B，以区域裁剪图为输入的 EMA 自教师，Vision-OPD-6K |
| [**VCSD**](examples/reproduction/vcsd/README.md) | **蒸馏目标** | [Visual Contrastive Self-Distillation](https://arxiv.org/abs/2607.21556)<br>Yijun Liang et al. · `2607.21556` | arXiv | [GitHub](https://github.com/joliang17/VCSD) | 从 EMA 教师自蒸馏：以原图分布按其与全黑图的对比锐化后作为前向 KL 的目标，不需要答案或奖励。 | Qwen3-VL-2B/4B/8B、Qwen3.5-2B/4B/9B，EMA 自教师，ViRL39K |
| [**VGS**](examples/reproduction/vgs/README.md) | **蒸馏目标** · 辅助目标 | [Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](https://arxiv.org/abs/2606.00564)<br>Hee Suk Yoon et al. · `2606.00564` | ICML 2026 (Spotlight) | - | 把教师分解为语言先验和视觉增益：在反向 KL 之外，学生拟合自己的纯文本分布乘以教师有图/无图的概率比。 | Qwen3-VL-2B / 4B 学生，经 GRPO 训练的 Qwen3-VL-8B 教师，Vision-SR1-47K |
| [**VA-OPD**](examples/reproduction/va_opd/README.md) | **信用分配** | [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924)<br>Ruiqi Liu et al. · `2605.21924` | arXiv | - | 按教师的视觉优势（原图下比像素化图像下高出的对数概率）逐回答、逐 token 加权蒸馏。 | Qwen3-VL-2B 学生，4B / 8B / 32B 教师，Geometry3K / ViRL39K |
| [**S-OPD**](examples/reproduction/s_opd/README.md) | **辅助目标** | [Is Better Teacher Supervision Enough? Unlocking Student-side Learning in Multimodal On-Policy Distillation](https://arxiv.org/abs/2609.39120)<br>Siyuan Liu et al. · `2609.39120` | arXiv | [GitHub](https://github.com/Sirilaw/S-OPD) | 在基于采样 token 的 OPD 上加两项学生侧损失：在遮挡会降低教师对数概率的 token 上，拉开学生在原图与遮挡图上的预测；同时拉近学生在原图与加噪图上的预测。 | Qwen3-VL-2B / 4B 学生，Qwen3-VL-8B 或经 GRPO 训练的 8B 教师，ViRL39K / Geometry3K |

EasyR1 原有的算法（GRPO、DAPO、GSPO、CISPO、SAPO、REINFORCE++、RLOO、ReMax）仍然可用。OPD 方法从教师学习，不使用可验证奖励；它们共用教师（`worker.teacher`）和两种基础目标：基于采样 token 的 OPD（`algorithm.adv_estimator=teacher_log_ratio`）和基于完整下一 token 分布的 OPD（`algorithm.distill_loss_coef`），见 [docs/algorithm_parameters.md](docs/algorithm_parameters.md#on-policy-distillation)。

## 🚀 快速开始

**1. 安装环境**（Linux、CUDA 12.8 驱动、conda）：

```bash
git clone https://github.com/zsxm1998/awesome-perception-aware-rlvr.git
cd awesome-perception-aware-rlvr
bash scripts/install_env.sh          # 创建 conda 环境 "parlvr"（torch 2.10、vLLM 0.19、flash-attn 2.8.3）
conda activate parlvr
QWEN35_FASTPATH_ONLY=1 bash scripts/install_env.sh   # 可选，仅 Qwen3.5 模型需要：补装其加速算子
```

也提供了基于 `vllm/vllm-openai:v0.19.0` 的 [Dockerfile](Dockerfile)。

**2. 准备数据**（下载到 `./data`，见 [data/README.md](data/README.md)）：

```bash
bash scripts/prepare_data.sh papo            # 某个方法的训练 / 验证数据
bash scripts/prepare_data.sh all             # 或全部数据
bash scripts/prepare_eval_data.sh papo       # PAPO 评测套件中的基准
```

如果访问 `huggingface.co` 较慢或受限，可设置 `HF_ENDPOINT=https://hf-mirror.com`。

**3. 训练**：

```bash
bash examples/reproduction/papo/qwen2_5_vl_7b_grpo_papo.sh                       # PAPO-G，论文设定
bash examples/comparison/rlvr_qwen3_vl_4b/cgpo.sh                                 # 统一对比中的 CGPO
N_GPUS_PER_NODE=4 bash examples/reproduction/papo/qwen2_5_vl_7b_grpo.sh trainer.total_epochs=1   # 任意覆盖参数
```

checkpoint 保存在 `checkpoints/<project>/<experiment>/global_step_*`（脚本只保留最新一次保存和验证奖励最高的一次，`trainer.save_limit=1`）；日志输出到终端，同时写入同一目录下的 `experiment_log.jsonl`（在线记录可设置 `LOGGER='["console","wandb"]'` 或 `swanlab`）。

**4. 评测**：

```bash
bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo
bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct --suite papo         # 基座模型
```

评测脚本会自动把 FSDP checkpoint 合并成 Hugging Face 格式，在所有可见 GPU 上用 vLLM 跑完套件中的每个基准，并输出汇总表。详见 [eval/README.md](eval/README.md)。

**5. 释放磁盘空间**（训练结束后）：

```bash
python3 scripts/finalize_run.py checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo   # --keep last|best|all|N,...，--dry-run
```

每个保存的 step 都带优化器状态，大小是模型本身的 3 到 4 倍（4B 模型为 34 GB）。该脚本默认把最后一步和验证奖励最高的一步保留为 `global_step_N/actor` 中的 Hugging Face 权重（4B 模型每步约 9 GB），删除优化器状态和其余 step，此后该实验无法再续训。`--keep` 接受逗号分隔的集合：单独的 `last` 或 `best`、`all`（保留所有已保存的 step，例如分析训练动态，训练时需设 `trainer.save_limit=-1` 才会全部保存）或具体步数（`--keep last,50,100`）。统一对比的主表取最后一步，验证最优步另列一张表并去掉验证集对应的基准（验证集往往同时是评测基准，按它挑步会使该基准偏高）；验证最优步也可用作教师。传入单个 `.../global_step_N` 时只处理这一步，不动其他 step。`scripts/eval.sh` 可以直接评测收尾后的实验。

## 📁 仓库结构

```
.
├── examples/
│   ├── reproduction/<method>/   # 各论文原设定：common.sh（公共参数）+ 每个实验一个脚本
│   ├── comparison/              # 统一设定对比：索引页，每个对比一个目录
│   │   ├── rlvr_qwen3_vl_4b/    #   RLVR：每个方法一个脚本，共用 common.sh
│   │   └── opd_qwen3_vl_2b/     #   OPD：每个方法一个脚本，共用 common.sh
│   ├── config.yaml              # 基础训练配置
│   ├── agent_env/               # 通过 HTTP 提供的 agent 环境（示例）
│   ├── reward_function/         # 奖励函数
│   └── format_prompt/、system_prompt/、chat_template/
├── eval/                    # 一键评测（注册表、loader、scorer、prepare/）
├── scripts/                 # install_env.sh、prepare_data.sh、prepare_eval_data.sh、eval.sh、finalize_run.py、launcher.sh
├── verl/                    # 训练框架（EasyR1 分支），包含所有方法的实现
├── docs/                    # 算法参数说明、新增方法教程、BibTeX、插图（assets/）
├── data/                    # 下载的数据（不纳入 git）
└── tests/                   # 单元测试
```

各方法在 `verl/trainer/` 中被实现为可组合的模块：辅助（反事实）图像视图、token 级视觉敏感度信号、token 选择、优势缩放、辅助损失和额外奖励；OPD 另有教师和蒸馏损失（`distillation.py`）。所有开关见 [docs/algorithm_parameters.md](docs/algorithm_parameters.md)，多个方法共有的实现细节（如 k3 KL 估计的梯度方向、损失平均方式）见 [docs/implementation_notes.md](docs/implementation_notes.md)，新增方法见 [docs/add_method.md](docs/add_method.md)。多轮 agent 训练在 agent 环境中进行，环境可以在进程内，也可以通过 HTTP 提供（[docs/agent_environments.md](docs/agent_environments.md)）。

## 📊 统一设定对比

两个对比各用一套统一设定训练已复现的方法，并在同样的 19 项基准上评测，分为三组：数学推理、依赖视觉的推理、感知与幻觉。[examples/comparison/](examples/comparison/README_zh.md) 列出了这两个对比，并说明了[这些结果能说明什么、不能说明什么](examples/comparison/README_zh.md#关于结果的说明)。

- **[RLVR](examples/comparison/rlvr_qwen3_vl_4b/README_zh.md)**：除 DeepEyes 外的所有 RLVR 方法（[原因](examples/comparison/rlvr_qwen3_vl_4b/README_zh.md#为什么不包含-deepeyes)）在 Qwen3-VL-4B-Instruct 上训练，使用相同的数据（ViRL39K / MMK12）和相同的 GRPO 超参数，只有方法相关的参数不同，各方法沿用其论文中的图像扰动方式。
- **[OPD](examples/comparison/opd_qwen3_vl_2b/README_zh.md)**：两种基础 OPD 目标、VA-OPD、VGS、VCSD、S-OPD 以 Qwen3-VL-2B 为学生，以经 GRPO 训练的 Qwen3-VL-4B 为教师（VCSD 改为从学生自身的 EMA 蒸馏），并与同一学生上的 GRPO 对照，使用相同的数据、训练量和评测；Vision-OPD 需要区域标注，这份数据没有。

每个对比的 README 给出每个方法在各组和各基准上的分数（最后一步和验证最优步）以及训练时长。

## 📈 评测

评测系统覆盖所复现论文使用的基准的并集：数学与多模态推理（Geometry3K、MathVista、We-Math、MMK12、MathVerse、MathVerse-V、LogicVista、CLEVR 计数、MMMU-Pro、MMMU、DynaMath、MathVision），感知与幻觉（POPE、HallusionBench、MME、MMStar、BLINK、AI2D、TextVQA、CFPO 的反事实基准 C-VQA-Real 与 MARS-Bench、SEED-Bench 等），基于定位的推理（GRIT 的 VSR / TallyQA / GQA / OVDEval、refCOCO），以及高分辨率感知（V*、HR-Bench、MME-RealWorld-Lite、ZoomBench）。每个基准都有下载脚本，每篇论文都有评测套件（`--suite papo`、`--suite deepeyes` 等）。无论用哪个套件，本仓库训练的检查点都按训练时的方式提问（提示词、图像尺寸、对话模板与分词器，从其训练目录读取，[详见](eval/README.md#prompts)）。新增基准只需要 loader、scorer 和注册表条目，教程见 [eval/README.md](eval/README.md#adding-a-new-benchmark)。

> [!IMPORTANT]
> 各论文的评测协议并不相同（PAPO/VPPO/PGPO/DVRP/CFPO 使用温度 1.0 下基于规则的 avg@8；ToR/VEPO/NoisyRollout/VAPO/GRIT/DeepEyes/Vision-OPD 使用贪心解码加 LLM 裁判）。本仓库用同一套评测系统评测所有模型，因此请与本仓库中跑出的基线比较，而不要直接与跨论文摘抄的数字比较。

## 🔧 配置

所有训练脚本都支持以下环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODEL_PATH` | 由脚本指定 | 策略模型的 Hugging Face id 或本地路径 |
| `TEACHER_PATH` | 由脚本指定 | OPD 脚本的教师模型 |
| `DATA_ROOT` | `./data` | 训练数据目录 |
| `LOGGER` | `["console","file"]` | 可加入 `"wandb"`、`"swanlab"`、`"tensorboard"`、`"mlflow"` |
| `LOGGER_FINISH_TIMEOUT` | `300` | 训练结束时最多等待 SwanLab 收尾的秒数；超时后实验照常结束，尚未上传的 SwanLab 数据可能在线上缺失（可用 `swanlab sync` 上传本地备份） |
| `N_GPUS_PER_NODE` | 由脚本指定 | 每个节点的 GPU 数 |
| `NNODES` | `1` | 节点数（多机训练需先启动 Ray 集群） |
| `EXPERIMENT_NAME` | 脚本名 | 实验名；checkpoint 保存在 `checkpoints/<project>/<experiment>` |

任何配置项都可以用 `key=value` 追加在命令后面（见 [examples/config.yaml](examples/config.yaml) 和 [verl/trainer/config.py](verl/trainer/config.py)）。设置 `USE_MODELSCOPE_HUB=1` 可从 ModelScope 下载模型。

**Qwen3-VL Instruct 模型中的 `<think>`。** 这些模型的分词器带有 `<think>`、`</think>` 两个附加 token，但它们从未被训练（只有 Thinking 版会用到）。格式提示词要求写 `<think> ... </think>` 时，这两个标签会被编码成未训练的 token，模型不会照做：在 256 道 ViRL39K 题上，Qwen3-VL-4B-Instruct 用原版分词器的格式奖励为 0.00，把标签当普通文字时为 0.47，准确率相同。因此默认（`worker.actor.model.plain_think_tokens=auto`）把这类 token 当普通文字处理：分词器照常加载（Hugging Face、ModelScope 或本地路径均可），删掉这两个附加 token 后保存在 `~/.cache/parlvr/tokenizers`（可用 `PARLVR_CACHE_DIR` 修改），训练、rollout 和评测都使用它，检查点也保存这份分词器。对话模板会用到这两个 token 的模型（Qwen3-VL Thinking）或本来没有它们的模型（Qwen2.5-VL、InternVL）不受影响。设置 `worker.actor.model.plain_think_tokens=false` 可保留原版分词器（评测从检查点的训练记录读取该设置；其他模型用 `--plain-think-tokens false`）。

## 📚 论文清单

以下条目对照 arXiv、OpenReview 和官方仓库核实（2026-10-01；感知导向的 OPD 论文于 2026-10-15）；只有在一手来源确认时才标注会议（来源见 [docs/paper_list_sources.md](docs/paper_list_sources.md)，BibTeX 见 [docs/paper_list.bib](docs/paper_list.bib)）。每张表内按时间倒序排列。每篇论文按其主要的感知导向改动所在的训练环节归类（见[已复现方法](#-已复现方法)）；本仓库实现的方法列在那里，这里不再重复。欢迎提交 PR 补充论文。

### 🎯 感知导向的 RLVR

面向大视觉语言模型、在某个训练环节加入感知导向改动的 RLVR；基于定位的推理与看图思考归入回答结构。

#### 训练数据与输入

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2025-12 | [Boosting RL-Based Visual Reasoning with Selective Adversarial Entropy Intervention](https://arxiv.org/abs/2512.10414)<br>Yang Yu et al. · `2512.10414` | arXiv | - | **SaEI**: Entropy-guided adversarial perturbation of visual inputs during RL sampling to enlarge the explored answer space. |
| 2025-09 | [Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models](https://arxiv.org/abs/2509.13031)<br>Yan Chen et al. · `2509.13031` | arXiv | [GitHub](https://github.com/cythu/PeBR-R1) | **PeBR-R1**: Two-stage RL: first perception (coarse and fine visual understanding), then reasoning, with dataset-level sampling. |
| 2025-06 | [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/abs/2506.10128)<br>Xiyao Wang et al. · `2506.10128` | NeurIPS 2025 | [GitHub](https://github.com/si0wang/ViCrit) | **ViCrit**: RL proxy task: localize one injected visual hallucination in a human-written caption; exact-match verifiable reward. |
| 2025-06 | [Revisiting Visual Understanding in Multimodal Reasoning through a Lens of Image Perturbation](https://arxiv.org/abs/2506.09736)<br>Yuting Li et al. · `2506.09736` | arXiv | [GitHub](https://github.com/YutingLi0606/Vision-Matters) | **Vision-Matters**: Distractor concatenation, dominance-preserving mixup and random rotation perturbations plugged into SFT, DPO or GRPO. |
| 2025-05 | [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](https://arxiv.org/abs/2505.14231)<br>Sule Bai et al. · `2505.14231` | arXiv | [GitHub](https://github.com/AMAP-ML/UniVG-R1) | **UniVG-R1**: Universal visual grounding: CoT cold-start SFT then rule-based RL with difficulty-aware weight adjustment. |
| 2025-04 | [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](https://arxiv.org/abs/2504.07954)<br>En Yu et al. · `2504.07954` | NeurIPS 2025 | [GitHub](https://github.com/linkangheng/PR1) | **Perception-R1 (Yu et al.)**: Rule-based GRPO for perception policies (grounding, counting, OCR, detection); studies perceptual complexity and reward design. |
| 2025-04 | [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/abs/2504.07615)<br>Haozhan Shen et al. · `2504.07615` | arXiv | [GitHub](https://github.com/om-ai-lab/VLM-R1) | **VLM-R1**: R1-style rule-based RL framework for VLMs on visual understanding tasks; better generalization than SFT. |
| 2025-03 | [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)<br>Ziyu Liu et al. · `2503.01785` | ICCV 2025 | [GitHub](https://github.com/Liuziyu77/Visual-RFT) | **Visual-RFT**: GRPO with verifiable perception rewards (e.g., IoU) for detection, reasoning grounding and fine-grained classification. |

#### 回答结构

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-08 | [ReGround: Restoring Visual Grounding in Multi-Step Reasoning through Self-Diagnosis and Visual Re-Examination](https://arxiv.org/abs/2608.04385)<br>Lei Peng et al. · `2608.04385` | ACM MM 2026 | [GitHub](https://github.com/sespoir/ReGround) | **ReGround**: SFT then GRPO teach self-diagnosis of grounding failures and selective image re-injection during multi-step reasoning. |
| 2026-07 | [Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning](https://arxiv.org/abs/2607.01191)<br>Hongxing Li et al. · `2607.01191` | arXiv | [GitHub](https://github.com/ZJU-REAL/Perceive-to-Reason) | **P2R**: Perceiver localizes evidence, Reasoner answers from annotated image and crops; alternating perception/reasoning GRPO. |
| 2026-06 | [Thinking with Visual Grounding](https://arxiv.org/abs/2606.16122)<br>Junkai Zhang et al. · `2606.16122` | arXiv | [GitHub](https://github.com/Jun-Kai-Zhang/visually_grounded_thinking) | **Visually grounded thinking**: Interleaves thoughts with point/box groundings; grounding-aware RL adds dense grounding rewards to answer rewards. |
| 2026-05 | [iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](https://arxiv.org/abs/2605.31096)<br>Chang-Bin Zhang et al. · `2605.31096` | ICML 2026 | [GitHub](https://github.com/Visual-AI/iVGR) | **iVGR**: Dual-stream RL with a consistency reward internalizes localization into textual CoT; no explicit boxes needed at inference. |
| 2026-03 | [Seeing with You: Perception-Reasoning Coevolution for Multimodal Reasoning](https://arxiv.org/abs/2603.28618)<br>Ziqi Miao et al. · `2603.28618` | arXiv | [GitHub](https://github.com/Dtc7w3PQ/PRCO) | **PRCO**: One shared policy plays Observer (question-tailored evidence caption, utility reward) and Solver (answer, outcome reward). |
| 2026-02 | [What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom](https://arxiv.org/abs/2602.01334)<br>Yan Ma et al. · `2602.01334` | ICML 2026 | [GitHub](https://github.com/GAIR-NLP/Med) | **MED**: Analysis of crop-and-zoom tool-use RL: gains are mostly intrinsic; RL mainly reduces tool-induced harm. |
| 2025-12 | [Deep But Reliable: Advancing Multi-turn Reasoning for Thinking with Images](https://arxiv.org/abs/2512.17306)<br>Wenhao Yang et al. · `2512.17306` | arXiv | - | **DRIM**: Multi-turn tool-based thinking with images; redundancy-penalized policy optimization rewards self-reflection and multi-scale exploration. |
| 2025-12 | [Thinking with Images via Self-Calling Agent](https://arxiv.org/abs/2512.08511)<br>Wenxi Yang et al. · `2512.08511` | arXiv | [GitHub](https://github.com/YWenxi/think-with-images-through-self-calling) | **sCoT**: Recasts interleaved multimodal CoT as language-only CoT that calls parameter-sharing subagents; optimized with GRPO. |
| 2025-12 | [Thinking with Programming Vision: Towards a Unified View for Thinking with Images](https://arxiv.org/abs/2512.03746)<br>Zirun Guo et al. · `2512.03746` | CVPR 2026 | [GitHub](https://github.com/ByteDance-BandAI/CodeVision) | **CodeVision**: Code as a universal tool for image operations; SFT then RL with a dense process reward. |
| 2025-11 | [CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](https://arxiv.org/abs/2511.19820)<br>Miguel Carvalho et al. · `2511.19820` | CVPR 2026 Workshop (GRAIL-V) | [GitHub](https://github.com/miguelscarv/cropvlm) | **CropVLM**: RL-trained external cropping model (no box labels) that zooms into relevant regions for any target VLM. |
| 2025-11 | [DeepEyesV2: Toward Agentic Multimodal Model](https://arxiv.org/abs/2511.05271)<br>Jack Hong et al. · `2511.05271` | ICLR 2026 | [GitHub](https://github.com/Visual-Agent/DeepEyesV2) | **DeepEyesV2**: Agentic MLLM invoking code execution and web search; cold-start SFT then RL; introduces RealX-Bench. |
| 2025-09 | [Mixture-of-Visual-Thoughts: Exploring Context-Adaptive Reasoning Mode Selection for General Visual Reasoning](https://arxiv.org/abs/2509.22746)<br>Zejun Li et al. · `2509.22746` | ICLR 2026 | [GitHub](https://github.com/Future-Living-Lab/mixture-of-visual-thoughts) | **MoVT / AdaVaR**: Unifies multiple visual reasoning modes via SFT; AdaGRPO induces context-adaptive mode selection. |
| 2025-09 | [Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search](https://arxiv.org/abs/2509.07969)<br>Xin Lai et al. · `2509.07969` | ICLR 2026 | [GitHub](https://github.com/Mini-o3/Mini-o3) | **Mini-o3**: Deep multi-turn visual search: Visual Probe dataset, diverse cold-start trajectories, over-turn masking in RL. |
| 2025-09 | [Reinforced Visual Perception with Tools](https://arxiv.org/abs/2509.01656)<br>Zetong Zhou et al. · `2509.01656` | arXiv | [GitHub](https://github.com/ls-kelvin/REVPT) | **ReVPT**: GRPO-based RL that teaches MLLMs to reason with a suite of four visual perception tools. |
| 2025-08 | [Thyme: Think Beyond Images](https://arxiv.org/abs/2508.11630)<br>Yi-Fan Zhang et al. · `2508.11630` | ICLR 2026 | [GitHub](https://github.com/yfzhang114/Thyme) | **Thyme**: Generates and executes code for image manipulation and computation; SFT then RL with GRPO-ATS. |
| 2025-07 | [Look-Back: Implicit Visual Re-focusing in MLLM Reasoning](https://arxiv.org/abs/2507.03019)<br>Shuo Yang et al. · `2507.03019` | arXiv | [GitHub](https://github.com/PKU-YuanGroup/Look-Back) | **Look-Back**: SFT then RL teach MLLMs to autonomously re-focus on visual inputs late in reasoning, without image re-injection. |
| 2025-07 | [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](https://arxiv.org/abs/2507.07999)<br>Haochen Wang et al. · `2507.07999` | ICLR 2026 | [GitHub](https://github.com/Haochen-Wang409/TreeVGR) | **TreeVGR**: RL jointly supervising localization and reasoning for traceable evidence; introduces the TreeBench benchmark. |
| 2025-07 | [High-Resolution Visual Reasoning via Multi-Turn Grounding-Based Reinforcement Learning](https://arxiv.org/abs/2507.05920)<br>Xinyu Huang et al. · `2507.05920` | ACL 2026 Findings | [GitHub](https://github.com/EvolvingLMMs-Lab/MGPO) | **MGPO**: Multi-turn RL cropping sub-images at model-predicted grounding coordinates, using only answer-correctness reward. |
| 2025-06 | [Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing](https://arxiv.org/abs/2506.09965)<br>Junfei Wu et al. · `2506.09965` | NeurIPS 2025 | [GitHub](https://github.com/AntResearchNLP/ViLaSR) | **ViLaSR**: Drawing to reason in space (boxes, auxiliary lines); cold start, reflective rejection sampling, then RL. |
| 2025-06 | [Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning](https://arxiv.org/abs/2506.04034)<br>Qing Jiang et al. · `2506.04034` | ICLR 2026 | [GitHub](https://github.com/IDEA-Research/Rex-Thinker) | **Rex-Thinker**: Object referring as CoT over candidate instances; cold-start SFT then GRPO; learns to abstain when nothing matches. |
| 2025-05 | [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.14677)<br>Jiaer Xia et al. · `2505.14677` | TMLR | [GitHub](https://github.com/maifoundations/Visionary-R1) | **Visionary-R1**: Caption-reason-answer output format trained with pure RL to mitigate shortcut learning. |
| 2025-05 | [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/abs/2505.23678)<br>Gabriel Sarch et al. · `2505.23678` | NeurIPS 2025 | [GitHub](https://github.com/Gabesarch/grounded-rl) | **ViGoRL**: RL anchors each reasoning step to image coordinates; multi-turn variant zooms into predicted coordinates. |
| 2025-05 | [ACTIVE-o3: Empowering MLLMs with Active Perception via Pure Reinforcement Learning](https://arxiv.org/abs/2505.21457)<br>Muzhi Zhu et al. · `2505.21457` | ICML 2026 | [GitHub](https://github.com/aim-uofa/Active-o3) | **ACTIVE-o3**: GRPO framework giving MLLMs active perception (region selection) with a dual-form reward, no region supervision. |
| 2025-05 | [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.20272)<br>Meng Cao et al. · `2505.20272` | arXiv | [GitHub](https://github.com/zzzhhzzz/Ground-R1) | **Ground-R1**: Thinking-with-images trained with Scale Relative Policy Optimization to de-bias reward learning across evidence-region sizes. |
| 2025-05 | [Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning](https://arxiv.org/abs/2505.19702)<br>Minheng Ni et al. · `2505.19702` | NeurIPS 2025 | [GitHub](https://github.com/kodenii/Point-RFT) | **Point-RFT**: Format finetuning on 71K point-grounded rationales, then reinforcement finetuning for visual document understanding. |
| 2025-05 | [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/abs/2505.19255)<br>Mingyuan Wu et al. · `2505.19255` | ICLR 2026 | [GitHub](https://github.com/VTool-R1/VTool-R1) | **VTool-R1**: RL with Python visual-editing tools yields interleaved text-image CoT using only outcome rewards. |
| 2025-05 | [VLM-R³: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192)<br>Chaoya Jiang et al. · `2505.16192` | NeurIPS 2025 | - | **VLM-R³**: Region-Conditioned RPO rewards choosing regions, crop/zoom transformations, and weaving sub-images into interleaved CoT. |
| 2025-05 | [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966)<br>Haozhe Wang et al. · `2505.15966` | NeurIPS 2025 | [GitHub](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) | **Pixel Reasoner**: Pixel-space operations (zoom-in, select-frame); instruction tuning then curiosity-driven RL. |
| 2025-05 | [Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs](https://arxiv.org/abs/2505.15436)<br>Xintong Zhang et al. · `2505.15436` | arXiv | [GitHub](https://github.com/xtong-zhang/Chain-of-Focus) | **Chain-of-Focus**: Adaptive search-and-zoom on key regions; SFT on MM-CoF, then RL with outcome and format rewards. |
| 2025-05 | [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246)<br>Ziyu Liu et al. · `2505.14246` | arXiv | [GitHub](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT) | **Visual-ARFT**: Agentic RFT for web search and code-based image manipulation (crop, rotate); introduces MAT benchmark. |
| 2025-05 | [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617)<br>Zhaochen Su et al. · `2505.08617` | arXiv | [GitHub](https://github.com/zhaochen0110/OpenThinkIMG) | **OpenThinkIMG**: Open framework for tool-augmented LVLMs; V-ToolRL learns adaptive vision-tool invocation from task success. |

#### 奖励

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-08 | [Evidence-RL: Towards Evidence-intensive Visual Reasoning](https://arxiv.org/abs/2608.08021)<br>Haojie Huang et al. · `2608.08021` | NeurIPS 2026 | [GitHub](https://github.com/evidencerl/code) | **Evidence-RL (CED)**: Neutralizes object evidence regions vs. matched non-evidence regions; GRPO rewards correct answers that causally rely on evidence. |
| 2026-05 | [Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning](https://arxiv.org/abs/2605.14054)<br>Haozhe Wang et al. · `2605.14054` | ICML 2026 | - | **MoCA**: Perception Verification via a blindfolded text reasoner rewards perception separately; credit routed to bad seeing vs. bad thinking. |
| 2026-04 | [Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization](https://arxiv.org/abs/2604.08476)<br>Sai Srinivas Kancheti et al. · `2604.08476` | COLM 2026 | - | **Faithful GRPO**: GRPO with logical-consistency and visual-grounding constraints enforced via Lagrangian dual ascent. |
| 2026-02 | [Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs](https://arxiv.org/abs/2602.08241)<br>Siqu Ou et al. · `2602.08241` | arXiv | - | **SAYO**: RL with a region-level visual-attention reward that aligns optimization with visually grounded reasoning steps. |
| 2025-12 | [Learning When to Look: A Disentangled Curriculum for Strategic Perception in Multimodal Reasoning](https://arxiv.org/abs/2512.17227)<br>Siqi Yang et al. · `2512.17227` | CVPR 2026 Findings | [GitHub](https://github.com/gaozilve-max/learning-when-to-look) | **Learning When to Look**: Disentangled SFT curriculum, then RL with a Pivotal Perception Reward teaching when to look. |
| 2025-09 | [VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding](https://arxiv.org/abs/2509.24776)<br>Yizhuo Ding et al. · `2509.24776` | arXiv | - | **VTPerception-R1**: Perception-augmented SFT, then RL with visual, textual and consistency perception rewards. |
| 2025-09 | [DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning](https://arxiv.org/abs/2509.20912)<br>Tianrun Xu et al. · `2509.20912` | ICML 2026 | [GitHub](https://github.com/tinnel123666888/defacto) | **DeFacto**: Positive, counterfactual (evidence masked) and random-masking training with GRPO rewards for evidence-answer consistency. |
| 2025-08 | [Self-Rewarding Vision-Language Model via Reasoning Decomposition](https://arxiv.org/abs/2508.19652)<br>Zongxia Li et al. · `2508.19652` | arXiv | [GitHub](https://github.com/zli12321/Vision-SR1) | **Vision-SR1**: Self-reward checks that generated visual descriptions suffice to answer without the image; decoupled visual and answer advantages. |
| 2025-06 | [Perception-R1: Advancing Multimodal Reasoning Capabilities of MLLMs via Visual Perception Reward](https://arxiv.org/abs/2506.07218)<br>Tong Xiao et al. · `2506.07218` | ICLR 2026 | [GitHub](https://github.com/tongxiao2002/Perception-R1) | **Perception-R1 (Xiao et al.)**: Judging LLM scores consistency between responses and visual annotations as a visual perception reward in RLVR. |

#### 信用分配

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-09 | [Reinforcing Multimodal Reasoning via Token-Level Perception-Grounded Advantage Estimation](https://arxiv.org/abs/2609.39168)<br>Zhihan Zhang et al. · `2609.39168` | ACM MM 2026 | [GitHub](https://github.com/Zhihan72/TPAE) | **TPAE**: Scores each token's consistency with vision-dependency/entropy patterns of correct rollouts to modulate sequence-level advantages. |
| 2026-09 | [Anchoring What Matters: A Dual-Level Learning Framework for Visually-Grounded Multimodal Reasoning](https://arxiv.org/abs/2609.18057)<br>Xinxin Song et al. · `2609.18057` | arXiv | - | **PIVOT**: Replays visually-grounded past trajectories as anchors; allocates extra advantage by local visual support and downstream impact. |
| 2026-06 | [PRPO: Perception-Reinforced Policy Optimization via Token-Level Dynamic Advantage Reshaping](https://arxiv.org/abs/2606.08708)<br>Qiming Li et al. · `2606.08708` | arXiv | - | **PRPO**: Robust Visual Dependency finds grounded, perturbation-stable tokens; Perceptual Advantage Reshaping amplifies their token-level advantages. |
| 2026-06 | [DyCo-RL: Dynamic Cross-Modal Coordination for Visual Reasoning](https://arxiv.org/abs/2606.08035)<br>Hangui Lin et al. · `2606.08035` | arXiv | [GitHub](https://github.com/Sammy20207109/DyCo-RL) | **DyCo-RL**: Assigns tokens visual/text roles via Fisher-Rao attention shifts; reweights advantages by attention-role alignment. |
| 2026-05 | [Structured Role-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2605.07274)<br>Bingqing Jiang et al. · `2605.07274` | arXiv | - | **SRPO**: Role-aware token weights: perception tokens by original-vs-corrupted image dependency, reasoning tokens by consistency with perception. |
| 2026-04 | [Improving Vision-language Models with Perception-centric Process Reward Models](https://arxiv.org/abs/2604.24583)<br>Yingqian Min et al. · `2604.24583` | CVPR 2026 | [GitHub](https://github.com/RUCAIBox/Perceval) | **Perceval**: Perception-centric PRM flags hallucinated image claims; RL applies token-level penalties on those spans instead of sequence-level advantages. |
| 2025-08 | [COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](https://arxiv.org/abs/2508.04182)<br>Peizheng Guo et al. · `2508.04182` | CVPR 2026 | - | **COPO**: Token-level causal-completeness reward (sufficiency/necessity) builds causally informed GRPO advantages to reduce hallucination. |

#### 辅助目标

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-07 | [SIVA-RL: Sensitivity-Invariance Visual Alignment for Multimodal Reinforcement Learning](https://arxiv.org/abs/2607.13931)<br>Cheng Tang et al. · `2607.13931` | arXiv | [GitHub](https://github.com/tchenglv520/SIVA-RL) | **SIVA-RL**: Within-image PatchSwap interventions; audited reward drop routes each pair to sensitivity or invariance alignment. |
| 2026-05 | [Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR](https://arxiv.org/abs/2605.30912)<br>Ruina Hu et al. · `2605.30912` | EMNLP 2026 | [GitHub](https://github.com/Nrich-sunny/Attend-to-Evidence) | **EASE**: On high-reward rollouts only, supervises response-to-image attention toward annotated evidence regions (training-time labels). |
| 2026-05 | [Reinforcing Multimodal Reasoning Against Visual Degradation](https://arxiv.org/abs/2605.09262)<br>Rui Liu et al. · `2605.09262` | arXiv | - | **ROMA**: Teacher-forced corrupted views of clean trajectories, worst-case token KL, correctness-conditioned regularization for degradation robustness. |
| 2026-01 | [CPPO: Contrastive Perception Policy Optimization for VLM Agents](https://arxiv.org/abs/2601.00501)<br>Ahmad Rezaei et al. · `2601.00501` | ICML 2026 Workshop (AIWILD) | [GitHub](https://github.com/vbdi/cppo) | **CPPO**: Contrastive Perception Loss applied to perception tokens detected by entropy shifts under perturbed images. |
| 2025-09 | [Perception-Consistency Multimodal Large Language Models Reasoning via Caption-Regularized Policy Optimization](https://arxiv.org/abs/2509.21854)<br>Songjun Tu et al. · `2509.21854` | arXiv | - | **CapPO**: Caption-based consistency regularization between image- and caption-conditioned responses, plus KL-weighted advantage estimation. |

### 🧑‍🏫 感知导向的 OPD

面向大视觉语言模型的 OPD，其教师、目标或 token 权重由视觉证据决定：教师看到图像的特权视图、不同视图之间的对比，或每个 token 的视觉依赖。

#### 输入与特权信息

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [Distilling Routed 3D Privilege for Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2610.12355)<br>Hongxing Li et al. · `2610.12355` | arXiv | [GitHub](https://github.com/ZJU-REAL/GPD) | **GPD**: Depth, semantic and bird's-eye-view cues rendered as text are routed to the self-teacher as privilege for spatial reasoning; a privileged KL on incorrect trajectories augments GRPO. |
| 2026-10 | [Where-OPD: Spatially Guided On-Policy Self-Distillation of MLLMs with Synthetic Scenes](https://arxiv.org/abs/2610.02117)<br>Sophia Sirko-Galouchenko et al. · `2610.02117` | arXiv | [GitHub](https://github.com/sirkosophia/Where-OPD) | **Where-OPD**: The self-teacher receives textual, spatially grounded guidance on the visual elements a query needs, from procedurally generated scenes with known object positions. |
| 2026-09 | [RS-OPSD: Reliable Privileged On-Policy-Self-Distillation for Ultra-High-Resolution Remote Sensing VQA](https://arxiv.org/abs/2609.38072)<br>Chengjie Jiang et al. · `2609.38072` | arXiv | - | **RS-OPSD**: Zoomed evidence crops as the self-teacher's privilege for ultra-high-resolution remote sensing VQA, with safeguards against context loss and unreliable teachers (GeoEvidence-6K). |
| 2026-09 | [MM-OPD: Towards One More Bottleneck Between Perception and Reasoning](https://arxiv.org/abs/2609.32690)<br>Jintao Tong et al. · `2609.32690` | arXiv | - | **MM-OPD**: A frozen self-teacher also sees a caption or the generating code of the image (a symbolic view); residual token targets correct the image-conditioned prediction toward the right evidence. |
| 2026-09 | [Progressive-View On-Policy Distillation for Regional-to-Global Transfer in Multimodal LLMs](https://arxiv.org/abs/2609.32333)<br>Shanfeng Huang et al. · `2609.32333` | arXiv | - | **PVD**: Progressive views move the student from the region crop to the full image through a padded crop; a regional-advantage weight reallocates the token supervision. |
| 2026-08 | [ArmorOCR: Grounded Adversarial Visual Perception via Observation-Transferred Self-Distillation](https://arxiv.org/abs/2608.20122)<br>Linhan Cao et al. · `2608.20122` | arXiv | [GitHub](https://github.com/ant-research/ArmorOCR) | **ArmorOCR**: Self-distillation from privileged transformed observations for adversarial OCR, then GRPO for grounded OCR; introduces the AdvSpot benchmark. |
| 2026-08 | [Self-Supervised Visual On-Policy Distillation](https://arxiv.org/abs/2608.14144)<br>Yijiang Li et al. · `2608.14144` | arXiv | - | **S²VOPD**: Removes information from the student instead of adding it to the teacher: the teacher sees the original image, the student a strongly augmented view; no annotations, rewards or stronger teacher. |
| 2026-08 | [ADOPD: Reference-Privileged On-Policy Distillation for MLLM-Based Industrial Anomaly Detection](https://arxiv.org/abs/2608.09789)<br>Jingtai He et al. · `2608.09789` | arXiv | [GitHub](https://github.com/withTai/ADOPD) | **ADOPD**: A reference-privileged teacher for industrial anomaly detection; its likelihood gap between matched and mismatched references calibrates each response's weight. |
| 2026-07 | [RP-OPSD: Resolution-Privileged On-Policy Self-Distillation for Multimodal Large Language Models](https://arxiv.org/abs/2607.24447)<br>Qihui Zhu et al. · `2607.24447` | arXiv | [GitHub](https://github.com/sansanyuchen/RP-OPSD) | **RP-OPSD**: Resolution as privilege: the student samples from a low-resolution image and the self-teacher scores with the original resolution. |
| 2026-07 | [Self-Boosting Vision-Language Models with Noisy Student On-Policy Self-Distillation](https://arxiv.org/abs/2607.23125)<br>Shuai Wang et al. · `2607.23125` | arXiv | - | **NOPD**: Noisy-student self-distillation: the model learns on corrupted inputs from its own clean-input predictions, without external models or answers. |
| 2026-06 | [Learning Visual Spatial Planning from Symbolic State via Modality-Gap-Aware Self-Distillation](https://arxiv.org/abs/2606.06076)<br>Haocheng Luo et al. · `2606.06076` | arXiv | [GitHub](https://github.com/Oranger-l/MGSD) | **MGSD**: Self-distillation guided by the symbolic state for visual spatial planning, after a grounding cold start; inference uses the image only. |
| 2026-06 | [ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation](https://arxiv.org/abs/2606.05718)<br>Kanghui Tian et al. · `2606.05718` | arXiv | [GitHub](https://github.com/tiankanghui/ViCuR) | **ViCuR**: Replaces answer-side privilege with query-related visual cues and adds a cue recovery module, so the student can recover the cues from its own input. |

#### 回答结构

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-09 | [On-Policy Visual Evidence Distillation](https://arxiv.org/abs/2609.36838)<br>Shaohang Wei et al. · `2609.36838` | arXiv | [GitHub](https://github.com/sylvain-wei/ReVuE) | **ReVuE**: OPD for visual agents that interleave reasoning with image operations; corrections target the failing stage: acquiring, reading or grounding the evidence. |
| 2026-06 | [Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation](https://arxiv.org/abs/2606.19120)<br>Sihan Wang et al. · `2606.19120` | arXiv | - | **ViGOS**: The student describes the image before reasoning; an image-only perception teacher supervises the description and a privileged teacher the reasoning, against reference-answer shortcuts. |
| 2026-06 | [Thinking Without Images: Internalizing Visual Manipulation with On-Policy Self-Distillation](https://arxiv.org/abs/2606.08719)<br>Yishuo Cai et al. · `2606.08719` | arXiv | [GitHub](https://github.com/walkeralan123/Imagine-OPD) | **Imagine-OPD**: A teacher with zoomed evidence views supervises the student's imagined zoom-in reasoning, internalizing thinking with images without tool calls. |

#### 蒸馏目标

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [CV-OPD: Contrastive Vision On-policy Distillation](https://arxiv.org/abs/2610.13264)<br>Haowei Wang et al. · `2610.13264` | arXiv | - | **CV-OPD**: Compares the teacher's distributions under a positive and a negative crop on the same student prefix and adds a contrastive term, so that the student relies on the visual evidence. |
| 2026-09 | [LEGO-OPD: Factorized Teacher Composition for Multimodal On-Policy Distillation](https://arxiv.org/abs/2610.00333)<br>Jaeyun Shin et al. · `2610.00333` | arXiv | - | **LEGO-OPD**: Composes a language-expert prior and a grounding-expert factor into one teacher distribution, so the visual grounding signal transfers without the VLM teacher's language prior. |
| 2026-09 | [Distill the Visual Evidence, Not Just the Answer: Cross-World On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2609.38777)<br>Yuanhao Sun et al. · `2609.38777` | arXiv | [GitHub](https://github.com/baokou-fw2/CWAD) | **CW-OPD**: Builds two visual worlds that differ in answer-critical evidence; the student distills the teacher's cross-world change of response, not only its answer. |
| 2026-09 | [OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation](https://arxiv.org/abs/2609.16459)<br>Chenhao Qiu et al. · `2609.16459` | arXiv | [GitHub](https://github.com/Echochef/OPD-Aha) | **OPD-Aha**: Reconstructs the target from the teacher's preference with the real image over a visual null, so that the student learns to reflect when its reasoning contradicts the image. |
| 2026-08 | [Distill What the Student Can See: Fisher-Projected On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2608.01263)<br>Leyan Xue et al. · `2608.01263` | arXiv | - | **FP-OPD**: Distills only the teacher corrections the student can realize, projecting the teacher-student gap onto the student's visual tangent space estimated from image perturbations. |
| 2026-07 | [VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation](https://arxiv.org/abs/2607.28590)<br>Kangning Zhang et al. · `2607.28590` | arXiv | [GitHub](https://github.com/DeepExperience/VAD_Multimodal_OPD) | **VAD**: The teacher scores each prefix with the evidence crop present and degraded; only the part of its correction along this visual direction forms the target. |

#### 信用分配

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [VC-OPD: Visual Counterfactual On-Policy Distillation for Grounded Vision-Language Reasoning](https://arxiv.org/abs/2610.14939)<br>Jiahe Feng · `2610.14939` | arXiv | - | **VC-OPD**: The teacher scores each student prefix on the original and on a counterfactual image with the instance evidence corrupted; the gap reweights the distillation toward image-dependent tokens. |
| 2026-09 | [Evidence-Aligned Multimodal On-Policy Self-Distillation for Fine-Grained Visual Understanding](https://arxiv.org/abs/2609.34672)<br>Nanxing Hu et al. · `2609.34672` | arXiv | - | **EAD**: Keeps the crop-conditioned teacher as the target but weights its corrections by a separate evidence reference that removes model-state and lost-context effects. |
| 2026-09 | [BAS-OPD: Budget-Aware Selective On-Policy Self-Distillation for Fine-Grained Multimodal Perception](https://arxiv.org/abs/2609.25891)<br>Zihan Chen et al. · `2609.25891` | arXiv | - | **BAS-OPD**: Selects which rollouts query the privileged-view teacher under a budget (random, uncertainty-based or a learned utility) for fine-grained perception. |
| 2026-08 | [Perception Before Supervision: Self-Contained Visual Distillation from Counterfactual Blind Spots](https://arxiv.org/abs/2608.09931)<br>Shravan Venkatraman et al. · `2608.09931` | BMVC 2026 | [GitHub](https://github.com/mbzuai-oryx/CVPD) | **CVPD**: Finds visual blind spots in the model's own responses (zooming into a region sharpens the answer, removing it changes nothing) and distills them without external supervision. |
| 2026-08 | [OPD-V: Visual On-Policy Self-Distillation with Modality Balance](https://arxiv.org/abs/2608.05131)<br>Aniri et al. · `2608.05131` | arXiv | [GitHub](https://github.com/aniri15/OPD-V) | **OPD-V**: A positive (zoom-in) and a negative (masked) self-teacher; their logit margin defines a modality-balance trust region for selective distillation. |
| 2026-07 | [Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners](https://arxiv.org/abs/2607.28336)<br>Feng Xiong et al. · `2607.28336` | arXiv | - | **PCD**: Strengthens distillation only where downstream failure and teacher-student disagreement both point to a correctable perception error, with separated perception and reasoning rollouts. |
| 2026-07 | [Med-OPD: Improving Medical Vision-Language Models via Evidence-Aware On-Policy Distillation](https://arxiv.org/abs/2607.16303)<br>Yunhang Qian et al. · `2607.16303` | arXiv | [GitHub](https://github.com/yunhang8658/MedOPD) | **Med-OPD**: A medical evidence advantage from a teacher counterfactual focuses distillation on the diagnosis-critical, evidence-dependent tokens of medical VLMs. |
| 2026-06 | [V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning](https://arxiv.org/abs/2606.25319)<br>Haoxiang Sun et al. · `2606.25319` | arXiv | [GitHub](https://github.com/eVI-group-SCU/V-Zero) | **V-Zero**: Answer-label-free OPD for fine-grained visual reasoning: a question-relevant crop and a negative view gate the token-level distillation. |

### 📏 评测基准与数据集

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2025-04 | [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837)<br>Haozhe Wang et al. · `2504.08837` | NeurIPS 2025 | [GitHub](https://github.com/TIGER-AI-Lab/VL-Rethinker) | **VL-Rethinker / ViRL39K**: GRPO with Selective Sample Replay and Forced Rethinking; releases the ViRL39K RL training set. |
| 2025-03 | [MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning](https://arxiv.org/abs/2503.07365)<br>Fanqing Meng et al. · `2503.07365` | arXiv | [GitHub](https://github.com/ModalMinds/MM-EUREKA) | **MM-Eureka / MMK12**: Introduces MMK12, a multimodal K12 math dataset with human-verified answers, plus rule-based RL models. |
| 2024-10 | [DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models](https://arxiv.org/abs/2411.00836)<br>Chengke Zou et al. · `2411.00836` | ICLR 2025 | [GitHub](https://github.com/DynaMath/DynaMath) | **DynaMath**: Dynamic visual math benchmark generating question variants to test reasoning robustness. |
| 2024-09 | [MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2409.02813)<br>Xiang Yue et al. · `2409.02813` | ACL 2025 | [GitHub](https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu-pro) | **MMMU-Pro**: Harder MMMU: filters text-answerable questions, augments options, adds a vision-only (question-in-image) setting. |
| 2024-08 | [Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models](https://arxiv.org/abs/2408.15556)<br>Wenbin Wang et al. · `2408.15556` | AAAI 2025 | [GitHub](https://github.com/DreamMr/HR-Bench) | **HR-Bench**: Benchmark of 4K and 8K images for high-resolution perception; also proposes training-free DC². |
| 2024-08 | [MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?](https://arxiv.org/abs/2408.13257)<br>Yi-Fan Zhang et al. · `2408.13257` | ICLR 2025 | [GitHub](https://github.com/MME-Benchmarks/MME-RealWorld) | **MME-RealWorld**: 29,429 human-annotated QA pairs over 43 subtasks on high-resolution real-world images. |
| 2024-07 | [LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts](https://arxiv.org/abs/2407.04973)<br>Yijia Xiao et al. · `2407.04973` | arXiv | [GitHub](https://github.com/Yijia-Xiao/LogicVista) | **LogicVista**: 448 multiple-choice questions evaluating logical reasoning in visual contexts across 5 tasks. |
| 2024-07 | [We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](https://arxiv.org/abs/2407.01284)<br>Runqi Qiao et al. · `2407.01284` | ACL 2025 | [GitHub](https://github.com/We-Math/We-Math) | **We-Math**: 6.5K visual math problems over 67 knowledge concepts with a four-dimensional diagnostic metric. |
| 2024-03 | [Are We on the Right Way for Evaluating Large Vision-Language Models?](https://arxiv.org/abs/2403.20330)<br>Lin Chen et al. · `2403.20330` | NeurIPS 2024 | [GitHub](https://github.com/MMStar-Benchmark/MMStar) | **MMStar**: 1,500 human-selected vision-indispensable samples minimizing visual-free answering and data leakage. |
| 2024-03 | [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://arxiv.org/abs/2403.14624)<br>Renrui Zhang et al. · `2403.14624` | ECCV 2024 | [GitHub](https://github.com/ZrrSkywalker/MathVerse) | **MathVerse**: 2,612 diagram math problems, each in six versions varying text vs. diagram information (15K test samples). |
| 2024-02 | [Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset](https://arxiv.org/abs/2402.14804)<br>Ke Wang et al. · `2402.14804` | NeurIPS 2024 (Datasets and Benchmarks) | [GitHub](https://github.com/mathllm/MATH-V) | **MATH-Vision**: 3,040 competition math problems with visual contexts across 16 disciplines and 5 difficulty levels. |
| 2023-12 | [V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135)<br>Penghao Wu et al. · `2312.14135` | CVPR 2024 | [GitHub](https://github.com/penghao-wu/vstar) | **V\* / V\*Bench**: Introduces V\*Bench for fine visual detail search in high-resolution images, alongside the SEAL visual-search MLLM. |
| 2023-11 | [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502)<br>Xiang Yue et al. · `2311.16502` | CVPR 2024 | [GitHub](https://github.com/MMMU-Benchmark/MMMU) | **MMMU**: 11.5K college-level multimodal questions across 6 disciplines and 30 subjects. |
| 2023-10 | [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://arxiv.org/abs/2310.14566)<br>Tianrui Guan et al. · `2310.14566` | CVPR 2024 | [GitHub](https://github.com/tianyi-lab/HallusionBench) | **HallusionBench**: 346 images with 1,129 expert-written questions diagnosing language hallucination and visual illusion. |
| 2023-10 | [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://arxiv.org/abs/2310.02255)<br>Pan Lu et al. · `2310.02255` | ICLR 2024 | [GitHub](https://github.com/lupantech/MathVista) | **MathVista**: 6,141 examples from 28 existing datasets plus three new ones (IQTest, FunctionQA, PaperQA). |
| 2023-05 | [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)<br>Yifan Li et al. · `2305.10355` | EMNLP 2023 | [GitHub](https://github.com/RUCAIBox/POPE) | **POPE**: Polling-based object probing evaluation for object hallucination in LVLMs. |
| 2021-05 | [Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning](https://arxiv.org/abs/2105.04165)<br>Pan Lu et al. · `2105.04165` | ACL 2021 | [GitHub](https://github.com/lupantech/InterGPS) | **Inter-GPS / Geometry3K**: Introduces Geometry3K (3,002 geometry problems with formal-language annotations) and the Inter-GPS solver. |

## 🤝 贡献

欢迎贡献：

- **补充论文**：提交 PR，在中英文两份 README 的清单中按时间倒序各添加一行（arXiv 链接、会议、代码链接），然后运行 `python scripts/check_docs.py --fix`。
- **新增方法**：参考 [docs/add_method.md](docs/add_method.md)，新增 `examples/reproduction/<method>/` 目录，包含 README 和基线脚本。
- **新增评测基准**：参考 [eval/README.md](eval/README.md#adding-a-new-benchmark)。
- **提交复现结果**：用 "Reproduction results" 模板开 issue，提供运行命令、日志和评测汇总。

开 PR 之前，先用 `pip install pre-commit && pre-commit install` 配好检查（只需一次），再运行 `pre-commit run --all-files` 和 `make test`。[贡献指南](.github/CONTRIBUTING_zh.md)（[English](.github/CONTRIBUTING.md)）介绍了每项检查、提交信息格式和提交 PR 的步骤。

## 📝 引用

如果本仓库对你有帮助，请引用 CGPO 以及你所使用方法的原论文：

```bibtex
@inproceedings{zhang2026cgpo,
  title     = {{CGPO}: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning},
  author    = {Zhang, Shengxuming and Zhou, Linyun and Lou, Hengrui and Wang, Zhenyang and Zhang, Xiuming and Feng, Zunlei},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  location  = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {10},
  doi       = {10.1145/3767308.3835969},
  url       = {https://doi.org/10.1145/3767308.3835969}
}
```

各复现方法的 BibTeX 见对应目录的 README 和 [docs/references.bib](docs/references.bib)。

## 🙏 致谢

本仓库基于 [EasyR1](https://github.com/hiyouga/EasyR1) 和 [veRL](https://github.com/volcengine/verl) 构建，使用 [vLLM](https://github.com/vllm-project/vllm) 进行 rollout 和评测。感谢所有被复现方法的作者公开论文、代码和数据，特别感谢 PAPO、VPPO、GRIT 和 DeepEyes 公开的训练与评测数据。

## 📄 许可证

Apache-2.0，见 [LICENSE](LICENSE)。脚本下载的数据集和模型遵循各自的许可证。
