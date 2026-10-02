# Paper list: verification notes

The paper list in the README was verified on 2026-10-01. This page records how each entry was checked and where each venue comes from.

## Notes on verification

- Every entry's arXiv page was fetched on 2026-10-01 (`https://arxiv.org/abs/<id>`). Titles are the **current** arXiv titles, which differ from v1 for some papers. Examples: 2506.09736 (v1: *Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning*), 2505.15436 (v1: *Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL*), 2601.00501 (v1: *CPPO: Contrastive Perception for Vision Language Policy Optimization*) and 2605.14054 (v1 ended *... for Vision-Language Reasoning*). Date is the month of arXiv v1.
- **Venue** is filled only when one of these sources confirms it: the arXiv comments or journal-ref field, an OpenReview venue record (`api2.openreview.net`), the paper's official repo README, or CVF Open Access. Otherwise the entry says `arXiv`. The per-entry sources are listed below.
- **Code** links were checked to resolve (HTTP 200) on 2026-10-01, and each one matches the paper (by abstract link, the arXiv HTML, the HF papers API, or the repo README). Some old URLs now redirect, and the table shows where they land: `NUS-TRAIL/NoisyRollout` → `real-absolute-AI/NoisyRollout`, `eric-ai-lab/GRIT` → `UCSB-AI/GRIT`, `yfzhang114/MME-RealWorld` → `MME-Benchmarks/MME-RealWorld`.
- Caveats:
  - **PGPO** (2604.01840): the abstract announces `github.com/Yzk1114/PGPO`, but it returns 404, so Code is `-`.
  - **VTPerception-R1** (2509.24776): the linked `github.com/yizhuoDi/VTPerceprion-R1` returns 404, so Code is `-`.
  - **CapPO** (2509.21854): `github.com/TU2021/CapPO` exists but the repository is empty, so Code is `-`.
  - **CPPO** (2601.00501): the repo README says "accepted to ICML 2026". OpenReview lists only an ICML 2026 **AIWILD workshop** record and no main-track record, while other ICML 2026 main-track papers do appear in the same search. The table therefore lists the workshop.
  - **MoCA** (2605.14054): the arXiv comment says "ICML 2026 Oral", but OpenReview says "ICML 2026 spotlight". The table lists only "ICML 2026".
  - **Evidence-RL** (2608.08021): NeurIPS 2026 is stated only in the official repo README. There is no arXiv comment and no OpenReview record yet.
  - **VEPO** (2606.03937): a co-author homepage reports EMNLP 2026 Findings. No allowed source confirms it, so the table keeps `arXiv`.
  - **PRPO** (2606.08708): third-party aggregators list it as a NeurIPS 2026 poster. OpenReview shows only a CoRR record, so the table keeps `arXiv`.
  - **Visual-ARFT** shares the Visual-RFT repository (`Visual-ARFT/` subfolder). **MMMU-Pro** lives in the MMMU repository (`mmmu-pro/` subfolder).
  - The two different **Perception-R1** papers are disambiguated by first author.

<details><summary>Per-entry venue sources</summary>

- TPAE (`2609.39168`): ACM MM 2026, from arXiv comments
- Evidence-RL (CED) (`2608.08021`): NeurIPS 2026, from official repo README
- ReGround (`2608.04385`): ACM MM 2026, from arXiv comments
- CFPO (`2606.23206`): ICML 2026, from arXiv comments; OpenReview
- EASE (`2605.30912`): EMNLP 2026, from arXiv comments
- MoCA (`2605.14054`): ICML 2026, from arXiv comments; OpenReview
- Perceval (`2604.24583`): CVPR 2026, from arXiv journal-ref; repo README
- VGPO (`2604.09349`): ACL 2026, from arXiv comments; repo README
- Faithful GRPO (`2604.08476`): COLM 2026, from OpenReview
- CPPO (`2601.00501`): ICML 2026 Workshop (AIWILD), from OpenReview (README states ICML 2026)
- Learning When to Look (`2512.17227`): CVPR 2026 Findings, from OpenReview; repo README
- VPPO (`2510.09285`): ICLR 2026, from arXiv comments; repo README
- VAPO (`2509.25848`): ICLR 2026, from arXiv comments; OpenReview
- COPO (`2508.04182`): CVPR 2026, from OpenReview; CVF Open Access
- PAPO (`2507.06448`): ICLR 2026, from OpenReview; repo README
- ViCrit (`2506.10128`): NeurIPS 2025, from OpenReview
- Perception-R1 (Xiao et al.) (`2506.07218`): ICLR 2026, from OpenReview
- Visionary-R1 (`2505.14677`): TMLR, from OpenReview
- NoisyRollout (`2504.13055`): NeurIPS 2025, from arXiv comments; OpenReview; repo README
- iVGR (`2605.31096`): ICML 2026, from arXiv comments; OpenReview
- MED (`2602.01334`): ICML 2026, from arXiv comments; OpenReview
- CodeVision (`2512.03746`): CVPR 2026, from OpenReview; repo README
- CropVLM (`2511.19820`): CVPR 2026 Workshop (GRAIL-V), from arXiv comments
- DeepEyesV2 (`2511.05271`): ICLR 2026, from arXiv comments; OpenReview
- MoVT / AdaVaR (`2509.22746`): ICLR 2026, from arXiv comments; OpenReview
- DeFacto (`2509.20912`): ICML 2026, from OpenReview; repo README
- Mini-o3 (`2509.07969`): ICLR 2026, from OpenReview; repo README
- Thyme (`2508.11630`): ICLR 2026, from OpenReview; repo README
- TreeVGR (`2507.07999`): ICLR 2026, from arXiv comments; OpenReview
- MGPO (`2507.05920`): ACL 2026 Findings, from arXiv comments
- ViLaSR (`2506.09965`): NeurIPS 2025, from OpenReview; repo README
- Rex-Thinker (`2506.04034`): ICLR 2026, from OpenReview; repo README
- ViGoRL (`2505.23678`): NeurIPS 2025, from OpenReview
- ACTIVE-o3 (`2505.21457`): ICML 2026, from arXiv comments; OpenReview; repo README
- Point-RFT (`2505.19702`): NeurIPS 2025, from OpenReview
- VTool-R1 (`2505.19255`): ICLR 2026, from arXiv comments; OpenReview; repo README
- VLM-R³ (`2505.16192`): NeurIPS 2025, from OpenReview
- Pixel Reasoner (`2505.15966`): NeurIPS 2025, from OpenReview; repo README
- GRIT (`2505.15879`): NeurIPS 2025, from arXiv journal-ref; OpenReview; repo README
- DeepEyes (`2505.14362`): ICLR 2026, from arXiv comments; OpenReview
- Perception-R1 (Yu et al.) (`2504.07954`): NeurIPS 2025, from OpenReview; repo README
- Visual-RFT (`2503.01785`): ICCV 2025, from repo README
- VL-Rethinker / ViRL39K (`2504.08837`): NeurIPS 2025, from OpenReview; repo README
- DynaMath (`2411.00836`): ICLR 2025, from arXiv comments
- MMMU-Pro (`2409.02813`): ACL 2025, from arXiv comments
- HR-Bench (`2408.15556`): AAAI 2025, from repo README
- MME-RealWorld (`2408.13257`): ICLR 2025, from arXiv comments; repo README
- We-Math (`2407.01284`): ACL 2025, from repo README
- MMStar (`2403.20330`): NeurIPS 2024, from repo README
- MathVerse (`2403.14624`): ECCV 2024, from arXiv comments; repo README
- MATH-Vision (`2402.14804`): NeurIPS 2024 (Datasets and Benchmarks), from OpenReview; repo README
- V\* / V\*Bench (`2312.14135`): CVPR 2024, from CVF Open Access
- MMMU (`2311.16502`): CVPR 2024, from arXiv comments
- HallusionBench (`2310.14566`): CVPR 2024, from arXiv comments; repo README
- MathVista (`2310.02255`): ICLR 2024, from arXiv comments; repo README
- POPE (`2305.10355`): EMNLP 2023, from arXiv comments; repo README
- Inter-GPS / Geometry3K (`2105.04165`): ACL 2021, from arXiv comments; repo README

</details>

