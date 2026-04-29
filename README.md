# Awesome-Large-Language-Diffusion-Models

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Maintained](https://img.shields.io/badge/Maintained-2026-blue?style=flat-square)

A comprehensive and structured list of research papers about **Large-Language-Diffusion-Models (dLLMs)**.

> Last major update: April 2026 — added latest 2026 arxiv works, NeurIPS 2025 spotlights, ICLR 2026 papers, frontier-scale dLLMs (LLaDA2.0/2.1, Dream-VLA, etc.), and a new section for Agentic / Tool-Use behavior.

---

## ⚙️ Framework (Taxonomy)

1. [**Surveys & Useful Resources**](#1-surveys--useful-resources)
2. [**Core Methodologies**](#2-core-methodologies)
    - [Discrete & Masked Diffusion](#21-discrete--masked-diffusion)
    - [Continuous & Latent Space Diffusion](#22-continuous--latent-space-diffusion)
    - [AR-to-Diffusion Adaptation](#23-ar-to-diffusion-adaptation)
    - [Hybrid AR-Diffusion (Block / Forcing)](#24-hybrid-ar-diffusion-block--forcing)
3. [**Reasoning & Policy Optimization**](#3-reasoning--policy-optimization)
    - [Reasoning & Planning](#31-reasoning--planning)
    - [Alignment & RL](#32-alignment--reinforcement-learning)
4. [**Token Ordering & Generation Strategies**](#4-token-ordering--generation-strategies)
5. [**System Efficiency & Acceleration**](#5-system-efficiency--acceleration)
    - [Caching & Memory Strategy](#51-caching--memory-strategy)
    - [Decoding & Sampling](#52-decoding--sampling)
    - [Distillation, Quantization & Sparsity](#53-distillation-quantization--sparsity)
    - [Inference Frameworks & Systems](#54-inference-frameworks--systems)
6. [**Multi-modal & Physical AI**](#6-multi-modal--physical-ai)
    - [Multi-modal dLLMs](#61-multi-modal-dllms)
    - [Vision-Language-Action (VLA)](#62-vision-language-action-vla)
    - [Autonomous Driving / World Models](#63-autonomous-driving--world-models)
7. [**Agentic & Tool-Use dLLMs**](#7-agentic--tool-use-dllms)
8. [**Theory, Guidance & Applications**](#8-theory-guidance--applications)
    - [Theory & Analysis](#81-theory--analysis)
    - [Guidance & Downstream Applications](#82-guidance--downstream-applications)
9. [**Seminal Diffusion Papers**](#9-seminal-diffusion-papers)

---

## 1. Surveys & Useful Resources

### 📚 Blogs & Reports
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Mercury (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [Dream-7B](https://hkunlp.github.io/blog/2025/dream/)
- [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/)
- [Discrete Diffusion Forcing (D2F)](https://zhijie-group.github.io/Discrete-Diffusion-Forcing/)
- [LLaDA2.X (InclusionAI / Ant Group)](https://github.com/inclusionAI/LLaDA2.X)
- [What are Diffusion Language Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generative Modeling by Estimating Gradients (Yang Song)](https://yang-song.net/blog/2021/score/)

### 📝 Survey & Perspective Papers
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) | 2023.03 | IJCAI | Early NAR-text survey |
| [A Survey of Diffusion Models in NLP](https://arxiv.org/abs/2305.14671) | 2023.05 | Arxiv | Early NLP survey |
| [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759) | 2025.06 | Arxiv | dLLM + dMLLM |
| [Diffusion-based Large Language Models Survey](https://www.techrxiv.org/users/952417/articles/1321784-diffusion-based-large-language-models-survey) | 2025.08 | TechRxiv | - |
| [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/pdf/2508.08712v2) | 2025.08 | Arxiv | - |
| [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875) | 2025.08 | Arxiv | VILA-Lab; comprehensive |
| [Efficient Diffusion Language Models: A Comprehensive Survey](https://github.com/FelixMessi/Awesome-Efficient-dLLMs) | 2026.01 | TechRxiv | Efficiency-focused |
| [Top 10 Open Challenges Steering the Future of Diffusion Language Model and Its Variants](https://arxiv.org/abs/2601.14041) | 2026.01 | Arxiv | Perspective / roadmap |

---

## 2. Core Methodologies

### 2.1 Discrete & Masked Diffusion
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DiffusER: Discrete Diffusion via Edit-based Reconstruction](https://arxiv.org/abs/2210.16886) | 2022.10 | ICLR | <7B |
| [SSD-LM: Semi-autoregressive Simplex-based Diffusion for Modular Control](https://aclanthology.org/2023.acl-long.647.pdf) | 2022.10 | ACL | <7B, Simplex |
| [DiffusionBERT: Improving Generative Masked Language Models](https://aclanthology.org/2023.acl-long.248.pdf) | 2022.11 | ACL | <7B, Masked |
| [A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/abs/2302.05737) | 2023.02 | COLM | <7B |
| [David helps Goliath: Inference-Time Collaboration Between Small and Large Diffusion LMs](https://arxiv.org/abs/2305.14771) | 2023.05 | NAACL | >7B, Scale-collaboration |
| [TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379) | 2023.05 | EACL | <7B, Simplex |
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | 2023.08 | Arxiv | >7B, Scaling |
| [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)](https://proceedings.mlr.press/v235/lou24a.html) | 2023.10 | ICML | <7B, Discrete |
| [Simplified and Generalized Masked Diffusion for Discrete Data (MD4)](https://arxiv.org/pdf/2406.04329) | 2024.06 | NeurIPS | - |
| [Simple and Effective Masked Diffusion Language Models (MDLM)](https://arxiv.org/abs/2406.07524) | 2024.06 | NeurIPS | <7B, Masked |
| [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (RADD)](https://arxiv.org/abs/2406.03736) | 2024.06 | ICLR | <7B, Masked |
| [Scaling up Masked Diffusion Models on Text (SMDM)](https://arxiv.org/abs/2410.18514) | 2024.10 | ICLR | <7B, 1.1B Scaling |
| [Energy-Based Diffusion Language Models for Text Generation (EDLM)](https://arxiv.org/abs/2410.21357) | 2024.10 | ICLR | <7B |
| [Conditional MASK Discrete Diffusion Language Model](https://arxiv.org/abs/2411.06438) | 2024.11 | EMNLP | <7B |
| [Non-Markovian Discrete Diffusion with Causal Language Models](https://arxiv.org/abs/2502.09767v1) | 2025.02 | NeurIPS | <7B |
| [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) | 2025.02 | NeurIPS | >7B, LLaDA-8B |
| [Anchored Diffusion Language Model (ADLM)](https://arxiv.org/abs/2505.18456) | 2025.05 | NeurIPS | >7B; ANELBO objective |
| [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429v2) | 2025.06 | Arxiv | >7B, Context Scaling |
| [Esoteric Language Models (Eso-LMs)](https://arxiv.org/pdf/2506.01928) | 2025.06 | Arxiv | AR + MDM hybrid |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487v1) | 2025.08 | Arxiv | >7B, Dream-7B |
| [Sequential Diffusion Language Models](https://arxiv.org/abs/2509.24007v1) | 2025.09 | Arxiv | >7B |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025.09 | Arxiv | >7B, 7B-A1B MoE from scratch |
| [UltraLLaDA: Scaling Context to 128K](https://arxiv.org/abs/2510.10481) | 2025.10 | Arxiv | >7B, Context Scaling |
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632) | 2025.10 | NeurIPS | - |
| [Masked Diffusion Models as Energy Minimization](https://arxiv.org/abs/2509.13866v1) | 2025.10 | NeurIPS | <7B |
| [Soft-Masked Diffusion Language Models](https://arxiv.org/abs/2510.17206v1) | 2025.10 | Arxiv | <7B |
| [Variational Masked Diffusion Models](https://arxiv.org/abs/2510.23606v1) | 2025.10 | Arxiv | <7B |
| [Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way](https://arxiv.org/abs/2510.24605v1) | 2025.10 | Arxiv | >7B, Variable Length |
| [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276) | 2025.11 | Arxiv | Data efficiency |
| [DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone](https://arxiv.org/abs/2511.15927) | 2025.11 | Arxiv | Non-Transformer Backbone |
| [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923v1) | 2025.11 | Arxiv | >7B |
| [C2DLM: Causal Concept-Guided Diffusion Large Language Models](https://arxiv.org/abs/2511.22146v1) | 2025.11 | Arxiv | >7B |
| [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745v1) | 2025.12 | Arxiv | >100B, MoE; Ant Group |
| [LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676) | 2026.02 | Arxiv | Editable State Evolution |
| [Introspective Diffusion Language Models (I-DLM)](https://arxiv.org/abs/2604.11035) | 2026.04 | Arxiv | Introspective consistency |

### 2.2 Continuous & Latent Space Diffusion
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) | 2022.05 | NeurIPS | <7B, Embedding |
| [DiffuSeq: Sequence to Sequence Text Generation](https://arxiv.org/abs/2210.08933) | 2022.10 | ICLR | <7B, Embedding |
| [Latent Diffusion for Language Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html) | 2022.12 | NeurIPS | <7B, Latent |
| [Diffusion Glancing Transformer for Parallel Sequence to Sequence Learning](https://arxiv.org/abs/2212.10240) | 2022.12 | NAACL | <7B |
| [Empowering Diffusion Models on the Embedding Space for Text Generation](https://arxiv.org/abs/2212.09412) | 2022.12 | NAACL | <7B, Embedding |
| [Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise](https://arxiv.org/abs/2212.11685) | 2022.12 | ICML | <7B, Embedding |
| [DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises](https://arxiv.org/abs/2302.10025) | 2023.02 | TACL | <7B, Embedding |
| [Likelihood-Based Diffusion Language Models (Plaid)](https://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html) | 2023.05 | NeurIPS | <7B, Plaid 1B |
| [PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model](https://arxiv.org/abs/2306.02531) | 2023.06 | NeurIPS | <7B, Latent |
| [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/pdf/2506.09018) | 2025.06 | Arxiv | - |
| [Coevolutionary Continuous Discrete Diffusion: Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025.10 | Arxiv | >7B; CCDD |

### 2.3 AR-to-Diffusion Adaptation
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models (DiffuLLM)](https://openreview.net/forum?id=j1tSLYKwg8) | 2024.10 | ICLR | >7B, GPT2/LLaMA2 Adaptation |
| [Large Language Models to Diffusion Finetuning](https://arxiv.org/abs/2501.15781) | 2025.01 | ICML | >7B |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) | 2025.02 | ACL | >7B, Adapted from Mistral |
| [SDAR: A Synergistic Diffusion-AutoRegression Paradigm](https://arxiv.org/abs/2510.06303) | 2025.10 | Arxiv | >7B, Qwen3-based BD |
| [From Next-Token to Next-Block: Principled Adaptation Path](https://arxiv.org/abs/2512.06776) | 2025.11 | Arxiv | >7B, Adaptation Path |
| [Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](https://arxiv.org/abs/2512.14067v1) | 2025.12 | Arxiv | >7B |
| [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745v1) | 2025.12 | Arxiv | >7B, AR→dLLM at 100B |

### 2.4 Hybrid AR-Diffusion (Block / Forcing)
> A new section: hybrids that interleave block-level AR with intra-block diffusion, or "forcing" approaches that retain causal masks for KV-cache reuse.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)](https://arxiv.org/abs/2503.09573) | 2025.03 | ICLR | <7B, Interpolation |
| [D2F: Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192) | 2025.08 | ICLR | >7B, Faster-than-AR |
| [Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding](https://arxiv.org/abs/2508.19529v1) | 2025.08 | Arxiv | >7B |
| [SDAR: Synergistic Diffusion-AutoRegression Paradigm](https://arxiv.org/abs/2510.06303) | 2025.10 | Arxiv | >7B, Block hybrid |
| [Encoder-Decoder Block Diffusion Language Models for Efficient Training and Inference (E2D2)](https://neurips.cc/virtual/2025/poster/119836) | 2025.10 | NeurIPS | Block Enc-Dec |
| [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328) | 2025.09 | Arxiv | >7B, Block Decoding |
| [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737v1) | 2025.12 | Arxiv | Causal-attn diffusion |
| [ReFusion: Diffusion LLM with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586v1) | 2025.12 | Arxiv | Slot-level interleaving |
| [Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models](https://arxiv.org/abs/2602.04399) | 2026.02 | Arxiv | Adaptive block |
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) | 2026.02 | Arxiv | Block + speculative |

---

## 3. Reasoning & Policy Optimization

### 3.1 Reasoning & Planning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion of Thought: Chain-of-Thought Reasoning in dLLMs](https://arxiv.org/abs/2402.07754) | 2024.02 | NeurIPS | <7B, CoT Foundation |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/pdf/2410.14157) | 2024.10 | ICLR | <7B |
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2024.10 | Arxiv | Planning |
| [d1: Scaling Reasoning in dLLMs via RL](https://arxiv.org/abs/2504.12216) | 2025.04 | NeurIPS | >7B, Reasoning scaling |
| [Reinforcing the Diffusion Chain of Lateral Thought](https://arxiv.org/abs/2505.10446) | 2025.05 | NeurIPS | >7B |
| [Thinking Inside the Mask: In-Place Prompting in dLLMs](https://arxiv.org/pdf/2508.10736) | 2025.08 | Arxiv | >7B |
| [Reinforced Context Order Recovery for Adaptive Reasoning](https://arxiv.org/pdf/2508.13070) | 2025.08 | Arxiv | <7B, Planning |
| [d2: Improved Techniques for Training Reasoning dLLMs](https://www.arxiv.org/abs/2509.21474) | 2025.09 | Arxiv | >7B |
| [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573) | 2025.10 | Arxiv | >7B |
| [Beyond Surface Reasoning: Unveiling Long CoT Capacity](https://arxiv.org/abs/2510.09544) | 2025.10 | Arxiv | >7B |
| [Coevolutionary Continuous Discrete Diffusion: Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025.10 | Arxiv | >7B |
| [On the Reasoning Abilities of Masked Diffusion Language Models](https://arxiv.org/abs/2510.13117v1) | 2025.10 | Arxiv | >7B |
| [Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning](https://arxiv.org/abs/2510.15244v2) | 2025.10 | Arxiv | Collaboration |
| [Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning](https://arxiv.org/abs/2510.27469v1) | 2025.10 | Arxiv | >7B |

### 3.2 Alignment & Reinforcement Learning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295) | 2025.03 | Arxiv | >7B |
| [DiFFPO: Training dLLMs to Reason Fast and Furious via RL](https://arxiv.org/pdf/2510.02212) | 2025.05 | Arxiv | >7B, Direct Preference |
| [LLaDA 1.5: Variance-Reduced Preference Optimization](https://arxiv.org/abs/2505.19223) | 2025.05 | Arxiv | >7B |
| [wd1: Weighted Policy Optimization for Reasoning](https://arxiv.org/pdf/2507.08838) | 2025.07 | Arxiv | >7B |
| [Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](https://arxiv.org/abs/2508.12398v1) | 2025.08 | Arxiv | >7B, Safety |
| [Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](https://arxiv.org/abs/2507.19227v1) | 2025.07 | Arxiv | Safety |
| [The Devil behind the mask: An emergent safety vulnerability](https://arxiv.org/pdf/2507.11097v1) | 2025.07 | Arxiv | Safety |
| [MDPO: Overcoming the Training-Inference Divide](https://arxiv.org/abs/2508.13148) | 2025.08 | Arxiv | >7B |
| [Reward-Weighted Sampling: Enhancing Non-Autoregressive Characteristics in Masked Diffusion LLMs](https://arxiv.org/abs/2509.00707) | 2025.08 | EMNLP | >7B |
| [Inpainting-Guided Policy Optimization for dLLMs](https://arxiv.org/abs/2509.10396) | 2025.09 | Arxiv | >7B |
| [Taming Masked Diffusion via Consistency Trajectory RL](https://arxiv.org/abs/2509.23924) | 2025.09 | Arxiv | >7B |
| [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning](https://arxiv.org/abs/2509.25171) | 2025.09 | Arxiv | >7B |
| [Revolutionizing RL Framework for Diffusion Large Language Models](https://arxiv.org/pdf/2509.06949) | 2025.09 | Arxiv | >7B |
| [A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](https://arxiv.org/abs/2509.23286v1) | 2025.09 | Arxiv | Safety |
| [DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models](https://arxiv.org/abs/2509.24296v1) | 2025.09 | Arxiv | Safety |
| [RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance](https://arxiv.org/abs/2509.25604v1) | 2025.09 | Arxiv | >7B |
| [AGRPO: Simple Policy Gradients for Reasoning with Diffusion Language Models](https://arxiv.org/abs/2510.04019) | 2025.10 | Arxiv | >7B |
| [Improving Reasoning via Group Diffusion Policy Optimization (GDPO)](https://arxiv.org/pdf/2510.08554) | 2025.10 | Arxiv | >7B |
| [Step-Aware Policy Optimization for Reasoning](https://arxiv.org/abs/2510.01544) | 2025.10 | Arxiv | >7B |
| [MRO: Enhancing Reasoning via Multi-Reward Optimization](https://arxiv.org/abs/2510.21473) | 2025.10 | Arxiv | >7B |
| [Enhancing Reasoning via Distribution Matching Policy Optimization](https://arxiv.org/abs/2510.08233) | 2025.10 | Arxiv | >7B |
| [Boundary-Guided Policy Optimization for Memory-efficient RL](https://arxiv.org/abs/2510.11683) | 2025.10 | Arxiv | >7B |
| [SPG: Sandwiched Policy Gradient for Masked Diffusion](https://arxiv.org/abs/2510.09541) | 2025.10 | Arxiv | >7B |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025.10 | Arxiv | >7B |
| [Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052v2) | 2025.10 | Arxiv | >7B |
| [Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective](https://arxiv.org/abs/2512.03759v1) | 2025.12 | Arxiv | >7B |
| [d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2512.09675v1) | 2025.12 | Arxiv | >7B |
| [DARE: Diffusion Large Language Models Alignment and Reinforcement Executor](https://arxiv.org/abs/2604.04215) | 2026.04 | Arxiv | Unified RL framework |
| [DiRL: An Efficient Post-Training Framework for Diffusion Language Models](https://arxiv.org/abs/2512.22234) | 2025.12 | Arxiv | Post-training |
| [Efficient and Stable Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2602.08905) | 2026.02 | Arxiv | Variance reduction |
| [Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation](https://arxiv.org/abs/2601.07152) | 2026.01 | Arxiv | Multi-agent RL |

---

## 4. Token Ordering & Generation Strategies

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [SSD-LM: Semi-autoregressive Simplex-based Diffusion for Modular Control](https://aclanthology.org/2023.acl-long.647.pdf) | 2022.10 | ACL | <7B, Blockwise |
| [AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation](https://arxiv.org/abs/2305.09515) | 2023.05 | NeurIPS | <7B, AR-like noise |
| [Train for the Worst, Plan for the Best: Understanding Token Ordering](https://arxiv.org/pdf/2502.06768) | 2025.02 | ICML | <7B, Ordering Analysis |
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion LMs](https://arxiv.org/abs/2503.09573) | 2025.03 | ICLR | <7B, Interpolation |
| [Review, Remask, Refine (R3): Process-Guided Block Diffusion](https://arxiv.org/pdf/2507.08018v1) | 2025.07 | ICML MOSS | >7B, Block-wise |
| [Any-Order Flexible Length Masked Diffusion](https://arxiv.org/pdf/2509.01025) | 2025.09 | Arxiv | <7B, Order Flexibility |
| [Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models](https://arxiv.org/abs/2509.23653v1) | 2025.09 | Arxiv | >7B, Remasking |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025.10 | NeurIPS | <7B, Edit preservation |
| [Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models](https://arxiv.org/abs/2510.05090v1) | 2025.10 | Arxiv | >7B, Unmasking |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025.10 | Arxiv | >7B, Unmasking |
| [Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing](https://arxiv.org/abs/2510.21961v1) | 2025.10 | Arxiv | >7B, Unmasking |
| [Diffusion Language Model Inference with Monte Carlo Tree Search](https://arxiv.org/abs/2512.12168v1) | 2025.12 | Arxiv | >7B, MCTS |
| [Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty](https://arxiv.org/abs/2512.21336v1) | 2025.12 | Arxiv | >7B, Unmasking |
| [Adaptation to Intrinsic Dependence in Diffusion Language Models](https://arxiv.org/abs/2602.20126) | 2026.02 | Arxiv | Distribution-agnostic schedule |
| [D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding](https://arxiv.org/abs/2603.19146) | 2026.03 | Arxiv | Diversity-aware decoding |

---

## 5. System Efficiency & Acceleration

### 5.1 Caching & Memory Strategy
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025.05 | NeurIPS | >7B |
| [FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467) | 2025.05 | Arxiv | >7B |
| [Fast-dLLM: Training-free Acceleration via KV Cache + Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025.05 | Arxiv | NVIDIA; KV cache + parallel |
| [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295v1) | 2025.06 | Arxiv | >7B |
| [d^2Cache: Accelerating via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094) | 2025.09 | Arxiv | >7B |
| [Attention Is All You Need for KV Cache in dLLMs](https://arxiv.org/abs/2510.14973) | 2025.10 | Arxiv | >7B |
| [Attention Sinks in Diffusion Language Models](https://arxiv.org/abs/2510.15731) | 2025.10 | Arxiv | >7B |
| [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737v1) | 2025.12 | Arxiv | >7B, Causal cache |
| [Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding (COVER)](https://arxiv.org/abs/2602.06161) | 2026.02 | Arxiv | KV-override verification |
| [Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing](https://arxiv.org/abs/2602.02159) | 2026.02 | Arxiv | Long-context sparsity |
| [Mosaic: Unlocking Long-Context Inference for Diffusion LLMs via Global Memory Planning and Dynamic Peak Taming](https://arxiv.org/abs/2601.06562) | 2026.01 | Arxiv | Long-context memory |
| [Residual Context Diffusion Language Models](https://arxiv.org/abs/2601.22954) | 2026.01 | Arxiv | Recycle discarded tokens |

### 5.2 Decoding & Sampling
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://arxiv.org/abs/2408.05636) | 2024.08 | NAACL | <7B, Speculative Decoding |
| [Wide-In, Narrow-Out: Revokable Decoding for Effective dLLMs](https://arxiv.org/pdf/2507.18578) | 2025.07 | Arxiv | >7B |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding (APD)](https://arxiv.org/abs/2506.00413v1) | 2025.05 | NeurIPS | >7B |
| [DLM-One: Diffusion Language Models for One-Step Generation](https://arxiv.org/pdf/2506.00290) | 2025.06 | Arxiv | <7B |
| [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848v2) | 2025.06 | Arxiv | >7B |
| [Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037) | 2025.06 | Arxiv | >7B |
| [Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models](https://arxiv.org/abs/2508.00819) | 2025.08 | Arxiv | >7B |
| [DPad: Efficient Diffusion Language Models with Suffix Dropout](https://arxiv.org/abs/2508.14148v2) | 2025.08 | Arxiv | >7B |
| [Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](https://arxiv.org/abs/2509.15188v1) | 2025.09 | NeurIPS | >7B |
| [AdaBlock-dLLM: Semantic-Aware Inference via Adaptive Block Size](https://arxiv.org/pdf/2509.26432) | 2025.09 | Arxiv | >7B |
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | 2025.09 | Arxiv | >7B |
| [Learning to Parallel: Accelerating dLLMs via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188) | 2025.09 | Arxiv | >7B |
| [Spiffy: Multiplying Acceleration via Lossless Speculative Decoding](https://arxiv.org/pdf/2509.18085) | 2025.09 | Arxiv | >7B, Speculative Decoding |
| [DiffuSpec: Unlocking dLLMs for Speculative Decoding](https://www.arxiv.org/pdf/2510.02358) | 2025.09 | Arxiv | >7B, Speculative Decoding |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](https://arxiv.org/abs/2510.19304) | 2025.10 | Arxiv | >7B |
| [Saber: Efficient Sampling with Backtracking Enhanced Remasking](https://arxiv.org/abs/2510.18165) | 2025.10 | Arxiv | >7B |
| [CreditDecoding: Parallel Decoding with Trace Credits](https://arxiv.org/abs/2510.06133) | 2025.10 | Arxiv | >7B |
| [Accelerating dLLM Inference via Local Determinism Propagation](https://arxiv.org/abs/2510.07081) | 2025.10 | Arxiv | >7B |
| [Self Speculative Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2510.04147) | 2025.10 | Arxiv | >7B, Speculative Decoding |
| [SpecDiff-2: Scaling Diffusion Drafter Alignment](https://arxiv.org/abs/2511.00606) | 2025.11 | Arxiv | >7B, Speculative Decoding |
| [Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models](https://arxiv.org/abs/2511.21759v1) | 2025.11 | Arxiv | >7B |
| [Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models](https://arxiv.org/abs/2512.02044v1) | 2025.11 | Arxiv | >7B |
| [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](https://arxiv.org/abs/2512.20573) | 2025.12 | Arxiv | >7B, Speculative Decoding |
| [Fast-Decoding via Progress-Aware Confidence Schedules](https://arxiv.org/abs/2512.02892) | 2025.12 | Arxiv | >7B |
| [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586v1) | 2025.12 | Arxiv | >7B |
| [Context-Aware Initialization for Reducing Generative Path Length in Diffusion Language Models](https://arxiv.org/abs/2512.19004v1) | 2025.12 | Arxiv | >7B |
| [DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference](https://arxiv.org/abs/2601.19278) | 2026.01 | Arxiv | Speculative drafting |
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) | 2026.02 | Arxiv | Block + speculative |
| [Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models](https://arxiv.org/abs/2602.04399) | 2026.02 | Arxiv | Entropy-adaptive blocks |

### 5.3 Distillation, Quantization & Sparsity
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2410.21035) | 2024.10 | ICLR | <7B, Distillation |
| [Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558) | 2025.08 | Arxiv | >7B, Sparsity |
| [DLLMQuant: Quantizing Diffusion-based Large Language Models](https://arxiv.org/abs/2508.14090) | 2025.08 | Arxiv | >7B, Quantization |
| [Quantization Meets dLLMs: Post-training Quantization Study](https://arxiv.org/pdf/2508.14896) | 2025.08 | Arxiv | >7B, Quantization |
| [FS-DFM: Few-Step Diffusion Language Model](https://arxiv.org/abs/2509.20624) | 2025.09 | Arxiv | >7B |
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014) | 2025.09 | Arxiv | >7B, Sparsity |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025.09 | Arxiv | >7B, MoE |
| [Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](https://arxiv.org/abs/2509.25035v2) | 2025.10 | Arxiv | >7B, Distillation |
| [CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269) | 2025.11 | Arxiv | >7B, Consistency |

### 5.4 Inference Frameworks & Systems
> **New section**: production-grade frameworks and runtime engineering for dLLMs.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666v2) | 2025.10 | Arxiv | Modular framework, >1100 TPS |
| [JetEngine (SDAR)](https://github.com/JetAstra/SDAR) | 2025.10 | Repo | Lightweight engine for SDAR (3700+ TPS on H200) |
| [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298v1) | 2025.06 | Arxiv | Inception Labs commercial dLLM |
| [Seed Diffusion: Large-Scale dLLM with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025.08 | Arxiv | ByteDance code-focused dLLM |

---

## 6. Multi-modal & Physical AI

### 6.1 Multi-modal dLLMs
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Dual Diffusion for Unified Image Generation and Understanding](https://arxiv.org/pdf/2501.00289) | 2025.01 | Arxiv | Unified Task |
| [Unified Multimodal Discrete Diffusion (UniDisc)](https://arxiv.org/abs/2503.20853) | 2025.03 | Arxiv | Unified Diffusion |
| [LaViDa: A Large Diffusion LLM for Multimodal Understanding](https://arxiv.org/abs/2505.16839) | 2025.05 | NeurIPS Spotlight | Understanding |
| [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) | 2025.05 | NeurIPS | Native Multimodal |
| [Dimple: Discrete Diffusion Multimodal LLM with Parallel Decoding](https://arxiv.org/abs/2505.16990) | 2025.05 | Arxiv | Parallel Multimodal |
| [LLaDA-V: Diffusion LLMs with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) | 2025.06 | Arxiv | Visual Tuning |
| [Muddit: Liberating Generation Beyond Text-to-Image](https://arxiv.org/pdf/2505.23606) | 2025.05 | Arxiv | Multi-modal |
| [Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564) | 2025.06 | Arxiv | Unified Generation |
| [Diffuse Everything: Multimodal Diffusion on Arbitrary Spaces](https://www.arxiv.org/abs/2506.07903) | 2025.06 | ICML | Arbitrary Spaces |
| [TBAC-UniImage: Unified Understanding and Generation by Ladder-Side Diffusion Tuning](https://arxiv.org/abs/2508.08098) | 2025.08 | Arxiv | Tencent ladder-side tuning |
| [Lumina-DiMOO: Omni Diffusion LLM for Generation](https://arxiv.org/abs/2510.06308) | 2025.10 | Arxiv | Omni-generation |
| [MMaDA-Parallel: Thinking-Aware Editing and Generation](https://arxiv.org/pdf/2511.09611) | 2025.11 | Arxiv | Parallel Multimodal |
| [DiffusionVL: Translating AR Models into Diffusion VL Models](https://arxiv.org/abs/2512.15713) | 2025.12 | Arxiv | VL Adaptation |
| [SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding](https://arxiv.org/abs/2512.14068) | 2025.12 | Arxiv | Block-diffusion VL |
| [Dream-VL: Open Vision-Language Model with Diffusion Backbone](https://arxiv.org/abs/2512.22615) | 2025.12 | Arxiv | dVLM from Dream-7B |
| [LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models](https://arxiv.org/abs/2602.14147) | 2026.02 | Arxiv | Unified RL post-training |
| [Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion](https://arxiv.org/abs/2603.06577) | 2026.03 | Arxiv | Any-to-any (text/speech/image) |
| [LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation](https://arxiv.org/abs/2604.20796) | 2026.04 | Arxiv | SigLIP-VQ + block diffusion |
| [Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM](https://arxiv.org/abs/2604.06832) | 2026.04 | Arxiv | AR-VLM → block-diffusion VLM |
| [Analyzing Diffusion and Autoregressive VLMs in Multimodal Embedding Space](https://arxiv.org/abs/2602.06056) | 2026.02 | Arxiv | Embedding analysis |
| [Dynin-Omni: Omnimodal Unified Large Diffusion Language Model](https://arxiv.org/abs/2604.00007) | 2026.03 | Arxiv | Omnimodal |


### 6.2 Vision-Language-Action (VLA)
> **Scope note**: this section covers VLA models that *use a diffusion/masked-diffusion language model as the backbone* (dVLM-based VLA) **or** apply *discrete diffusion as the action-decoding mechanism* (not continuous diffusion action heads grafted onto an AR VLM). Pure continuous-diffusion-policy VLAs such as DiVLA (Wen et al., 2024), HybridVLA, and ProgressVLA are intentionally excluded because their language model is autoregressive — only the action head is diffusion-based.

**(a) dVLM-backbone VLA** — language backbone itself is a diffusion language model.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [LLaDA-VLA: Vision Language Diffusion Action Models](https://arxiv.org/abs/2509.06932) | 2025.06 | Arxiv | First LLaDA(d-VLM)-based VLA |
| [dVLA: Diffusion VLA with Multimodal Chain-of-Thought](https://arxiv.org/pdf/2509.25681) | 2025.09 | Arxiv | dLLM backbone + multimodal CoT |
| [Dream-VLA: Open Vision-Language-Action Model with Diffusion Backbone](https://arxiv.org/abs/2512.22615) | 2025.12 | Arxiv | dVLA from Dream-7B; first dLLM pretrained VLA |
| [MMaDA-VLA: Large Diffusion VLA with Unified Multi-Modal Instruction and Generation](https://arxiv.org/abs/2603.25406) | 2026.03 | Arxiv | Native discrete-diffusion VLA from MMaDA |

**(b) Discrete-diffusion action decoding** — language backbone may still be AR-VLM, but action chunks are decoded via discrete diffusion. Closely tied to dLLM literature for inference techniques.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Discrete Diffusion VLA: Action Decoding in VLA Policies](https://arxiv.org/abs/2508.20072) | 2025.08 | Arxiv | Unified-transformer + discrete-diffusion actions |
| [E0: Enhancing Generalization and Fine-Grained Control in VLA Models via Tweedie Discrete Diffusion](https://arxiv.org/abs/2511.21542) | 2025.11 | Arxiv | AR-VLM backbone + Tweedie discrete diffusion on action tokens |

### 6.3 Autonomous Driving / World Models
> **Scope note**: works that apply *discrete diffusion / masked-diffusion language modeling* to driving trajectories, action codebooks, or tokenized world states. Continuous trajectory-diffusion planners (e.g., classical Diffusion Policy applied to driving) are out of scope.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | 2023.11 | ICLR | Discrete diffusion on tokenized point-cloud world model |
| [ReflectDrive: Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving](https://arxiv.org/abs/2509.20109) | 2025.09 | Arxiv | dLLM finetuned on discretized 2D driving space |
| [Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion](https://arxiv.org/abs/2602.20577) | 2026.02 | Arxiv | Discrete action codebook + masked diffusion |

---

## 7. Agentic & Tool-Use dLLMs
> **New section** — emerging line: how dLLMs behave as agents (planning, multi-turn, tool calling). Critical for connecting dLLMs to robotics and physical-AI agent stacks.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check](https://arxiv.org/abs/2601.12979) | 2026.01 | Arxiv | Embodied + tool-call eval |
| [Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation](https://arxiv.org/abs/2601.07152) | 2026.01 | Arxiv | Multi-agent RL |
| [DLLM Agent: See Farther, Run Faster](https://arxiv.org/abs/2602.07451) | 2026.02 | Arxiv | dLLM-as-agent comparison |

---

## 8. Theory, Guidance & Applications

### 8.1 Theory & Analysis
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Can Diffusion Model Achieve Better Performance in Text Generation? Bridging the Gap between Training and Inference!](https://arxiv.org/abs/2305.04465) | 2023.05 | ACL Findings | <7B |
| [TEncDM: Understanding the Properties of the Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097) | 2024.02 | AAAI | <7B |
| [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://arxiv.org/abs/2409.02908) | 2024.10 | ICLR | <7B |
| [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622) | 2025.02 | NeurIPS | TER vs SER analysis |
| [Generalized Interpolating Discrete Diffusion (GIDD)](https://arxiv.org/abs/2503.04482) | 2025.03 | ICML | Noising |
| [Understanding the Quality-Diversity Trade-off in Diffusion Language Models](https://arxiv.org/abs/2503.10683v1) | 2025.03 | ICML | Quality-Diversity Trade-off |
| [Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](https://arxiv.org/abs/2505.22165v1) | 2025.05 | ACL | <7B |
| [The Diffusion Duality](https://arxiv.org/abs/2506.10892) | 2025.06 | ICML | <7B, Theoretical Duality |
| [Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior](https://arxiv.org/pdf/2507.07586) | 2025.07 | Arxiv | <7B |
| [Time Is a Feature: Exploiting Temporal Dynamics in dLLMs](https://arxiv.org/pdf/2508.09138v1) | 2025.08 | Arxiv | Temporal focus |
| [Diffusion LLMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) | 2025.08 | Arxiv | Semantic focus |
| [What Makes Diffusion Language Models Super Data Learners?](https://arxiv.org/pdf/2510.04071) | 2025.10 | Arxiv | Data efficiency |
| [Why mask diffusion does not work](https://arxiv.org/pdf/2510.03289) | 2025.10 | Arxiv | Failure analysis |
| [Empirical Analysis of Decoding Biases in Masked Diffusion Models](https://arxiv.org/abs/2508.13021v3) | 2025.10 | Arxiv | Decoding Bias |
| [Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models](https://arxiv.org/abs/2510.04146v1) | 2025.10 | Arxiv | Speed Analysis |
| [On the Role of Discreteness in Diffusion LLMs](https://arxiv.org/abs/2512.22630v1) | 2025.12 | Arxiv | Discreteness analysis |
| [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](https://arxiv.org/abs/2510.04767) | 2025.10 | ICLR | Benchmark |
| [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276) | 2025.11 | Arxiv | Data learner analysis |
| [Adaptation to Intrinsic Dependence in Diffusion Language Models](https://arxiv.org/abs/2602.20126) | 2026.02 | Arxiv | Distribution-agnostic schedule theory |
| [Confidence-Based Decoding is Provably Efficient for Diffusion Language Models](https://arxiv.org/abs/2603.22248) | 2026.03 | Arxiv | First theory of confidence-based decoding |

### 8.2 Guidance & Downstream Applications
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DiffusEmp: A Diffusion Model-Based Framework with Multi-Grained Control for Empathetic Response Generation](https://arxiv.org/abs/2306.01657) | 2023.06 | ACL | Dialogue |
| [DiffuDetox: A Mixed Diffusion Model for Text Detoxification](https://arxiv.org/abs/2306.08505) | 2023.06 | ACL Findings | Detoxification |
| [PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation](https://arxiv.org/abs/2306.08456) | 2023.06 | AAAI | Poetry Generation |
| [ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer](https://arxiv.org/abs/2308.15459) | 2023.08 | AAAI | Text Style Transfer |
| [P^3SUM: Preserving Author's Perspective in News Summarization with Diffusion Language Models](https://arxiv.org/abs/2311.09741) | 2023.11 | NAACL | Summarization |
| [DiffuCOMET: Contextual Commonsense Knowledge Diffusion](https://arxiv.org/abs/2402.17011) | 2024.02 | ACL | Commonsense |
| [DiffusionDialog: A Diffusion Model for Diverse Dialog Generation with Latent Space](https://arxiv.org/abs/2404.06760) | 2024.04 | LREC-COLING | Dialogue |
| [Diffusion Guided Language Modeling](https://arxiv.org/abs/2408.04220) | 2024.08 | ACL Findings | Control |
| [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](https://arxiv.org/abs/2411.03250) | 2024.11 | ACL Findings | Data Synthesis |
| [Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](https://arxiv.org/abs/2412.11333) | 2024.12 | ACL | Text Segmentation |
| [EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](https://arxiv.org/abs/2502.19765) | 2025.02 | ACL | Text Editing |
| [Constrained Discrete Diffusion](https://arxiv.org/abs/2503.09790) | 2025.03 | NeurIPS | Constraint |
| [Planning with Diffusion Models for Target-Oriented Dialogue Systems](https://arxiv.org/abs/2504.16858) | 2025.04 | ACL | Dialogue |
| [CtrlDiff: Boosting dLLMs with Dynamic Block Prediction](https://arxiv.org/abs/2505.14455) | 2025.05 | Arxiv | Control |
| [Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective](https://arxiv.org/abs/2505.15045) | 2025.05 | Arxiv | Embedding |
| [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) | 2025.05 | Arxiv | Constrained Decoding |
| [Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design](https://arxiv.org/abs/2505.22524) | 2025.05 | ICLR | SMC test-time scaling |
| [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298v1) | 2025.06 | Arxiv | Code |
| [DiffuCoder: Improving Masked Diffusion for Code Generation](https://arxiv.org/abs/2506.20639) | 2025.06 | Arxiv | Code |
| [Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](https://arxiv.org/abs/2507.04504v1) | 2025.07 | Arxiv | Control |
| [Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement](https://arxiv.org/abs/2507.19081v1) | 2025.07 | Arxiv | Summarization |
| [Improving Text Style Transfer using Masked Diffusion Language Models with Inference-time Scaling](https://arxiv.org/abs/2508.10995v2) | 2025.08 | Arxiv | Text Style Transfer |
| [Seed Diffusion: Large-Scale dLLM with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025.08 | Arxiv | Code |
| [TreeDiff: AST-Guided Code Generation with Diffusion LLMs](https://arxiv.org/abs/2508.01473) | 2025.08 | Arxiv | Code (syntax-aware) |
| [Beyond Autoregression: Empirical Study for Code Generation](https://arxiv.org/abs/2509.11252) | 2025.09 | Arxiv | Code |
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2024.10 | Arxiv | Control |
| [Syntax-Guided Diffusion Language Models with User-Integrated Personalization](https://arxiv.org/abs/2510.01028v1) | 2025.10 | Arxiv | Personalization |
| [TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models](https://arxiv.org/abs/2510.01274v1) | 2025.10 | Arxiv | Hallucination |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025.10 | NeurIPS | Control |
| [Diffusion Language Models for Speech Recognition](https://arxiv.org/abs/2604.14001) | 2026.04 | Arxiv | ASR rescoring (MDLM/USDM) |
| [CAGenMol: Condition-Aware Diffusion Language Model for Goal-Directed Molecular Generation](https://arxiv.org/abs/2604.11483) | 2026.04 | Arxiv | Molecular generation |

---

## 9. Seminal Diffusion Papers

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) | 2015.03 | ICML | Formulation |
| [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) | 2020.06 | NeurIPS | - |
| [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) | 2020.10 | ICLR | - |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2020.11 | ICLR | - |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 2021.05 | NeurIPS | CG |
| [Structured Denoising Diffusion in Discrete State-Spaces (D3PM)](https://arxiv.org/abs/2107.03006) | 2021.07 | NeurIPS | Discrete |
| [Vector Quantized Diffusion Model (VQ-Diffusion)](https://arxiv.org/abs/2111.14822) | 2021.11 | CVPR | VQ |
| [High-Resolution Image Synthesis with Latent Diffusion (LDM)](https://arxiv.org/abs/2112.10752) | 2021.12 | CVPR | - |
| [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512) | 2022.02 | ICLR | Distillation |
| [DPM-Solver: Fast ODE Solver for Sampling](https://arxiv.org/abs/2206.00927) | 2022.06 | NeurIPS | - |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2022.07 | NeurIPS | CFG |
| [Analog Bits: Generating Discrete Data using Diffusion](https://arxiv.org/abs/2208.04202) | 2022.08 | ICLR | Self-conditioning |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | 2022.12 | ICCV | Scalable focus |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023.03 | ICML | - |

---

## 🤝 Contact
* Maintainers: jake630@snu.ac.kr / wjk9904@snu.ac.kr 
* Contributions via Pull Requests are welcome!
