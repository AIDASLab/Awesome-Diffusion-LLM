# Awesome-Large-Language-Diffusion-Models

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Maintained](https://img.shields.io/badge/Maintained-2025-blue?style=flat-square)

A comprehensive and structured list of research papers about **Large-Language-Diffusion-Models (dLLMs)**.

---

## ⚙️ Framework (Taxonomy)

1. [**Surveys & Useful Resources**](#1-surveys--useful-resources)
2. [**Core Methodologies**](#2-core-methodologies)
    - [Discrete & Masked Diffusion](#21-discrete--masked-diffusion)
    - [Continuous & Latent Space Diffusion](#22-continuous--latent-space-diffusion)
    - [AR-to-Diffusion Adaptation](#23-ar-to-diffusion-adaptation)
3. [**Reasoning & Policy Optimization**](#3-reasoning--policy-optimization)
    - [Reasoning & Planning](#31-reasoning--planning)
    - [Alignment & RL](#32-alignment--reinforcement-learning)
4. [**Token Ordering**](#4-token-ordering)
5. [**System Efficiency & Acceleration**](#5-system-efficiency--acceleration)
    - [Caching & Memory Strategy](#51-caching--memory-strategy)
    - [Decoding & Sampling](#52-decoding--sampling)
    - [Distillation, Quantization & Sparsity](#53-distillation-quantization--sparsity)
6. [**Multi-modal & Physical AI**](#6-multi-modal--physical-ai)
    - [Multi-modal dLLMs](#61-multi-modal-dllms)
    - [Vision-Language-Action (VLA)](#62-vision-language-action-vla)
7. [**Theory, Guidance & Applications**](#7-theory-guidance--applications)
8. [**Seminal Diffusion Papers**](#8-seminal-diffusion-papers)
---

## 1. Surveys & Useful Resources

### 📚 Blogs & Reports
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Dream-7B](https://hkunlp.github.io/blog/2025/dream/)
- [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/)
- [What are Diffusion Language Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generative Modeling by Estimating Gradients](https://yang-song.net/blog/2021/score/)

### 📝 Survey Papers
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759) | 2025.06 | Arxiv | - |
| [Diffusion-based Large Language Models Survey](https://www.researchgate.net/profile/Junhao-Song-3/publication/394262235) | 2025.08 | Arxiv | - |
| [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/pdf/2508.08712v2) | 2025.08 | Arxiv | - |

---

## 2. Core Methodologies

### 2.1 Discrete & Masked Diffusion
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DiffusionBERT: Improving Generative Masked Language Models](https://aclanthology.org/2023.acl-long.248.pdf) | 2022.11 | ACL | <7B, Masked |
| [DiffusER: Discrete Diffusion via Edit-based Reconstruction](https://arxiv.org/abs/2210.16886) | 2022.10 | ICLR | <7B |
| [SSD-LM: Semi-autoregressive Simplex-based Diffusion for Modular Control](https://aclanthology.org/2023.acl-long.647.pdf) | 2022.10 | ACL | <7B, Simplex |
| [A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/abs/2302.05737) | 2023.02 | COLM | <7B |
| [David helps Goliath: Inference-Time Collaboration Between Small and Large Diffusion LMs](https://arxiv.org/abs/2305.14771) | 2023.05 | NAACL | >7B, Scale-collaboration |
| [TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379) | 2023.05 | EACL | <7B, Simplex |
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | 2023.08 | Arxiv | >7B, Scaling |
| [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://proceedings.mlr.press/v235/lou24a.html) | 2023.10 | ICML | <7B, Discrete |
| [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/pdf/2406.04329) | 2024.06 | NeurIPS | - |
| [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) | 2024.06 | NeurIPS | <7B, Masked |
| [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736) | 2024.06 | ICLR | <7B, Masked |
| [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) | 2024.10 | ICLR | <7B, 1.1B Scaling |
| [Energy-Based Diffusion Language Models for Text Generation](https://arxiv.org/abs/2410.21357) | 2024.10 | ICLR | <7B, EDLM |
| [Conditional MASK Discrete Diffusion Language Model](https://arxiv.org/abs/2411.06438) | 2024.11 | EMNLP | <7B |
| [Non-Markovian Discrete Diffusion with Causal Language Models](https://arxiv.org/abs/2502.09767v1) | 2025.02 | NeurIPS | <7B |
| [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) | 2025.02 | NeurIPS | >7B, LLaDA-8B |
| [Anchored Diffusion Language Model](https://arxiv.org/abs/2505.18456) | 2025.05 | NeurIPS | >7B |
| [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429v2) | 2025.06 | Arxiv | >7B, Context Scaling |
| [Esoteric Language Models](https://arxiv.org/pdf/2506.01928) | 2025.06 | Arxiv | - |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487v1) | 2025.08 | Arxiv | >7B |
| [Sequential Diffusion Language Models](https://arxiv.org/abs/2509.24007v1) | 2025.09 | Arxiv | >7B |
| [UltraLLaDA: Scaling Context to 128K](https://arxiv.org/abs/2510.10481) | 2025.10 | Arxiv | >7B, Context Scaling |
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632) | 2025.10 | Arxiv | - |
| [Masked Diffusion Models as Energy Minimization](https://arxiv.org/abs/2509.13866v1) | 2025.10 | NeurIPS | <7B |
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632v1) | 2025.10 | NeurIPS | <7B |
| [Soft-Masked Diffusion Language Models](https://arxiv.org/abs/2510.17206v1) | 2025.10 | Arxiv | <7B |
| [Variational Masked Diffusion Models](https://arxiv.org/abs/2510.23606v1) | 2025.10 | Arxiv | <7B |
| [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923v1) | 2025.11 | Arxiv | >7B |
| [C2DLM: Causal Concept-Guided Diffusion Large Language Models](https://arxiv.org/abs/2511.22146v1) | 2025.11 | Arxiv | >7B |

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
| [Likelihood-Based Diffusion Language Models](https://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html) | 2023.05 | NeurIPS | <7B, Plaid1B |
| [PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model](https://arxiv.org/abs/2306.02531) | 2023.06 | NeurIPS | <7B, Latent |
| [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/pdf/2506.09018) | 2025.06 | Arxiv | - |


### 2.3 AR-to-Diffusion Adaptation
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://openreview.net/forum?id=j1tSLYKwg8) | 2024.10 | ICLR | >7B, GPT2/LLaMA2 Adaptation |
| [Large Language Models to Diffusion Finetuning](https://arxiv.org/abs/2501.15781) | 2025.01 | ICML | >7B |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) | 2025.02 | ACL | >7B, Adapted from Mistral |
| [SDAR: A Synergistic Diffusion-AutoRegression Paradigm](https://arxiv.org/abs/2510.06303) | 2025.10 | Arxiv | >7B, Synergistic Training |
| [From Next-Token to Next-Block: Principled Adaptation Path](https://arxiv.org/abs/2512.06776) | 2025.11 | Arxiv | >7B, Adaptation Path |
| [Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](https://arxiv.org/abs/2512.14067v1) | 2025.12 | Arxiv | >7B |
| [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745v1) | 2025.12 | Arxiv | >7B |

---

## 3. Reasoning & Policy Optimization

### 3.1 Reasoning & Planning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion of Thought: Chain-of-Thought Reasoning in dLLMs](https://arxiv.org/abs/2402.07754) | 2024.02 | NeurIPS | <7B, CoT Foundation |
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2024.10 | Arxiv | Planning |
| [d1: Scaling Reasoning in dLLMs via RL](https://arxiv.org/abs/2504.12216) | 2025.04 | NeurIPS | >7B, Reasoning scaling |
| [Reinforcing the Diffusion Chain of Lateral Thought](https://arxiv.org/abs/2505.10446) | 2025.05 | NeurIPS | >7B |
| [Thinking Inside the Mask: In-Place Prompting in dLLMs](https://arxiv.org/pdf/2508.10736) | 2025.08 | Arxiv | >7B |
| [Reinforced Context Order Recovery for Adaptive Reasoning](https://arxiv.org/pdf/2508.13070) | 2025.08 | Arxiv | <7B, Planning |
| [d2: Improved Techniques for Training Reasoning dLLMs](https://www.arxiv.org/abs/2509.21474) | 2025.09 | Arxiv | >7B |
| [Beyond Surface Reasoning: Unveiling Long CoT Capacity](https://arxiv.org/abs/2510.09544) | 2025.10 | Arxiv | >7B |
| [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573) | 2025.10 | Arxiv | >7B |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/pdf/2410.14157) | 2024.10 | ICLR | <7B |
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
| [Principled and Tractable RL for Reasoning with dLLMs](https://arxiv.org/pdf/2510.04019) | 2025.10 | Arxiv | >7B |
| [Improving Reasoning via Group Diffusion Policy Optimization](https://arxiv.org/pdf/2510.08554) | 2025.10 | Arxiv | >7B |
| [Step-Aware Policy Optimization for Reasoning](https://arxiv.org/abs/2510.01544) | 2025.10 | Arxiv | >7B |
| [MRO: Enhancing Reasoning via Multi-Reward Optimization](https://arxiv.org/abs/2510.21473) | 2025.10 | Arxiv | >7B |
| [Enhancing Reasoning via Distribution Matching Policy Optimization](https://arxiv.org/abs/2510.08233) | 2025.10 | Arxiv | >7B |
| [Boundary-Guided Policy Optimization for Memory-efficient RL](https://arxiv.org/abs/2510.11683) | 2025.10 | Arxiv | >7B |
| [SPG: Sandwiched Policy Gradient for Masked Diffusion](https://arxiv.org/abs/2510.09541) | 2025.10 | Arxiv | >7B |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025.10 | Arxiv | >7B |
| [Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052v2) | 2025.10 | Arxiv | >7B |
| [Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective](https://arxiv.org/abs/2512.03759v1) | 2025.12 | Arxiv | >7B |
| [d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2512.09675v1) | 2025.12 | Arxiv | >7B |

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
| [Diffusion Language Model Inference with Monte Carlo Tree Search](https://arxiv.org/abs/2512.12168v1) | 2025.12 | Arxiv | >7B, Unmasking |
| [Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty](https://arxiv.org/abs/2512.21336v1) | 2025.12 | Arxiv | >7B, Unmasking |

---

## 5. System Efficiency & Acceleration

### 5.1 Caching & Memory Strategy
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025.05 | NeurIPS | >7B |
| [FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467) | 2025.05 | Arxiv | >7B |
| [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295v1) | 2025.06 | Arxiv | >7B |
| [d^2Cache: Accelerating via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094) | 2025.09 | Arxiv | >7B |
| [Attention Is All You Need for KV Cache in dLLMs](https://arxiv.org/abs/2510.14973) | 2025.10 | Arxiv | >7B |
| [Attention Sinks in Diffusion Language Models](https://arxiv.org/abs/2510.15731) | 2025.10 | Arxiv | >7B |
| [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666v2) | 2025.10 | Arxiv | >7B |
| [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737v1) | 2025.12 | Arxiv | >7B |

### 5.2 Decoding & Sampling
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Wide-In, Narrow-Out: Revokable Decoding for Effective dLLMs](https://arxiv.org/pdf/2507.18578?) | 2025.07 | Arxiv | >7B |
| [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://arxiv.org/abs/2408.05636) | 2024.08 | NAACL | <7B, Speculative Decoding |
| [Fast-dLLM: Training-free Acceleration via Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025.05 | Arxiv | >7B, Parallel Decoding |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413v1) | 2025.05 | NeurIPS | >7B |
| [DLM-One: Diffusion Language Models for One-Step Generation](https://arxiv.org/pdf/2506.00290) | 2025.06 | Arxiv | <7B |
| [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848v2) | 2025.06 | Arxiv | >7B |
| [Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037) | 2025.06 | Arxiv | >7B |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](https://arxiv.org/abs/2510.19304) | 2025.10 | Arxiv | >7B |
| [Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models](https://arxiv.org/abs/2508.00819) | 2025.08 | Arxiv | >7B |
| [DPad: Efficient Diffusion Language Models with Suffix Dropout](https://arxiv.org/abs/2508.14148v2) | 2025.08 | Arxiv | >7B |
| [Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding](https://arxiv.org/abs/2508.19529v1) | 2025.08 | Arxiv | >7B |
| [Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](https://arxiv.org/abs/2509.15188v1) | 2025.09 | NeurIPS | >7B |
| [AdaBlock-dLLM: Semantic-Aware Inference via Adaptive Block Size](https://arxiv.org/pdf/2509.26432) | 2025.09 | Arxiv | >7B |
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | 2025.09 | Arxiv | >7B |
| [Learning to Parallel: Accelerating dLLMs via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188) | 2025.09 | Arxiv | >7B |
| [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328) | 2025.09 | Arxiv | >7B, Block Decoding |
| [Spiffy: Multiplying Acceleration via Lossless Speculative Decoding](https://arxiv.org/pdf/2509.18085) | 2025.09 | Arxiv | >7B, Speculative Decoding |
| [DiffuSpec: Unlocking dLLMs for Speculative Decoding](https://www.arxiv.org/pdf/2510.02358) | 2025.09 | Arxiv | >7B, Speculative Decoding |
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


### 5.3 Distillation, Quantization & Sparsity
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Beyond Autoregression: Fast LLMs via Self-Distillation](https://arxiv.org/abs/2410.21035) | 2024.10 | ICLR | <7B, Distillation |
| [Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558) | 2025.08 | Arxiv | >7B, Sparsity |
| [DLLMQuant: Quantizing Diffusion-based Large Language Models](https://arxiv.org/abs/2508.14090) | 2025.08 | Arxiv | >7B, Quantization |
| [Quantization Meets dLLMs: Post-training Quantization Study](https://arxiv.org/pdf/2508.14896) | 2025.08 | Arxiv | >7B, Quantization |
| [FS-DFM: Few-Step Diffusion Language Model](https://arxiv.org/abs/2509.20624) | 2025.09 | Arxiv | >7B |
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014) | 2025.09 | Arxiv | >7B, Sparsity |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025.09 | Arxiv | >7B, MoE |
| [Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](https://arxiv.org/abs/2509.25035v2) | 2025.10 | Arxiv | >7B, Distillation |
| [CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269) | 2025.11 | Arxiv | >7B, Consistency |

---

## 6. Multi-modal & Physical AI

### 6.1 Multi-modal dLLMs
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Dual Diffusion for Unified Image Generation and Understanding](https://arxiv.org/pdf/2501.00289) | 2025.01 | Arxiv | Unified Task |
| [Unified Multimodal Discrete Diffusion](https://arxiv.org/abs/2503.20853) | 2025.03 | Arxiv | Unified Diffusion |
| [LaViDa: A Large Diffusion LLM for Multimodal Understanding](https://arxiv.org/abs/2505.16839) | 2025.05 | Arxiv | Understanding |
| [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) | 2025.05 | NeurIPS | Native Multimodal |
| [Dimple: Discrete Diffusion Multimodal LLM with Parallel Decoding](https://arxiv.org/abs/2505.16990) | 2025.05 | Arxiv | Parallel Multimodal |
| [Muddit: Liberating Generation Beyond Text-to-Image](https://arxiv.org/pdf/2505.23606) | 2025.05 | Arxiv | Multi-modal |
| [Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564) | 2025.06 | Arxiv | Unified Generation |
| [Diffuse Everything: Multimodal Diffusion on Arbitrary Spaces](https://www.arxiv.org/abs/2506.07903) | 2025.06 | ICML | Arbitrary Spaces |
| [LLaDA-V: Diffusion LLMs with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) | 2025.06 | Arxiv | Visual Tuning |
| [Lumina-DiMOO: Omni Diffusion LLM for Generation](https://arxiv.org/abs/2510.06308) | 2025.10 | Arxiv | Omni-generation |
| [MMaDA-Parallel: Thinking-Aware Editing and Generation](https://arxiv.org/pdf/2511.09611) | 2025.11 | Arxiv | Parallel Multimodal |
| [DiffusionVL: Translating AR Models into Diffusion VL Models](https://arxiv.org/abs/2512.15713) | 2025.12 | Arxiv | VL Adaptation |



### 6.2 Vision-Language-Action (VLA)
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [LLaDA-VLA: Vision Language Diffusion Action Models](https://arxiv.org/abs/2509.06932) | 2025.06 | Arxiv | VLA Framework |
| [Discrete Diffusion VLA: Action Decoding in VLA Policies](https://arxiv.org/abs/2508.20072) | 2025.08 | Arxiv | VLA Action Decoding |
| [dVLA: Diffusion VLA with Multimodal Chain-of-Thought](https://arxiv.org/pdf/2509.25681) | 2025.09 | Arxiv | VLA Reasoning |

---

## 7. Theory, Guidance & Applications

### 7.1 Theory & Analysis
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Can Diffusion Model Achieve Better Performance in Text Generation? Bridging the Gap between Training and Inference!](https://arxiv.org/abs/2305.04465) | 2023.05 | ACL Findings | <7B |
| [TEncDM: Understanding the Properties of the Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097) | 2024.02 | AAAI | <7B |
| [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://arxiv.org/abs/2409.02908) | 2024.10 | ICLR | <7B |
| [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622) | 2025.02 | NeurIPS | Limits analysis |
| [Generalized Interpolating Discrete Diffusion](https://arxiv.org/abs/2503.04482) | 2025.03 | ICML | Noising |
| [Understanding the Quality-Diversity Trade-off in Diffusion Language Models](https://arxiv.org/abs/2503.10683v1) | 2025.03 | ICML | Quality-Diversity Trade-off |
| [Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](https://arxiv.org/abs/2505.22165v1) | 2025.05 | ACL | <7B |
| [The Diffusion Duality](https://arxiv.org/abs/2506.10892) | 2025.06 | ICML | <7B, Theoretical Duality |
| [Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior](https://arxiv.org/pdf/2507.07586) | 2025.07 | ArXiv | <7B |
| [Time Is a Feature: Exploiting Temporal Dynamics in dLLMs](https://arxiv.org/pdf/2508.09138v1) | 2025.08 | Arxiv | Temporal focus |
| [Diffusion LLMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) | 2025.08 | Arxiv | Semantic focus |
| [What Makes Diffusion Language Models Super Data Learners?](https://arxiv.org/pdf/2510.04071) | 2025.10 | Arxiv | Data efficiency |
| [Why mask diffusion does not work](https://arxiv.org/pdf/2510.03289) | 2025.10 | Arxiv | Failure analysis |
| [Empirical Analysis of Decoding Biases in Masked Diffusion Models](https://arxiv.org/abs/2508.13021v3) | 2025.10 | Arxiv | Decoding Bias |
| [Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models](https://arxiv.org/abs/2510.04146v1) | 2025.10 | Arxiv | Speed Analysis |
| [On the Role of Discreteness in Diffusion LLMs](https://arxiv.org/abs/2512.22630v1) | 2025.12 | Arxiv | Speed Analysis |

### 7.2 Guidance & Downstream Applications
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
| [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298v1) | 2025.06 | Arxiv | Code |
| [DiffuCoder: Improving Masked Diffusion for Code Generation](https://arxiv.org/abs/2506.20639) | 2025.06 | Arxiv | Code |
| [Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](https://arxiv.org/abs/2507.04504v1) | 2025.07 | Arxiv | Control |
| [Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement](https://arxiv.org/abs/2507.19081v1) | 2025.07 | Arxiv | Summarization |
| [Improving Text Style Transfer using Masked Diffusion Language Models with Inference-time Scaling](https://arxiv.org/abs/2508.10995v2) | 2025.08 | Arxiv | Text Style Transfer |
| [Seed Diffusion: Large-Scale dLLM with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025.08 | Arxiv | Code |
| [Beyond Autoregression: Empirical Study for Code Generation](https://arxiv.org/abs/2509.11252) | 2025.09 | Arxiv | Code |
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2024.10 | Arxiv | Control |
| [Syntax-Guided Diffusion Language Models with User-Integrated Personalization](https://arxiv.org/abs/2510.01028v1) | 2025.10 | Arxiv | Personalization |
| [TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models](https://arxiv.org/abs/2510.01274v1) | 2025.10 | Arxiv | Hallucination |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025.10 | NeurIPS | Control|

---

## 8. Seminal Diffusion Papers

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) | 2015.03 | ICML | Formulation |
| [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) | 2020.06 | NeurIPS | - |
| [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) | 2020.10 | ICLR | - |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2020.11 | ICLR | - |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 2021.05 | NeurIPS | CG |
| [Structured Denoising Diffusion in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) | 2021.07 | NeurIPS | Discrete |
| [Vector Quantized Diffusion Model (VQ-Diffusion)](https://arxiv.org/abs/2111.14822) | 2021.11 | CVPR | VQ |
| [High-Resolution Image Synthesis with Latent Diffusion](https://arxiv.org/abs/2112.10752) | 2021.12 | CVPR | - |
| [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512) | 2022.02 | ICLR | Distillation |
| [DPM-Solver: Fast ODE Solver for Sampling](https://arxiv.org/abs/2206.00927) | 2022.06 | NeurIPS | - |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2022.07 | NeurIPS | CFG |
| [Analog Bits: Generating Discrete Data using Diffusion](https://arxiv.org/abs/2208.04202) | 2022.08 | ICLR | Self-conditioning |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | 2022.12 | ICCV | Scalable focus |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023.03 | ICML | - |

---

## 🤝 Contact
* Maintainers: jake630@snu.ac.kr / wjk9904@snu.ac.kr / qicher@snu.ac.kr
* Contributions via Pull Requests are welcome!
