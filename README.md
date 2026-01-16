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
| [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759) | 2025 | Arxiv | - |
| [Diffusion-based Large Language Models Survey](https://www.researchgate.net/profile/Junhao-Song-3/publication/394262235) | 2025 | Arxiv | - |
| [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/pdf/2508.08712v2) | 2025 | Arxiv | - |

---

## 2. Core Methodologies

### 2.1 Discrete & Masked Diffusion
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) | 2025 | Arxiv | >7B, LLaDA-8B |
| [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) | 2024 | ICLR | <7B, 1.1B Scaling |
| [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) | 2024 | NeurIPS | <7B, Masked |
| [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/pdf/2406.04329) | 2024 | NeurIPS | - |
| [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://proceedings.mlr.press/v235/lou24a.html) | 2024 | ICML | <7B, Discrete |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487v1) | 2025 | Arxiv | >7B |
| [UltraLLaDA: Scaling Context to 128K](https://arxiv.org/abs/2510.10481) | 2025 | Arxiv | >7B, Context Scaling |
| [Esoteric Language Models](https://arxiv.org/pdf/2506.01928) | 2025 | Arxiv | - |
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632) | 2025 | Arxiv | - |
| [DiffusionBERT: Improving Generative Masked Language Models](https://aclanthology.org/2023.acl-long.248.pdf) | 2023 | ACL | <7B, Masked |
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | 2023 | Arxiv | >7B, Scaling |
| [David helps Goliath: Inference-Time Collaboration Between Small and Large Diffusion LMs](https://arxiv.org/abs/2305.14771) | 2023 | NAACL | >7B, Scale-collaboration |

### 2.2 Continuous & Latent Space Diffusion
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) | 2022 | NeurIPS | <7B, Embedding |
| [DiffuSeq: Sequence to Sequence Text Generation](https://arxiv.org/abs/2210.08933) | 2023 | ICLR | <7B, Embedding |
| [Latent Diffusion for Language Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html) | 2023 | NeurIPS | <7B, Latent |
| [Likelihood-Based Diffusion Language Models](https://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html) | 2023 | NeurIPS | <7B, Plaid1B |
| [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/pdf/2506.09018) | 2025 | Arxiv | - |

### 2.3 AR-to-Diffusion Adaptation
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [SDAR: A Synergistic Diffusion-AutoRegression Paradigm](https://arxiv.org/abs/2510.06303) | 2025 | Arxiv | >7B, Synergistic Training |
| [From Next-Token to Next-Block: Principled Adaptation Path](https://arxiv.org/abs/2512.06776) | 2025 | Arxiv | >7B, Adaptation Path |
| [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://openreview.net/forum?id=j1tSLYKwg8) | 2025 | ICLR | >7B, GPT2/LLaMA2 Adaptation |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) | 2025 | ACL | >7B, Adapted from Mistral |
| [Large Language Models to Diffusion Finetuning](https://arxiv.org/abs/2501.15781) | 2025 | Arxiv | >7B |

---

## 3. Reasoning & Policy Optimization

### 3.1 Reasoning & Planning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [d1: Scaling Reasoning in dLLMs via RL](https://arxiv.org/abs/2504.12216) | 2025 | Arxiv | >7B, Reasoning scaling |
| [d2: Improved Techniques for Training Reasoning dLLMs](https://www.arxiv.org/abs/2509.21474) | 2025 | Arxiv | >7B |
| [Diffusion of Thought: Chain-of-Thought Reasoning in dLLMs](https://arxiv.org/abs/2402.07754) | 2024 | NeurIPS | >7B, CoT Foundation |
| [Thinking Inside the Mask: In-Place Prompting in dLLMs](https://arxiv.org/pdf/2508.10736) | 2025 | Arxiv | >7B |
| [Beyond Surface Reasoning: Unveiling Long CoT Capacity](https://arxiv.org/abs/2510.09544) | 2025 | Arxiv | >7B |
| [Reinforcing the Diffusion Chain of Lateral Thought](https://arxiv.org/abs/2505.10446) | 2025 | Arxiv | >7B |
| [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573) | 2025 | Arxiv | >7B |
| [Reinforced Context Order Recovery for Adaptive Reasoning](https://arxiv.org/pdf/2508.13070) | 2025 | Arxiv | <7B, Planning |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/pdf/2410.14157) | 2025 | ICLR | <7B |

### 3.2 Alignment & Reinforcement Learning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DiFFPO: Training dLLMs to Reason Fast and Furious via RL](https://arxiv.org/pdf/2510.02212) | 2025 | Arxiv | >7B, Direct Preference |
| [LLaDA 1.5: Variance-Reduced Preference Optimization](https://arxiv.org/abs/2505.19223) | 2025 | Arxiv | >7B |
| [MDPO: Overcoming the Training-Inference Divide](https://arxiv.org/abs/2508.13148) | 2025 | Arxiv | >7B |
| [wd1: Weighted Policy Optimization for Reasoning](https://arxiv.org/pdf/2507.08838) | 2025 | Arxiv | >7B |
| [Principled and Tractable RL for Reasoning with dLLMs](https://arxiv.org/pdf/2510.04019) | 2025 | Arxiv | >7B |
| [Improving Reasoning via Group Diffusion Policy Optimization](https://arxiv.org/pdf/2510.08554) | 2025 | Arxiv | >7B |
| [Step-Aware Policy Optimization for Reasoning](https://arxiv.org/abs/2510.01544) | 2025 | Arxiv | >7B |
| [Inpainting-Guided Policy Optimization for dLLMs](https://arxiv.org/abs/2509.10396) | 2025 | Arxiv | >7B |
| [MRO: Enhancing Reasoning via Multi-Reward Optimization](https://arxiv.org/abs/2510.21473) | 2025 | Arxiv | >7B |
| [Enhancing Reasoning via Distribution Matching Policy Optimization](https://arxiv.org/abs/2510.08233) | 2025 | Arxiv | >7B |
| [Boundary-Guided Policy Optimization for Memory-efficient RL](https://arxiv.org/abs/2510.11683) | 2025 | Arxiv | >7B |
| [Taming Masked Diffusion via Consistency Trajectory RL](https://arxiv.org/abs/2509.23924) | 2025 | Arxiv | >7B |
| [SPG: Sandwiched Policy Gradient for Masked Diffusion](https://arxiv.org/abs/2510.09541) | 2025 | Arxiv | >7B |
| [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning](https://arxiv.org/abs/2509.25171) | 2025 | Arxiv | >7B |
| [Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295) | 2025 | Arxiv | >7B |
| [Revolutionizing RL Framework for Diffusion Large Language Models](https://arxiv.org/pdf/2509.06949) | 2025 | Arxiv | >7B |
| [Improving Discrete Diffusion Unmasking Policies Beyond Reference Policies](https://arxiv.org/abs/2510.05725) | 2025 | Arxiv | >7B |
| [Coevolutionary Continuous Discrete Diffusion: Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025 | Arxiv | >7B |

---

## 4. Token Ordering

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Train for the Worst, Plan for the Best: Understanding Token Ordering](https://arxiv.org/pdf/2502.06768) | 2025 | ICML | <7B, Ordering Analysis |
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion LMs](https://arxiv.org/abs/2503.09573) | 2025 | ICLR | <7B, Interpolation |
| [SSD-LM: Semi-autoregressive Simplex-based Diffusion for Modular Control](https://aclanthology.org/2023.acl-long.647.pdf) | 2023 | ACL | <7B, Blockwise |
| [AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation](https://arxiv.org/abs/2305.09515) | 2023 | NeurIPS | <7B, AR-like noise |
| [Any-Order Flexible Length Masked Diffusion](https://arxiv.org/pdf/2509.01025) | 2025 | Arxiv | <7B, Order Flexibility |
| [Review, Remask, Refine (R3): Process-Guided Block Diffusion](https://arxiv.org/pdf/2507.08018v1) | 2025 | ICML | >7B, Block-wise |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025 | NeurIPS | <7B, Edit preservation |

---

## 5. System Efficiency & Acceleration

### 5.1 Caching & Memory Strategy
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025 | Arxiv | >7B |
| [d^2Cache: Accelerating via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094) | 2025 | Arxiv | >7B |
| [Accelerating dLLM Inference via Efficient KV Caching](https://arxiv.org/pdf/2505.21467) | 2025 | Arxiv | >7B |
| [Attention Is All You Need for KV Cache in dLLMs](https://arxiv.org/abs/2510.14973) | 2025 | Arxiv | >7B |
| [Attention Sinks in Diffusion Language Models](https://arxiv.org/abs/2510.15731) | 2025 | Arxiv | >7B |

### 5.2 Decoding & Sampling
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Fast-dLLM: Training-free Acceleration via Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025 | Arxiv | >7B, Parallel Decoding |
| [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328) | 2025 | Arxiv | >7B, Block Decoding |
| [Spiffy: Multiplying Acceleration via Lossless Speculative Decoding](https://arxiv.org/pdf/2509.18085) | 2025 | Arxiv | >7B |
| [DiffuSpec: Unlocking dLLMs for Speculative Decoding](https://www.arxiv.org/pdf/2510.02358) | 2025 | Arxiv | >7B |
| [Saber: Efficient Sampling with Backtracking Enhanced Remasking](https://arxiv.org/abs/2510.18165) | 2025 | Arxiv | >7B |
| [CreditDecoding: Parallel Decoding with Trace Credits](https://arxiv.org/abs/2510.06133) | 2025 | Arxiv | >7B |
| [Accelerating dLLM Inference via Local Determinism Propagation](https://arxiv.org/abs/2510.07081) | 2025 | Arxiv | >7B |
| [Self Speculative Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2510.04147) | 2025 | Arxiv | >7B |
| [Wide-In, Narrow-Out: Revokable Decoding for Effective dLLMs](https://arxiv.org/pdf/2507.18578?) | 2025 | Arxiv | >7B |
| [SpecDiff-2: Scaling Diffusion Drafter Alignment](https://arxiv.org/abs/2511.00606) | 2025 | Arxiv | >7B |
| [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](https://arxiv.org/abs/2512.20573) | 2025 | Arxiv | >7B |
| [Fast-Decoding via Progress-Aware Confidence Schedules](https://arxiv.org/abs/2512.02892) | 2025 | Arxiv | >7B |
| [DLM-One: Diffusion Language Models for One-Step Generation](https://arxiv.org/pdf/2506.00290) | 2025 | Arxiv | <7B |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](https://arxiv.org/abs/2510.19304) | 2025 | Arxiv | >7B |
| [Accelerating LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413) | 2025 | Arxiv | >7B |
| [Accelerating dLLMs with SlowFast Sampling](https://arxiv.org/pdf/2506.10848) | 2025 | Arxiv | >7B |
| [AdaBlock-dLLM: Semantic-Aware Inference via Adaptive Block Size](https://arxiv.org/pdf/2509.26432) | 2025 | Arxiv | >7B |
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | 2025 | Arxiv | >7B |
| [Learning to Parallel: Accelerating dLLMs via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188) | 2025 | Arxiv | >7B |

### 5.3 Distillation, Quantization & Sparsity
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Beyond Autoregression: Fast LLMs via Self-Distillation](https://arxiv.org/abs/2410.21035) | 2025 | ICLR | <7B, Distillation |
| [CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269) | 2025 | Arxiv | >7B, Consistency |
| [FS-DFM: Few-Step Diffusion Language Model](https://arxiv.org/abs/2509.20624) | 2025 | Arxiv | >7B |
| [Quantization Meets dLLMs: Post-training Quantization Study](https://arxiv.org/pdf/2508.14896) | 2025 | Arxiv | >7B, Quantization |
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014) | 2025 | Arxiv | >7B, Sparsity |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025 | Arxiv | >7B, MoE |

---

## 6. Multi-modal & Physical AI

### 6.1 Multi-modal dLLMs
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) | 2025 | Arxiv | Native Multimodal |
| [MMaDA-Parallel: Thinking-Aware Editing and Generation](https://arxiv.org/pdf/2511.09611) | 2025 | Arxiv | Parallel Multimodal |
| [Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564) | 2025 | Arxiv | Unified Generation |
| [Lumina-DiMOO: Omni Diffusion LLM for Generation](https://arxiv.org/abs/2510.06308) | 2025 | Arxiv | Omni-generation |
| [DiffusionVL: Translating AR Models into Diffusion VL Models](https://arxiv.org/abs/2512.15713) | 2025 | Arxiv | VL Adaptation |
| [Diffuse Everything: Multimodal Diffusion on Arbitrary Spaces](https://www.arxiv.org/abs/2506.07903) | 2025 | ICML | Arbitrary Spaces |
| [LLaDA-V: Diffusion LLMs with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) | 2025 | Arxiv | Visual Tuning |
| [Unified Multimodal Discrete Diffusion](https://arxiv.org/abs/2503.20853) | 2025 | Arxiv | Unified Diffusion |
| [LaViDa: A Large Diffusion LLM for Multimodal Understanding](https://arxiv.org/abs/2505.16839) | 2025 | Arxiv | Understanding |
| [Dimple: Discrete Diffusion Multimodal LLM with Parallel Decoding](https://arxiv.org/abs/2505.16990) | 2025 | Arxiv | Parallel Multimodal |
| [Dual Diffusion for Unified Image Generation and Understanding](https://arxiv.org/pdf/2501.00289) | 2025 | Arxiv | Unified Task |
| [Muddit: Liberating Generation Beyond Text-to-Image](https://arxiv.org/pdf/2505.23606) | 2025 | Arxiv | Multi-modal |

### 6.2 Vision-Language-Action (VLA)
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Discrete Diffusion VLA: Action Decoding in VLA Policies](https://arxiv.org/abs/2508.20072) | 2025 | Arxiv | VLA Action Decoding |
| [LLaDA-VLA: Vision Language Diffusion Action Models](https://arxiv.org/abs/2509.06932) | 2025 | Arxiv | VLA Framework |
| [dVLA: Diffusion VLA with Multimodal Chain-of-Thought](https://arxiv.org/pdf/2509.25681) | 2025 | Arxiv | VLA Reasoning |

---

## 7. Theory, Guidance & Applications

### 7.1 Theory & Analysis
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Time Is a Feature: Exploiting Temporal Dynamics in dLLMs](https://arxiv.org/pdf/2508.09138v1) | 2025 | Arxiv | Temporal focus |
| [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622) | 2025 | NeurIPS | Limits analysis |
| [What Makes Diffusion Language Models Super Data Learners?](https://arxiv.org/pdf/2510.04071) | 2025 | Arxiv | Data efficiency |
| [Why mask diffusion does not work](https://arxiv.org/pdf/2510.03289) | 2025 | Arxiv | Failure analysis |
| [The Diffusion Duality](https://arxiv.org/abs/2506.10892) | 2025 | ICML | <7B, Theoretical Duality |
| [Diffusion LLMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) | 2025 | Arxiv | Semantic focus |
| [Generalized Interpolating Discrete Diffusion](https://openreview.net/pdf?id=rvZv7sDPV9) | 2025 | ICML | <7B |
| [Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior](https://arxiv.org/pdf/2507.07586) | 2025 | ArXiv | <7B |

### 7.2 Guidance & Downstream Applications
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) | 2025 | Arxiv | Constrained Decoding |
| [DiffuCoder: Improving Masked Diffusion for Code Generation](https://arxiv.org/abs/2506.20639) | 2025 | Arxiv | Code |
| [Beyond Autoregression: Empirical Study for Code Generation](https://arxiv.org/abs/2509.11252) | 2025 | Arxiv | Code |
| [Seed Diffusion: Large-Scale dLLM with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025 | Arxiv | Code |
| [Planning with Diffusion Models for Target-Oriented Dialogue](https://arxiv.org/abs/2504.16858v1) | 2025 | ACL | Dialogue |
| [The Devil behind the mask: An emergent safety vulnerability](https://arxiv.org/pdf/2507.11097v1) | 2025 | Arxiv | Safety |
| [CtrlDiff: Boosting dLLMs with Dynamic Block Prediction](https://arxiv.org/abs/2505.14455) | 2025 | Arxiv | Control |

---

## 8. Seminal Diffusion Papers

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) | 2015 | ICML | Formulation |
| [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) | 2020 | NeurIPS | - |
| [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) | 2021 | ICLR | - |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2021 | ICLR | - |
| [High-Resolution Image Synthesis with Latent Diffusion](https://arxiv.org/abs/2112.10752) | 2022 | CVPR | - |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | 2023 | ICCV | Scalable focus |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023 | ICML | - |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 2021 | NeurIPS | CG |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2021 | NeurIPS | CFG |
| [DPM-Solver: Fast ODE Solver for Sampling](https://arxiv.org/abs/2206.00927) | 2022 | NeurIPS | - |
| [Vector Quantized Diffusion Model (VQ-Diffusion)](https://arxiv.org/abs/2111.14822) | 2022 | CVPR | VQ |
| [Analog Bits: Generating Discrete Data using Diffusion](https://arxiv.org/abs/2208.04202) | 2023 | ICLR | Self-conditioning |
| [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512) | 2022 | ICLR | Distillation |
| [Structured Denoising Diffusion in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) | 2021 | NeurIPS | Discrete |

---

## 🤝 Contact
* Maintainers: jake630@snu.ac.kr / wjk9904@snu.ac.kr / qicher@snu.ac.kr
* Contributions via Pull Requests are welcome!
