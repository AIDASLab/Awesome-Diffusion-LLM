# Awesome-Large-Language-Diffusion-Models

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![Maintained](https://img.shields.io/badge/Maintained-2026-blue?style=flat-square)

A comprehensive and structured list of research papers about **Large-Language-Diffusion-Models (dLLMs)**.

> Last major update: October 2026 — added 176 papers through Oct 10 (152 Sep–Oct, 24 backfills), updated six venues, and added §5.5 Speculative Decoding & Draft Verification.

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
    - [Speculative Decoding & Draft Verification](#55-speculative-decoding--draft-verification)
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
- [LLaDA2.X (InclusionAI / Ant Group)](https://github.com/inclusionAI/LLaDA2.X)
- [W1-4B-dLLM (Whaletech AI)](https://huggingface.co/WhaletechAI/W1-4B-dLLM-Base) ([Demo](https://vibe.whaletech.ai/))
- [What are Diffusion Language Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Generative Modeling by Estimating Gradients (Yang Song)](https://yang-song.net/blog/2021/score/)
- [DiffusionGemma Explained (ML@Berkeley)](https://mlberkeley.substack.com/p/the-annotated-diffusiongemma)

### 📝 Survey & Perspective Papers
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) | 2023.03 | IJCAI | Early NAR-text survey |
| [A Survey of Diffusion Models in NLP](https://arxiv.org/abs/2305.14671) | 2023.05 | Arxiv | Early NLP survey |
| [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759) | 2025.06 | Arxiv | dLLM + dMLLM |
| [Diffusion-based Large Language Models Survey](https://www.techrxiv.org/users/952417/articles/1321784-diffusion-based-large-language-models-survey) | 2025.08 | TechRxiv | - |
| [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/pdf/2508.08712v2) | 2025.08 | Arxiv | - |
| [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875) | 2025.08 | Arxiv | VILA-Lab; comprehensive |
| [Efficient Diffusion Language Models: A Comprehensive Survey](https://www.authorea.com/users/1021451/articles/1381451-efficient-diffusion-language-models-a-comprehensive-survey) | 2026.01 |  | Efficiency-focused |
| [Top 10 Open Challenges Steering the Future of Diffusion Language Model and Its Variants](https://arxiv.org/abs/2601.14041) | 2026.01 | Arxiv | Perspective / roadmap |
| [A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models](https://arxiv.org/abs/2605.22586) | 2026.05 | Arxiv | Tutorial, Diffusion theory |
| [Accelerating Masked Diffusion Large Language Models: A Survey of Efficient Inference Techniques](https://arxiv.org/abs/2607.12829) | 2026.07 | IJCAI | A Survey of Efficient Inference Techniques |
| [Discrete Diffusion Models: A Unified Framework from Tokenization to Generation](https://arxiv.org/abs/2607.13431) | 2026.07 | Arxiv | Backfill, Discrete diffusion survey |
| [Is Multimodal Speculative Decoding Ready for Diffusion-Based Parallel Drafting? A Survey and Empirical Diagnosis](https://arxiv.org/abs/2608.20743) | 2026.08 | Arxiv | Speculative decoding, Summarization |
| [Diffusion Language Models for Mobile Edge Agentic AI: Foundations, Applications, and Challenges](https://arxiv.org/abs/2609.04778) | 2026.09 | Arxiv | Survey, Mobile agents |
| [Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference](https://arxiv.org/abs/2609.14850) | 2026.09 | Arxiv | Thesis, Generation strategies |

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
| [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429v2) | 2025.06 | AAAI | >7B, Context Scaling |
| [Esoteric Language Models (Eso-LMs)](https://arxiv.org/pdf/2506.01928) | 2025.06 | ICML | AR + MDM hybrid |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487v1) | 2025.08 | Arxiv | >7B, Dream-7B |
| [Sequential Diffusion Language Models](https://arxiv.org/abs/2509.24007v1) | 2025.09 | Arxiv | >7B |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025.09 | Arxiv | >7B, 7B-A1B MoE from scratch |
| [UltraLLaDA: Scaling Context to 128K](https://arxiv.org/abs/2510.10481) | 2025.10 | Arxiv | >7B, Context Scaling |
| [Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](https://arxiv.org/abs/2510.08632) | 2025.10 | NeurIPS | - |
| [Masked Diffusion Models as Energy Minimization](https://arxiv.org/abs/2509.13866v1) | 2025.10 | NeurIPS | <7B |
| [Soft-Masked Diffusion Language Models](https://arxiv.org/abs/2510.17206v1) | 2025.10 | ICLR | <7B |
| [Variational Masked Diffusion Models](https://arxiv.org/abs/2510.23606v1) | 2025.10 | Arxiv | <7B |
| [Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way](https://arxiv.org/abs/2510.24605v1) | 2025.10 | Arxiv | >7B, Variable Length |
| [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276) | 2025.11 | Arxiv | Data efficiency |
| [DiffuMamba: High-Throughput Diffusion LMs with Mamba Backbone](https://arxiv.org/abs/2511.15927) | 2025.11 | Arxiv | Non-Transformer Backbone |
| [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923v1) | 2025.11 | Arxiv | >7B |
| [C2DLM: Causal Concept-Guided Diffusion Large Language Models](https://arxiv.org/abs/2511.22146v1) | 2025.11 | Arxiv | >7B |
| [Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models](https://arxiv.org/abs/2601.07351) | 2026.01 | ACL | Soft tokens, Masked |
| [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745v1) | 2025.12 | Arxiv | >100B, MoE; Ant Group |
| [LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676) | 2026.02 | Arxiv | Editable State Evolution |
| [Introspective Diffusion Language Models (I-DLM)](https://arxiv.org/abs/2604.11035) | 2026.04 | Arxiv | Introspective consistency |
| [W1-4B-dLLM (WhaletechAI)](https://huggingface.co/WhaletechAI/W1-4B-dLLM-Base) | 2026.04 | HF Model | 4B open dLLM; [demo](https://vibe.whaletech.ai/) |
| [Scaling Beyond Masked Diffusion Language Models](https://arxiv.org/abs/2602.15014) | 2026.02 | Arxiv | Uniform-state & interpolating diffusion scaling |
| [dLLM: Simple Diffusion Language Modeling](https://arxiv.org/abs/2602.22661) | 2026.02 | Arxiv | Unified open-source dLLM framework |
| [Generalized Discrete Diffusion from Snapshots](https://arxiv.org/abs/2603.21342) | 2026.03 | Arxiv | Unified arbitrary noising framework |
| [Diffutron: A Masked Diffusion Language Model for Turkish Language](https://arxiv.org/abs/2603.20466) | 2026.03 | Arxiv | Multilingual, Turkish MDM |
| [Expert-Choice Routing Enables Adaptive Computation in Diffusion Language Models](https://arxiv.org/abs/2604.01622) | 2026.04 | COLM | MoE, Expert-choice routing |
| [Rethinking Token Prediction: Tree-Structured Diffusion Language Model](https://arxiv.org/abs/2604.03537) | 2026.04 | Arxiv | Tree-structured token prediction |
| [Drifting Objectives for Refining Discrete Diffusion Language Models](https://arxiv.org/abs/2605.19470) | 2026.05 | Arxiv | TokenDrift, anti-symmetric objective |
| [Forward-Free Diffusion Language Models with BPTT-Free Looped Refinement](https://arxiv.org/abs/2606.08357) | 2026.06 | Arxiv | - |
| [Semantic DLM+: Improving Diffusion Language Models through Bias-variance Trade-off in Transition Kernel Design](https://arxiv.org/abs/2606.15327) | 2026.06 | Arxiv | Improving Diffusion Language Models through Bias-variance |
| [VoidPadding: Let [VOID] Handle Padding in Masked Diffusion Language Models so that [EOS] Can Focus on Semantic Termination](https://arxiv.org/abs/2606.17999) | 2026.06 | Arxiv | Code |
| [Sumi: Open Uniform Diffusion Language Model from Scratch](https://arxiv.org/abs/2606.19005) | 2026.06 | Arxiv | Open Uniform Diffusion Language Model from |
| [Improved Large Language Diffusion Models](https://arxiv.org/abs/2606.25331) | 2026.06 | Arxiv | - |
| [Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation](https://arxiv.org/abs/2606.27732) | 2026.06 | Arxiv | KV cache, Serving |
| [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788) | 2026.07 | Arxiv | Few-step, Unmasking order |
| [PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention](https://arxiv.org/abs/2607.25157) | 2026.07 | Arxiv | Pretrained Discrete Masked Diffusion Language Modeling |
| [DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146) | 2026.07 | Arxiv | MoE, Speculative decoding |
| [Mean-to-Score Discrete Diffusion: Posterior-Mean Denoisers for Score Entropy](https://arxiv.org/abs/2607.21372) | 2026.07 | Arxiv | Posterior-Mean Denoisers for Score Entropy |
| [LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models](https://arxiv.org/abs/2608.03457) | 2026.08 | Arxiv | MoE |
| [MDLMPE: Distribution Aware Positional Encoding for Masked Diffusion Language Models](https://arxiv.org/abs/2608.03769) | 2026.08 | Arxiv | Distribution Aware Positional Encoding for Masked |
| [Retrofitting Linear Attention into Diffusion Language Models](https://arxiv.org/abs/2608.06628) | 2026.08 | Arxiv | Block diffusion, Serving |
| [Reducing Pretraining-Generation Mismatch in Diffusion Language Models](https://arxiv.org/abs/2608.09424) | 2026.08 | Arxiv | - |
| [A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models](https://arxiv.org/abs/2609.06324) | 2026.09 | Arxiv | Coupled-noise training |
| [Representation-based Masked Diffusion Model](https://arxiv.org/abs/2609.12382) | 2026.09 | Arxiv | Conditional masked modeling |
| [DA-DLM: Explicitly Modeling Token Dependencies in Diffusion Language Models](https://arxiv.org/abs/2609.15070) | 2026.09 | Arxiv | DAG architecture |
| [Register Tokens for Bounded-State Reasoning in Diffusion Language Models](https://arxiv.org/abs/2609.16372) | 2026.09 | Arxiv | Register tokens, Bounded state |
| [Circuit Hypernetworks for Quantum-Augmented Diffusion Language Models](https://arxiv.org/abs/2609.24657) | 2026.09 | Arxiv | Quantum residual architecture |
| [Less Uniform Discrete Diffusion is More Powerful and Scalable](https://arxiv.org/abs/2609.35817) | 2026.09 | Arxiv | Uniform diffusion objective |
| [Learning to Re-Draft: A Variational Stackelberg Game for Discrete Diffusion](https://arxiv.org/abs/2609.35166) | 2026.09 | Arxiv | Learned corruption objective |
| [E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models](https://arxiv.org/abs/2609.37533) | 2026.09 | Arxiv | Non-factorized reverse model |
| [Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation](https://arxiv.org/abs/2609.37924) | 2026.09 | Arxiv | Anchor-denoiser architecture |
| [On Trajectory-Aware Training for Masked Diffusion Language Models](https://arxiv.org/abs/2609.37974) | 2026.09 | Arxiv | Differentiable diffusion training |
| [Alpha Diffusion Language Models: Factorization Alone Is Not the Problem](https://arxiv.org/abs/2609.38066) | 2026.09 | Arxiv | Sequence-level training loss |
| [Acceleration of Diffusion Language Model through Discrete Average Generator](https://arxiv.org/abs/2609.38364) | 2026.09 | Arxiv | CTMC training objective |
| [Know When to Hold 'em: Correct-Token Retention in Uniform-State Diffusion Language Models](https://arxiv.org/abs/2610.01275) | 2026.10 | Arxiv | Correct-token retention loss |
| [ALoDLM: Adaptively Looped Diffusion Language Models](https://arxiv.org/abs/2610.04198) | 2026.10 | Arxiv | Adaptive recurrent architecture |

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
| [Coevolutionary Continuous Discrete Diffusion: Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025.10 | ICML | >7B; CCDD |
| [Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning](https://arxiv.org/abs/2602.20528) | 2026.02 | Arxiv | Latent planning + AR hybrid |
| [CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think](https://arxiv.org/abs/2603.02547) | 2026.03 | Arxiv | Contextual AR decoder for continuous diffusion |
| [LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling](https://arxiv.org/abs/2604.11748) | 2026.04 | Arxiv | Flow matching via Bregman divergence |
| [Scaling Properties of Continuous Diffusion Spoken Language Models](https://arxiv.org/abs/2604.24416) | 2026.04 | Arxiv | Continuous diffusion SLM, scaling laws |
| [Towards Closing the Autoregressive Gap via Entropy-Gated Continuous Bitstream Diffusion](https://arxiv.org/abs/2605.07013) | 2026.05 | Arxiv | Continuous bitstream diffusion |
| [TextLDM: Language Modeling with Continuous Latent Diffusion](https://arxiv.org/abs/2605.07748) | 2026.05 | Arxiv | DiT-style flow matching for text |
| [How to Train Your Latent Diffusion Language Model Jointly With the Latent Space](https://arxiv.org/abs/2605.07933) | 2026.05 | Arxiv | Joint latent encoder+diffusion training |
| [BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion](https://arxiv.org/abs/2605.11577) | 2026.05 | Arxiv | Bitwise continuous diffusion head |
| [Language Generation as Optimal Control: Closed-Loop Diffusion in Latent Control Space](https://arxiv.org/abs/2605.14531) | 2026.05 | Arxiv | HJB-based latent optimal control |
| [Continuous Diffusion Scales Competitively with Discrete Diffusion for Language](https://arxiv.org/abs/2605.18530) | 2026.05 | Arxiv | RePlaid scaling law, continuous vs discrete |
| [Simplex Relaxation for Discrete Diffusion](https://arxiv.org/abs/2608.10615) | 2026.08 | Arxiv | - |
| [AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling](https://arxiv.org/abs/2608.02602) | 2026.08 | Arxiv | Continuous text latents, Ascend |
| [Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders](https://arxiv.org/abs/2609.27248) | 2026.09 | Arxiv | Continuous text autoencoder |
| [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 2026.09 | Arxiv | Continuous representations |
| [One Latent, Many Tokens: Jointly Learning Compressed Embeddings for Efficient Language Diffusion](https://arxiv.org/abs/2609.33698) | 2026.09 | Arxiv | Compressed continuous latents |
| [Simplex Diffusion Models](https://arxiv.org/abs/2609.35553) | 2026.09 | Arxiv | Simplex continuous diffusion |
| [Clock Diffusion: Efficient Semi-Autoregressive Continuous Diffusion Language Models](https://arxiv.org/abs/2610.00894) | 2026.10 | Arxiv | Continuous, Sliding-window model |
| [Scaling and Distilling Text Embeddings for Better Diffusibility](https://arxiv.org/abs/2610.01016) | 2026.10 | Arxiv | Text embedding geometry |
| [Hierarchical Continuous Diffusion Language Models](https://arxiv.org/abs/2610.02193) | 2026.10 | Arxiv | Hierarchical continuous states |
| [Large Language Continuous Diffusion Models](https://arxiv.org/abs/2610.02665) | 2026.10 | Arxiv | Continuous, Sigma |
| [Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2610.08738) | 2026.10 | Arxiv | Hierarchical continuous representations |
| [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 2026.10 | TMLR | Prompt-noising objective |
| [Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization](https://arxiv.org/abs/2610.09311) | 2026.10 | Arxiv | Compressed block latents |

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
| [Where Should Diffusion Enter a Language Model? Geometry-Guided Hidden-State Replacement](https://arxiv.org/abs/2605.14368) | 2026.05 | Arxiv | Geometry-guided layer insertion |
| [Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation](https://arxiv.org/abs/2606.06712) | 2026.06 | Arxiv | Distillation |
| [Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](https://arxiv.org/abs/2606.26493) | 2026.06 | Arxiv | MoE, Block diffusion |
| [UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective](https://arxiv.org/abs/2607.24507) | 2026.07 | Arxiv | Adapting Autoregressive Language Models into Discrete |
| [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751) | 2026.09 | Arxiv | AR-to-Diffusion, Qwen |
| [DreamingGoose: Staged Distillation from Autoregressive Transformers to Bidirectional Recurrent Diffusion Language Models](https://arxiv.org/abs/2609.34253) | 2026.09 | Arxiv | AR conversion, Recurrence |
| [d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models](https://arxiv.org/abs/2609.35362) | 2026.09 | Arxiv | AR-to-block-diffusion distillation |
| [Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs](https://arxiv.org/abs/2610.02657) | 2026.10 | Arxiv | Low-budget AR conversion |

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
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) | 2026.02 | ICML | Block + speculative |
| [Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2604.08964) | 2026.04 | ACL | Anchor-based cross-block decoding |
| [When to Commit? Towards Variable-Size Self-Contained Blocks for Discrete Diffusion Language Models](https://arxiv.org/abs/2604.23994) | 2026.04 | Arxiv | Variable-size blocks |
| [Dynamic Chunking for Diffusion Language Models](https://arxiv.org/abs/2605.15676) | 2026.05 | Arxiv | Content-defined semantic chunks |
| [FLARE: Diffusion for Hybrid Language Model](https://arxiv.org/abs/2606.01774) | 2026.06 | Arxiv | Serving |
| [BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers](https://arxiv.org/abs/2606.02241) | 2026.06 | Arxiv | Few-step, Block diffusion |
| [SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs](https://arxiv.org/abs/2606.04964) | 2026.06 | Arxiv | Block diffusion |
| [Multi-Block Diffusion Language Models](https://arxiv.org/abs/2606.29215) | 2026.06 | Arxiv | Block diffusion |
| [Adaptive Block Diffusion: Resolving Training-Inference Mismatch in Diffusion Language Models](https://arxiv.org/abs/2606.29275) | 2026.06 | Arxiv | Block diffusion |
| [Beyond Block Boundaries: Multi-Block Editing for Diffusion Large Language Models](https://arxiv.org/abs/2607.22663) | 2026.06 | Arxiv | Training-free, KV cache |
| [Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding](https://arxiv.org/abs/2607.01775) | 2026.07 | ICML | KV cache, Block diffusion |
| [Training Hybrid Block Diffusion Language Models with Partial Bidirectionality](https://arxiv.org/abs/2607.02805) | 2026.07 | Arxiv | Block diffusion, Serving |
| [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](https://arxiv.org/abs/2607.05722) | 2026.07 | Arxiv | Serving |
| [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010) | 2026.09 | Arxiv | Hybrid AR-Diffusion |
| [Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale](https://arxiv.org/abs/2609.11998) | 2026.09 | Arxiv | Block diffusion, Recurrent state |
| [Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference](https://arxiv.org/abs/2609.19868) | 2026.09 | Arxiv | Hybrid AR-Diffusion |

---

## 3. Reasoning & Policy Optimization

### 3.1 Reasoning & Planning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Diffusion of Thought: Chain-of-Thought Reasoning in dLLMs](https://arxiv.org/abs/2402.07754) | 2024.02 | NeurIPS | <7B, CoT Foundation |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/pdf/2410.14157) | 2024.10 | ICLR | <7B |
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2025.09 | Arxiv | Planning |
| [d1: Scaling Reasoning in dLLMs via RL](https://arxiv.org/abs/2504.12216) | 2025.04 | NeurIPS | >7B, Reasoning scaling |
| [Reinforcing the Diffusion Chain of Lateral Thought](https://arxiv.org/abs/2505.10446) | 2025.05 | NeurIPS | >7B |
| [Thinking Inside the Mask: In-Place Prompting in dLLMs](https://arxiv.org/pdf/2508.10736) | 2025.08 | Arxiv | >7B |
| [Reinforced Context Order Recovery for Adaptive Reasoning](https://arxiv.org/pdf/2508.13070) | 2025.08 | Arxiv | <7B, Planning |
| [d2: Improved Techniques for Training Reasoning dLLMs](https://www.arxiv.org/abs/2509.21474) | 2025.09 | Arxiv | >7B |
| [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573) | 2025.10 | Arxiv | >7B |
| [Beyond Surface Reasoning: Unveiling Long CoT Capacity](https://arxiv.org/abs/2510.09544) | 2025.10 | Arxiv | >7B |
| [Coevolutionary Continuous Discrete Diffusion: Latent Reasoner](https://arxiv.org/abs/2510.03206) | 2025.10 | ICML | >7B |
| [On the Reasoning Abilities of Masked Diffusion Language Models](https://arxiv.org/abs/2510.13117v1) | 2025.10 | Arxiv | >7B |
| [Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning](https://arxiv.org/abs/2510.15244v2) | 2025.10 | Arxiv | Collaboration |
| [Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning](https://arxiv.org/abs/2510.27469v1) | 2025.10 | Arxiv | >7B |
| [Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching](https://arxiv.org/abs/2602.22871) | 2026.02 | Arxiv | Step-level rationale stitching |
| [Reasoning or Rationalization? The Role of Justifications in Masked Diffusion Models for Fact Verification](https://arxiv.org/abs/2603.01190) | 2026.03 | Arxiv | CoT dynamics analysis |
| [Diffusion LLMs can think EoS-by-EoS](https://arxiv.org/abs/2603.05197) | 2026.03 | Arxiv | EoS-guided reasoning via padding |
| [LogicDiff: Logic-Guided Denoising Improves Reasoning in Masked Diffusion Language Models](https://arxiv.org/abs/2603.26771) | 2026.03 | Arxiv | Logic-guided unmasking order |
| [Learnability-Informed Fine-Tuning of Diffusion Language Models](https://arxiv.org/abs/2605.22939) | 2026.05 | Arxiv | LIFT, SFT with learnability schedule |
| [d-OPSD: On-Policy Self-Distillation for Diffusion Language Models](https://arxiv.org/abs/2606.18195) | 2026.06 | Arxiv | On-policy self-distillation, [Code](https://github.com/xingzhejun/d-opsd-code) |
| [Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models](https://arxiv.org/abs/2606.16700) | 2026.06 | Arxiv | - |
| [DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models](https://arxiv.org/abs/2606.19257) | 2026.06 | Arxiv | 8B, Block diffusion |
| [Diffusion-Proof: Recipe for Formal Theorem Proving Beyond Auto-Regressive Generation](https://arxiv.org/abs/2606.19315) | 2026.06 | Arxiv | Block diffusion |
| [Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models](https://arxiv.org/abs/2606.23567) | 2026.06 | Arxiv | Unmasking order |
| [Escaping Confidence Trap: Evolutionary Decoding for Mathematical Reasoning in Diffusion LLMs](https://arxiv.org/abs/2608.00605) | 2026.08 | Arxiv | Training-free, Unmasking order |
| [Semantic Reasoning Denoising: Correcting Language Model Reasoning with Semantic Operators](https://arxiv.org/abs/2608.22090) | 2026.08 | Arxiv | Correcting Language Model Reasoning with Semantic |
| [DCGC: Draft-Conditioned Global Correction for Complex Reasoning with Masked Diffusion Models](https://arxiv.org/abs/2608.25428) | 2026.08 | Arxiv | Draft-Conditioned Global Correction for Complex Reasoning |
| [CaLR: Causal Latent Revision for Robust Diffusion Reasoning](https://arxiv.org/abs/2609.20981) | 2026.09 | Arxiv | Latent reasoning search |
| [LOCKR: A Hidden-State Trajectory-Guided Planner for Detecting and Repairing Stable-but-Wrong Lock-In in Diffusion Language Models](https://arxiv.org/abs/2609.27220) | 2026.09 | Arxiv | Planning, Test-time repair |
| [Towards Efficient Reasoning: Learning Causal Shortcuts for Diffusion Language Models](https://arxiv.org/abs/2609.28272) | 2026.09 | Arxiv | Reasoning curriculum |
| [Reasoning with Continuous Latent Diffusion](https://arxiv.org/abs/2609.35694) | 2026.09 | Arxiv | Continuous diffusion reasoning |
| [Blackboard Intelligence Can Surpass Autoregressive on Globally Constrained Problems](https://arxiv.org/abs/2609.38806) | 2026.09 | Arxiv | Globally constrained search |

### 3.2 Alignment & Reinforcement Learning
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295) | 2025.03 | Arxiv | >7B |
| [DiFFPO: Training dLLMs to Reason Fast and Furious via RL](https://arxiv.org/pdf/2510.02212) | 2025.10 | Arxiv | >7B, Direct Preference |
| [LLaDA 1.5: Variance-Reduced Preference Optimization](https://arxiv.org/abs/2505.19223) | 2025.05 | Arxiv | >7B |
| [wd1: Weighted Policy Optimization for Reasoning](https://arxiv.org/pdf/2507.08838) | 2025.07 | ICLR | >7B |
| [Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](https://arxiv.org/abs/2508.12398v1) | 2025.08 | AAAI | >7B, Safety |
| [Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](https://arxiv.org/abs/2507.19227v1) | 2025.07 | Arxiv | Safety |
| [The Devil behind the mask: An emergent safety vulnerability](https://arxiv.org/pdf/2507.11097v1) | 2025.07 | ICLR | Safety |
| [MDPO: Overcoming the Training-Inference Divide](https://arxiv.org/abs/2508.13148) | 2025.08 | Arxiv | >7B |
| [Reward-Weighted Sampling: Enhancing Non-Autoregressive Characteristics in Masked Diffusion LLMs](https://arxiv.org/abs/2509.00707) | 2025.08 | EMNLP | >7B |
| [Inpainting-Guided Policy Optimization for dLLMs](https://arxiv.org/abs/2509.10396) | 2025.09 | Arxiv | >7B |
| [Taming Masked Diffusion via Consistency Trajectory RL](https://arxiv.org/abs/2509.23924) | 2025.09 | Arxiv | >7B |
| [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning](https://arxiv.org/abs/2509.25171) | 2025.09 | Arxiv | >7B |
| [Revolutionizing RL Framework for Diffusion Large Language Models](https://arxiv.org/pdf/2509.06949) | 2025.09 | Arxiv | >7B |
| [A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](https://arxiv.org/abs/2509.23286v1) | 2025.09 | ICLR | Safety |
| [DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models](https://arxiv.org/abs/2509.24296v1) | 2025.09 | Arxiv | Safety |
| [RFG: Test-Time Scaling for Diffusion Large Language Model Reasoning with Reward-Free Guidance](https://arxiv.org/abs/2509.25604v1) | 2025.09 | Arxiv | >7B |
| [AGRPO: Simple Policy Gradients for Reasoning with Diffusion Language Models](https://arxiv.org/abs/2510.04019) | 2025.10 | Arxiv | >7B |
| [Improving Reasoning via Group Diffusion Policy Optimization (GDPO)](https://arxiv.org/pdf/2510.08554) | 2025.10 | Arxiv | >7B |
| [Step-Aware Policy Optimization for Reasoning](https://arxiv.org/abs/2510.01544) | 2025.10 | Arxiv | >7B |
| [MRO: Enhancing Reasoning via Multi-Reward Optimization](https://arxiv.org/abs/2510.21473) | 2025.10 | NeurIPS | >7B |
| [Enhancing Reasoning via Distribution Matching Policy Optimization](https://arxiv.org/abs/2510.08233) | 2025.10 | Arxiv | >7B |
| [Boundary-Guided Policy Optimization for Memory-efficient RL](https://arxiv.org/abs/2510.11683) | 2025.10 | Arxiv | >7B |
| [SPG: Sandwiched Policy Gradient for Masked Diffusion](https://arxiv.org/abs/2510.09541) | 2025.10 | Arxiv | >7B |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025.10 | ICLR | >7B |
| [Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052v2) | 2025.10 | Arxiv | >7B |
| [Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective](https://arxiv.org/abs/2512.03759v1) | 2025.12 | Arxiv | >7B |
| [d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2512.09675v1) | 2025.12 | Arxiv | >7B |
| [DARE: Diffusion Large Language Models Alignment and Reinforcement Executor](https://arxiv.org/abs/2604.04215) | 2026.04 | Arxiv | Unified RL framework |
| [DiRL: An Efficient Post-Training Framework for Diffusion Language Models](https://arxiv.org/abs/2512.22234) | 2025.12 | Arxiv | Post-training |
| [Efficient and Stable Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2602.08905) | 2026.02 | Arxiv | Variance reduction |
| [Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation](https://arxiv.org/abs/2601.07152) | 2026.01 | Arxiv | Multi-agent RL |
| [Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages](https://arxiv.org/abs/2603.12554) | 2026.03 | Arxiv | MDP formulation, entropy-guided steps |
| [TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models](https://arxiv.org/abs/2604.00666) | 2026.04 | Arxiv | Trajectory-ranked SFT |
| [Relative Score Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2605.10218) | 2026.05 | Arxiv | RSPO, RLVR for dLLMs |
| [Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models](https://arxiv.org/abs/2605.13043) | 2026.05 | Arxiv | Safety, contrastive steering |
| [Beyond Mode-Seeking RL: Trajectory-Balance Post-Training for Diffusion Language Models](https://arxiv.org/abs/2605.13935) | 2026.05 | Arxiv | TraFL, trajectory-balance objective |
| [Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2606.04396) | 2026.06 | Arxiv | MoE, Unmasking order |
| [Back on Track: Aligning Rewards and States for Reasoning in Diffusion Large Language Models](https://arxiv.org/abs/2606.08501) | 2026.06 | EMNLP | RL |
| [A2D2: Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding](https://arxiv.org/abs/2606.13565) | 2026.06 | Arxiv | Unmasking order |
| [SLIM-RL: Risk-Budgeted Random-Masking RL for Diffusion LLMs Without Trajectory Slicing](https://arxiv.org/abs/2607.00208) | 2026.06 | Arxiv | Training-free, RL |
| [dVLA-RL: Reinforcement Learning over Denoising Trajectories for Discrete Diffusion Vision-Language-Action Models](https://arxiv.org/abs/2606.23623) | 2026.06 | Arxiv | Backfill, VLA trajectory-level RL |
| [A Continuous-Time Reinforcement Learning Framework for Fine-Tuning Discrete Diffusion Models](https://arxiv.org/abs/2607.14522) | 2026.07 | Arxiv | RL |
| [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/abs/2607.15200) | 2026.07 | COLM | Remasking, Unmasking order |
| [Beyond On-Policy Exploration: Integrating External Policy Rollouts for Reinforcement Learning in Diffusion Language Models](https://arxiv.org/abs/2608.01717) | 2026.08 | Arxiv | RL |
| [CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2609.13060) | 2026.09 | Arxiv | RL curriculum |
| [Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models](https://arxiv.org/abs/2609.25927) | 2026.09 | Arxiv | RL, Informed masking |
| [ForkLeft: Entropy-First Rollouts for Prefix-Aligned Autoregressive-to-Diffusion Distillation](https://arxiv.org/abs/2609.32448) | 2026.09 | Arxiv | Capability distillation |
| [CT-OPD: Counterfactual Trace On-Policy Distillation for Diffusion Vision-Language Models](https://arxiv.org/abs/2609.32781) | 2026.09 | Arxiv | Multimodal on-policy distillation |
| [Fine-Tuning Diffusion Language Models with Context Selection and Target Weighting](https://arxiv.org/abs/2609.38385) | 2026.09 | Arxiv | Supervised fine-tuning |
| [Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models](https://arxiv.org/abs/2610.00661) | 2026.09 | Arxiv | Stepwise risk-sensitive GRPO |
| [IDRF: Inverse-Distilled Reward Fine-tuning of Masked Discrete Diffusion Models](https://arxiv.org/abs/2610.03641) | 2026.10 | Arxiv | Reward fine-tuning |
| [Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2610.03665) | 2026.10 | Arxiv | Pivot-focused self-distillation |
| [Stabilizing language models under continual learning via condition-anchored distillation](https://arxiv.org/abs/2610.06940) | 2026.10 | Arxiv | Continual adaptation |
| [Towards Unbiased On-Policy Distillation for Block Diffusion Language Models](https://arxiv.org/abs/2610.05373) | 2026.10 | Arxiv | Stable on-policy distillation |
| [Representation-Space MMD for Diffusion Language Models](https://arxiv.org/abs/2610.06648) | 2026.10 | Arxiv | MMD post-training |
| [Enhancing Diffusion Language Models with Autoregressive Post-Training Weights](https://arxiv.org/abs/2610.08108) | 2026.10 | Arxiv | Training-free weight transfer |

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
| [Unlocking the Potential of Diffusion Language Models through Template Infilling](https://arxiv.org/abs/2510.13870) | 2025.10 | ACL 2026 Oral | Template-then-fill generation; [code](https://github.com/JunHoo-Lee/Template-Infilling) |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025.10 | NeurIPS | <7B, Edit preservation |
| [Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models](https://arxiv.org/abs/2510.05090v1) | 2025.10 | Arxiv | >7B, Unmasking |
| [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies](https://arxiv.org/abs/2510.05725) | 2025.10 | ICLR | >7B, Unmasking |
| [Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing](https://arxiv.org/abs/2510.21961v1) | 2025.10 | Arxiv | >7B, Unmasking |
| [Diffusion Language Model Inference with Monte Carlo Tree Search](https://arxiv.org/abs/2512.12168v1) | 2025.12 | Arxiv | >7B, MCTS |
| [Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty](https://arxiv.org/abs/2512.21336v1) | 2025.12 | Arxiv | >7B, Unmasking |
| [Adaptation to Intrinsic Dependence in Diffusion Language Models](https://arxiv.org/abs/2602.20126) | 2026.02 | Arxiv | Distribution-agnostic schedule |
| [Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration](https://arxiv.org/abs/2603.02760) | 2026.03 | ACL | Self-evaluation, Flexible length |
| [D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding](https://arxiv.org/abs/2603.19146) | 2026.03 | Arxiv | Diversity-aware decoding |
| [Improving Sampling for Masked Diffusion Models via Information Gain](https://arxiv.org/abs/2602.18176) | 2026.02 | Arxiv | Info-Gain sampler |
| [DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2603.15340) | 2026.03 | Arxiv | Dependency-aware unmasking |
| [Diffusion Language Models Are Natively Length-Aware](https://arxiv.org/abs/2603.06123) | 2026.03 | Arxiv | Length-aware EoS generation |
| [Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models](https://arxiv.org/abs/2604.00375) | 2026.04 | Arxiv | Quality vs exploration trade-off |
| [Remask, Don't Replace: Token-to-Mask Refinement in Masked Diffusion Language Models](https://arxiv.org/abs/2604.18738) | 2026.04 | Arxiv | T2M refinement, LLaDA2.1 analysis |
| [Edit-Based Refinement for Parallel Masked Diffusion Language Models](https://arxiv.org/abs/2605.09603) | 2026.05 | ICML | ME-DLM, edit-based post-correction |
| [When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation for Diffusion Language Models](https://arxiv.org/abs/2605.28181) | 2026.05 | Arxiv | Suffix anchor, confidence modulation |
| [Supportive Token Revealing for Fast Diffusion Language Model Decoding](https://arxiv.org/abs/2606.04236) | 2026.06 | Arxiv | Training-free |
| [NAVIRA: Decoupled Stochastic Remasking for Masked Diffusion Language Models](https://arxiv.org/abs/2606.06031) | 2026.06 | Arxiv | Remasking, Unmasking order |
| [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232) | 2026.06 | Arxiv | Training-free, Remasking |
| [Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models](https://arxiv.org/abs/2606.12273) | 2026.06 | ACL | Unmasking order, RL |
| [Who Should Lead Decoding Now? Tracking Reliable Trajectories for Ensembling Masked Diffusion Language Models](https://arxiv.org/abs/2606.16281) | 2026.06 | Arxiv | - |
| [Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens](https://arxiv.org/abs/2606.16847) | 2026.06 | Arxiv | Training-free, Remasking |
| [When to Plan, When to Polish: Noise Level as a Granularity Axis for Diffusion Language Models](https://arxiv.org/abs/2606.21802) | 2026.06 | Arxiv | Noise Level as a Granularity Axis |
| [TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding](https://arxiv.org/abs/2607.03236) | 2026.07 | Arxiv | Training-free |
| [Don't Commit Alone: Joint Token Commitment in Diffusion Language Models](https://arxiv.org/abs/2607.04469) | 2026.07 | Arxiv | Joint Token Commitment in Diffusion Language |
| [Reinforcing the Generation Order of Multimodal Masked Diffusion Models](https://arxiv.org/abs/2607.08056) | 2026.07 | Arxiv | RL, dMLLM |
| [ST-Veto: Spatio-Temporal Token Veto for Diffusion MLLMs via Taylor Prediction and Visual Grounding](https://arxiv.org/abs/2607.17884) | 2026.07 | ICML | Training-free, Unmasking order |
| [Rethinking the Generation Order of Block Diffusion Language Models](https://arxiv.org/abs/2607.24306) | 2026.07 | Arxiv | Training-free, Unmasking order |
| [Commit Locally, Exit Globally: Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models](https://arxiv.org/abs/2607.28166) | 2026.07 | Arxiv | Coordinating Adaptive Sampling and Early Exit |
| [Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs](https://arxiv.org/abs/2608.22367) | 2026.08 | EMNLP | dMLLM, Context-aware token ordering, [Code](https://github.com/zhaoyk-sysu/CACD-dMLLM) |
| [Ripple-Pivot Search: Active Parallel Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2608.11742) | 2026.08 | Arxiv | Training-free, Unmasking order |
| [Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference](https://arxiv.org/abs/2608.26574) | 2026.08 | Arxiv | Training-free, Remasking |
| [Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models](https://arxiv.org/abs/2608.26580) | 2026.08 | Arxiv | Unmasking order, dMLLM |
| [Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs](https://arxiv.org/abs/2608.26641) | 2026.08 | EMNLP Findings | Training-free, Hallucination |
| [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion](https://arxiv.org/abs/2609.01043) | 2026.09 | Arxiv | Persistent-context commitment |
| [Early-Bird Decoding: Accelerating Diffusion LLMs with Learnable Block Sizes and Parallel Sampling](https://arxiv.org/abs/2609.16450) | 2026.09 | Arxiv | Token grouping, Mask policy |
| [Low-Confidence Remasking Traps Flexibility: Realizing Arbitrary-Order Potential for Diverse Rollouts in Diffusion LLMs](https://arxiv.org/abs/2609.34509) | 2026.09 | Arxiv | Remasking, Commitment |
| [CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models](https://arxiv.org/abs/2610.08833) | 2026.09 | Arxiv | Confidence-drift remasking |
| [Reliable Parallel Decoding in Masked Diffusion Language Models](https://arxiv.org/abs/2609.36452) | 2026.09 | Arxiv | Entropy-based commitment |
| [Fork-dLLM: Avoiding the Flexibility Trap in Diffusion Language Models](https://arxiv.org/abs/2609.39859) | 2026.09 | Arxiv | Token routing, AR fallback |
| [Bayesian Entropy-based Reordering for Calibrated Diffusion Language Models](https://arxiv.org/abs/2610.05125) | 2026.10 | Arxiv | Bayesian token commitment |

---

## 5. System Efficiency & Acceleration

### 5.1 Caching & Memory Strategy
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/pdf/2505.15781) | 2025.05 | NeurIPS | >7B |
| [FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467) | 2025.05 | Arxiv | >7B |
| [Fast-dLLM: Training-free Acceleration via KV Cache + Parallel Decoding](https://arxiv.org/abs/2505.22618) | 2025.05 | Arxiv | NVIDIA; KV cache + parallel |
| [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295v1) | 2025.06 | ICML | >7B |
| [d^2Cache: Accelerating via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094) | 2025.09 | ICLR | >7B |
| [Attention Is All You Need for KV Cache in dLLMs](https://arxiv.org/abs/2510.14973) | 2025.10 | Arxiv | >7B |
| [Attention Sinks in Diffusion Language Models](https://arxiv.org/abs/2510.15731) | 2025.10 | Arxiv | >7B |
| [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737v1) | 2025.12 | Arxiv | >7B, Causal cache |
| [Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding (COVER)](https://arxiv.org/abs/2602.06161) | 2026.02 | Arxiv | KV-override verification |
| [Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing](https://arxiv.org/abs/2602.02159) | 2026.02 | Arxiv | Long-context sparsity |
| [Mosaic: Unlocking Long-Context Inference for Diffusion LLMs via Global Memory Planning and Dynamic Peak Taming](https://arxiv.org/abs/2601.06562) | 2026.01 | Arxiv | Long-context memory |
| [Residual Context Diffusion Language Models](https://arxiv.org/abs/2601.22954) | 2026.01 | Arxiv | Recycle discarded tokens |
| [MAGE: All-[MASK] Block Already Knows Where to Look in Diffusion LLM](https://arxiv.org/abs/2602.14209) | 2026.02 | Arxiv | MASK-guided sparse attention, block dLLM |
| [MetaState: Persistent Working Memory for Discrete Diffusion Language Models](https://arxiv.org/abs/2603.01331) | 2026.03 | Arxiv | GRU-style cross-step memory |
| [DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](https://arxiv.org/abs/2603.08026) | 2026.03 | ICML | Saliency-based partial attention |
| [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489) | 2026.03 | Arxiv | Entropy-guided KV cache refresh |
| [LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models](https://arxiv.org/abs/2604.12056) | 2026.04 | Arxiv | Locality-aware sparse KV, block dLLM |
| [PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models](https://arxiv.org/abs/2605.20813) | 2026.05 | Arxiv | Column-sparse attention, periodic refresh |
| [Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](https://arxiv.org/abs/2606.02955)| 2026.06 | ICML Workshop |  |
| [Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models](https://arxiv.org/abs/2606.10537) | 2026.06 | EMNLP | Training-free, Long context |
| [HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval](https://arxiv.org/abs/2606.21633) | 2026.06 | Arxiv | KV cache, Block diffusion |
| [Affix Cache for Diffusion Large Language Models](https://arxiv.org/abs/2608.26140) | 2026.06 | Arxiv | Serving |
| [LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models](https://arxiv.org/abs/2607.16339) | 2026.07 | Arxiv | Training-free, Quantization |
| [Archer: Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion Language Models](https://arxiv.org/abs/2608.08086) | 2026.08 | Arxiv | Training-free |
| [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/abs/2609.26796) | 2026.09 | Arxiv | KV cache, Fused kernels |
| [SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models](https://arxiv.org/abs/2610.04875) | 2026.10 | Arxiv | Branch-state reuse, Speculation |
| [Mask-Guided KV Cache Eviction in Block Diffusion Language Models](https://arxiv.org/abs/2610.06996) | 2026.10 | Arxiv | KV selection and eviction |
### 5.2 Decoding & Sampling
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://arxiv.org/abs/2408.05636) | 2024.08 | NAACL | <7B, Speculative Decoding |
| [Wide-In, Narrow-Out: Revokable Decoding for Effective dLLMs](https://arxiv.org/pdf/2507.18578) | 2025.07 | Arxiv | >7B |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding (APD)](https://arxiv.org/abs/2506.00413v1) | 2025.05 | NeurIPS | >7B |
| [DLM-One: Diffusion Language Models for One-Step Generation](https://arxiv.org/pdf/2506.00290) | 2025.06 | Arxiv | <7B |
| [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848v2) | 2025.06 | Arxiv | >7B |
| [Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037) | 2025.06 | ICML | >7B |
| [Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models](https://arxiv.org/abs/2508.00819) | 2025.08 | Arxiv | >7B |
| [DPad: Efficient Diffusion Language Models with Suffix Dropout](https://arxiv.org/abs/2508.14148v2) | 2025.08 | Arxiv | >7B |
| [Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](https://arxiv.org/abs/2509.15188v1) | 2025.09 | NeurIPS | >7B |
| [AdaBlock-dLLM: Semantic-Aware Inference via Adaptive Block Size](https://arxiv.org/pdf/2509.26432) | 2025.09 | ICLR | >7B |
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488) | 2025.09 | Arxiv | >7B |
| [Learning to Parallel: Accelerating dLLMs via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188) | 2025.09 | Arxiv | >7B |
| [Spiffy: Multiplying Acceleration via Lossless Speculative Decoding](https://arxiv.org/pdf/2509.18085) | 2025.09 | ICML Workshop | >7B, Speculative Decoding |
| [DiffuSpec: Unlocking dLLMs for Speculative Decoding](https://www.arxiv.org/pdf/2510.02358) | 2025.09 | Arxiv | >7B, Speculative Decoding |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](https://arxiv.org/abs/2510.19304) | 2025.10 | ICLR | >7B |
| [Saber: Efficient Sampling with Backtracking Enhanced Remasking](https://arxiv.org/abs/2510.18165) | 2025.10 | ACL | >7B |
| [CreditDecoding: Parallel Decoding with Trace Credits](https://arxiv.org/abs/2510.06133) | 2025.10 | ACL | >7B |
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
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) | 2026.02 | ICML | Block + speculative |
| [Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models](https://arxiv.org/abs/2602.04399) | 2026.02 | Arxiv | Entropy-adaptive blocks |
| [Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding](https://arxiv.org/abs/2602.23792) | 2026.02 | Arxiv | DiCo, three-phase parallel decoding |
| [Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models](https://arxiv.org/abs/2603.04893) | 2026.03 | Arxiv | Diverse sampling, Pass@k |
| [S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation](https://arxiv.org/abs/2603.25702) | 2026.03 | Arxiv | Self-speculation, block diffusion |
| [Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models](https://arxiv.org/abs/2604.02560) | 2026.04 | Arxiv | DEMASK, dependency predictor |
| [DualDiffusion: A Speculative Decoding Strategy for Masked Diffusion Models](https://arxiv.org/abs/2604.05250) | 2026.04 | Arxiv | Speculative + causal drafter |
| [Accelerating Speculative Decoding with Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) | 2026.04 | Arxiv | Draft trees for block diffusion |
| [Stability-Weighted Decoding for Diffusion Language Models](https://arxiv.org/abs/2604.17068) | 2026.04 | Arxiv | KL-based token stability metric |
| [R²-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction](https://arxiv.org/abs/2604.18995) | 2026.04 | Arxiv | Spatial + temporal redundancy reduction |
| [Focus on the Core: Empowering Diffusion Large Language Models by Self-Contrast](https://arxiv.org/abs/2605.01373) | 2026.05 | Arxiv | Self-contrast, HD token focus |
| [Factorization-Error-Free Discrete Diffusion Language Model via Speculative Decoding](https://arxiv.org/abs/2605.14305) | 2026.05 | Arxiv | FeF-DLLM, prefix-conditioned factorization |
| [PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding](https://arxiv.org/abs/2605.15609) | 2026.05 | Arxiv | Parallel speculative, hierarchical acceptance |
| [Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers](https://arxiv.org/abs/2605.16941) | 2026.05 | Arxiv | WINO revokable parallel decoding |
| [Cost-Aware Diffusion Draft Trees for Speculative Decoding](https://arxiv.org/abs/2606.01813) | 2026.06 | Arxiv | Speculative decoding, Block diffusion |
| [DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding](https://arxiv.org/abs/2606.02091) | 2026.06 | Arxiv | Speculative decoding, Block diffusion |
| [SimSD: Simple Speculative Decoding in Diffusion Language Models](https://arxiv.org/abs/2606.02544) | 2026.06 | Arxiv | Training-free, KV cache |
| [SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding](https://arxiv.org/abs/2606.04974) | 2026.06 | Arxiv | Block diffusion |
| [Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge](https://arxiv.org/abs/2606.08048) | 2026.06 | ICML | - |
| [AsyncLane: Decoupling Refinement from Advancement in Diffusion Language Model Decoding](https://arxiv.org/abs/2606.08411) | 2026.06 | Arxiv | Training-free, Block diffusion |
| [Unified Energy for Invariant and Independent Decoding in Diffusion Language Models](https://arxiv.org/abs/2606.09159) | 2026.06 | Arxiv | - |
| [Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2606.10829) | 2026.06 | NeurIPS | Training-free |
| [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/abs/2606.11552) | 2026.06 | Arxiv | Speculative decoding |
| [Mean-Field Parallel Decoding for Discrete Diffusion Language Models](https://arxiv.org/abs/2606.15805) | 2026.06 | Arxiv | Training-free |
| [LESS Is More: Mutual-Stability Sampling for Diffusion Language Models](https://arxiv.org/abs/2606.16908) | 2026.06 | Arxiv | Training-free, Unmasking order |
| [Speculative Refinement: A Hybrid Autoregressive Diffusion Decoding Strategy and Its Behavior Across Benchmarks](https://arxiv.org/abs/2606.27474) | 2026.06 | Arxiv | Training-free |
| [$x$-Prediction Flow: Efficient Continuous Decoding for Masked Diffusion Language Models](https://arxiv.org/abs/2606.29066) | 2026.06 | Arxiv | Unmasking order, RL |
| [BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding](https://arxiv.org/abs/2606.31315) | 2026.06 | Arxiv | Speculative decoding |
| [PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding](https://arxiv.org/abs/2607.22634) | 2026.06 | Arxiv | Speculative decoding, Serving |
| [GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring](https://arxiv.org/abs/2608.20375) | 2026.06 | Arxiv | Speculative decoding, Serving |
| [Accelerating Discrete Diffusion Models with Parallel-In-Time Sampling](https://arxiv.org/abs/2607.00773) | 2026.07 | Arxiv | - |
| [DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding](https://arxiv.org/abs/2607.08642) | 2026.07 | Arxiv | Training-free, Speculative decoding |
| [Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models](https://arxiv.org/abs/2607.15655) | 2026.07 | Arxiv | Few-step |
| [FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models](https://arxiv.org/abs/2607.17652) | 2026.07 | Arxiv | Training-free, KV cache |
| [AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters](https://arxiv.org/abs/2607.19223) | 2026.07 | Arxiv | Speculative decoding, Distillation |
| [AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding](https://arxiv.org/abs/2607.25852) | 2026.07 | Arxiv | Speculative decoding, Serving |
| [Speculative Correction: Draft-then-Refine Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.02625) | 2026.07 | Arxiv | Training-free, Speculative decoding |
| [CORA-Diff: Confidence-Oriented Residual Acceptance for Efficient Diffusion Language Model Inference](https://arxiv.org/abs/2608.11235) | 2026.07 | Arxiv | Training-free, Block diffusion |
| [DAVET: Denoising-Aware Visual Evidence Trajectory Allocation for Diffusion Vision-Language Models](https://arxiv.org/abs/2608.01821) | 2026.08 | Arxiv | Training-free, Compression |
| [xPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding](https://arxiv.org/abs/2608.02438) | 2026.08 | Arxiv | Speculative decoding, Serving |
| [DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding](https://arxiv.org/abs/2608.05448) | 2026.08 | Arxiv | Speculative decoding, Block diffusion |
| [LibraSpec: Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization](https://arxiv.org/abs/2608.08721) | 2026.08 | Arxiv | Training-free, Speculative decoding |
| [CForce: Boosting Parallel Decoding for dLLMs via Consistency Forcing](https://arxiv.org/abs/2608.13925) | 2026.08 | Arxiv | Distillation |
| [From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding](https://arxiv.org/abs/2608.14787) | 2026.08 | Arxiv | Speculative decoding |
| [LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding](https://arxiv.org/abs/2608.20530) | 2026.08 | Arxiv | Speculative decoding, Serving |
| [CAI-DLLM: Convergence Aware Inference for Diffusion Language Models](https://arxiv.org/abs/2608.22646) | 2026.08 | Arxiv | Training-free, Long context |
| [Accelerating Diffusion Language Models via Structured Suffix Modeling](https://arxiv.org/abs/2608.23167) | 2026.08 | Arxiv | Training-free, KV cache |
| [Prefix-Denoising Consistency: Test-Time Verification for Diffusion Language Models](https://arxiv.org/abs/2608.25311) | 2026.08 | Arxiv | Unmasking order |
| [Survival-Guided Length Control for Efficient Diffusion Language Models](https://arxiv.org/abs/2608.26374) | 2026.08 | EMNLP | Training-free |
| [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.27514) | 2026.08 | Arxiv | Speculative decoding, Unmasking order |
| [ReTrace: Rejected-Trajectory Conditioning for Speculative Decoding](https://arxiv.org/abs/2608.29748) | 2026.08 | Arxiv | Speculative decoding, Code |
| [Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2608.30427) | 2026.08 | Arxiv | Speculative decoding |
| [CARVE: Verified Expansion for Variable-Length Generation in Diffusion Language Models](https://arxiv.org/abs/2608.30922) | 2026.08 | EMNLP Findings | Training-free, Block diffusion |
| [PACE-dLLM: Elastic Block Decoding via Confidence Cliff Estimation for Diffusion Language Models](https://arxiv.org/abs/2609.26249) | 2026.08 | Arxiv | Elastic decoding horizon |
| [Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models](https://arxiv.org/abs/2609.02108) | 2026.09 | EMNLP | Adaptive-length infilling |
| [Routing by Reasoning Need: Trajectory-Aware Decoding Control for Diffusion Vision-Language Models](https://arxiv.org/abs/2609.11315) | 2026.09 | EMNLP Findings | Adaptive decoding budget |
| [Rethinking Soft Tokens for Parallel Decoding in Diffusion Language Models](https://arxiv.org/abs/2609.37391) | 2026.09 | Arxiv | Soft-token sampling |
| [Self-Repulsive Sampling for Diffusion Language Models](https://arxiv.org/abs/2609.39560) | 2026.09 | Arxiv | Diverse joint sampling |
| [Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling](https://arxiv.org/abs/2610.03034) | 2026.10 | Arxiv | Adaptive stochastic solver |

### 5.3 Distillation, Quantization & Sparsity
| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2410.21035) | 2024.10 | ICLR | <7B, Distillation |
| [Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558) | 2025.08 | Arxiv | >7B, Sparsity |
| [DLLMQuant: Quantizing Diffusion-based Large Language Models](https://arxiv.org/abs/2508.14090) | 2025.08 | Arxiv | >7B, Quantization |
| [Quantization Meets dLLMs: Post-training Quantization Study](https://arxiv.org/pdf/2508.14896) | 2025.08 | Arxiv | >7B, Quantization |
| [FS-DFM: Few-Step Diffusion Language Model](https://arxiv.org/abs/2509.20624) | 2025.09 | ICLR | >7B |
| [SparseD: Sparse Attention for Diffusion Language Models](https://arxiv.org/abs/2509.24014) | 2025.09 | Arxiv | >7B, Sparsity |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389v1) | 2025.09 | Arxiv | >7B, MoE |
| [Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](https://arxiv.org/abs/2509.25035v2) | 2025.10 | Arxiv | >7B, Distillation |
| [CDLM: Consistency Diffusion Language Models For Faster Sampling](https://arxiv.org/abs/2511.19269) | 2025.11 | Arxiv | >7B, Consistency |
| [Sink-Aware Pruning for Diffusion Language Models](https://arxiv.org/abs/2602.17664) | 2026.02 | Arxiv | Unstable sink pruning |
| [FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](https://arxiv.org/abs/2604.05551) | 2026.04 | ACL Findings | Few-step distillation, S2S |
| [On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks](https://arxiv.org/abs/2604.20079) | 2026.04 | Arxiv | GPTQ/HAWQ on code dLLMs |
| [Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models](https://arxiv.org/abs/2604.26951) | 2026.04 | Arxiv | Cross-architecture dLLM distillation |
| [TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM](https://arxiv.org/abs/2605.09536) | 2026.05 | Arxiv | Trajectory self-distillation |
| [Infinite Mask Diffusion for Few-Step Distillation](https://arxiv.org/abs/2605.10518) | 2026.05 | Arxiv | IMDM, stochastic infinite-state mask |
| [Self-Distilled Trajectory-Aware Boltzmann Modeling for Diffusion Language Models](https://arxiv.org/abs/2605.11854) | 2026.05 | Arxiv | TABOM, Boltzmann ranking objective |
| [DiLaDiff: Distilled Latent-Augmented Diffusion for Language Modeling](https://arxiv.org/abs/2605.23605) | 2026.05 | Arxiv | Latent + consistency distillation |
| [STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models](https://arxiv.org/abs/2606.04945) | 2026.06 | Arxiv | Quantization, Unmasking order |
| [FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models](https://arxiv.org/abs/2606.06547) | 2026.06 | ICML | Quantization |
| [Seeing the End at Step Zero: Accelerating Diffusion MLLMs via MLP Sparsity-Aware Truncation](https://arxiv.org/abs/2607.14557) | 2026.07 | Arxiv | Training-free, Serving |
| [Trace-Based On-Policy Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2607.16872) | 2026.07 | Arxiv | Distillation, RL |
| [Multi-Mask Diffusion Language Models for Few-Step Generation](https://arxiv.org/abs/2607.19686) | 2026.07 | COLM | Distillation, Few-step |
| [Neuromorphic Diffusion Language Models: Addressing Compute and Memory Bottlenecks via Sparsity and Block Denoising](https://arxiv.org/abs/2607.24841) | 2026.07 | Arxiv | Block diffusion, Serving |
| [REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models](https://arxiv.org/abs/2608.01784) | 2026.08 | Arxiv | MoE, Training-free |
| [OPTD: On-Policy Transition Distillation with Consistency-Guided Adaptive Compression for Few-Step Diffusion Language Models](https://arxiv.org/abs/2608.02942) | 2026.08 | Arxiv | Distillation, Few-step |
| [SelFusion: Self-distillation for Diffusion Language Models](https://arxiv.org/abs/2608.22898) | 2026.08 | ACL | Distillation |
| [Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models](https://arxiv.org/abs/2609.15177) | 2026.09 | Arxiv | Temporal distillation |
| [Hesitation-Aware On-Policy Distillation for Diffusion Language Models](https://arxiv.org/abs/2609.33301) | 2026.09 | Arxiv | Few-step distillation |
| [JustQuant: You Don't Need Smoothing, SVD, or Rotation for 4-Bit Activation Quantization](https://arxiv.org/abs/2609.33601) | 2026.09 | Arxiv | 4-bit activation quantization |
| [Distribution Matching Distillation for Continuous Diffusion Language Models](https://arxiv.org/abs/2609.40235) | 2026.09 | Arxiv | Continuous diffusion distillation |
| [ITC-MoE: Importance-guided Token-aware Compression for MoE Diffusion Language Models](https://arxiv.org/abs/2610.01296) | 2026.10 | Arxiv | MoE low-rank compression |
| [Preserving Mathematical Reasoning in Compressed Diffusion Language Models via Trajectory-Aware Low-Rank Approximation](https://arxiv.org/abs/2610.03326) | 2026.10 | Arxiv | Trajectory-aware low-rank compression |
| [SoloQ: Calibration-Free Quantization for Diffusion Language Models](https://arxiv.org/abs/2610.07121) | 2026.10 | Arxiv | Calibration-free 4-bit quantization |
| [Omni-Diffusion-Distill: Few-Step Distillation of Unified Multimodal Diffusion Large Language Models](https://arxiv.org/abs/2610.10990) | 2026.10 | Arxiv | Multimodal few-step distillation |

### 5.4 Inference Frameworks & Systems
> **New section**: production-grade frameworks and runtime engineering for dLLMs.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [dlmserve](https://github.com/iOptimizeThings/dlmserve) | 2026.05 | Repo | First OSS serving engine for diffusion LMs (LLaDA family); OpenAI-compatible HTTP, step-level batching (2.5x HF), LocalLeap (1.8x). MIT. |
| [FOCUS: DLLMs Know How to Tame Their Compute Bound](https://arxiv.org/abs/2601.23278) | 2026.01 | ICML | Training-free inference system; token eviction for higher throughput |
| [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666v2) | 2025.10 | Arxiv | Modular framework, >1100 TPS |
| [JetEngine (SDAR)](https://github.com/JetAstra/SDAR) | 2025.10 | Repo | Lightweight engine for SDAR (3700+ TPS on H200) |
| [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298v1) | 2025.06 | Arxiv | Inception Labs commercial dLLM |
| [Seed Diffusion: Large-Scale dLLM with High-Speed Inference](https://lf3-static.bytednsdoc.com/obj/eden-cn/hyvsmeh7uhobf/sdiff_updated.pdf) | 2025.08 | Arxiv | ByteDance code-focused dLLM |
| [TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload](https://arxiv.org/abs/2605.20179) | 2026.05 | Arxiv | MoE expert offload, lossless |
| [Efficient On-Device Diffusion LLM Inference with Mobile NPU](https://arxiv.org/abs/2606.13740) | 2026.06 | Arxiv | KV cache, Speculative decoding |
| [DiLaServe: High SLO Attainment Serving for Diffusion Language Models](https://arxiv.org/abs/2606.29094) | 2026.06 | Arxiv | Serving |
| [Sangam: Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/abs/2607.04206) | 2026.07 | Arxiv | Serving |
| [BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving](https://arxiv.org/abs/2607.08930) | 2026.07 | Arxiv | Serving |
| [Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware](https://arxiv.org/abs/2608.23807) | 2026.08 | Arxiv | Serving |
| [FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving](https://arxiv.org/abs/2608.12932) | 2026.08 | Arxiv | Backfill, Driving inference system |
| [TailSieve: Partial-Rollout-Guided Tail Routing for LLM Rollouts](https://arxiv.org/abs/2608.22788) | 2026.08 | Arxiv | Backfill, Rollout routing |
| [Hardware Acceleration of Block-Diffusion LLM for Edge Devices](https://arxiv.org/abs/2609.01084) | 2026.09 | Arxiv | Hardware, Block diffusion |
| [Epoch: Compiling Diffusion Blocks for Sparse MoE Serving](https://arxiv.org/abs/2609.09748) | 2026.09 | Arxiv | MoE serving |
| [Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training](https://arxiv.org/abs/2609.19242) | 2026.09 | Arxiv | Distributed training |
| [OLED-MoE: Accelerating MoE-Based dLLM Inference via Inter-Iteration Locality-Aware Expert Offloading](https://arxiv.org/abs/2609.33385) | 2026.09 | EuroSys | MoE serving, EuroSys |
| [Faster Block-Diffusion Serving with Distribution-Free Risk Guarantees](https://arxiv.org/abs/2609.33887) | 2026.09 | Arxiv | Serving calibration |
| [Denoising Surface: Modeling and Predicting Inference Cost for Diffusion LLM Serving](https://arxiv.org/abs/2610.00499) | 2026.09 | Arxiv | Serving scheduler |
| [DynaTE: Accelerating Diffusion LLMs via Dynamic Token Execution](https://arxiv.org/abs/2610.11284) | 2026.10 | Arxiv | Hardware-software co-design |

### 5.5 Speculative Decoding & Draft Verification
> **Scope**: diffusion-based helper drafting, self-speculation, draft trees, and target verification/repair; AR-only speculative decoding is excluded.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147) | 2026.07 | Arxiv | Backfill, Confidence-scheduled drafting |
| [From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2608.02123) | 2026.08 | Arxiv | Backfill, Parent-conditioned draft trees |
| [Approximate Speculative Decoding](https://arxiv.org/abs/2608.03447) | 2026.08 | Arxiv | Backfill, Approximate verification |
| [TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2609.22098) | 2026.08 | Arxiv | Calibrated load-adaptive trees |
| [DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees](https://arxiv.org/abs/2608.13524) | 2026.08 | Arxiv | Backfill, Diffusion draft trees |
| [Verification-Aware Training for Speculative Decoding](https://arxiv.org/abs/2608.30135) | 2026.08 | Arxiv | Backfill, Verification-aware training |
| [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 2026.08 | Arxiv | Multimodal diffusion drafting |
| [DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding](https://arxiv.org/abs/2609.06498) | 2026.09 | Arxiv | Verifier-state reuse |
| [Carryover Drafting: Recycling Rejected States for Speculative Decoding](https://arxiv.org/abs/2609.14717) | 2026.09 | Arxiv | Rejected-state recycling |
| [H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache](https://arxiv.org/abs/2609.24197) | 2026.09 | Arxiv | Hybrid speculative drafter |
| [When Parallel Drafter Meets Parallel Speculative Decoding](https://arxiv.org/abs/2609.27396) | 2026.09 | Arxiv | Overlapped draft verification |
| [From Position Risks to Block Survival: Faster Generation for Diffusion Language Models](https://arxiv.org/abs/2609.33390) | 2026.09 | Arxiv | Draft correction |
| [Tsubame: Tree Replay for Diffusion-Based Speculative Decoding](https://arxiv.org/abs/2609.33652) | 2026.09 | Arxiv | Speculative drafting |
| [Reciprocal Guidance: Orchestrating Draft and Verify Budgets for Advancing the Diffusion-AR Self-Speculation Frontier](https://arxiv.org/abs/2609.34388) | 2026.09 | Arxiv | Load-aware verification |
| [BV Loss: Block Verification-Aware Loss for Block Diffusion Speculative Decoding](https://arxiv.org/abs/2609.34832) | 2026.09 | Arxiv | Drafter training objective |
| [Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding](https://arxiv.org/abs/2609.36173) | 2026.09 | Arxiv | Layer-wise drafter conditioning |
| [LongSpark: Efficient speculative decoding with a fixed-cost parallel drafter](https://arxiv.org/abs/2609.37029) | 2026.09 | Arxiv | Fixed-cost diffusion drafting |
| [DScale: Scaling Block-Diffusion Speculative Decoding with Adaptive Verification](https://arxiv.org/abs/2609.37532) | 2026.09 | Arxiv | Adaptive verification |
| [DEdit: Iterative Draft Editing for Speculative Decoding](https://arxiv.org/abs/2609.38510) | 2026.09 | Arxiv | Iterative diffusion drafting |
| [Recovering Off-Policy Supervision for Speculative Decoding](https://arxiv.org/abs/2609.38795) | 2026.09 | Arxiv | Off-policy drafter supervision |
| [UBTree: Parallel Tree Drafting via Unigram and Bigram Models for Speculative Decoding](https://arxiv.org/abs/2609.39972) | 2026.09 | Arxiv | Diffusion draft trees |
| [DRelay: Global Draft Context for Prefix-Aware Parallel Speculative Decoding Repair](https://arxiv.org/abs/2610.01439) | 2026.10 | Arxiv | Prefix-aware draft repair |
| [SharpDraft: Accelerating Long-Context Speculative Decoding with Cardinality-Aware Query Scaling](https://arxiv.org/abs/2610.05106) | 2026.10 | Arxiv | Long-context draft attention |
| [AdaSpark: Adaptive DSpark with Online Learning for Tree Verification and N-gram Fill](https://arxiv.org/abs/2610.05774) | 2026.10 | Arxiv | Online verification scheduling |
| [D-Loop: Looped Diffusion Drafting for Speculative Decoding](https://arxiv.org/abs/2610.06011) | 2026.10 | Arxiv | Looped diffusion drafter |
| [DLoop: Looped Speculative Decoding](https://arxiv.org/abs/2610.07659) | 2026.10 | Arxiv | Looped draft verification |
| [Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds](https://arxiv.org/abs/2610.10411) | 2026.10 | Arxiv | Expected decoding rounds |

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
| [Muddit: Liberating Generation Beyond Text-to-Image](https://arxiv.org/pdf/2505.23606) | 2025.05 | ICLR | Multi-modal |
| [Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564) | 2025.06 | Arxiv | Unified Generation |
| [Diffuse Everything: Multimodal Diffusion on Arbitrary Spaces](https://www.arxiv.org/abs/2506.07903) | 2025.06 | ICML | Arbitrary Spaces |
| [TBAC-UniImage: Unified Understanding and Generation by Ladder-Side Diffusion Tuning](https://arxiv.org/abs/2508.08098) | 2025.08 | Arxiv | Tencent ladder-side tuning |
| [Lumina-DiMOO: Omni Diffusion LLM for Generation](https://arxiv.org/abs/2510.06308) | 2025.10 | Arxiv | Omni-generation |
| [MMaDA-Parallel: Thinking-Aware Editing and Generation](https://arxiv.org/pdf/2511.09611) | 2025.11 | Arxiv | Parallel Multimodal |
| [DiffusionVL: Translating AR Models into Diffusion VL Models](https://arxiv.org/abs/2512.15713) | 2025.12 | Arxiv | VL Adaptation |
| [SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding](https://arxiv.org/abs/2512.14068) | 2025.12 | Arxiv | Block-diffusion VL |
| [Dream-VL: Open Vision-Language Model with Diffusion Backbone](https://arxiv.org/abs/2512.22615) | 2025.12 | Arxiv | dVLM from Dream-7B |
| [LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models](https://arxiv.org/abs/2602.14147) | 2026.02 | Arxiv | Unified RL post-training |
| [Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion](https://arxiv.org/abs/2603.06577) | 2026.03 | ICML | Any-to-any (text/speech/image) |
| [LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation](https://arxiv.org/abs/2604.20796) | 2026.04 | Arxiv | SigLIP-VQ + block diffusion |
| [Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM](https://arxiv.org/abs/2604.06832) | 2026.04 | Arxiv | AR-VLM → block-diffusion VLM |
| [Analyzing Diffusion and Autoregressive VLMs in Multimodal Embedding Space](https://arxiv.org/abs/2602.06056) | 2026.02 | Arxiv | Embedding analysis |
| [Dynin-Omni: Omnimodal Unified Large Diffusion Language Model](https://arxiv.org/abs/2604.00007) | 2026.03 | Arxiv | Omnimodal |
| [UAT: Unified Audio-Text Diffusion for Audio Generation, Editing, and Captioning](https://arxiv.org/abs/2606.04939) | 2026.06 | Arxiv | Unified Audio-Text Diffusion for Audio Generation, |
| [Efficient Reinforcement for Visual-Textual Thinking with Discrete Diffusion Model](https://arxiv.org/abs/2606.14792) | 2026.06 | Arxiv | RL, dMLLM |
| [PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models](https://arxiv.org/abs/2606.19534) | 2026.06 | Arxiv | dMLLM |
| [Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes](https://arxiv.org/abs/2607.13188) | 2026.07 | Arxiv | Training-free, Remasking |
| [Does More Retrieved Evidence Help Visual Retrieval-Augmented Generation with Diffusion Language Models?](https://arxiv.org/abs/2608.07006) | 2026.08 | Arxiv | Training-free, Retrieval |
| [LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes](https://arxiv.org/abs/2609.03796) | 2026.09 | Arxiv | Unified multimodal, Image |
| [Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding](https://arxiv.org/abs/2609.26638) | 2026.09 | Arxiv | OCR, Hybrid model |
| [BiMoGen: Bidirectional Motion-Text Generation via Unified Masked Discrete Diffusion](https://arxiv.org/abs/2609.35407) | 2026.09 | NeurIPS | Bidirectional motion-text generation |
| [GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed](https://arxiv.org/abs/2609.39600) | 2026.09 | Arxiv | Text and grounding generation |
| [Gestalt: Large Multimodal Interplay Model](https://arxiv.org/abs/2610.00576) | 2026.09 | Arxiv | Unified multimodal model |


### 6.2 Vision-Language-Action (VLA)
> **Scope note**: this section covers VLA models that *use a diffusion/masked-diffusion language model as the backbone* (dVLM-based VLA) **or** apply *discrete diffusion as the action-decoding mechanism* (not continuous diffusion action heads grafted onto an AR VLM). Pure continuous-diffusion-policy VLAs such as DiVLA (Wen et al., 2024), HybridVLA, and ProgressVLA are intentionally excluded because their language model is autoregressive — only the action head is diffusion-based.

**(a) dVLM-backbone VLA** — language backbone itself is a diffusion language model.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [LLaDA-VLA: Vision Language Diffusion Action Models](https://arxiv.org/abs/2509.06932) | 2025.09 | Arxiv | First LLaDA(d-VLM)-based VLA |
| [dVLA: Diffusion VLA with Multimodal Chain-of-Thought](https://arxiv.org/pdf/2509.25681) | 2025.09 | Arxiv | dLLM backbone + multimodal CoT |
| [Dream-VLA: Open Vision-Language-Action Model with Diffusion Backbone](https://arxiv.org/abs/2512.22615) | 2025.12 | Arxiv | dVLA from Dream-7B; first dLLM pretrained VLA |
| [MMaDA-VLA: Large Diffusion VLA with Unified Multi-Modal Instruction and Generation](https://arxiv.org/abs/2603.25406) | 2026.03 | Arxiv | Native discrete-diffusion VLA from MMaDA |
| [Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model](https://arxiv.org/abs/2609.13053) | 2026.09 | Arxiv | VLA, Diffused language branch |

**(b) Discrete-diffusion action decoding** — language backbone may still be AR-VLM, but action chunks are decoded via discrete diffusion. Closely tied to dLLM literature for inference techniques.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Discrete Diffusion VLA: Action Decoding in VLA Policies](https://arxiv.org/abs/2508.20072) | 2025.08 | ICML | Unified-transformer + discrete-diffusion actions |
| [E0: Enhancing Generalization and Fine-Grained Control in VLA Models via Tweedie Discrete Diffusion](https://arxiv.org/abs/2511.21542) | 2025.11 | Arxiv | AR-VLM backbone + Tweedie discrete diffusion on action tokens |
| [TBD-VLA: Temporal Block Diffusion Vision Language Action Model](https://arxiv.org/abs/2606.07895) | 2026.06 | Arxiv | Backfill, Discrete action tokens, Temporal blocks |
| [Humanoid Loco-Manipulation With Discrete VLA Model](https://arxiv.org/abs/2609.35709) | 2026.09 | Arxiv | Discrete action tokens, Humanoid |
| [SUAVE: Unified Video-Action Models via Masked Diffusion](https://arxiv.org/abs/2610.04009) | 2026.10 | Arxiv | Masked discrete video-action model |

### 6.3 Autonomous Driving / World Models
> **Scope note**: works that apply *discrete diffusion / masked-diffusion language modeling* to driving trajectories, action codebooks, or tokenized world states. Continuous trajectory-diffusion planners (e.g., classical Diffusion Policy applied to driving) are out of scope.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | 2023.11 | ICLR | Discrete diffusion on tokenized point-cloud world model |
| [ReflectDrive: Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving](https://arxiv.org/abs/2509.20109) | 2025.09 | Arxiv | dLLM finetuned on discretized 2D driving space |
| [Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion](https://arxiv.org/abs/2602.20577) | 2026.02 | Arxiv | Discrete action codebook + masked diffusion |
| [Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving](https://arxiv.org/abs/2605.23163) | 2026.05 | Arxiv | Block-diffusion VLA, speculative scaffold decoding |
| [WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA](https://arxiv.org/abs/2608.01035) | 2026.08 | Arxiv | Backfill, Driving VLA, AR-to-diffusion distillation |

---

## 7. Agentic & Tool-Use dLLMs
> **New section** — emerging line: how dLLMs behave as agents (planning, multi-turn, tool calling). Critical for connecting dLLMs to robotics and physical-AI agent stacks.

| Paper Title | Year | Venue | Remark |
| :--- | :---: | :---: | :--- |
| [The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check](https://arxiv.org/abs/2601.12979) | 2026.01 | Arxiv | Embodied + tool-call eval |
| [Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation](https://arxiv.org/abs/2601.07152) | 2026.01 | Arxiv | Multi-agent RL |
| [DLLM Agent: See Farther, Run Faster](https://arxiv.org/abs/2602.07451) | 2026.02 | Arxiv | dLLM-as-agent comparison |
| [Continuous Interaction Diffusion: A Diffusion-Native Runtime for Asynchronous Tool-Augmented Reasoning](https://arxiv.org/abs/2608.10438) | 2026.08 | Arxiv | Backfill, Conceptual tool runtime |
| [LLaDA-UI: Bringing Block-wise Diffusion to Vision-Language GUI Agents](https://arxiv.org/abs/2609.13287) | 2026.09 | Arxiv | GUI agent |
| [Does This Action Still Explain the Task? Reverse Scoring for Diffusion Language Model Agents](https://arxiv.org/abs/2609.38536) | 2026.09 | Arxiv | Agentic generation, Reflection |
| [Plan-and-Patch: Diffusion Language Models for Agentic Planning](https://arxiv.org/abs/2610.10786) | 2026.10 | Arxiv | Agentic plan repair |

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
| [Empirical Analysis of Decoding Biases in Masked Diffusion Models](https://arxiv.org/abs/2508.13021) | 2025.10 | ACL | Decoding Bias, UNCODE (formerly PC-Sampler) |
| [Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models](https://arxiv.org/abs/2510.04146v1) | 2025.10 | Arxiv | Speed Analysis |
| [On the Role of Discreteness in Diffusion LLMs](https://arxiv.org/abs/2512.22630v1) | 2025.12 | Arxiv | Speed Analysis |
| [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](https://arxiv.org/abs/2510.04767) | 2025.10 | ICLR |  |
| [Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration](https://arxiv.org/abs/2603.02760) | 2026.03 | ACL | Self-evaluation, Generalization analysis |
| [On the Role of Discreteness in Diffusion LLMs](https://arxiv.org/abs/2512.22630v1) | 2025.12 | Arxiv | Discreteness analysis |
| [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](https://arxiv.org/abs/2510.04767) | 2025.10 | ICLR | Benchmark |
| [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276) | 2025.11 | Arxiv | Data learner analysis |
| [Adaptation to Intrinsic Dependence in Diffusion Language Models](https://arxiv.org/abs/2602.20126) | 2026.02 | Arxiv | Distribution-agnostic schedule theory |
| [Confidence-Based Decoding is Provably Efficient for Diffusion Language Models](https://arxiv.org/abs/2603.22248) | 2026.03 | Arxiv | First theory of confidence-based decoding |
| [Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding?](https://arxiv.org/abs/2602.23225) | 2026.02 | Arxiv | AR-like decoding analysis, NAP approach |
| [Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects](https://arxiv.org/abs/2603.02333) | 2026.03 | Arxiv | Memorization, privacy analysis |
| [Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](https://arxiv.org/abs/2603.07475) | 2026.03 | Arxiv | Layer skipping, representation analysis |
| [Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison](https://arxiv.org/abs/2603.22075) | 2026.03 | Arxiv | Controlled AR vs MDM empirical study |
| [Why Gaussian Diffusion Models Fail on Discrete Data?](https://arxiv.org/abs/2604.02028) | 2026.04 | Arxiv | Multimodal sampling interval theory |
| [Generative Frontiers: Why Evaluation Matters for Diffusion Language Models](https://arxiv.org/abs/2604.02718) | 2026.04 | Arxiv | Evaluation methodology critique |
| [Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](https://arxiv.org/abs/2604.10556) | 2026.04 | ACL Findings | Hallucination patterns analysis |
| [Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](https://arxiv.org/abs/2604.10567) | 2026.04 | ICML | Proximity bias analysis |
| [Measuring Temporal Linguistic Emergence in Diffusion Language Models](https://arxiv.org/abs/2604.23235) | 2026.04 | Arxiv | Temporal probing, linguistic emergence |
| [Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data](https://arxiv.org/abs/2604.26841) | 2026.04 | Arxiv | Associative memory theory |
| [Understanding and Accelerating the Training of Masked Diffusion Language Models](https://arxiv.org/abs/2605.13026) | 2026.05 | Arxiv | Bell-shaped time sampling, training speed |
| [Uncertainty Quantification for Large Language Diffusion Models](https://arxiv.org/abs/2605.14570) | 2026.05 | Arxiv | UQ, hallucination detection for dLLMs |
| [Continuous Language Diffusion as a Decoder-Interface Problem](https://arxiv.org/abs/2606.08810) | 2026.06 | Arxiv | - |
| [Neither Parallel Nor Sequential: How DiffusionGemma Actually Commits Tokens](https://arxiv.org/abs/2606.14620) | 2026.06 | Arxiv | MoE |
| [Diffusion Language Models: An Experimental Analysis](https://arxiv.org/abs/2606.19475) | 2026.06 | Arxiv | Unmasking order |
| [Understanding Parallel Samplers in Masked Diffusion via Random Walks on Graphs](https://arxiv.org/abs/2606.22976) | 2026.06 | Arxiv | Unmasking order |
| [When Top-1 Fails: Calibrating LoRA Monitors for Masked Diffusion LMs](https://arxiv.org/abs/2606.24119) | 2026.06 | Arxiv | Calibrating LoRA Monitors for Masked Diffusion |
| [Understanding Evaluation Illusion in Diffusion Large Language Models](https://arxiv.org/abs/2606.29228) | 2026.06 | Arxiv | - |
| [CaRE Compute-aware Remasking Evaluation Protocol for Masked Diffusion Language Models](https://arxiv.org/abs/2607.24763) | 2026.06 | Arxiv | Remasking, Unmasking order |
| [Where Quality Breaks in Compressed Short-Text Generation: Staged Bottleneck Localization](https://arxiv.org/abs/2607.24176) | 2026.07 | FRUCT | Staged codec-vs-generation bottleneck diagnosis; [project](https://aogavrilov.com/publications/where-quality-breaks/) |
| [Low Perplexity is Repetition: A One-Dimensional Self-Conditioning Attractor in Continuous Diffusion LMs](https://arxiv.org/abs/2607.00588) | 2026.07 | Arxiv | A One-Dimensional Self-Conditioning Attractor in Continuous |
| [Subliminal Clocks: Latent Time Modelling in Diffusion Language Models](https://arxiv.org/abs/2607.01774) | 2026.07 | EMNLP | Latent Time Modelling in Diffusion Language |
| [Accepted Prefixes Are Not All You Need: A Negative Result on PEFT-Based Block-Diffusion Drafting](https://arxiv.org/abs/2607.12422) | 2026.07 | Arxiv | Speculative decoding |
| [Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models](https://arxiv.org/abs/2607.15893) | 2026.07 | Arxiv | A Mechanistic Analysis of In-Context Learning |
| [Beyond the Bidirectional Promise: Re-evaluating the Robustness of Diffusion Language Models](https://arxiv.org/abs/2607.27386) | 2026.07 | Arxiv | Safety attack |
| [Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models](https://arxiv.org/abs/2607.29079) | 2026.07 | Arxiv | Training-free, KV cache |
| [What Does a Discrete Diffusion Model Learn?](https://arxiv.org/abs/2607.05381) | 2026.07 | Arxiv | Backfill, Discrete diffusion theory |
| [Answer First, Reason Later: When Commitment Order Costs Accuracy in Diffusion Language Models](https://arxiv.org/abs/2608.05687) | 2026.08 | Arxiv | When Commitment Order Costs Accuracy in |
| [Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models](https://arxiv.org/abs/2608.06529) | 2026.08 | Arxiv | Why Predictive Feedback Fails in Diffusion |
| [Scaling Inherently Interpretable Language Models](https://arxiv.org/abs/2608.07594) | 2026.08 | Arxiv | - |
| [Unsure but Certain: Uncovering the Representation-Confidence Gap in Diffusion Language Models](https://arxiv.org/abs/2608.08791) | 2026.08 | Arxiv | Uncovering the Representation-Confidence Gap in Diffusion |
| [Minimax Optimality of Score-Entropy Discrete Diffusion](https://arxiv.org/abs/2608.20635) | 2026.08 | Arxiv | - |
| [Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling](https://arxiv.org/abs/2608.25505) | 2026.08 | Arxiv | - |
| [Separating quantum circuits from classical LLMs](https://arxiv.org/abs/2608.03962) | 2026.08 | Arxiv | Backfill, Quantum complexity |
| [Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting](https://arxiv.org/abs/2608.27339) | 2026.08 | Arxiv | Backfill, Drafting information floors |
| [Reliability Challenges in Diffusion Vision-Language Models](https://arxiv.org/abs/2609.01318) | 2026.09 | Arxiv | Multimodal reliability evaluation |
| [Evaluating Losslessness in Speculative Decoding Under Finite-Precision Inference](https://arxiv.org/abs/2609.15504) | 2026.09 | Arxiv | Finite-precision analysis |
| [What does FFN compression change downstream? Same-state causal restoration in diffusion language models](https://arxiv.org/abs/2609.31685) | 2026.09 | Arxiv | FFN causal diagnosis |
| [Parallelism, critical windows, and separations among diffusion language models](https://arxiv.org/abs/2609.20539) | 2026.09 | Arxiv | Parallelism theory |
| [Limits of Confidence in Diffusion](https://arxiv.org/abs/2609.20581) | 2026.09 | Arxiv | Confidence and dependence theory |
| [Schedule optimization for tau-leaping in masked discrete diffusion](https://arxiv.org/abs/2609.21960) | 2026.09 | Arxiv | Tau-leaping theory |
| [Optimizers for Diffusion Models: A Controlled Benchmark](https://arxiv.org/abs/2609.23055) | 2026.09 | Arxiv | Optimizer benchmark |
| [Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis](https://arxiv.org/abs/2609.30841) | 2026.09 | Arxiv | Safety landscape analysis |
| [Does Uniform Discrete Diffusion Need Time?](https://arxiv.org/abs/2609.30977) | 2026.09 | Arxiv | Uniform-diffusion time analysis |
| [The Detectability Gap: Hidden Heterogeneity in Hallucination Detection Across Language Models](https://arxiv.org/abs/2609.35860) | 2026.09 | NeurIPS Workshop | Detectability analysis |
| [Unmask the State: When Does State Adaptation Matter for Masked Diffusion Language Models](https://arxiv.org/abs/2609.33355) | 2026.09 | Arxiv | Denoising-state analysis |
| [Simple Diffusion Language Models Are More Effective Few-Step Generators Than Reported](https://arxiv.org/abs/2609.33947) | 2026.09 | Arxiv | Few-step evaluation |
| [Before the Token Commits: Trajectory-Level Benchmarking of Visual Hallucinations in Diffusion VLMs](https://arxiv.org/abs/2609.34772) | 2026.09 | Arxiv | Multimodal hallucination analysis |
| [Why Deterministic PRM Guidance Underperforms in Discrete Diffusion Reasoning](https://arxiv.org/abs/2609.35472) | 2026.09 | NeurIPS | Compute-matched guidance evaluation |
| [A Heckler in the Hidden State: Correctness Signals in Diffusion Language Models](https://arxiv.org/abs/2609.36783) | 2026.09 | Arxiv | Code correctness probes |
| [How Should Diffusion Language Models Edit Code?](https://arxiv.org/abs/2609.38257) | 2026.09 | Arxiv | Code editing evaluation |
| [Two Clocks in Diffusion MLLMs: When Answers Stabilize Before Rationales Unfold](https://arxiv.org/abs/2610.00953) | 2026.10 | Arxiv | Multimodal commitment analysis |
| [Temporally-Resolved Token Attribution Reveals the Generation Dynamics of Diffusion Language Models](https://arxiv.org/abs/2610.01177) | 2026.10 | Arxiv | Token attribution |
| [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 2026.10 | Arxiv | Text-distribution evaluation |
| [Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size](https://arxiv.org/abs/2610.07655) | 2026.10 | Arxiv | Uniform-diffusion minimax theory |
| [Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval](https://arxiv.org/abs/2610.08716) | 2026.10 | Arxiv | Controlled retrieval comparison |
| [The Lattice of Transition Laws](https://arxiv.org/abs/2610.11216) | 2026.10 | Arxiv | Transition-law theory |

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
| [Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models](https://arxiv.org/abs/2509.23146v1) | 2025.09 | Arxiv | Control |
| [Syntax-Guided Diffusion Language Models with User-Integrated Personalization](https://arxiv.org/abs/2510.01028v1) | 2025.10 | Arxiv | Personalization |
| [TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models](https://arxiv.org/abs/2510.01274v1) | 2025.10 | Arxiv | Hallucination |
| [Don't Let It Fade: Preserving Edits via Token Timestep Allocation](https://arxiv.org/abs/2510.26200) | 2025.10 | NeurIPS | Control |
| [Diffusion Language Models for Speech Recognition](https://arxiv.org/abs/2604.14001) | 2026.04 | Arxiv | ASR rescoring (MDLM/USDM) |
| [CAGenMol: Condition-Aware Diffusion Language Model for Goal-Directed Molecular Generation](https://arxiv.org/abs/2604.11483) | 2026.04 | Arxiv | Molecular generation |
| [TabDLM: Free-Form Tabular Data Generation via Joint Numerical-Language Diffusion](https://arxiv.org/abs/2602.22586) | 2026.02 | Arxiv | Tabular generation, masked diffusion |
| [Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models](https://arxiv.org/abs/2603.17677) | 2026.03 | Arxiv | RAG, retrieval-prior conflict handling |
| [DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning](https://arxiv.org/abs/2603.16459) | 2026.03 | EMNLP Findings | Hallucination detection |
| [Unlocking Prompt Infilling Capability for Diffusion Language Models](https://arxiv.org/abs/2604.03677) | 2026.04 | Arxiv | Prompt infilling via full-sequence masking |
| [DiffuMask: Diffusion Language Model for Token-level Prompt Pruning](https://arxiv.org/abs/2604.06627) | 2026.04 | Arxiv | Prompt compression |
| [BiMol-Diff: A Unified Diffusion Framework for Molecular Generation and Captioning](https://arxiv.org/abs/2604.24089) | 2026.04 | Arxiv | Molecule generation + captioning |
| [HIVE: Hidden-Evidence Verification for Hallucination Detection in Diffusion Large Language Models](https://arxiv.org/abs/2604.26139) | 2026.04 | Arxiv | Hallucination detection, denoising dynamics |
| [Chainwash: Multi-Step Rewriting Attacks on Diffusion Language Model Watermarks](https://arxiv.org/abs/2605.05503) | 2026.05 | Arxiv | Watermark attack, security |
| [DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models](https://arxiv.org/abs/2605.07210) | 2026.05 | Arxiv | Dense retrieval with dLLMs |
| [Guidance Is Not a Hyperparameter: Learning Dynamic Control in Diffusion Language Models](https://arxiv.org/abs/2605.07701) | 2026.05 | Arxiv | Dynamic CFG via RL |
| [Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.10971) | 2026.05 | Arxiv | Adaptive guidance schedule, SAE analysis |
| [Constrained Code Generation with Discrete Diffusion](https://arxiv.org/abs/2605.16829) | 2026.05 | Arxiv | Neurosymbolic constrained code generation |
| [Prompt Compression in Diffusion Large Language Models: Evaluating LLMLingua-2 on LLaDA](https://arxiv.org/abs/2605.17932) | 2026.05 | Arxiv | Prompt compression study |
| [Machine Unlearning for Masked Diffusion Language Models](https://arxiv.org/abs/2605.18253) | 2026.05 | Arxiv | MDU, unlearning framework |
| [Knowledge Editing in Masked Diffusion Language Models](https://arxiv.org/abs/2606.03924) | 2026.06 | EMNLP | Unmasking order, Knowledge editing |
| [MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models](https://arxiv.org/abs/2606.04027) | 2026.06 | Arxiv | Safety attack |
| [Global Sketch-Based Watermarking for Diffusion Language Models](https://arxiv.org/abs/2606.04486) | 2026.06 | Arxiv | Watermarking |
| [Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](https://arxiv.org/abs/2606.04535) | 2026.06 | ACL | Training-free |
| [Self-Augmenting Retrieval for Diffusion Language Models](https://arxiv.org/abs/2606.06474) | 2026.06 | ICML | Training-free, Serving |
| [TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering](https://arxiv.org/abs/2606.08408) | 2026.06 | Arxiv | Automatic In-place Text Sequence Modification via |
| [Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts](https://arxiv.org/abs/2606.12807) | 2026.06 | Arxiv | Few-step, Remasking |
| [TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for Masked Diffusion Language Models](https://arxiv.org/abs/2606.12841) | 2026.06 | Arxiv | MoE, Training-free |
| [Self-Generated Error Training for Token Editing in Diffusion Language Models](https://arxiv.org/abs/2606.17175) | 2026.06 | Arxiv | - |
| [TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning](https://arxiv.org/abs/2606.31166) | 2026.06 | Arxiv | Diffusion Language Models for Text-Attributed Graph |
| [TRE: Training-Free Hallucination Detection for Diffusion Language Models](https://arxiv.org/abs/2607.22661) | 2026.06 | Arxiv | Training-free, Hallucination |
| [Diffusion-GR2: Diffusion Generative Reasoning Re-ranker](https://arxiv.org/abs/2607.01170) | 2026.07 | Arxiv | Distillation, Serving |
| [Discrete Diffusion Language Models for Interactive Radiology Report Drafting](https://arxiv.org/abs/2607.01436) | 2026.07 | Arxiv | MoE, Clinical |
| [Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata](https://arxiv.org/abs/2607.07026) | 2026.07 | Arxiv | Remasking, Block diffusion |
| [Audio-Native Speech Recognition with a Frozen Discrete-Diffusion Language Model](https://arxiv.org/abs/2607.13013) | 2026.07 | Arxiv | MoE |
| [Diffusion Language Model for Recommendation](https://arxiv.org/abs/2607.21519) | 2026.07 | Arxiv | - |
| [Exploring More to Solve More: Boosting Diversity in Text Diffusion Models via Entropy-Based Guidance](https://arxiv.org/abs/2608.00024) | 2026.07 | Arxiv | Training-free |
| [DeMTS: Denoising Trajectories as Multivariate Time Series for Hallucination Detection in Diffusion Language Models](https://arxiv.org/abs/2608.14632) | 2026.07 | Arxiv | Hallucination |
| [EchoChange: A Diffusion Language Model with Dual Pass Remasking for Factual Remote Sensing Disaster Change Captioning](https://arxiv.org/abs/2608.01856) | 2026.08 | Arxiv | Remasking, dMLLM |
| [Stoicheia: Character-Level Masked Diffusion for Ancient Greek Textual Restoration, Parsing, and Metrical Scansion](https://arxiv.org/abs/2608.07249) | 2026.08 | Arxiv | Character-Level Masked Diffusion for Ancient Greek |
| [Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits](https://arxiv.org/abs/2608.07430) | 2026.08 | Arxiv | Safety attack |
| [Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models](https://arxiv.org/abs/2608.08082) | 2026.08 | Arxiv | When Classifier-Free Guidance Becomes Unnecessary in |
| [Diffuse to Compress: Leveraging Diffusion LMs for Lossless Compression](https://arxiv.org/abs/2608.11249) | 2026.08 | Arxiv | Serving, Compression |
| [Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo](https://arxiv.org/abs/2608.20123) | 2026.08 | Arxiv | - |
| [SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields](https://arxiv.org/abs/2608.20839) | 2026.08 | EMNLP Findings | Watermarking, Unmasking order |
| [Length-Adaptive Decoding for Masked Diffusion Machine Translation](https://arxiv.org/abs/2608.22274) | 2026.08 | EMNLP | Training-free, Unmasking order |
| [DRRG: A Discrete Diffusion Framework for Radiology Report Generation](https://arxiv.org/abs/2608.24105) | 2026.08 | Arxiv | Clinical |
| [Wiener Representation Filtering for VLM Hallucination Suppression](https://arxiv.org/abs/2608.08167) | 2026.08 | Arxiv | Backfill, Hallucination suppression |
| [BiGraph-Diffuse: A Bidirectional Diffusion Language Model with Graph-Structured Retrieval For Mental Health Counseling](https://arxiv.org/abs/2609.29519) | 2026.08 | Arxiv | Counseling, Graph retrieval |
| [Weak Ties, Strong Signals: Efficient Training Data Detection in Diffusion LLMs via Independent Token Sampling](https://arxiv.org/abs/2609.22145) | 2026.08 | Arxiv | Training-data detection |
| [DiffPDE: Masked Diffusion Language Models as PDE Solver](https://arxiv.org/abs/2608.30532) | 2026.08 | Arxiv | Backfill, PDE code repair |
| [Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models](https://arxiv.org/abs/2609.00495) | 2026.08 | EMNLP | Training-free safety commitment |
| [Membership Inference in Fine-tuned Diffusion Language Models via Token-level Memorization Asymmetry](https://arxiv.org/abs/2609.00873) | 2026.09 | Arxiv | Membership inference |
| [DIVA: Exploiting Cross-Step Conditional Propagation for Visual Jailbreaks in Discrete Diffusion Vision-Language Models](https://arxiv.org/abs/2609.05525) | 2026.09 | EMNLP | Safety, Jailbreak |
| [In-Place Instruction Following in Diffusion Language Models](https://arxiv.org/abs/2609.07160) | 2026.09 | Arxiv | Controllable generation |
| [Decomposition-Guided Diffusion Language Models for Inertial Confinement Fusion Prediction](https://arxiv.org/abs/2609.07756) | 2026.09 | Arxiv | LLaDA-based scientific prediction |
| [DenMark: Robust Semantic Watermarking for Diffusion Language Models](https://arxiv.org/abs/2609.14257) | 2026.09 | Arxiv | Semantic watermarking |
| [Efficient One-to-Many Translation with Joint Multi-Stream Diffusion](https://arxiv.org/abs/2609.16312) | 2026.09 | Arxiv | Machine translation |
| [How to Guide Your Language Flow](https://arxiv.org/abs/2609.19356) | 2026.09 | Arxiv | Continuous-text guidance |
| [The Temporal Tug-of-War: Visualizing and Detecting RAG Conflicts in Diffusion Models via Trajectory Variance](https://arxiv.org/abs/2609.31684) | 2026.09 | EMNLP Workshop | RAG conflict detection |
| [LADDER: Graph-Guided Diffusion Language Models for Efficient Multi-Hop Reasoning](https://arxiv.org/abs/2609.24346) | 2026.09 | Arxiv | Graph retrieval guidance |
| [Constraints Are Graphs, Not Chains: Exact Decoding for Diffusion Language Models](https://arxiv.org/abs/2609.32900) | 2026.09 | Arxiv | Constraint guidance |
| [Safety Reconstructed: Generative Modeling via Masked Diffusion Builds Strong Safety Guardrails](https://arxiv.org/abs/2609.33634) | 2026.09 | Arxiv | Safety, Prompt rewriting |
| [One Sequence, Many Decodings: CAGenMol-2 Recasts Drug Design as Masked Molecular Inference](https://arxiv.org/abs/2609.34301) | 2026.09 | Arxiv | Molecular text generation |
| [TANGO: Watermarking Masked Diffusion Language Models in Token Pairs](https://arxiv.org/abs/2609.35224) | 2026.09 | Arxiv | Watermarking |
| [Twist, Don't Tilt: Trajectory-Exact Constrained Decoding for Masked Diffusion Models](https://arxiv.org/abs/2609.35609) | 2026.09 | Arxiv | Constraint guidance |
| [Grab a Coffee: Future-Aware Guidance for Discrete Diffusion with Compiled Objectives](https://arxiv.org/abs/2609.35924) | 2026.09 | Arxiv | Compiled constraint guidance |
| [FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models](https://arxiv.org/abs/2609.36202) | 2026.09 | Arxiv | Reward guidance |
| [HyperZip: Efficient Data Compression through Personalized Diffusion LLMs with Hypernetworks](https://arxiv.org/abs/2609.36357) | 2026.09 | Arxiv | Lossless data compression |
| [Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering](https://arxiv.org/abs/2610.05894) | 2026.10 | Arxiv | Safety, Activation steering |

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
