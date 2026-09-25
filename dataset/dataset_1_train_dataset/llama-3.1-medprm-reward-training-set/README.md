---
license: mit
library_name: datasets
tags:
- medical
- biomedical
- process-reward-model
- medical-ai
- retrieval-augmented-generation
task_categories:
- text-generation
---

# Med-PRM-Reward (Version 1.0)

🚀 Med-PRM-Reward is among the first Process Reward Models (PRMs) specifically designed for the medical domain. Unlike conventional PRMs, it enhances its verification capabilities by integrating clinical knowledge through retrieval-augmented generation (RAG). Med-PRM-Reward demonstrates exceptional performance in scaling-test-time computation, particularly outperforming majority‐voting ensembles on complex medical reasoning tasks. Moreover, its scalability is not limited to Llama-3.1-8B-Instruct: it delivers similarly outstanding results in scaling-test-time computation across multiple other medical‐specialized models. Notably, when combined with llama-3-meerkat-8b-v1.0, it became the first sub-10B small language model to surpass a score of 80 on the MedQA (4-option) benchmark.


📄 Paper: [Med-PRM-Reward: Medical Reasoning Models with Stepwise, Guideline‑verified Process Rewards](https://arxiv.org/abs/2506.11474)

💻 Code: https://github.com/eth-medical-ai-lab/Med-PRM

🌐 Project Page: https://Med-PRM.github.io