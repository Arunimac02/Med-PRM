Source: `dataset/dataset_4_scored_dataset/Llama-3.1-8B-Instruct-gemini_label-filter_yes-ep3-20260731_191535-RAG_yes__sol64_2_test_dataset.json`

AgentClinic columns are mapped from `nejm` / `osce` data_source values as stand-ins (no `agent_clinic` data_source exists in this scored dataset) -- treat as approximate, not a confirmed match to the paper's AgentClinic benchmark.

| Benchmark | N | Best-of-N | SC | SC+RM hybrid (PRM_min_score) | SC+RM hybrid (PRM_score) |
|---|---|---|---|---|---|
| MedQA-4 | 1273 | 76.98% | 75.18% | 77.69% | 77.93% |
| MedQA-5 | 1273 | 73.37% | 70.7% | 72.82% | 72.82% |
| MedMCQA | 500 | 65.6% | 63.4% | 65.8% | 65.8% |
| MMLU-Med | 1089 | 82.55% | 81.63% | 82.74% | 82.64% |
| DDXPlus | 500 | 78.4% | 75.2% | 76.4% | 76.6% |
| AgentClinic (NEJM*) | 120 | 55.83% | 51.67% | 53.33% | 53.33% |
| AgentClinic (MedQA*) | 214 | 74.3% | 74.77% | 75.7% | 75.7% |
| MedQA* Average | - | 75.18% | 72.94% | 75.25% | 75.38% |
| Average | - | 72.43% | 70.36% | 72.07% | 72.12% |
