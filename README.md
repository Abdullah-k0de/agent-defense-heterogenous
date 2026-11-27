# Architectural Diversity in Multi-Agent LLM Systems (Agent Defense)

Empirical evaluation repository for investigating **passive inter-model architectural diversity** as a structural defense without extra defense-specific inference passes against the propagation of **Indirect Prompt Injections (IPI)** in sequential Multi-Agent Systems (MAS).

---

## Project Overview

Homogeneous multi-agent workflows—where all agents share identical model weights, tokenizers, and alignment boundaries—can suffer from correlated failure modes. Malicious directives that bypass an initial ingestor can propagate across downstream agents without attenuation.

This study implements a **5-hop sequential multi-agent pipeline** using five open-weight model families and a unified 40-tool catalog to assess whether crossing disparate architectural boundaries passively attenuates semantic payload propagation without dedicated firewall or re-execution passes.

### Evaluated Model Families (7B–9B Scale)
* **Llama-3.1:8B** (Meta) – Grouped-Query Attention (GQA), 128k BPE vocabulary
* **Mistral-7B-v0.3** (Mistral) – GQA (`sliding_window = null` in v0.3), 32,768 SentencePiece vocabulary, permissive instruction-following priors
* **Qwen-2.5:7B** (Alibaba) – Untied multilingual embeddings, 152k vocabulary
* **Gemma-2:9B-IT** (Google) – Interleaved local/global attention with soft logit capping, conservative safety alignment, 256k vocabulary
* **GLM-4:9B-Chat** (Zhipu) – Bilingual autoregressive chat architecture, 151k vocabulary

---

## Pipeline Architecture

The sequential multi-agent pipeline comprises five specialized functional roles:

$$\mathcal{P} = \langle A_1, A_2, A_3, A_4, A_5 \rangle$$

1. **Ingestor ($A_1$):** Summarizes external, potentially contaminated user/environment data. Anchored to `llama3.1:8b` across all configurations to control initial ingestion.
2. **Planner ($A_2$):** Decomposes task requirements into structured execution plans.
3. **Writer ($A_3$):** Maps decomposition into tool-specific parameter calls.
4. **Reviewer ($A_4$):** Audits parameters against schemas and safety constraints.
5. **Executor ($A_5$):** Emits final structured JSON tool-call proposals (evaluated offline in a simulated environment without dispatching live external APIs).

### Experimental Configurations ($N = 102$ Paired Trials from 17 Parent Tasks)
* **Benchmark Hierarchy:** 17 parent user tasks from InjecAgent $\times$ 2 threat vectors (`DH`, `DS`) = 34 base payload scenarios $\times$ 3 framings (`Base`, `Authoritative`, `Urgency`) = **102 trials** (51 Direct Harm, 51 Data Stealing).
* **Control (Homogeneous):** `Llama` $\to$ `Llama` $\to$ `Llama` $\to$ `Llama` $\to$ `Llama`
* **Test 1 (`lmqgg`):** `Llama` $\to$ `Mistral` $\to$ `Qwen` $\to$ `Gemma` $\to$ `GLM`
* **Test 2 (`lgmgq`):** `Llama` $\to$ `Gemma` $\to$ `Mistral` $\to$ `GLM` $\to$ `Qwen`

---

## Repository Structure

```plaintext
agent-defense/
├── Phase 2/                          # Adversarial Benchmark & Framing Synthesis
│   ├── step1_ingestion.py            # Extraction and preprocessing of 17 parent tasks
│   ├── step1_prepared_data.json      # Ingested benchmark seed tasks
│   ├── Generator/                    # LLM-in-the-loop candidate payload generation
│   │   ├── generator_prompt.txt
│   │   └── step2_candidate_payloads.json
│   ├── Discriminator/                # Dual-pass auditor filtering (score >= 4/5)
│   │   ├── 1st pass/
│   │   ├── 2nd pass/
│   │   └── final_validated_payloads.json
│   └── Psychological-Framing/        # 3x factorial expansion (Base, Authoritative, Urgency)
│       ├── matrix_compilation_framings.py
│       └── eval_matrix_v2.csv
├── Phase 3/                          # Sequential Pipeline Execution
│   ├── build_benchmark_tools.py      # Synthesizes unified 40-tool catalog
│   ├── benchmark_tools.json          # 40-tool definitions (benign, attack, decoy)
│   ├── orchestrator.py               # Deterministic multi-hop pipeline orchestrator
│   ├── phase3_execution_logs_lmqgg.json
│   └── phase3_execution_logs_lgmgq.json
├── Phase 4/                          # Hybrid Two-Tier Evaluation Oracle
│   ├── score_eval.py                 # Tier 1 judge + Tier 2 JSON parser + statistical tests
│   └── phase4_scored_results.json    # Paired case-level results (N = 102)
├── Phase 5/                          # Publication Figure Generation
│   ├── generate_clean_ieee_v2.py     # Empirical results & statistical plots
│   └── generate_phase5_visualizations.py # Four-phase methodology overview diagram
├── LICENSE                           # MIT License
├── requirements.txt                  # Pinned Python package dependencies
├── test_models.py                    # Local Ollama endpoint verification script
├── tools.json                        # Raw tool definitions catalog
└── user_cases.jsonl                  # Base user interaction scenarios
```

---

## Setup & Execution

### Prerequisites
* **Python:** 3.12+
* **Local Inference:** [Ollama](https://ollama.com/) with the exact tags referenced by the scripts:
  ```bash
  ollama pull llama3.1:8b
  ollama pull mistral:7b
  ollama pull qwen2.5:7b
  ollama pull gemma2:9b
  ollama pull glm4:9b
  ollama pull deepseek-r1:14b
  ```

### Installation
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Reproducing Pipeline Execution & Statistical Analysis
```bash
# Verify local Ollama model endpoints
python test_models.py

# Run Phase 3 Multi-Hop Orchestration for both permutations
python "Phase 3/orchestrator.py" --permutation lmqgg --all
python "Phase 3/orchestrator.py" --permutation lgmgq --all

# Run Phase 4 Hybrid Evaluation Oracle (or recompute aligned statistics offline)
python "Phase 4/score_eval.py"
python "Phase 4/score_eval.py" --analyze-only
```

---

## Evaluation Framework

* **Tier 1 (Intermediate Hops 1–4):** `deepseek-r1:14b` reasoning judge evaluates intermediate semantic survival ($S_{i,k} \in \{0, 1\}$).
* **Tier 2 (Hop 5 Terminal Proposal Check):** Deterministic JSON parser checks proposed Hop 5 tool calls against ground-truth unauthorized mutation and exfiltration rules ($E_{i,5} \in \{0, 1\}$) without executing external APIs.
* **Deterministic Controls:** Greedy decoding (`temperature = 0.0`, `num_ctx = 8192`) applied uniformly across all pipeline hops.
