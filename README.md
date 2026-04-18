# StateTune

**Transforming LLM-Assisted EDA Flow Tuning into a Stateful, Closed-Loop Process**

StateTune reformulates LLM-assisted EDA flow parameter tuning as a closed-loop, state-carrying process. Its optimizer state is a **typed, evidence-gated persistent optimization memory** that is updated by every evaluation and shared between candidate generation and budget allocation. On top of this memory, an **EHVI-guided, runtime-aware promotion policy** ranks quick-stage candidates by expected Pareto frontier gain per unit of runtime cost.

Evaluated on a Cadence Genus/Innovus industrial flow across six benchmark blocks (2 nodes × 3 designs), StateTune attains the **best final hypervolume on all six blocks**, the best WNS and best power on all six, and best-or-tied-best area on all six, against five baselines (BO-qEHVI, Optuna-TPE, Random, RankTuner ICCAD'24, CROP ICCAD'25).

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Reproducing the Paper](#reproducing-the-paper)
- [Extending StateTune](#extending-statetune)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Persistent optimization memory** `M_t = {P_t, R_t, F_t, S_t, H_t, Π_t, T_t}` — typed artifacts (prior, rules, failures, sensitivities, history, Pareto, runtime) carried across iterations, not a growing prompt transcript.
- **Evidence gating** — hard rules require `k = 12` corroborating full-valid observations before activation; active hard rules are capped at 8 and periodically trimmed. Removing gating costs 43.4% hypervolume.
- **EHVI-guided runtime-aware promotion** — candidates scored by `EHVI(x) / (t̂_ucb(x) + ε)` using a multi-fidelity GP over quick/full stages and a *k*NN runtime model.
- **Five-mode knowledge-guided search** — `explore / exploit / diversify / repair / promote`, conditioned on memory, with frontier-driven buffer refresh.
- **Dual-model inference** — a reasoning model for CoT failure/sensitivity diagnosis, a fast generation model for parameter formatting.
- **Multi-fidelity scheduling** — configurable quick-stage endpoint (CTS by default); full-stage routing runs only when promoted.
- **Fair baseline framework** — re-implementations of RankTuner (ICCAD'24) and CROP (ICCAD'25) share the same execution, promotion, and budget-accounting pipelines.

---

## Repository Layout

```
.
├── benchmark/               # RTL + SDC for JPEG / AES / IBEX (ASAP7 & NanGate45)
│   ├── aes/
│   ├── ibex/
│   └── jpeg/
├── configs/                 # One directory per (PDK × design) pair
│   ├── asap7_jpeg/          #   baseline_tuner.yaml  = StateTune
│   ├── asap7_aes/           #   baseline_bo.yaml / baseline_optuna.yaml / ...
│   ├── asap7_ibex/          #   baseline_ranktuner.yaml / baseline_crop.yaml
│   ├── nangate_jpeg/
│   ├── nangate_aes/
│   ├── nangate_ibex/
│   ├── ablation_asap7_jpeg/            # ablation variants (see Section 5.2)
│   └── configs_sensitivity_asap7_jpeg/ # k-threshold sweep (Section 5.3)
├── baselines/               # CROP / RankTuner proposer implementations
│   ├── crop_proposer.py
│   ├── ranktuner_proposer.py
│   └── run_baseline.py
├── src/EDATuner/
│   ├── main.py              # Main tuning loop
│   ├── agents/              # Knowledge Agent, MO-EHVI agent, tuner interface
│   ├── knowledge/           # M_t components: prior, failure, sensitivity, rules, RAG
│   ├── llm/                 # LLM client, CoT parser, token tracker
│   ├── optimization/        # Candidate buffer, k-NN runtime model
│   ├── runners/             # Cadence / OpenROAD backends, remote scheduler
│   └── utils/
├── tools/                   # State recovery helpers
│   ├── recover_state.py
│   └── recover_baseline_state.py
├── credentials.json.example # Template for remote Cadence credentials (see below)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

### Software

| Component | Version tested | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.12 used in paper |
| Cadence Genus | 21.17 | License required |
| Cadence Innovus | 21.18 | License required |
| LSF (or equivalent) | — | Remote job scheduling |
| PDKs | ASAP7, NanGate45 | See [Sources](#acknowledgements) |

StateTune can also drive OpenROAD via `src/EDATuner/runners/openroad_runner.py` if a Cadence flow is not available, but the paper numbers are reported on Cadence.

### Python Dependencies

```
PyYAML
numpy
pydantic>=2
requests
torch>=2.1
gpytorch>=1.11
botorch>=0.10
```

Install with:

```bash
pip install -r requirements.txt
```

Versions are pinned to the API surface this repo actually uses (e.g. `botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement`, `SingleTaskMultiFidelityGP`, `PairwiseGP`). BoTorch / GPyTorch / PyTorch are tightly coupled — if you override one, override all three together.

### RAG dependency (required)

StateTune's Knowledge Agent is grounded in the **RAG-EDA** corpus and retriever ([Pu et al., TCAD 2025](https://github.com/lesliepy99/RAG-EDA)). It is a hard dependency in the default configuration (`rag.enable: true`); removing RAG costs **46.1% hypervolume** in ablation (Section 5.2 of the paper). Clone it into `./third_party/RAG-EDA/` so the default paths in `configs/*/baseline_tuner.yaml` resolve:

```bash
mkdir -p third_party
git clone https://github.com/lesliepy99/RAG-EDA.git third_party/RAG-EDA
```

Expected layout after cloning:

```
./third_party/RAG-EDA/benchmark/openroad_documentation.json   # corpus
./third_party/RAG-EDA/benchmark/ORD-QA.jsonl                  # QA pairs
./third_party/RAG-EDA/MiniLM-L6-v2/                           # embedding model
```

To run StateTune *without* RAG (not recommended; reproduces the `w/o RAG` ablation variant), set `rag.enable: false` in the YAML.

---

## Installation

```bash
git clone https://github.com/<your-org>/StateTune.git
cd StateTune

# (recommended) isolate dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# REQUIRED: fetch the RAG-EDA corpus used by the Knowledge Agent
mkdir -p third_party
git clone https://github.com/lesliepy99/RAG-EDA.git third_party/RAG-EDA

# Copy credentials template and fill in your cluster info
cp credentials.json.example credentials.json
# then edit credentials.json  (see Configuration § below)
```

---

## Configuration

### 1. LLM API keys

StateTune reads API keys from environment variables **only** — nothing is persisted to disk.

| Variable | Used for |
|---|---|
| `PROVIDER_API_KEY` or `OPENAI_API_KEY` | Generation model (DeepSeek-V3.2 in the paper) via any OpenAI-compatible endpoint |
| `DASHSCOPE_API_KEY` | Qwen-series reasoning models via Alibaba DashScope |
| `DEEPSEEK_API_KEY` | Optional fallback for the Knowledge Agent |

Example:

```bash
export PROVIDER_API_KEY=sk-...
export DASHSCOPE_API_KEY=sk-...
```

Model names are set in the YAML (`llm.cot_model`, `llm.generation_model`). The client auto-routes Qwen models to DashScope and other models to the provider endpoint.

### 2. Remote Cadence credentials

Each config references `./credentials.json` for connecting to the remote host that runs Genus/Innovus. Copy and fill in the template:

```bash
cp credentials.json.example credentials.json
```

`credentials.json.example`:

```json
{
  "host": "your.cluster.hostname",
  "port": 22,
  "username": "your_user",
  "auth": {
    "type": "key",
    "key_path": "~/.ssh/id_rsa"
  }
}
```

`credentials.json` is listed in `.gitignore`; do **not** commit it.

### 3. Remote paths and LSF

Inside every `configs/*/baseline_tuner.yaml`, update the `cadence.remote` block:

```yaml
cadence:
  remote:
    remote_root: "/your/project/path"
    rtl_dir:     "/your/project/path/benchmark/jpeg"
    plat_dir:    "/your/project/path/platform/asap7"
    sdc_file:    "/your/project/path/benchmark/jpeg/jpeg_encoder15_7nm.sdc"
    genus_bin:   ""                      # empty = use $PATH on the remote
    innovus_bin: ""
    tool_env_script: ""                  # e.g. "source /tools/cadence/setup.sh"
    lsf_group_user: "your_lsf_group"
```

The same substitution is needed in every benchmark directory under `configs/`.

### 4. Budget knobs

Key fields in the YAML:

```yaml
fidelity:
  quick_target: "cts"          # configurable; "placement" also supported
  full_target:  "route"
  risk_beta:    2.0            # β in t̂_ucb(x)  (Eq. 6)
  promote_prob: 0.35

ka:
  min_full_valid_for_hard_rules: 12   # k in evidence gating  (Section 3.2)
  max_active_hard_rules: 8

tuning:
  total_seconds: 21600         # wall-clock budget
  max_concurrency: 48
  use_ehvi: true
```

---

## Quick Start

Run StateTune on ASAP7-JPEG:

```bash
python -m src.EDATuner.main --config configs/asap7_jpeg/baseline_tuner.yaml
```

Outputs land under `runs/<project_name>/`:

```
runs/baseline_EDATuner/
├── history.jsonl            # Every evaluation (params, QoR, fidelity, runtime)
├── memory/                  # Snapshots of M_t (rules, failures, sensitivities, prior)
├── pareto.json              # Final Pareto frontier
├── hv_trace.csv             # Anytime hypervolume trajectory
└── logs/
```

Recover from an interrupted run:

```bash
python tools/recover_state.py --run_dir runs/baseline_EDATuner
```

---

## Reproducing the Paper

All six benchmark blocks use the same 192 thread·h budget (4 threads × 48 h); ablation and sensitivity studies use 48 thread·h (1 thread × 48 h) as stated in the paper.

### Main comparison — Table 6

```bash
# StateTune on the 2×3 matrix
for cfg in configs/asap7_jpeg/baseline_tuner.yaml \
           configs/asap7_aes/baseline_tuner.yaml  \
           configs/asap7_ibex/baseline_tuner.yaml \
           configs/nangate_jpeg/baseline_tuner.yaml \
           configs/nangate_aes/baseline_tuner.yaml \
           configs/nangate_ibex/baseline_tuner.yaml
do
  python -m src.EDATuner.main --config "$cfg"
done

# Baselines (BO-qEHVI / Optuna-TPE / Random / RankTuner / CROP)
for cfg in configs/asap7_jpeg/baseline_{bo,optuna,random,ranktuner,crop}.yaml; do
  python -m baselines.run_baseline --config "$cfg"
done
```

Repeat the baseline loop for each of the other five `configs/<pdk>_<design>/` directories.

### Ablation study — Table 8

```bash
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_full.yaml
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_no_persistence.yaml
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_no_ka.yaml
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_no_rag.yaml
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_no_cot.yaml
python -m src.EDATuner.main --config configs/ablation_asap7_jpeg/ablation_no_ehvi.yaml
```

### Evidence-gating sensitivity — Figure 5

```bash
for k in 4 8 12 16 24; do
  python -m src.EDATuner.main \
    --config configs/configs_sensitivity_asap7_jpeg/sensitivity_k${k}.yaml
done
```

### Memory poisoning — Section 5.4

Toggle `ka.persist_rules_gating: false` in the config to disable evidence gating; run the ASAP7-JPEG 48 thread·h setup for both the `Pure` (gated) and `Poisoned` (ungated) variants and compare `memory/validated_rules.json`.

### Cross-design transfer — Section 5.5

```bash
# Train memory on the five non-JPEG blocks first (the main-comparison runs produce this).
# Then launch ASAP7-JPEG with the transferred artifacts:
python -m src.EDATuner.main --config configs/asap7_jpeg/knowledge_transfer.yaml
```

`knowledge_transfer.yaml` aggregates validated artifacts from sibling runs and loads them via `ka.bootstrap_memory_dirs`.

### Reproducibility — Table 7

Repeat each benchmark three times with a different `tuning.seed`:

```bash
for s in 1 2 3; do
  python -m src.EDATuner.main --config configs/asap7_jpeg/baseline_tuner_seed.yaml \
    --override tuning.seed=$s work_dir="./runs/seed_${s}"
done
```

---

## Extending StateTune

- **New tool backend** — add a class implementing `EDARunner` in `src/EDATuner/runners/` and register it in `runner_factory.py`.
- **New parameter space** — edit `tuning.param_space` in the YAML; `main.py` auto-discovers types (`float`, `int`, `categorical`).
- **New LLM provider** — point `PROVIDER_API_KEY` at any OpenAI-compatible endpoint; Qwen traffic is routed to DashScope automatically by model-name regex in `llm/openai_client.py`.
- **New memory artifact type** — extend `knowledge/kb.py::KB` and the rule-validation hook in `agents/ka_agent.py`; statistical gating (Mann–Whitney *U*, α = 0.05) is implemented centrally so new types inherit it.

---

## Known Limitations

- The benchmark matrix covers six blocks (JPEG / AES / IBEX × ASAP7 / NanGate45). Larger industrial SoCs have not been evaluated; the flat memory structure may benefit from a hierarchical organization when memory grows to thousands of rules.
- A Cadence license and a working LSF (or equivalent) scheduler are required to reproduce Table 6. The OpenROAD backend in `openroad_runner.py` is provided for smoke testing but was not used for the reported numbers.
- LLM token spend for a full 6-block suite is ~6.0 M tokens (~74% prompt, ~26% completion). Budget accordingly.

---

## Citation

If you use StateTune in academic work, please cite:

```bibtex
@inproceedings{statetune2026,
  title     = {StateTune: Transforming LLM-Assisted EDA Flow Tuning into a
               Stateful, Closed-Loop Process},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE/ACM International Conference on
               Computer-Aided Design (ICCAD)},
  year      = {2026}
}
```

---

## License

Released under the MIT License. See [`LICENSE`](./LICENSE).

Benchmark RTL under `benchmark/` retains its original upstream license; see each subdirectory's `LICENSE` file.

---

## Acknowledgements

- Benchmarks: [ibex](https://github.com/lowRISC/ibex) (Apache-2.0), OpenCores AES and JPEG encoder.
- PDKs: [ASAP7](http://asap.asu.edu/) (Clark *et al.*, 2016), [NanGate45 / FreePDK45](https://si2.org/open-source-cell-library/) (Stine *et al.*, 2007).
- RAG corpus: [RAG-EDA](https://github.com/lesliepy99/RAG-EDA) (Pu *et al.*, TCAD 2025).
- Baselines: [RankTuner](https://doi.org/10.1145/3676536.3676782) (Xu *et al.*, ICCAD 2024), [CROP](https://arxiv.org/abs/2507.02128) (Pan *et al.*, 2025), [BoTorch](https://botorch.org/) qEHVI, [Optuna](https://optuna.org/) TPE.
