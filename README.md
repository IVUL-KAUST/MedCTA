<h1 align="center">🩺 MedCTA</h1>

<h3 align="center">
  A Benchmark for Clinical Tool Agents
</h3>

<p align="center">
  <em>Evaluating multimodal medical agents through realistic clinical tool-use workflows.</em>
</p>

<p align="center">
<p align="center">
   <a href="https://arxiv.org/abs/2606.11702">
    <img src="https://img.shields.io/badge/arXiv-2606.11702-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" />
  </a>
  <a href="https://ivul-kaust.github.io/MedCTA/">
    <img src="https://img.shields.io/badge/Project-Page-ffb6c1?style=for-the-badge&logo=githubpages&logoColor=white" />
  </a>

  <a href="https://huggingface.co/datasets/IVUL-KAUST/MedCTA">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-ffd166?style=for-the-badge&logo=huggingface&logoColor=black" />
  </a>

  <a href="#-leaderboard">
    <img src="https://img.shields.io/badge/Leaderboard-MedCTA-9b5de5?style=for-the-badge&logo=mediamarkt&logoColor=white" />
  </a>

  <a href="#-evaluate-on-medcta">
    <img src="https://img.shields.io/badge/Evaluation-OpenCompass-00bbf9?style=for-the-badge&logo=python&logoColor=white" />
  </a>

  <a href="#-citation">
    <img src="https://img.shields.io/badge/Citation-BibTeX-f15bb5?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tasks-107-ffcad4?style=flat-square" />
  <img src="https://img.shields.io/badge/Tools-5-bde0fe?style=flat-square" />
  <img src="https://img.shields.io/badge/Models-18-caffbf?style=flat-square" />
  <img src="https://img.shields.io/badge/Rollouts-1,926-fdffb6?style=flat-square" />
  <img src="https://img.shields.io/badge/Clinical-Multimodal-e0bbff?style=flat-square" />
</p>

---

<div align="center">

### ✨ Clinical tool-use evaluation, but make it realistic.

MedCTA is a benchmark for evaluating **clinical tool agents** on clinician-verified, multimodal medical tasks.  
It tests whether an agent can select the right tools, call them correctly, use evidence faithfully, and reach a clinically meaningful final answer.

</div>

---

## 🌸 Table of Contents

- [🌟 Introduction](#-introduction)
- [🧠 Why MedCTA?](#-why-medcta)
- [📚 Dataset at a Glance](#-dataset-at-a-glance)
- [🧰 Tool Library](#-tool-library)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [🏆 Leaderboard](#-leaderboard)
- [🔍 Failure Diagnostics](#-failure-diagnostics)
- [🚀 Evaluate on MedCTA](#-evaluate-on-medcta)
- [📝 Citation](#-citation)

---

## 🌟 Introduction

> Clinical reasoning is not a single-step QA problem.  
> It is multimodal, iterative, tool-dependent, and evidence-sensitive.

**MedCTA** evaluates how well LLM-based agents can operate as **clinical tool agents** in realistic medical scenarios.

Unlike static medical QA benchmarks, MedCTA requires agents to:

- understand multimodal clinical inputs,
- decide which tools are needed,
- call tools with valid arguments,
- integrate intermediate observations,
- retrieve evidence when necessary,
- and produce clinically faithful final answers.

The benchmark is built around **step-implicit clinical tasks**.  
The model receives a clinical objective, but it is not explicitly told which tools to use or in what order.

---

## 🧠 Why MedCTA?

Many existing medical benchmarks mainly test final-answer accuracy.  
MedCTA goes deeper.

It asks:

<div align="center">

| Question | What MedCTA Measures |
|---|---|
| 🧭 Can the agent plan? | Tool selection and trajectory fidelity |
| 🧰 Can it use tools correctly? | Argument validity and execution reliability |
| 🩻 Can it understand images? | Multimodal evidence extraction |
| 🔎 Can it retrieve useful knowledge? | Search and evidence integration |
| 🧮 Can it compute when needed? | Numerical and symbolic reasoning |
| 🧑‍⚕️ Can it answer clinically? | Faithfulness, completeness, and final goal accuracy |

</div>

MedCTA is designed to reveal not only **whether** an agent gets the answer right, but also **how** it reaches that answer.

---

## 📚 Dataset at a Glance

<div align="center">

| 🌷 Item | 💡 Value |
|---|---:|
| Clinician-verified tasks | **107** |
| Executable tools | **5** |
| Benchmarked models | **18** |
| Autonomous rollouts | **1,926** |
| Human annotation time | **321 hours** |
| Anatomical regions / body systems | **34** |
| Tool steps per task | **2–4** |
| Average tool-execution steps | **3.1** |

</div>

---

## 🧩 Task Format

Each task can be viewed as:

<div align="center">

```text
(X, Q, U, π, A)
```

</div>

where:

| Symbol | Meaning |
|---|---|
| `X` | Multimodal clinical context |
| `Q` | Step-implicit clinical query |
| `U` | Hidden sufficient tool subset |
| `π` | Reference interaction trajectory |
| `A` | Final clinical outcome |

This structure allows MedCTA to evaluate both the **process** and the **final answer**.

---

## 🧰 Tool Library

MedCTA uses five deployed tools that cover perception, retrieval, localization, and reasoning.

<div align="center">

| Tool | Role | Typical Use |
|---|---|---|
| `OCR` | Text extraction | Reading reports, labels, tables, scanned documents |
| `ImageDescription` | Global visual understanding | Summarizing medical images |
| `RegionAttributeDescription` | Local visual inspection | Describing specific regions or findings |
| `GoogleSearch` | External evidence retrieval | Looking up clinical or biomedical knowledge |
| `Calculator` | Numerical reasoning | Computing values, ratios, expressions, or scores |

</div>

These tools make MedCTA closer to real clinical workflows, where an agent must actively gather and combine information before answering.

---

## 📏 Evaluation Metrics

MedCTA evaluates agents from three complementary angles.

### 🪜 1. Step-by-step Tool-use Fidelity

| Metric | Meaning |
|---|---|
| `InstAcc` | Instruction-following accuracy |
| `ToolAcc` | Tool-selection accuracy |
| `ArgAcc` | Tool-argument prediction accuracy |
| `SummAcc` | Intermediate summarization accuracy |

### 🧑‍⚕️ 2. Clinical Reasoning Quality

| Metric | Meaning |
|---|---|
| `Facc` | Clinical faithfulness |
| `Cs` | Multimodal context integration |
| `Scomp` | Semantic completeness |

### 🎯 3. Final Outcome Accuracy

| Metric | Meaning |
|---|---|
| `Gacc` | Final goal / answer accuracy |

Together, these metrics help diagnose whether an agent failed because of poor tool selection, invalid arguments, weak evidence use, premature stopping, or incorrect clinical reasoning.

---

## 🏆 Leaderboard

<div align="center">

### Autonomous Clinical Tool-use Performance

</div>

> `*` denotes closed-source/API-based evaluation.

<details open>
<summary><b>💛 API-based and closed models</b></summary>

<br/>

| Model | Developer | Inst | Tool | Arg | Summ | Facc | Cs | Scomp | Gacc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GPT-5.4*` | OpenAI | 35.27 | 23.46 | 12.61 | 35.51 | 17.52 | 14.21 | 18.60 | **31.54** |
| `GPT-5.4-mini*` | OpenAI | 5.36 | 6.74 | 3.23 | 0.93 | 16.47 | 10.56 | 17.29 | 28.31 |
| `GPT-5.4-nano*` | OpenAI | 33.93 | 18.18 | 12.02 | 34.24 | 18.43 | 11.96 | 14.77 | 20.30 |
| `Claude-opus-4-6*` | Anthropic | 24.78 | 8.80 | 0.59 | 39.25 | 14.11 | 14.86 | 23.83 | 31.32 |
| `Claude-sonnet-4-6*` | Anthropic | 23.66 | 4.99 | 0.00 | 33.64 | 12.77 | 12.90 | 20.19 | 25.33 |
| `Claude-haiku-4-5*` | Anthropic | 27.46 | 13.78 | 4.69 | 43.93 | 9.35 | 3.36 | 14.11 | 23.08 |
| `Gemini-3-flash*` | Google | 3.35 | 17.30 | 0.00 | 5.61 | 11.31 | 8.60 | 15.98 | 25.87 |
| `Gemini-3-flash-lite*` | Google | 2.90 | 8.21 | 0.00 | 1.87 | 10.75 | 6.82 | 14.58 | 23.64 |

</details>

<details open>
<summary><b>💚 Open-source models</b></summary>

<br/>

| Model | Developer | Inst | Tool | Arg | Summ | Facc | Cs | Scomp | Gacc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-9B` | Qwen | 44.20 | 14.37 | 13.78 | 29.91 | 10.37 | 17.10 | 13.36 | 21.64 |
| `Qwen3-8B` | Qwen | 33.93 | 10.56 | 7.04 | 32.71 | 8.50 | 10.09 | 11.50 | **27.80** |
| `DeepSeek-R1-Distill-7B` | DeepSeek | 10.49 | 3.52 | 0.00 | 7.48 | 2.62 | 0.84 | 3.36 | 10.61 |
| `Deepseek-llm-7b-chat` | DeepSeek | 11.61 | 6.45 | 0.00 | 4.67 | 4.30 | 2.62 | 4.02 | 11.00 |
| `DeepSeek-V2-Lite-Chat` | DeepSeek | 11.83 | 11.14 | 0.29 | 0.00 | 3.83 | 3.55 | 6.54 | 6.96 |
| `Llama-3.1-8B-Instruct` | Meta | 23.66 | 7.92 | 0.00 | 6.54 | 7.94 | 5.42 | 11.21 | 18.94 |
| `Llama-3.2-3B-Instruct` | Meta | 18.53 | 1.76 | 0.00 | 4.67 | 3.08 | 1.68 | 5.14 | 11.29 |
| `Mistral-7B` | Mistral | 18.75 | 14.66 | 0.00 | 9.35 | 2.52 | 1.87 | 3.46 | 9.40 |
| `Phi-4` | Microsoft | 20.09 | 6.45 | 0.00 | 14.02 | 6.36 | 3.36 | 6.17 | 10.65 |
| `GPT-oss-20B` | OpenAI | 1.79 | 0.00 | 0.00 | 0.00 | 1.31 | 0.56 | 1.68 | 3.18 |

</details>

---

## 🔍 Failure Diagnostics

MedCTA shows that clinical tool agents often struggle not only with final reasoning, but also with **tool routing**, **protocol stability**, and **premature stopping**.

### 🌱 Autonomous vs. Gold Tool Routing

Providing the gold tool route substantially improves final outcome accuracy, showing that tool planning remains a major bottleneck.

<div align="center">

| Model | Auto `Gacc` | Gold `Gacc` | Gain |
|---|---:|---:|---:|
| `GPT-5.4` | 31.54 | 49.50 | **+17.96** |
| `Claude-opus-4-6` | 31.32 | 66.40 | **+35.08** |
| `Qwen3.5-9B` | 21.64 | 49.50 | **+27.86** |

</div>

### 🧪 Rollout-Level Diagnostics

<div align="center">

| Diagnostic | Value | Main Issue |
|---|---:|---|
| API error rate | **64.2%** | Protocol instability |
| Under-call rate | **99.2%** | Premature stopping |
| Protocol failure | **58.3%** | Rollout breakdown |
| Tool-selection failure | **41.6%** | Incorrect actions |

</div>

These results suggest that reliable clinical agents need stronger controllers, not just stronger language or vision backbones.

---

# 🚀 Evaluate on MedCTA

This repository follows the OpenCompass-style evaluation pipeline with AgentLego tools and LMDeploy model serving.

If you want to add a new agent wrapper or integrate a different LLM endpoint, see:

```text
docs/ADDING_NEW_AGENT_OR_LLM.md
```

---

## 1. Prepare the MedCTA Dataset

Clone this repository.

```shell
git clone <ANONYMIZED_REPOSITORY_URL>
cd MedCTA
```

Create the dataset directory.

```shell
mkdir -p ./opencompass/data
```

Download the MedCTA dataset from the release file and place it under:

```text
./opencompass/data/
```

The expected file structure is:

```text
MedCTA/
├── agentlego
├── opencompass
│   ├── data
│   │   ├── medcta_dataset
│   ├── ...
├── ...
```

---

## 2. Prepare Your Model

### 2.1 Download model weights

```shell
pip install -U huggingface_hub

# huggingface-cli download --resume-download hugging/face/repo/name \
#   --local-dir your/local/path \
#   --local-dir-use-symlinks False

huggingface-cli download --resume-download Qwen/Qwen1.5-7B-Chat \
  --local-dir ~/models/qwen1.5-7b-chat \
  --local-dir-use-symlinks False
```

### 2.2 Install LMDeploy

```shell
conda create -n lmdeploy python=3.10
conda activate lmdeploy
```

For CUDA 12:

```shell
pip install lmdeploy
```

For CUDA 11+:

```shell
export LMDEPLOY_VERSION=0.4.0
export PYTHON_VERSION=310

pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Launch a model service

```shell
# lmdeploy serve api_server path/to/your/model \
#   --server-port [port_number] \
#   --model-name [your_model_name]

lmdeploy serve api_server ~/models/qwen1.5-7b-chat \
  --server-port 12580 \
  --model-name qwen1.5-7b-chat
```

---

## 3. Deploy Tools

### 3.1 Install AgentLego

```shell
conda create -n agentlego python=3.11.9
conda activate agentlego

cd agentlego

pip install -r requirements_all.txt
pip install agentlego
pip install -e .

mim install mmengine
mim install mmcv==2.1.0
```

Then open:

```text
~/anaconda3/envs/agentlego/lib/python3.11/site-packages/transformers/modeling_utils.py
```

and change:

```python
_supports_sdpa = False
```

to:

```python
_supports_sdpa = True
```

### 3.2 Configure API keys

To use the GoogleSearch and MathOCR tools, first obtain:

- a Serper API key from `https://serper.dev`
- a Mathpix API key from `https://mathpix.com/`

Then export them:

```shell
export SERPER_API_KEY='your_serper_key_for_google_search_tool'
export MATHPIX_APP_ID='your_mathpix_key_for_mathocr_tool'
export MATHPIX_APP_KEY='your_mathpix_key_for_mathocr_tool'
```

### 3.3 Start the tool server

```shell
agentlego-server start \
  --port 16181 \
  --extra ./benchmark.py `cat benchmark_toollist.txt` \
  --host 0.0.0.0
```

---

## 4. Start Evaluation

### 4.1 Install OpenCompass

```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass

cd agentlego
pip install -e .

cd ../opencompass
pip install -e .
```

Recommended package versions:

```text
huggingface_hub==0.25.2
transformers==4.40.1
```

---

## 5. Configure the Evaluation

Modify:

```text
configs/eval_medcta_bench.py
```

The IP and port of `openai_api_base` should match your LMDeploy model service.  
The IP and port of `tool_server` should match your AgentLego tool server.

```python
models = [
    dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```

---

## 6. Step-by-step Evaluation Mode

For step-by-step mode:

- comment out `tool_server`
- enable `tool_meta`
- set infer and eval mode to `every_with_gt`

In:

```text
configs/eval_medcta_bench.py
```

use:

```python
models = [
    dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        # tool_server='http://10.140.0.138:16181',
        tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```

In:

```text
configs/datasets/gta_bench.py
```

use:

```python
gta_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
)

gta_bench_eval_cfg = dict(
    evaluator=dict(type=GTABenchEvaluator, mode='every_with_gt')
)
```

---

## 7. End-to-end Evaluation Mode

For end-to-end mode:

- enable `tool_server`
- comment out `tool_meta`
- set infer and eval mode to `every`

In:

```text
configs/eval_medcta_bench.py
```

use:

```python
models = [
    dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        # tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```

In:

```text
configs/datasets/gta_bench.py
```

use:

```python
gta_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every'),
)

gta_bench_eval_cfg = dict(
    evaluator=dict(type=GTABenchEvaluator, mode='every')
)
```

---

## 8. Run Inference and Evaluation

### Infer only

```shell
python run.py configs/eval_medcta_bench.py \
  --max-num-workers 32 \
  --debug \
  --mode infer
```

### Evaluate only

```shell
# srun -p llmit -q auto python run.py configs/eval_medcta_bench.py \
#   --max-num-workers 32 \
#   --debug \
#   --reuse [time_stamp_of_prediction_file] \
#   --mode eval

srun -p llmit -q auto python run.py configs/eval_medcta_bench.py \
  --max-num-workers 32 \
  --debug \
  --reuse 20240628_115514 \
  --mode eval
```

### Infer and evaluate

```shell
python run.py configs/eval_medcta_bench.py \
  -p llmit \
  -q auto \
  --max-num-workers 32 \
  --debug
```

---

## 🧡 Project Structure

```text
MedCTA/
├── agentlego/                  # Tool deployment and AgentLego integration
├── docs/                       # Documentation for adding agents or LLMs
├── opencompass/                # OpenCompass evaluation framework
│   ├── data/
│   │   └── medcta_dataset/     # MedCTA dataset directory
│   └── configs/
├── clinical_accuracy.py        # Clinical reasoning evaluation
├── goal_accuracy.py            # Final goal accuracy evaluation
├── README.md
└── LICENSE.txt
```

---

## 📝 Citation

If you use MedCTA in your research, please cite the paper once the citation is available.

```bibtex
@misc{medcta,
      title={MedCTA: A Benchmark for Clinical Tool Agents}, 
      author={Tajamul Ashraf and Hyewon Jeong and Fida Mohammad Thoker and Bernard Ghanem},
      year={2026},
      eprint={2606.11702},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.11702}, 

```

---

## 💌 Acknowledgements

MedCTA builds on the OpenCompass-style evaluation ecosystem and AgentLego-based tool execution framework.

We thank the annotators, clinicians, and researchers who contributed to the benchmark design, validation, and evaluation.

---

<div align="center">

### 🌷 MedCTA

<strong>Clinical tool agents should not only answer — they should observe, reason, verify, and act.</strong>

<br/>

Made for careful, multimodal, clinically grounded agent evaluation.

</div>
