import os
import json
import csv
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


# =========================
# HARDCODED CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN") = ""
EVAL_MODEL = "gpt-5.4"

JSON_PATH = ""
OUT_DIR = ""

MAX_WORKERS = 12
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120
SAVE_EVERY = 10


# =========================================================
# PROMPTS
# =========================================================
F_ACC_SYSTEM_PROMPT = """You are a medical trajectory evaluator.

Evaluate only Clinical Faithfulness (F_acc).

Definition:
Clinical Faithfulness measures whether the predicted reasoning process follows a clinically valid step-by-step workflow.

Focus ONLY on:
- sequence of reasoning
- whether later conclusions are supported by earlier steps
- whether the workflow is clinically sensible
- whether there are logical jumps or contradictions in the reasoning path

Ignore:
- minor factual wording issues
- missing details unless they break the reasoning chain
- whether all findings are fully covered
- the final answer itself

Important:
- This metric is about LOGICAL WORKFLOW, not completeness.
- A trajectory can be factually incomplete but still faithful.
- A trajectory can have a good-looking final answer but low faithfulness if the reasoning path is poor.
- Evaluate only from the trajectory content provided.

Scoring guide:
- 1.0 = clinically coherent stepwise reasoning, no major logic flaws
- 0.7 = mostly reasonable workflow with some weak jumps
- 0.4 = partially logical but important workflow issues
- 0.1 = mostly illogical or unsupported reasoning
- 0.0 = contradictory or clinically nonsensical reasoning

Return JSON only:
{
  "score": number
}
"""

F_ACC_USER_PROMPT = """Metric: Clinical Faithfulness (F_acc)

Gold trajectory:
{gold_traj}

Predicted trajectory:
{pred_traj}
"""

C_S_SYSTEM_PROMPT = """You are a medical trajectory evaluator.

Evaluate only Context Integration Score (C_s).

Definition:
Context Integration Score measures how well the predicted trajectory uses the available multimodal evidence.

Focus ONLY on:
- whether tool outputs are actually used
- whether image findings, OCR outputs, region descriptions, and evidence are integrated into the reasoning
- whether the trajectory is grounded in available context rather than generic guessing

Ignore:
- whether the reasoning order is ideal
- whether every medical fact is correct
- whether the final answer is complete
- the final answer itself

Important:
- This metric is about EVIDENCE USAGE, not logic or correctness alone.
- A trajectory can be logically organized but still have low context integration if it ignores tool evidence.
- A trajectory can partially use evidence and should get partial credit.
- Evaluate only from the trajectory content provided.

Scoring guide:
- 1.0 = directly and effectively integrates relevant context
- 0.7 = uses some important context but not all
- 0.4 = weak or superficial use of context
- 0.1 = almost no meaningful evidence use
- 0.0 = ignores available context entirely or is unrelated

Return JSON only:
{
  "score": number
}
"""

C_S_USER_PROMPT = """Metric: Context Integration Score (C_s)

Gold trajectory:
{gold_traj}

Predicted trajectory:
{pred_traj}
"""

F_P_SYSTEM_PROMPT = """You are a medical trajectory evaluator.

Evaluate only Clinical Factual Precision (F_p).

Definition:
Clinical Factual Precision measures whether the medical claims in the trajectory are factually correct and non-hallucinated.

Focus ONLY on:
- correctness of diagnoses, anatomy, modality, measurements, locations, and medical claims inside the trajectory
- hallucinated findings
- fabricated tool interpretations
- clinically unsafe or false statements

Ignore:
- reasoning order
- whether every required finding is covered
- whether evidence integration is elegant
- the final answer itself

Important:
- This metric is about MEDICAL FACTUAL CORRECTNESS only.
- Penalize hallucinations strongly.
- If the trajectory contains correct facts plus extra wrong facts, reduce the score.
- Evaluate only from the trajectory content provided.

Scoring guide:
- 1.0 = medically precise and factually correct
- 0.7 = mostly correct with minor inaccuracies
- 0.4 = mix of correct and incorrect facts
- 0.1 = mostly false or hallucinated
- 0.0 = dangerously incorrect, fabricated, or unrelated

Return JSON only:
{
  "score": number
}
"""

F_P_USER_PROMPT = """Metric: Clinical Factual Precision (F_p)

Gold trajectory:
{gold_traj}

Predicted trajectory:
{pred_traj}
"""

S_COMP_SYSTEM_PROMPT = """You are a medical trajectory evaluator.

Evaluate only Semantic Completeness (S_comp).

Definition:
Semantic Completeness measures whether the trajectory covers all clinically necessary findings required by the task.

Focus ONLY on:
- whether required reasoning components and findings are present in the trajectory
- whether important findings are omitted
- whether the trajectory fully covers the task requirements

Ignore:
- reasoning order
- elegance of evidence usage
- small factual imprecision unless it removes a required component
- the final answer itself

Important:
- This metric is about COVERAGE and MISSING INFORMATION.
- A concise trajectory can score 1.0 if it includes all required findings.
- A logically good trajectory can still score low if key clinical findings are missing.
- Evaluate only from the trajectory content provided.

Scoring guide:
- 1.0 = all required clinical content is covered
- 0.7 = most important content covered, some omissions
- 0.4 = only partial coverage
- 0.1 = very incomplete
- 0.0 = missing essentially all required content or unrelated

Return JSON only:
{
  "score": number
}
"""

S_COMP_USER_PROMPT = """Metric: Semantic Completeness (S_comp)

Gold trajectory:
{gold_traj}

Predicted trajectory:
{pred_traj}
"""


# =========================================================
# HELPERS
# =========================================================
def normalize_message_list(msgs: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      1) [msg, msg, ...]
      2) [[msg, msg, ...]]
    """
    if not msgs:
        return []

    if isinstance(msgs, list):
        if len(msgs) > 0 and isinstance(msgs[0], dict):
            return [x for x in msgs if isinstance(x, dict)]

        if len(msgs) > 0 and isinstance(msgs[0], list):
            inner = msgs[0]
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]

    return []


def extract_final_text(msgs: Any) -> str:
    msgs = normalize_message_list(msgs)
    if not msgs:
        return ""

    for item in reversed(msgs):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
            and item.get("content").strip()
        ):
            return item["content"].strip()

    return ""


def compact_trajectory(msgs: Any) -> List[Dict[str, Any]]:
    msgs = normalize_message_list(msgs)
    out: List[Dict[str, Any]] = []

    for m in msgs:
        if not isinstance(m, dict):
            continue

        item: Dict[str, Any] = {"role": m.get("role", "")}

        if "thought" in m and isinstance(m["thought"], str) and m["thought"].strip():
            item["thought"] = m["thought"].strip()

        if "tool_calls" in m and isinstance(m["tool_calls"], list):
            calls = []
            for c in m["tool_calls"]:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function", {})
                if isinstance(fn, dict):
                    calls.append({
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {})
                    })
            item["tool_calls"] = calls

        if "content" in m:
            if isinstance(m["content"], str) and m["content"].strip():
                item["content"] = m["content"].strip()
            elif isinstance(m["content"], dict):
                item["content"] = m["content"]
            elif isinstance(m["content"], list):
                item["content"] = m["content"]
            elif m["content"] is None:
                item["content"] = None

        if "error" in m:
            item["error"] = m["error"]

        if "name" in m:
            item["name"] = m["name"]

        out.append(item)

    return out


def load_samples(json_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return list(data.items())

    if isinstance(data, list):
        return [(str(i), x) for i, x in enumerate(data)]

    raise ValueError("Unsupported JSON format. Expected dict or list.")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_output_paths(out_dir: str, model_name: str) -> Dict[str, str]:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return {
        "summary_txt": str(Path(out_dir) / f"{safe_model}_clinical_results.txt"),
        "raw_json": str(Path(out_dir) / f"{safe_model}_clinical_results.json"),
        "csv": str(Path(out_dir) / f"{safe_model}_clinical_results.csv"),
        "checkpoint_json": str(Path(out_dir) / f"{safe_model}_clinical_checkpoint.json"),
    }


def safe_mean(values: List[Optional[float]]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def extract_output_text_from_response(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    else:
        data = resp

    if isinstance(data, dict):
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()

        output = data.get("output", [])
        texts: List[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                for c in item.get("content", []):
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                        texts.append(c["text"])
        if texts:
            return "\n".join(texts).strip()

    return ""


def get_metric_prompt(metric_name: str) -> Tuple[str, str]:
    if metric_name == "F_acc":
        return F_ACC_SYSTEM_PROMPT, F_ACC_USER_PROMPT
    if metric_name == "C_s":
        return C_S_SYSTEM_PROMPT, C_S_USER_PROMPT
    if metric_name == "F_p":
        return F_P_SYSTEM_PROMPT, F_P_USER_PROMPT
    if metric_name == "S_comp":
        return S_COMP_SYSTEM_PROMPT, S_COMP_USER_PROMPT
    raise ValueError(f"Unsupported metric: {metric_name}")


# =========================================================
# API CALL
# =========================================================
def score_one_metric(
    client: OpenAI,
    sample_id: str,
    sample: Dict[str, Any],
    metric_name: str,
) -> Dict[str, Any]:
    gold_answer = extract_final_text(sample.get("gold", []))
    pred_answer = extract_final_text(sample.get("prediction", []))
    gold_traj = compact_trajectory(sample.get("gold", []))
    pred_traj = compact_trajectory(sample.get("prediction", []))

    system_prompt, user_template = get_metric_prompt(metric_name)
    user_prompt = user_template.format(
        gold_traj=json.dumps(gold_traj, ensure_ascii=False, indent=2),
        pred_traj=json.dumps(pred_traj, ensure_ascii=False, indent=2),
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=EVAL_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": f"{metric_name}_eval",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "score": {"type": "number"}
                            },
                            "required": ["score"]
                        },
                        "strict": True
                    }
                },
                timeout=REQUEST_TIMEOUT,
            )

            raw_text = extract_output_text_from_response(resp)
            parsed = json.loads(raw_text)

            score = parsed.get("score", None)
            if not isinstance(score, (int, float)):
                raise ValueError(f"Missing/invalid score in parsed response: {parsed}")

            score = max(0.0, min(1.0, float(score)))

            return {
                "sample_id": sample_id,
                "metric": metric_name,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "judge_score": score,
                "status": "ok",
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                sleep_s = min(8, 1.5 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    return {
        "sample_id": sample_id,
        "metric": metric_name,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "judge_score": None,
        "status": "error",
        "error": last_error,
    }


# =========================================================
# SAVING
# =========================================================
def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for r in results:
        sid = str(r["sample_id"])
        if sid not in grouped:
            grouped[sid] = {
                "sample_id": sid,
                "gold_answer": r.get("gold_answer", ""),
                "pred_answer": r.get("pred_answer", ""),
                "F_acc": None,
                "C_s": None,
                "F_p": None,
                "S_comp": None,
                "statuses": {},
                "errors": {},
            }

        metric = r.get("metric")
        grouped[sid][metric] = r.get("judge_score")
        grouped[sid]["statuses"][metric] = r.get("status", "")
        if r.get("status") == "error":
            grouped[sid]["errors"][metric] = r.get("error", "")

    per_sample = sorted(grouped.values(), key=lambda x: str(x.get("sample_id", "")))

    return {
        "model": EVAL_MODEL,
        "num_samples": len(per_sample),
        "num_metric_results": len(results),
        "num_scored_metric_results": sum(1 for x in results if isinstance(x.get("judge_score"), (int, float))),
        "num_error_metric_results": sum(1 for x in results if x.get("status") == "error"),
        "mean_F_acc": safe_mean([x.get("F_acc") for x in per_sample]),
        "mean_C_s": safe_mean([x.get("C_s") for x in per_sample]),
        "mean_F_p": safe_mean([x.get("F_p") for x in per_sample]),
        "mean_S_comp": safe_mean([x.get("S_comp") for x in per_sample]),
        "per_sample": per_sample,
    }


def save_summary_json(summary: Dict[str, Any], json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_summary_txt(summary: Dict[str, Any], txt_path: str) -> None:
    lines: List[str] = []
    lines.append(f"Model: {summary['model']}")
    lines.append(f"Samples: {summary['num_samples']}")
    lines.append(f"Metric Results: {summary['num_metric_results']}")
    lines.append(f"Scored Metric Results: {summary['num_scored_metric_results']}")
    lines.append(f"Error Metric Results: {summary['num_error_metric_results']}")
    lines.append(f"Mean F_acc: {summary['mean_F_acc']:.4f}")
    lines.append(f"Mean C_s: {summary['mean_C_s']:.4f}")
    lines.append(f"Mean F_p: {summary['mean_F_p']:.4f}")
    lines.append(f"Mean S_comp: {summary['mean_S_comp']:.4f}")
    lines.append("")
    lines.append("Per-sample results")
    lines.append("=" * 100)

    for item in summary["per_sample"]:
        lines.append(f"Sample ID: {item.get('sample_id', '')}")
        lines.append(f"Gold Answer: {item.get('gold_answer', '')}")
        lines.append(f"Pred Answer: {item.get('pred_answer', '')}")
        lines.append(f"F_acc: {item.get('F_acc', 'NA')}")
        lines.append(f"C_s: {item.get('C_s', 'NA')}")
        lines.append(f"F_p: {item.get('F_p', 'NA')}")
        lines.append(f"S_comp: {item.get('S_comp', 'NA')}")
        if item.get("errors"):
            lines.append(f"Errors: {item.get('errors', {})}")
        lines.append("-" * 100)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_summary_csv(summary: Dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "gold_answer",
            "pred_answer",
            "F_acc",
            "C_s",
            "F_p",
            "S_comp",
            "error_metrics",
        ])

        for item in summary["per_sample"]:
            writer.writerow([
                item.get("sample_id", ""),
                item.get("gold_answer", ""),
                item.get("pred_answer", ""),
                item.get("F_acc", ""),
                item.get("C_s", ""),
                item.get("F_p", ""),
                item.get("S_comp", ""),
                json.dumps(item.get("errors", {}), ensure_ascii=False),
            ])


def save_all(results: List[Dict[str, Any]], paths: Dict[str, str]) -> None:
    summary = build_summary(results)
    save_summary_json(summary, paths["raw_json"])
    save_summary_txt(summary, paths["summary_txt"])
    save_summary_csv(summary, paths["csv"])
    save_summary_json(summary, paths["checkpoint_json"])


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    ensure_dir(OUT_DIR)
    paths = get_output_paths(OUT_DIR, EVAL_MODEL)
    samples = load_samples(JSON_PATH)

    client = OpenAI(api_key=OPENAI_API_KEY)

    results: List[Dict[str, Any]] = []
    completed = 0
    metrics = ["F_acc", "C_s", "F_p", "S_comp"]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(score_one_metric, client, sample_id, sample, metric): (sample_id, metric)
            for sample_id, sample in samples
            for metric in metrics
        }

        for future in tqdm(as_completed(future_to_meta), total=len(future_to_meta), desc="Scoring clinical metrics"):
            sample_id, metric = future_to_meta[future]
            try:
                result = future.result()
            except Exception as e:
                sample = dict(samples)[sample_id]
                result = {
                    "sample_id": sample_id,
                    "metric": metric,
                    "gold_answer": extract_final_text(sample.get("gold", [])),
                    "pred_answer": extract_final_text(sample.get("prediction", [])),
                    "judge_score": None,
                    "status": "error",
                    "error": str(e),
                }

            results.append(result)
            completed += 1

            if completed % SAVE_EVERY == 0 or completed == len(future_to_meta):
                save_all(results, paths)

    summary = build_summary(results)

    print(f"Saved raw JSON results to: {paths['raw_json']}")
    print(f"Saved summary TXT results to: {paths['summary_txt']}")
    print(f"Saved CSV results to: {paths['csv']}")
    print("")
    print("==== Final Means ====")
    print(f"F_acc: {summary['mean_F_acc']:.4f}")
    print(f"C_s: {summary['mean_C_s']:.4f}")
    print(f"F_p: {summary['mean_F_p']:.4f}")
    print(f"S_comp: {summary['mean_S_comp']:.4f}")
    print(f"Scored Metric Results: {summary['num_scored_metric_results']}/{summary['num_metric_results']}")
    print(f"Error Metric Results: {summary['num_error_metric_results']}")


if __name__ == "__main__":
    main()