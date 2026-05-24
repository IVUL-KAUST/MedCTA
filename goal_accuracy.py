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
OPENAI_API_KEY = ""
EVAL_MODEL = "gpt-5.4"

JSON_PATH = ""
OUT_DIR = ""


MAX_WORKERS = 12          # 8 to 16 is a good range
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120     # seconds
SAVE_EVERY = 10           # periodic checkpointing


# =========================================================
# PROMPTS
# =========================================================
GOAL_ACCURACY_SYSTEM_PROMPT = """You are a medical answer evaluator.

Compare the predicted FINAL answer against the gold FINAL clinical answer.
Assign a score from 0.0 to 1.0 based on semantic clinical correctness.

CRITICAL RULE (very important):
- If the predicted answer explicitly contains the correct gold answer, assign a score of 1.0.
- Presence of the correct diagnosis/finding overrides extra guesses unless contradictory.

General rules:
- Give partial credit if only partially correct.
- Do NOT give 0.0 unless completely wrong or unrelated.
- Judge by clinical meaning, not wording.
- Synonyms count as correct.

Scoring guide:
- 1.0 = gold answer clearly present OR fully correct
- 0.8–0.95 = correct but minor imprecision
- 0.5–0.75 = partially correct
- 0.2–0.45 = weak overlap
- 0.0–0.1 = wrong/unrelated

Return JSON only:
{
  "score": number
}
"""

GOAL_ACCURACY_USER_PROMPT = """Gold final answer:
{gold_final}

Predicted final answer:
{pred_final}
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
        "summary_txt": str(Path(out_dir) / f"{safe_model}_goal_results.txt"),
        "raw_json": str(Path(out_dir) / f"{safe_model}_goal_results.json"),
        "csv": str(Path(out_dir) / f"{safe_model}_goal_results.csv"),
        "checkpoint_json": str(Path(out_dir) / f"{safe_model}_goal_checkpoint.json"),
    }


def safe_mean(values: List[Optional[float]]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def extract_output_text_from_response(resp: Any) -> str:
    """
    Robust extraction from OpenAI Responses API object.
    """
    # Most convenient path if available
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback: inspect model dump
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


# =========================================================
# API CALL
# =========================================================
def score_one_sample(
    client: OpenAI,
    sample_id: str,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    gold_answer = extract_final_text(sample.get("gold", []))
    pred_answer = extract_final_text(sample.get("prediction", []))

    user_prompt = GOAL_ACCURACY_USER_PROMPT.format(
        gold_final=gold_answer,
        pred_final=pred_answer,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=EVAL_MODEL,
                input=[
                    {"role": "system", "content": GOAL_ACCURACY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "goal_accuracy_eval",
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

            # clamp for safety
            score = max(0.0, min(1.0, float(score)))

            return {
                "sample_id": sample_id,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "judge_score": score,
                "status": "ok",
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                # exponential backoff + jitter
                sleep_s = min(8, 1.5 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    return {
        "sample_id": sample_id,
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
    return {
        "model": EVAL_MODEL,
        "num_samples": len(results),
        "num_scored": sum(1 for x in results if isinstance(x.get("judge_score"), (int, float))),
        "num_errors": sum(1 for x in results if x.get("status") == "error"),
        "mean_goal_accuracy": safe_mean([x.get("judge_score") for x in results]),
        "per_sample": sorted(results, key=lambda x: str(x.get("sample_id", ""))),
    }


def save_summary_json(summary: Dict[str, Any], json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_summary_txt(summary: Dict[str, Any], txt_path: str) -> None:
    lines: List[str] = []
    lines.append(f"Model: {summary['model']}")
    lines.append(f"Samples: {summary['num_samples']}")
    lines.append(f"Scored: {summary['num_scored']}")
    lines.append(f"Errors: {summary['num_errors']}")
    lines.append(f"Mean Goal Accuracy: {summary['mean_goal_accuracy']:.4f}")
    lines.append("")
    lines.append("Per-sample results")
    lines.append("=" * 100)

    for item in summary["per_sample"]:
        lines.append(f"Sample ID: {item.get('sample_id', '')}")
        lines.append(f"Gold Answer: {item.get('gold_answer', '')}")
        lines.append(f"Pred Answer: {item.get('pred_answer', '')}")
        lines.append(f"Judge Score: {item.get('judge_score', 'NA')}")
        if item.get("status") == "error":
            lines.append(f"Error: {item.get('error', '')}")
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
            "judge_score",
            "status",
            "error",
        ])

        for item in summary["per_sample"]:
            writer.writerow([
                item.get("sample_id", ""),
                item.get("gold_answer", ""),
                item.get("pred_answer", ""),
                item.get("judge_score", ""),
                item.get("status", ""),
                item.get("error", ""),
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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(score_one_sample, client, sample_id, sample): sample_id
            for sample_id, sample in samples
        }

        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Scoring samples"):
            sample_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                sample = dict(samples)[sample_id]
                result = {
                    "sample_id": sample_id,
                    "gold_answer": extract_final_text(sample.get("gold", [])),
                    "pred_answer": extract_final_text(sample.get("prediction", [])),
                    "judge_score": None,
                    "status": "error",
                    "error": str(e),
                }

            results.append(result)
            completed += 1

            if completed % SAVE_EVERY == 0 or completed == len(samples):
                save_all(results, paths)

    summary = build_summary(results)

    print(f"Saved raw JSON results to: {paths['raw_json']}")
    print(f"Saved summary TXT results to: {paths['summary_txt']}")
    print(f"Saved CSV results to: {paths['csv']}")
    print("")
    print("==== Final Mean ====")
    print(f"Goal Accuracy: {summary['mean_goal_accuracy']:.4f}")
    print(f"Scored: {summary['num_scored']}/{summary['num_samples']}")
    print(f"Errors: {summary['num_errors']}")


if __name__ == "__main__":
    main()