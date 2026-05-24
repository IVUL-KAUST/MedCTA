import os
import re
import json
from typing import List, Dict, Any

from google import genai
from google.genai import types
from openai import OpenAI

# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "gemini-3-flash-preview"
DATA_JSON = "/home2/tajamul/VLM_evaluation/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/home2/tajamul/VLM_evaluation/MCTA/image"

MAX_OUTPUT_TOKENS = 96

EVAL_MODEL_NAME = "gpt-5.4"
EVAL_OUTPUT_TOKENS = 64
BINARY_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. "
    "Answer the question strictly based on the visual evidence in the image. "
    "Provide precise, concise, and clinically accurate answers. "
    "Do not add explanations, reasoning, or extra details. "
    "Do not speculate beyond what is visible. "
    "If multiple items are asked, list them clearly, separated by commas. "
    "Use standard medical terminology. "
    "Return only the final answer."
)

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def sanitize_filename(text: str) -> str:
    text = text.strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


OUTPUT_TXT = f"./{sanitize_filename(MODEL_NAME)}_results.txt"
OUTPUT_JSON = f"./{sanitize_filename(MODEL_NAME)}_results.json"
OUTPUT_PREDS_TXT = f"./{sanitize_filename(MODEL_NAME)}_preds_and_scores.txt"


def load_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Empty data file: {json_path}")

    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not isinstance(data, list):
        raise ValueError("Data must be a JSON list or JSONL records")

    required_keys = {"id", "image", "question", "answer"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Sample {i} missing keys: {sorted(missing)}")

    return data


def resolve_image_path(img_path: str, image_root: str) -> str:
    candidates = []
    if os.path.isabs(img_path):
        candidates.append(img_path)
    else:
        candidates.append(os.path.join(image_root, img_path))
        candidates.append(os.path.join(image_root, os.path.basename(img_path)))
        candidates.append(os.path.abspath(img_path))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return candidates[0]


def evaluate_prediction(client: OpenAI, pred: str, gt: str) -> float:
    try:
        response = client.responses.create(
            model=EVAL_MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": GOAL_ACCURACY_SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Gold answer: {gt}\nPredicted answer: {pred}",
                        }
                    ],
                },
            ],
            max_output_tokens=EVAL_OUTPUT_TOKENS,
        )

        text = getattr(response, "output_text", "").strip()
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def run_inference(client: genai.Client, item: Dict[str, Any]) -> str:
    image = types.Part.from_bytes(
        data=open(item["resolved_image"], "rb").read(),
        mime_type=_guess_mime_type(item["resolved_image"]),
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            image,
            item["question"],
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        ),
    )

    text = getattr(response, "text", "")
    return (text or "").strip()


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def write_txt_report(summary: Dict[str, Any], txt_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {summary['model_name']}\n")
        f.write(f"Eval Model: {summary['eval_model_name']}\n")
        f.write(f"Total: {summary['total']}\n")
        f.write(f"Correct: {summary['correct']}\n")
        f.write(f"Final Accuracy: {summary['accuracy']:.6f}\n")
        f.write(f"Average Semantic Score: {summary['avg_semantic_score']:.6f}\n")
        f.write(f"Binary Threshold: {summary['binary_threshold']:.2f}\n")
        f.write(f"Metric: {summary['metric']}\n")
        f.write("\nDetailed results\n")
        f.write("=" * 100 + "\n")

        for row in summary["results"]:
            f.write(f"ID: {row['id']}\n")
            f.write(f"Image: {row['image']}\n")
            f.write(f"Resolved image: {row['resolved_image']}\n")
            f.write(f"Question: {row['question']}\n")
            f.write(f"Ground truth: {row['ground_truth']}\n")
            f.write(f"Prediction: {row['prediction']}\n")
            f.write(f"Semantic Score: {row['semantic_score']:.3f}\n")
            f.write(f"Binary Match: {row['match']}\n")
            f.write(f"Error: {row['error']}\n")
            f.write("-" * 100 + "\n")


def write_preds_scores_txt(results: List[Dict[str, Any]], txt_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(f"ID: {row['id']}\n")
            f.write(f"Prediction: {row['prediction']}\n")
            f.write(f"Semantic Score: {row['semantic_score']:.3f}\n")
            f.write(f"Binary Match: {row['match']}\n")
            f.write("\n")


def main() -> None:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please export it before running.")

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    eval_client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"EVAL_MODEL_NAME: {EVAL_MODEL_NAME}")
    print(f"DATA_JSON: {DATA_JSON}")
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"OUTPUT_JSON: {OUTPUT_JSON}")
    print(f"OUTPUT_TXT: {OUTPUT_TXT}")
    print(f"OUTPUT_PREDS_TXT: {OUTPUT_PREDS_TXT}")
    print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
    print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")
    print(f"Example image exists: {os.path.exists(os.path.join(IMAGE_ROOT, 'image_1.jpg'))}")

    data = load_data(DATA_JSON)
    print(f"Total samples: {len(data)}")

    results: List[Dict[str, Any]] = []

    for item in data:
        resolved_image = resolve_image_path(item["image"], IMAGE_ROOT)

        pred = ""
        semantic_score = 0.0
        binary_score = 0
        error = ""

        if not os.path.exists(resolved_image):
            error = f"missing_image: {resolved_image}"
        else:
            try:
                eval_item = dict(item)
                eval_item["resolved_image"] = resolved_image
                pred = run_inference(gemini_client, eval_item)
                semantic_score = evaluate_prediction(eval_client, pred, item["answer"])
                binary_score = 1 if semantic_score >= BINARY_THRESHOLD else 0
            except Exception as e:
                error = f"inference_error: {repr(e)}"

        row = {
            "id": item["id"],
            "image": item["image"],
            "resolved_image": resolved_image,
            "question": item["question"],
            "ground_truth": item["answer"],
            "prediction": pred,
            "semantic_score": semantic_score,
            "match": binary_score,
            "error": error,
        }

        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    def sort_key(x):
        try:
            return int(x["id"])
        except Exception:
            return str(x["id"])

    results = sorted(results, key=sort_key)

    total = len(results)
    correct = sum(int(x["match"]) for x in results)
    accuracy = correct / total if total else 0.0
    avg_semantic_score = (
        sum(float(x["semantic_score"]) for x in results) / total if total else 0.0
    )

    summary = {
        "model_name": MODEL_NAME,
        "eval_model_name": EVAL_MODEL_NAME,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_semantic_score": avg_semantic_score,
        "binary_threshold": BINARY_THRESHOLD,
        "metric": (
            "LLM semantic clinical evaluation score in [0,1]. "
            f"Binary correct = 1 if semantic_score >= {BINARY_THRESHOLD}, else 0. "
            "Final accuracy = correct / total."
        ),
        "results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_txt_report(summary, OUTPUT_TXT)
    write_preds_scores_txt(results, OUTPUT_PREDS_TXT)

    print("=" * 80)
    print(json.dumps({
        "total": total,
        "correct": correct,
        "final_accuracy": accuracy,
        "avg_semantic_score": avg_semantic_score,
        "output_json": OUTPUT_JSON,
        "output_txt": OUTPUT_TXT,
        "output_preds_txt": OUTPUT_PREDS_TXT,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()