import os
import re
import io
import json
import base64
from typing import List, Dict, Any

from PIL import Image
from openai import OpenAI

# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "MBZUAI/BiMediX2-8B-hf"
DATA_JSON = "/workspace/vlm_models/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/workspace/vlm_models/MCTA/image"

BASE_URL = "http://localhost:8000/v1"
API_KEY = "DUMMY_KEY"

MAX_TOKENS = 96
TEMPERATURE = 0.1
TOP_P = 0.9
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


def image_to_data_url(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def normalize_generated_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*assistant\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s,./+-]", "", s)
    return s


def simple_semantic_score(pred: str, gt: str) -> float:
    p = normalize_text(pred)
    g = normalize_text(gt)

    if not p:
        return 0.0
    if p == g:
        return 1.0
    if g in p:
        return 1.0
    if p in g:
        return 0.8

    p_set = set(re.split(r"[,;/]\s*|\s+", p))
    g_set = set(re.split(r"[,;/]\s*|\s+", g))
    p_set = {x for x in p_set if x}
    g_set = {x for x in g_set if x}

    if not p_set or not g_set:
        return 0.0

    overlap = len(p_set & g_set) / max(len(g_set), 1)
    if overlap >= 0.9:
        return 0.9
    if overlap >= 0.6:
        return 0.7
    if overlap >= 0.3:
        return 0.4
    return 0.0


def run_inference(client: OpenAI, item: Dict[str, Any]) -> str:
    image_data_url = image_to_data_url(item["resolved_image"])

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": item["question"],
                    },
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    text = completion.choices[0].message.content or ""
    return normalize_generated_text(text)


def write_txt_report(summary: Dict[str, Any], txt_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {summary['model_name']}\n")
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
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DATA_JSON: {DATA_JSON}")
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"OUTPUT_JSON: {OUTPUT_JSON}")
    print(f"OUTPUT_TXT: {OUTPUT_TXT}")
    print(f"OUTPUT_PREDS_TXT: {OUTPUT_PREDS_TXT}")
    print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
    print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")

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
                pred = run_inference(client, eval_item)
                semantic_score = simple_semantic_score(pred, item["answer"])
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
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_semantic_score": avg_semantic_score,
        "binary_threshold": BINARY_THRESHOLD,
        "metric": (
            "Simple local semantic score in [0,1]. "
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