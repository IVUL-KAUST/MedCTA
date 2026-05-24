import os
import re
import json
import sys
from typing import List, Dict, Any

import requests
import torch
from PIL import Image
from einops import repeat
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

# IMPORTANT: run this file from inside the cloned med-flamingo repo,
# or keep these sys.path additions as-is.
REPO_ROOT = "/workspace/vlm_models/med-flamingo"
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))

from src.utils import FlamingoProcessor  # noqa: E402


# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "med-flamingo/med-flamingo"
DATA_JSON = "/workspace/vlm_models/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/workspace/vlm_models/MCTA/image"

# Set this to your local HF-converted Llama-7B v1 path
LLAMA_7B_PATH = "/workspace/models/llama-7b-hf"

MAX_NEW_TOKENS = 96
BINARY_THRESHOLD = 0.8

# Judge via OpenAI Responses HTTP API
EVAL_MODEL_NAME = "gpt-5.4"
EVAL_OUTPUT_TOKENS = 64

SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. "
    "Answer the question strictly based on the visual evidence in the image. "
    "Provide precise, concise, and clinically accurate answers. "
    "Do not add explanations, reasoning, or extra details. "
    "Do not speculate beyond what is visible. "e 
    "If multiple items are asked, list them clearly, separated by commas. "
    "Use standard medical terminology. "
    "Return only the final answer."
)

GOAL_ACCURACY_SYSTEM_PROMPT = """You are a medical answer evaluator.

Compare the predicted FINAL answer against the gold FINAL clinical answer.
Assign a score from 0.0 to 1.0 based on semantic clinical correctness.

CRITICAL RULE:
- If the predicted answer explicitly contains the correct gold answer, assign a score of 1.0.
- Presence of the correct diagnosis/finding overrides extra guesses unless contradictory.

General rules:
- Give partial credit if only partially correct.
- Do NOT give 0.0 unless completely wrong or unrelated.
- Judge by clinical meaning, not wording.
- Synonyms count as correct.

Scoring guide:
- 1.0 = gold answer clearly present OR fully correct
- 0.8-0.95 = correct but minor imprecision
- 0.5-0.75 = partially correct
- 0.2-0.45 = weak overlap
- 0.0-0.1 = wrong/unrelated

Return JSON only:
{
  "score": number
}
"""

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

RESULT_DIR = "/workspace/vlm_models/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TXT = f"{RESULT_DIR}/med_flamingo_results.txt"
OUTPUT_JSON = f"{RESULT_DIR}/med_flamingo_results.json"
OUTPUT_PREDS_TXT = f"{RESULT_DIR}/med_flamingo_preds_and_scores.txt"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s.+:/(),-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def sanitize_filename(text: str) -> str:
    text = text.strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def load_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Empty data file: {json_path}")

    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

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


def clean_generation(text: str) -> str:
    text = text.strip()

    # Remove common Flamingo prompt remnants
    if "<answer>" in text.lower():
        text = text.split("<answer>", 1)[-1].strip()

    for marker in [
        "<|endofchunk|>",
        "<image>",
        "Question:",
    ]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    return text.strip()


def load_model_and_processor():
    accelerator = Accelerator()
    device = accelerator.device

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=LLAMA_7B_PATH,
        tokenizer_path=LLAMA_7B_PATH,
        cross_attn_every_n_layers=4,
    )

    checkpoint_path = hf_hub_download(MODEL_NAME, "model.pt", token=HF_TOKEN if HF_TOKEN else None)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)

    processor = FlamingoProcessor(tokenizer, image_processor)

    model = accelerator.prepare(model)
    model.eval()

    return accelerator, model, processor


def build_prompt(question: str) -> str:
    # Zero-shot single-image prompt in Med-Flamingo style
    return f"<image>{SYSTEM_PROMPT} Question: {question} Answer:"


def run_inference(accelerator, model, processor, item: Dict[str, Any]) -> str:
    image = Image.open(item["resolved_image"]).convert("RGB")
    prompt = build_prompt(item["question"])

    pixels = processor.preprocess_images([image])
    # Expected shape in demo path: b N T c h w
    pixels = repeat(pixels, "N c h w -> b N T c h w", b=1, T=1)

    tokenized = processor.encode_text(prompt)

    vision_x = pixels.to(accelerator.device)
    lang_x = tokenized["input_ids"].to(accelerator.device)
    attention_mask = tokenized["attention_mask"].to(accelerator.device)

    with torch.inference_mode():
        generated = model.generate(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
    pred = clean_generation(decoded)

    # Remove prompt echo if present
    if "Answer:" in pred:
        pred = pred.split("Answer:", 1)[-1].strip()

    return pred.strip()


def extract_responses_api_text(resp_json: Dict[str, Any]) -> str:
    text = resp_json.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks = []
    for item in resp_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            ctext = content.get("text")
            if isinstance(ctext, str) and ctext:
                chunks.append(ctext)
    return "\n".join(chunks).strip()


def evaluate_prediction(pred: str, gt: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    payload = {
        "model": EVAL_MODEL_NAME,
        "instructions": GOAL_ACCURACY_SYSTEM_PROMPT,
        "input": f"Gold answer: {gt_norm}\nPredicted answer: {pred_norm}",
        "max_output_tokens": EVAL_OUTPUT_TOKENS,
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "medical_eval_score",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["score"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            print("EVAL_HTTP_STATUS:", response.status_code)
            print("EVAL_HTTP_BODY:", response.text[:2000])
            return 0.0

        resp_json = response.json()
        raw = extract_responses_api_text(resp_json)
        print("RAW_EVAL_OUTPUT:", raw)

        data = json.loads(raw)
        score = float(data["score"])
        return max(0.0, min(1.0, score))

    except Exception as e:
        print("EVAL_ERROR:", repr(e))
        print("GT:", gt)
        print("PRED:", pred)
        return 0.0


def write_txt_report(summary: Dict[str, Any], txt_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {summary['model_name']}\n")
        f.write(f"Eval Model: {summary['eval_model_name']}\n")
        f.write(f"Total: {summary['total']}\n")
        f.write(f"Correct: {summary['correct']}\n")
        f.write(f"Final Accuracy: {summary['accuracy']:.6f}\n")
        f.write(f"Average Semantic Score: {summary['avg_semantic_score']:.6f}\n")
        f.write(f"Binary Threshold: {summary['binary_threshold']:.2f}\n")
        f.write(f"Metric: {summary['metric']}\n\n")
        f.write("Detailed results\n")
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
            f.write(f"Binary Match: {row['match']}\n\n")


def main() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    if not os.path.exists(LLAMA_7B_PATH):
        raise ValueError(f"LLAMA_7B_PATH does not exist: {LLAMA_7B_PATH}")

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"EVAL_MODEL_NAME: {EVAL_MODEL_NAME}")
    print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
    print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")
    print(f"LLAMA_7B_PATH exists: {os.path.exists(LLAMA_7B_PATH)}")

    data = load_data(DATA_JSON)
    print(f"Total samples: {len(data)}")

    accelerator, model, processor = load_model_and_processor()

    results: List[Dict[str, Any]] = []

    for item in data:
        resolved_image = resolve_image_path(item["image"], IMAGE_ROOT)
        pred, semantic_score, binary_score, error = "", 0.0, 0, ""

        if not os.path.exists(resolved_image):
            error = f"missing_image: {resolved_image}"
        else:
            try:
                eval_item = dict(item)
                eval_item["resolved_image"] = resolved_image

                pred = run_inference(accelerator, model, processor, eval_item)
                semantic_score = evaluate_prediction(pred, item["answer"])
                binary_score = 1 if semantic_score >= BINARY_THRESHOLD else 0
            except Exception as e:
                error = f"runtime_error: {repr(e)}"
                print(error)

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

    results = sorted(
        results,
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else str(x["id"])
    )

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