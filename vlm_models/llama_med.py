import os
import re
import json
from typing import List, Dict, Any

import requests
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init


# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "microsoft/llava-med-v1.5-mistral-7b"
MODEL_BASE = None

DATA_JSON = "/workspace/vlm_models/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/workspace/vlm_models/MCTA/image"

MAX_NEW_TOKENS = 96

# Judge with GPT-5.4 via direct Responses API call
EVAL_MODEL_NAME = "gpt-5.4"
EVAL_OUTPUT_TOKENS = 64
BINARY_THRESHOLD = 0.8

CONV_MODE = "mistral_instruct"

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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


def sanitize_filename(text: str) -> str:
    text = text.strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


OUTPUT_TXT = f"./{sanitize_filename(MODEL_NAME)}_results.txt"
OUTPUT_JSON = f"./{sanitize_filename(MODEL_NAME)}_results.json"
OUTPUT_PREDS_TXT = f"./{sanitize_filename(MODEL_NAME)}_preds_and_scores.txt"


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s.+:/()-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


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


def load_model_tokenizer_processor():
    disable_torch_init()

    model_name = get_model_name_from_path(MODEL_NAME)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_NAME,
        model_base=MODEL_BASE,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)

    model.eval()
    return tokenizer, model, image_processor


def build_prompt(question: str, model_config) -> str:
    qs = f"{SYSTEM_PROMPT}\nQuestion: {question}"

    if getattr(model_config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_inference(tokenizer, model, image_processor, item: Dict[str, Any]) -> str:
    image = Image.open(item["resolved_image"]).convert("RGB")
    prompt = build_prompt(item["question"], model.config)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.to(model.device, dtype=DTYPE)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if pred.startswith(prompt):
        pred = pred[len(prompt):].strip()

    return pred


def extract_responses_api_text(resp_json: Dict[str, Any]) -> str:
    # Preferred convenience field if present
    text = resp_json.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: walk response.output[*].content[*].text
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
    if not OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY in the environment.")

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"EVAL_MODEL_NAME: {EVAL_MODEL_NAME}")
    print(f"DATA_JSON: {DATA_JSON}")
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"OUTPUT_JSON: {OUTPUT_JSON}")
    print(f"OUTPUT_TXT: {OUTPUT_TXT}")
    print(f"OUTPUT_PREDS_TXT: {OUTPUT_PREDS_TXT}")
    print(f"HF_TOKEN set: {bool(HF_TOKEN)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"DEVICE: {DEVICE}")
    print(f"DTYPE: {DTYPE}")
    print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
    print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")
    print(f"Example image exists: {os.path.exists(os.path.join(IMAGE_ROOT, 'image_1.jpg'))}")

    data = load_data(DATA_JSON)
    print(f"Total samples: {len(data)}")

    tokenizer, model, image_processor = load_model_tokenizer_processor()

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

                pred = run_inference(tokenizer, model, image_processor, eval_item)
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