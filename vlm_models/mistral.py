import os
import re
import json
from typing import List, Dict, Any

import torch
from PIL import Image
from openai import OpenAI
from transformers import LlavaForConditionalGeneration, AutoProcessor

MODEL_NAME = "mistral-community/pixtral-12b"
DATA_JSON = "/workspace/vlm_models/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/workspace/vlm_models/MCTA/image"

MAX_NEW_TOKENS = 96
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script.")

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DEVICE = torch.device("cuda:0")


def sanitize_filename(text: str) -> str:
    text = text.strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


RESULT_DIR = "/workspace/vlm_models/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TXT = f"{RESULT_DIR}/{sanitize_filename(MODEL_NAME)}_results.txt"
OUTPUT_JSON = f"{RESULT_DIR}/{sanitize_filename(MODEL_NAME)}_results.json"
OUTPUT_PREDS_TXT = f"{RESULT_DIR}/{sanitize_filename(MODEL_NAME)}_preds_and_scores.txt"


def load_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Empty data file: {json_path}")

    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

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


def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        token=HF_TOKEN if HF_TOKEN else None,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model.eval()
    return model, processor


def build_messages(question: str):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}\nAnswer:"},
            ],
        },
    ]


def move_inputs_to_model_device(inputs: Dict[str, Any], model) -> Dict[str, Any]:
    try:
        target_device = model.language_model.device
    except Exception:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = DEVICE

    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(target_device)
        else:
            out[k] = v
    return out


def run_inference(model, processor, item: Dict[str, Any]) -> str:
    image = Image.open(item["resolved_image"]).convert("RGB")
    messages = build_messages(item["question"])

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = move_inputs_to_model_device(inputs, model)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[:, input_len:]

    pred = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return pred.strip()


def extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    try:
        chunks = []
        for item in response.output:
            if getattr(item, "type", "") != "message":
                continue
            for content in getattr(item, "content", []):
                if hasattr(content, "text") and content.text:
                    chunks.append(content.text)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


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

        text = extract_response_text(response)
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
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
        raise ValueError("OPENAI_API_KEY is not set. Export it in the shell first.")

    eval_client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"DEVICE: {DEVICE}")
    print(f"DTYPE: {DTYPE}")
    print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
    print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")

    data = load_data(DATA_JSON)
    print(f"Total samples: {len(data)}")

    model, processor = load_model_and_processor()

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
                pred = run_inference(model, processor, eval_item)
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