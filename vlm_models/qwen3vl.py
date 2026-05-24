import os
import re
import json
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from openai import OpenAI

# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATA_JSON = "/home2/tajamul/VLM_evaluation/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/home2/tajamul/VLM_evaluation/MCTA/image"

BATCH_SIZE = 1
MAX_NEW_TOKENS = 96
NUM_WORKERS = 2
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def sanitize_filename(text: str) -> str:
    text = text.strip().replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


OUTPUT_TXT = f"./{sanitize_filename(MODEL_NAME)}_results.txt"
OUTPUT_JSON = f"./{sanitize_filename(MODEL_NAME)}_results.json"
OUTPUT_PREDS_TXT = f"./{sanitize_filename(MODEL_NAME)}_preds_and_scores.txt"


def setup_distributed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


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


class EvalDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], image_root: str):
        self.samples = samples
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.samples[idx])
        img_path = item["image"]

        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        else:
            candidates.append(os.path.join(self.image_root, img_path))
            candidates.append(os.path.join(self.image_root, os.path.basename(img_path)))
            candidates.append(os.path.abspath(img_path))

        resolved = None
        for cand in candidates:
            if os.path.exists(cand):
                resolved = cand
                break

        if resolved is None:
            resolved = candidates[0]

        item["resolved_image"] = resolved
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )

    model.eval()
    return model, processor


def build_messages(question: str, image_path: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]


def run_inference(model, processor, item: Dict[str, Any], device: torch.device) -> str:
    messages = build_messages(item["question"], item["resolved_image"])

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    pred = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred.strip()


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


def gather_results(results: List[Dict[str, Any]], world_size: int, rank: int) -> List[Dict[str, Any]]:
    if world_size == 1:
        return results

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(results, gathered, dst=0)

    if rank == 0:
        merged = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    return []


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
            f.write(f"Rank: {row['rank']}\n")
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
    dist_info = setup_distributed()
    world_size = dist_info["world_size"]
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please export it before running.")

    eval_client = OpenAI(api_key=OPENAI_API_KEY)

    if rank == 0:
        print(f"MODEL_NAME: {MODEL_NAME}")
        print(f"EVAL_MODEL_NAME: {EVAL_MODEL_NAME}")
        print(f"DATA_JSON: {DATA_JSON}")
        print(f"IMAGE_ROOT: {IMAGE_ROOT}")
        print(f"OUTPUT_JSON: {OUTPUT_JSON}")
        print(f"OUTPUT_TXT: {OUTPUT_TXT}")
        print(f"OUTPUT_PREDS_TXT: {OUTPUT_PREDS_TXT}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"WORLD_SIZE: {world_size}")
        print(f"DATA_JSON exists: {os.path.exists(DATA_JSON)}")
        print(f"IMAGE_ROOT exists: {os.path.exists(IMAGE_ROOT)}")
        print(f"Example image exists: {os.path.exists(os.path.join(IMAGE_ROOT, 'image_1.jpg'))}")

    data = load_data(DATA_JSON)
    dataset = EvalDataset(data, IMAGE_ROOT)

    indices = list(range(rank, len(dataset), world_size))
    local_samples = [dataset[i] for i in indices]

    loader = DataLoader(
        local_samples,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    if rank == 0:
        print(f"Total samples: {len(data)}")
        print(f"Local samples on rank 0: {len(local_samples)}")

    model, processor = load_model_and_processor()
    model.to(device)

    local_results: List[Dict[str, Any]] = []

    for batch in loader:
        for item in batch:
            pred = ""
            semantic_score = 0.0
            binary_score = 0
            error = ""

            if not os.path.exists(item["resolved_image"]):
                error = f"missing_image: {item['resolved_image']}"
            else:
                try:
                    pred = run_inference(model, processor, item, device)
                    semantic_score = evaluate_prediction(eval_client, pred, item["answer"])
                    binary_score = 1 if semantic_score >= BINARY_THRESHOLD else 0
                except Exception as e:
                    error = f"inference_error: {repr(e)}"

            row = {
                "id": item["id"],
                "image": item["image"],
                "resolved_image": item["resolved_image"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": pred,
                "semantic_score": semantic_score,
                "match": binary_score,
                "error": error,
                "rank": rank,
            }

            local_results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    merged = gather_results(local_results, world_size, rank)

    if rank == 0:
        def sort_key(x):
            try:
                return int(x["id"])
            except Exception:
                return str(x["id"])

        merged = sorted(merged, key=sort_key)
        total = len(merged)
        correct = sum(int(x["match"]) for x in merged)
        accuracy = correct / total if total else 0.0
        avg_semantic_score = (
            sum(float(x["semantic_score"]) for x in merged) / total if total else 0.0
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
            "results": merged,
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        write_txt_report(summary, OUTPUT_TXT)
        write_preds_scores_txt(merged, OUTPUT_PREDS_TXT)

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

    cleanup_distributed()


if __name__ == "__main__":
    main()