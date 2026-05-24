import os
import re
import json
from typing import List, Dict, Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

# =========================
# Hardcoded configuration
# =========================
MODEL_NAME = "UbiquantAI/Fleming-VL-8B"
DATA_JSON = "/workspace/vlm_models/MCTA/vlm_dataset.jsonl"
IMAGE_ROOT = "/workspace/vlm_models/MCTA/image"

MAX_NEW_TOKENS = 128
BINARY_THRESHOLD = 0.8
INPUT_SIZE = 448
MAX_NUM = 12
USE_FLASH_ATTN = False   # keep False unless flash-attn is correctly installed

SYSTEM_PROMPT = (
    "Answer the question strictly based on the visual evidence in the image. "
    "Provide precise, concise, and clinically accurate answers. "
    "Do not add explanations, reasoning, or extra details. "
    "Do not speculate beyond what is visible. "
    "If multiple items are asked, list them clearly, separated by commas. "
    "Use standard medical terminology. "
    "Return only the final answer."
)

REASONING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user a concise "
    "final answer in a short word or phrase. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think>"
    "<answer> answer here </answer>"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_path: str, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_fleming():
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=USE_FLASH_ATTN,
        trust_remote_code=True
    ).eval()

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False
    )

    return model, tokenizer


def extract_final_answer(text: str) -> str:
    text = text.strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s,.-]", "", s)
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


def run_inference(model, tokenizer, item: Dict[str, Any]) -> str:
    pixel_values = load_image(
        item["resolved_image"],
        input_size=INPUT_SIZE,
        max_num=MAX_NUM
    )

    if torch.cuda.is_available():
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    else:
        pixel_values = pixel_values.to(torch.float32)

    full_question = (
        f"{REASONING_PROMPT}\n"
        f"{SYSTEM_PROMPT}\n"
        f"<image>\n"
        f"{item['question']}"
    )

    generation_config = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    response = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config
    )

    return extract_final_answer(response)


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
    model, tokenizer = load_fleming()

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
                pred = run_inference(model, tokenizer, eval_item)
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
            f"Binary correct = 1 if semantic_score >= {BINARY_THRESHOLD}, else 0."
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