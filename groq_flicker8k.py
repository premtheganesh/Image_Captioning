import os
import time
import json

from groq import Groq
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.spice.spice import Spice

# 1. Initialize Groq client
client = Groq(api_key=os.environ["ENTER_YOUR_GROQ_API_KEY(Removed ours)"])

# 2. List all models via the Models API and pick those under "meta-llama"
resp = client.get("/openai/v1/models")
all_models = resp.json()["data"]  # :contentReference[oaicite:0]{index=0}
vlm_models = [m["id"] for m in all_models
              if m["id"].startswith("meta-llama/llama-4")]

# For reference, Groq’s Vision page lists:
#   meta-llama/llama-4-scout-17b-16e-instruct
#   meta-llama/llama-4-maverick-17b-128e-instruct :contentReference[oaicite:1]{index=1}

# 3. Prepare Flickr8k: download and make JSONL for fine-tuning
dataset = load_dataset("atasoglu/flickr8k-dataset", data_dir="data") # splits: train, validation, test

def write_jsonl(split, path):
    with open(path, "w") as f:
        for item in dataset[split]:
            # use first caption as target
            caption = item["caption"][0]
            url = item["image_url"]
            # Our prompt: feed the image, then ask for description
            prompt = [
                {"type":"image_url","image_url":{"url": url}},
                {"type":"text","text":"Describe the image."}
            ]
            record = {"prompt": prompt, "completion": caption}
            f.write(json.dumps(record) + "\n")

os.makedirs("f8k_jsonl", exist_ok=True)
train_file = "f8k_jsonl/train.jsonl"
write_jsonl("train", train_file)

# 4. Upload training file for fine-tuning
upload = client.post(
    "/openai/v1/files",
    json={"purpose":"fine-tune"},
    files={"file": open(train_file, "rb")}
)
file_id = upload.json()["id"]

# 5. Kick off fine-tunes
fine_tuned = {}
for base_model in vlm_models:
    job = client.post(
        "/openai/v1/fine-tunes",
        json={
            "training_file": file_id,
            "model": base_model,
            "n_epochs": 3,
            "batch_size": 8
        }
    ).json()
    job_id = job["id"]
    # poll until done
    status = None
    while status not in {"succeeded","failed"}:
        time.sleep(30)
        status = client.get(f"/openai/v1/fine-tunes/{job_id}").json()["status"]
    if status == "succeeded":
        fine_model = client.get(f"/openai/v1/fine-tunes/{job_id}").json()["fine_tuned_model"]
        fine_tuned[base_model] = fine_model

# 6. Run inference on test set and evaluate
test_ds = dataset["test"]
references = [[cap] for item in test_ds for cap in [item["caption"][0]]]
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
spice = Spice()

results = {}

for base_model, ft_model in fine_tuned.items():
    preds = []
    for item in test_ds:
        msg = [
            {"type":"image_url","image_url":{"url": item["image_url"]}},
            {"type":"text","text":"Describe the image."}
        ]
        out = client.chat.completions.create(
            model=ft_model,
            messages=[{"role":"user","content": msg}]
        )
        preds.append(out.choices[0].message.content)

    # BLEU
    cw = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, preds, weights=(1,0,0,0), smoothing_function=cw)
    bleu2 = corpus_bleu(references, preds, weights=(0.5,0.5,0,0), smoothing_function=cw)
    bleu3 = corpus_bleu(references, preds, weights=(0.33,0.33,0.33,0), smoothing_function=cw)
    bleu4 = corpus_bleu(references, preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=cw)
    # METEOR
    meteor = sum(meteor_score([ref[0]], pred) for ref, pred in zip(references, preds)) / len(preds)
    # ROUGE-L
    rouge_l = sum(sc.score(ref[0], pred)["rougeL"].fmeasure for ref, pred in zip(references, preds)) / len(preds)
    # SPICE (slower)
    spice_score = spice.compute_score(
        {i:[ref[0]] for i, ref in enumerate(references)},
        {i:pred for i, pred in enumerate(preds)}
    )[1]  # average

    results[ft_model] = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor,
        "ROUGE-L": rouge_l,
        "SPICE": spice_score
    }

# 7. Print out
for m, scores in results.items():
    print(f"\n=== Results for {m} ===")
    for metric, v in scores.items():
        print(f"{metric}: {v:.4f}")
