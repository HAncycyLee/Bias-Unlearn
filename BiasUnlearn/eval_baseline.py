import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Evaluator import BiasEvaluator, ScoreEvaluator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "facebook/opt-1.3b"
# GOLD_FILE = "StereoSet/dev.json"
# OUT_DIR  = "outputs/baseline_opt13b_dev"
MODEL_NAME = "openai-community/gpt2-large"
GOLD_FILE = "StereoSet/dev.json"
OUT_DIR  = "outputs/baseline_gpt2-large_dev"
OUT_FILE = f"{OUT_DIR}/predictions.json"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else None,
        device_map="auto" if DEVICE == "cuda" else None
    ).eval()

    evaluator = BiasEvaluator(
        pretrained_class=MODEL_NAME,
        no_cuda=(DEVICE!="cuda"),
        batch_size=4,
        input_file=GOLD_FILE,
        intrasentence_model="GPT2LM",
        intrasentence_load_path=MODEL_NAME,
        tokenizer=tok,               # 关键：传 tokenizer 对象
        skip_intersentence=True,
        output_dir=OUT_DIR
    )

    preds = evaluator.evaluate(model)

    with open(OUT_FILE, "w") as f:
        json.dump(preds, f, indent=2)
    print("saved:", OUT_FILE)

    sc = ScoreEvaluator(preds, gold_file_path=GOLD_FILE)
    res = sc.get_overall_results()
    sc.pretty_print(res)

if __name__ == "__main__":
    main()