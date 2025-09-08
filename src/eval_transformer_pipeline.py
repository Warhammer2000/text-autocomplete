from __future__ import annotations

import pandas as pd
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

from .metrics import compute_rouge


def main() -> None:

	with open("configs/default.yaml", "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	data_cfg = cfg.get("data", {})
	val_csv = data_cfg.get("val_path", "data/val.csv")

	gen_cfg = cfg.get("transformer", {})
	model_name = gen_cfg.get("model_name", "distilgpt2")
	max_length = int(gen_cfg.get("max_length", 20))
	do_sample = bool(gen_cfg.get("do_sample", True))
	top_k = int(gen_cfg.get("top_k", 50))
	top_p = float(gen_cfg.get("top_p", 0.95))

	texts = pd.read_csv(val_csv)["text"].astype(str).tolist()

	device_index = 0 if torch.cuda.is_available() else -1

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	model = AutoModelForCausalLM.from_pretrained(model_name)
	pl = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device_index)

	preds = []
	refs = []
	for t in tqdm(texts[:200], desc="eval-gpt2", leave=False):
		words = t.split()
		if len(words) < 4:
			continue
		cut = int(len(words) * 0.75)
		prefix = " ".join(words[:cut])
		ref = " ".join(words[cut:])
		out = pl(
			prefix,
			max_new_tokens=max_length,
			do_sample=do_sample,
			top_k=top_k,
			top_p=top_p,
			truncation=True,
			pad_token_id=tokenizer.pad_token_id,
		)
		gen = out[0]["generated_text"][len(prefix):].strip()
		preds.append(gen)
		refs.append(ref)

	scores = compute_rouge(preds, refs)
	print("Transformer ROUGE:", scores)


	print("\nSamples (distilgpt2):")
	for t in texts[:5]:
		ws = t.split()
		if len(ws) < 4:
			continue
		cut = int(len(ws) * 0.75)
		prefix = " ".join(ws[:cut])
		ref = " ".join(ws[cut:])
		out = pl(
			prefix,
			max_new_tokens=max_length,
			do_sample=do_sample,
			top_k=top_k,
			top_p=top_p,
			truncation=True,
			pad_token_id=tokenizer.pad_token_id,
		)
		gen = out[0]["generated_text"][len(prefix):].strip()
		print(f"\n> prefix: {prefix}\n> ref:    {ref}\n> pred:   {gen}")


if __name__ == "__main__":
	main()

