from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .data_utils import encode_text, get_special_token_ids
from .lstm_model import LSTMLanguageModel
from .metrics import compute_rouge


def load_vocab(vocab_path: str) -> Dict[str, int]:

	with open(vocab_path, "r", encoding="utf-8") as f:
		return json.load(f)


@torch.no_grad()
def generate_completion(model: LSTMLanguageModel, prefix_ids: List[int], max_new_tokens: int, eos_id: int | None, device: torch.device) -> List[int]:

	prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
	gen = model.generate(prefix, max_new_tokens=max_new_tokens, end_token_id=eos_id, device=device)
	return gen[0].tolist()


def ids_to_text(ids: List[int], id_to_token: Dict[int, str], stop_id: int | None = None) -> str:

	tokens: List[str] = []
	for i in ids:
		if stop_id is not None and i == stop_id:
			break
		tokens.append(id_to_token.get(i, "<unk>"))
	return " ".join(tokens)


def main() -> None:

	with open("configs/default.yaml", "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("training", {}).get("device", "cuda") == "cuda" else "cpu")

	data_cfg = cfg.get("data", {})
	val_csv = data_cfg.get("val_path", "data/val.csv")

	model_dir = Path(cfg.get("output", {}).get("model_dir", "models"))
	weights_path = model_dir / "lstm_model.pt"
	vocab_path = model_dir / "vocab.json"
	config_path = model_dir / "lstm_config.json"

	if not weights_path.exists():
		raise FileNotFoundError(f"Weights not found: {weights_path}")
	if not vocab_path.exists():
		raise FileNotFoundError(f"Vocab not found: {vocab_path}")

	with open(config_path, "r", encoding="utf-8") as f:
		mcfg = json.load(f)
		i2t = {int(v): k for k, v in load_vocab(str(vocab_path)).items()}

	model = LSTMLanguageModel(
		vocab_size=int(mcfg["vocab_size"]),
		embedding_dim=int(mcfg["embedding_dim"]),
		hidden_dim=int(mcfg["hidden_dim"]),
		num_layers=int(mcfg["num_layers"]),
		dropout=float(mcfg["dropout"]),
		pad_token_id=int(mcfg["pad_token_id"]),
	).to(device)
	model.load_state_dict(torch.load(str(weights_path), map_location=device))
	model.eval()

	val_texts = pd.read_csv(val_csv)["text"].astype(str).tolist()
	t2i = {k: int(v) for k, v in load_vocab(str(vocab_path)).items()}
	special_ids = get_special_token_ids(t2i)
	seq_len = int(mcfg.get("sequence_length", 64))
	max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 20))

	preds: List[str] = []
	refs: List[str] = []

	for text in tqdm(val_texts[:1000], desc="eval-lstm", leave=False):
		ids = encode_text(text, t2i)
		if len(ids) < 4:
			continue
		cut = int(len(ids) * 0.75)
		prefix, tail = ids[:cut], ids[cut:]
		gen_ids = generate_completion(model, prefix, max_new_tokens=max_new_tokens, eos_id=special_ids["eos"], device=device)
		gen_tail = gen_ids[len(prefix):]
		preds.append(ids_to_text(gen_tail, {v: k for k, v in t2i.items()}, stop_id=special_ids["eos"]))
		refs.append(ids_to_text(tail, {v: k for k, v in t2i.items()}, stop_id=special_ids["eos"]))

	scores = compute_rouge(preds, refs)
	print("ROUGE:", scores)


if __name__ == "__main__":
	main()

