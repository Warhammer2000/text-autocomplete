from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_utils import build_vocabulary, encode_text, get_special_token_ids
from .metrics import compute_rouge
from .lstm_model import LSTMLanguageModel
from .next_token_dataset import NextTokenDataset, collate_batch


def set_seed(seed: int) -> None:

	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_texts(csv_path: str) -> List[str]:

	df = pd.read_csv(csv_path)
	if "text" not in df.columns:
		raise ValueError(f"CSV {csv_path} must contain 'text' column")
	return df["text"].astype(str).tolist()


def create_dataloaders(
	train_texts: List[str],
	val_texts: List[str],
	test_texts: List[str] | None,
	sequence_length: int,
	batch_size: int,
	max_vocab_size: int,
	device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, Dict[str, int], int]:


	token_to_id, _ = build_vocabulary(train_texts, max_vocab_size=max_vocab_size)
	special_ids = get_special_token_ids(token_to_id)


	train_seqs = [encode_text(t, token_to_id) for t in train_texts]
	val_seqs = [encode_text(t, token_to_id) for t in val_texts]
	test_seqs = [encode_text(t, token_to_id) for t in (test_texts or [])]

	train_ds = NextTokenDataset(train_seqs, sequence_length=sequence_length, pad_token_id=special_ids["pad"])
	val_ds = NextTokenDataset(val_seqs, sequence_length=sequence_length, pad_token_id=special_ids["pad"])
	test_ds = NextTokenDataset(test_seqs, sequence_length=sequence_length, pad_token_id=special_ids["pad"]) if test_seqs else None

	# Fallback: if sliding-window datasets are empty (too-short texts), build fixed-window padded samples
	if len(train_ds) == 0 or len(val_ds) == 0 or (test_ds is not None and len(test_ds) == 0):
		from typing import Sequence, Tuple as Tup

		def build_fixed_window_samples(seqs: Sequence[List[int]], seq_len: int, pad_id: int, eos_id: int) -> List[Tup[List[int], List[int]]]:
			samples: List[Tup[List[int], List[int]]] = []
			for s in seqs:
				if not s:
					continue
				x = list(s[:seq_len])
				y = list(s[1 : seq_len + 1])

				if len(x) < seq_len:
					x = x + [pad_id] * (seq_len - len(x))

				if len(y) < seq_len:
					missing = seq_len - len(y)
					if missing > 0:
						y = y + [eos_id] + [pad_id] * max(0, missing - 1)
				samples.append((x, y))
			return samples

		class FixedWindowDataset(torch.utils.data.Dataset):
			def __init__(self, samples: List[Tuple[List[int], List[int]]]) -> None:
				self.samples = samples
			def __len__(self) -> int:
				return len(self.samples)
			def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
				x, y = self.samples[idx]
				return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

		pad_id = special_ids["pad"]
		eos_id = special_ids["eos"]
		if len(train_ds) == 0:
			train_samples = build_fixed_window_samples(train_seqs, sequence_length, pad_id, eos_id)
			train_ds = FixedWindowDataset(train_samples)
		if len(val_ds) == 0:
			val_samples = build_fixed_window_samples(val_seqs, sequence_length, pad_id, eos_id)
			val_ds = FixedWindowDataset(val_samples)
		if test_ds is not None and len(test_ds) == 0:
			test_samples = build_fixed_window_samples(test_seqs, sequence_length, pad_id, eos_id)
			test_ds = FixedWindowDataset(test_samples)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=lambda b: collate_batch(b, pad_token_id=special_ids["pad"]),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=lambda b: collate_batch(b, pad_token_id=special_ids["pad"]),
	)
	test_loader = None
	if test_ds is not None:
		test_loader = DataLoader(
			test_ds,
			batch_size=batch_size,
			shuffle=False,
			num_workers=0,
			collate_fn=lambda b: collate_batch(b, pad_token_id=special_ids["pad"]),
		)
	return train_loader, val_loader, test_loader, token_to_id, len(token_to_id)


def train_one_epoch(
	model: LSTMLanguageModel,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
) -> float:

	model.train()
	running_loss = 0.0
	batches = 0
	for inputs, targets in tqdm(loader, total=len(loader), desc="train", leave=False):
		inputs = inputs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad()
		logits = model(inputs)
		loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
		loss.backward()
		optimizer.step()
		running_loss += float(loss.item())
		batches += 1
	return running_loss / max(1, batches)


@torch.no_grad()
def evaluate_loss(model: LSTMLanguageModel, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:

	model.eval()
	running_loss = 0.0
	batches = 0
	for inputs, targets in tqdm(loader, total=len(loader), desc="val", leave=False):
		inputs = inputs.to(device)
		targets = targets.to(device)
		logits = model(inputs)
		loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
		running_loss += float(loss.item())
		batches += 1
	return running_loss / max(1, batches)


@torch.no_grad()
def evaluate_rouge(
	model: LSTMLanguageModel,
	texts: List[str],
	token_to_id: Dict[str, int],
	max_new_tokens: int,
	device: torch.device,
) -> Dict[str, float]:

	model.eval()
	id_to_token = {v: k for k, v in token_to_id.items()}
	special_ids = get_special_token_ids(token_to_id)
	preds: List[str] = []
	refs: List[str] = []
	for t in texts[:1000]:
		ids = encode_text(t, token_to_id)
		if len(ids) < 4:
			continue
		cut = int(len(ids) * 0.75)
		prefix, tail = ids[:cut], ids[cut:]
		prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
		gen = model.generate(prefix_tensor, max_new_tokens=max_new_tokens, end_token_id=special_ids["eos"], device=device)[0].tolist()
		gen_tail = gen[len(prefix):]
		def ids_to_text_local(id_list: List[int]) -> str:
			words: List[str] = []
			for i in id_list:
				if i == special_ids["eos"]:
					break
				words.append(id_to_token.get(i, "<unk>"))
			return " ".join(words)
		preds.append(ids_to_text_local(gen_tail))
		refs.append(ids_to_text_local(tail))
	return compute_rouge(preds, refs)


def main() -> None:

	with open("configs/default.yaml", "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	seed = int(cfg.get("training", {}).get("seed", 42))
	set_seed(seed)

	device_str = cfg.get("training", {}).get("device", "cuda")
	device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

	sequence_length = int(cfg.get("tokenization", {}).get("sequence_length", 64))
	batch_size = int(cfg.get("training", {}).get("batch_size", 256))
	learning_rate = float(cfg.get("training", {}).get("learning_rate", 1e-3))
	num_epochs = int(cfg.get("training", {}).get("num_epochs", 5))
	max_vocab_size = int(cfg.get("tokenization", {}).get("max_vocab_size", 30000))

	data_cfg = cfg.get("data", {})
	train_csv = data_cfg.get("train_path", "data/train.csv")
	val_csv = data_cfg.get("val_path", "data/val.csv")
	test_csv = data_cfg.get("test_path", "data/test.csv")

	train_texts = load_texts(train_csv) if os.path.exists(train_csv) else []
	val_texts = load_texts(val_csv) if os.path.exists(val_csv) else []
	test_texts = load_texts(test_csv) if os.path.exists(test_csv) else []

	# Auto-prepare data if splits are empty
	if len(train_texts) == 0 or len(val_texts) == 0:
		print("Datasets are empty. Running src.prepare_data to (re)generate splits...")
		try:
			from importlib import import_module
			prepare = import_module("src.prepare_data")
			prepare.main()
			# Reload after preparing
			train_texts = load_texts(train_csv)
			val_texts = load_texts(val_csv)
			test_texts = load_texts(test_csv) if os.path.exists(test_csv) else []
		except Exception as exc:
			raise RuntimeError("Failed to prepare data automatically. Ensure raw CSV is present in data/.") from exc

	train_loader, val_loader, test_loader, token_to_id, vocab_size = create_dataloaders(
		train_texts,
		val_texts,
		test_texts,
		sequence_length=sequence_length,
		batch_size=batch_size,
		max_vocab_size=max_vocab_size,
		device=device,
	)

	# Safety: if still empty, provide actionable error
	if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
		raise ValueError(
			"No samples produced for training/validation. "
			"Try decreasing tokenization.sequence_length in configs/default.yaml, "
			"or verify that data CSVs contain non-empty 'text' rows."
		)

	model_cfg = cfg.get("model", {})
	embedding_dim = int(model_cfg.get("embedding_dim", 256))
	hidden_dim = int(model_cfg.get("hidden_dim", 128))
	num_layers = int(model_cfg.get("num_layers", 2))
	dropout = float(model_cfg.get("dropout", 0.1))
	pad_id = token_to_id.get("<pad>", 0)

	model = LSTMLanguageModel(
		vocab_size=vocab_size,
		embedding_dim=embedding_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
		pad_token_id=pad_id,
	).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

	for epoch in range(1, num_epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
		val_loss = evaluate_loss(model, val_loader, criterion, device)

		rouge_scores = evaluate_rouge(model, val_texts, token_to_id, max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 20)), device=device)
		print(f"Epoch {epoch:02d}/{num_epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | ROUGE1: {rouge_scores.get('rouge1', 0):.4f} | ROUGE2: {rouge_scores.get('rouge2', 0):.4f}")


		for sample_text in val_texts[:3]:
			ids = encode_text(sample_text, token_to_id)
			if len(ids) < 4:
				continue
			cut = int(len(ids) * 0.75)
			prefix, tail = ids[:cut], ids[cut:]
			gen = model.generate(torch.tensor([prefix], dtype=torch.long, device=device), max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 20)), end_token_id=get_special_token_ids(token_to_id)["eos"], device=device)[0].tolist()
			id_to_token = {v: k for k, v in token_to_id.items()}
			def to_text(id_list: List[int]) -> str:
				words: List[str] = []
				for i in id_list:
					if i == get_special_token_ids(token_to_id)["eos"]:
						break
					words.append(id_to_token.get(i, "<unk>"))
				return " ".join(words)
			print(f"> prefix: {to_text(prefix)}\n> ref:    {to_text(tail)}\n> pred:   {to_text(gen[len(prefix):])}")


	out_dir = Path(cfg.get("output", {}).get("model_dir", "models"))
	out_dir.mkdir(parents=True, exist_ok=True)

	weights_path = out_dir / "lstm_model.pt"
	vocab_path = out_dir / "vocab.json"
	config_path = out_dir / "lstm_config.json"

	torch.save(model.state_dict(), str(weights_path))
	with open(vocab_path, "w", encoding="utf-8") as f:
		json.dump(token_to_id, f, ensure_ascii=False)
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump({
			"vocab_size": vocab_size,
			"embedding_dim": embedding_dim,
			"hidden_dim": hidden_dim,
			"num_layers": num_layers,
			"dropout": dropout,
			"pad_token_id": pad_id,
			"sequence_length": sequence_length,
		}, f, ensure_ascii=False)

	print(f"Saved model to {weights_path}")
	print(f"Saved vocab to {vocab_path}")
	print(f"Saved config to {config_path}")


	if test_texts:
		print("\nFinal evaluation (test, ROUGE):")
		final_scores = evaluate_rouge(model, test_texts, token_to_id, max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 20)), device=device)
		print(final_scores)


if __name__ == "__main__":
	main()