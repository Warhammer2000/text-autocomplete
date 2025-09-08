from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple


SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>", "<eos>"]


def clean_text(text: str) -> str:

	import re

	normalized = str(text).lower()
	normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
	normalized = re.sub(r"@[\w_]+", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	return normalized


def tokenize(text: str) -> List[str]:

	if not text:
		return []
	return text.split()


def build_vocabulary(texts: Iterable[str], max_vocab_size: int = 30000, min_frequency: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:

	token_counter: Counter = Counter()
	for raw in texts:
		cleaned = clean_text(raw)
		tokens = tokenize(cleaned)
		token_counter.update(tokens)

	most_common = [token for token, freq in token_counter.most_common() if freq >= min_frequency]
	vocab_tokens = SPECIAL_TOKENS + most_common[: max(0, max_vocab_size - len(SPECIAL_TOKENS))]

	token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(vocab_tokens)}
	id_to_token: Dict[int, str] = {idx: token for token, idx in token_to_id.items()}
	return token_to_id, id_to_token


def encode_text(text: str, token_to_id: Dict[str, int], add_eos: bool = True) -> List[int]:

	tokens = tokenize(clean_text(text))
	unk_id = token_to_id.get("<unk>", 1)
	ids = [token_to_id.get(token, unk_id) for token in tokens]
	if add_eos and "<eos>" in token_to_id:
		ids.append(token_to_id["<eos>"])
	return ids


def create_input_target_windows(token_ids: List[int], sequence_length: int) -> List[Tuple[List[int], List[int]]]:

	pairs: List[Tuple[List[int], List[int]]] = []
	if sequence_length < 2:
		return pairs
	for start in range(0, max(0, len(token_ids) - 1)):
		end = start + sequence_length
		window = token_ids[start:end]
		target_window = token_ids[start + 1 : end + 1]
		if len(window) < sequence_length:
			break
		if len(target_window) < sequence_length:
			break
		pairs.append((window, target_window))
	return pairs


def get_special_token_ids(token_to_id: Dict[str, int]) -> Dict[str, int]:

	return {
		"pad": token_to_id.get("<pad>", 0),
		"unk": token_to_id.get("<unk>", 1),
		"eos": token_to_id.get("<eos>", 2),
	}


__all__ = [
	"clean_text",
	"tokenize",
	"build_vocabulary",
	"encode_text",
	"create_input_target_windows",
	"get_special_token_ids",
]

