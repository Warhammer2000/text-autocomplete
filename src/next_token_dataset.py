from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):

	def __init__(self, sequences: Sequence[List[int]], sequence_length: int, pad_token_id: int = 0) -> None:

		self.sequences = sequences
		self.sequence_length = int(sequence_length)
		self.pad_token_id = int(pad_token_id)

		self.samples: List[Tuple[List[int], List[int]]] = []
		for token_ids in self.sequences:
			if len(token_ids) < 2:
				continue
			for start in range(0, max(0, len(token_ids) - 1)):
				end = start + self.sequence_length
				window = token_ids[start:end]
				target = token_ids[start + 1 : end + 1]
				if len(window) < self.sequence_length or len(target) < self.sequence_length:
					break
				self.samples.append((window, target))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

		window, target = self.samples[idx]
		x = torch.tensor(window, dtype=torch.long)
		y = torch.tensor(target, dtype=torch.long)
		return x, y


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:

	inputs, targets = zip(*batch)
	inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
	targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
	return inputs_padded, targets_padded


__all__ = ["NextTokenDataset", "collate_batch"]