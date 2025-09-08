from __future__ import annotations

from typing import List

import torch
from torch import nn


class LSTMLanguageModel(nn.Module):

	def __init__(
		self,
		vocab_size: int,
		embedding_dim: int = 256,
		hidden_dim: int = 128,
		num_layers: int = 2,
		dropout: float = 0.1,
		pad_token_id: int = 0,
	) -> None:

		super().__init__()
		self.pad_token_id = int(pad_token_id)
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_id)
		self.lstm = nn.LSTM(
			input_size=embedding_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)
		self.proj = nn.Linear(hidden_dim, vocab_size)

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  

		emb = self.embedding(input_ids)
		output, _ = self.lstm(emb)
		logits = self.proj(output)    
		return logits

	@torch.no_grad()
	def generate(
		self,
		prefix_ids: torch.Tensor,
		max_new_tokens: int = 20,
		end_token_id: int | None = None,
		device: torch.device | None = None,
	) -> torch.Tensor:

		self.eval()
		if device is not None:
			prefix_ids = prefix_ids.to(device)

		generated = prefix_ids.clone()
		for _ in range(max_new_tokens):
			logits = self.forward(generated[:, -prefix_ids.shape[1]:])
			next_token_logits = logits[:, -1, :]
			next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
			generated = torch.cat([generated, next_token], dim=1)
			if end_token_id is not None:
				if torch.all(next_token.squeeze(1) == end_token_id):
					break
		return generated


__all__ = ["LSTMLanguageModel"]