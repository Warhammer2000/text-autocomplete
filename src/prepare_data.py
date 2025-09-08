from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from .data_utils import clean_text


def _read_any_csv(raw1: str, raw2: str) -> pd.DataFrame:


	if os.path.exists(raw2):
		df = pd.read_csv(raw2, encoding="latin-1", header=None)
		return df[[5]].rename(columns={5: "text"})


	if os.path.exists(raw1):
		try:
			df = pd.read_csv(raw1)
			if "text" not in df.columns:
				tmp = pd.read_csv(raw1, header=None, encoding="latin-1")
				df = tmp[[5]].rename(columns={5: "text"})
			return df
		except Exception:
			tmp = pd.read_csv(raw1, header=None, encoding="latin-1")
			return tmp[[5]].rename(columns={5: "text"})

	raise FileNotFoundError(
		"Place CSV with 'text' column at data/raw_dataset.csv or Kaggle CSV at data/training.1600000.processed.noemoticon.csv"
	)


def _maybe_limit_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:

	if max_rows is None or max_rows <= 0:
		return df
	return df.iloc[: int(max_rows)].reset_index(drop=True)


def _split_indices(n: int, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

	rng = np.random.default_rng(seed)
	indices = rng.permutation(n)
	n_test = int(n * test_ratio)
	n_val = int(n * val_ratio)
	n_train = max(0, n - n_val - n_test)
	train_idx = indices[:n_train]
	val_idx = indices[n_train : n_train + n_val]
	test_idx = indices[n_train + n_val :]
	return train_idx, val_idx, test_idx


def main() -> None:

	with open("configs/default.yaml", "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	data_cfg = cfg.get("data", {})
	raw1 = data_cfg.get("raw_path", "data/raw_dataset.csv")
	raw2 = "data/training.1600000.processed.noemoticon.csv"
	processed_path = data_cfg.get("processed_path", "data/dataset_processed.csv")
	train_path = data_cfg.get("train_path", "data/train.csv")
	val_path = data_cfg.get("val_path", "data/val.csv")
	test_path = data_cfg.get("test_path", "data/test.csv")
	val_ratio = float(data_cfg.get("val_ratio", 0.1))
	test_ratio = float(data_cfg.get("test_ratio", 0.1))


	max_rows_env = os.environ.get("MAX_ROWS")
	max_rows_cfg = data_cfg.get("max_rows")
	max_rows = int(max_rows_env) if max_rows_env is not None else int(max_rows_cfg) if max_rows_cfg is not None else 1000


	def nonempty_csv(path: str) -> bool:
		try:
			return os.path.exists(path) and pd.read_csv(path, nrows=1).shape[0] > 0
		except Exception:
			return False

	if all(nonempty_csv(p) for p in [train_path, val_path, test_path]):
		print("Data already prepared. Skipping.")
		return

	Path("data").mkdir(parents=True, exist_ok=True)


	df = _read_any_csv(raw1, raw2)
	df = df[["text"]].astype(str)
	df["text"] = df["text"].map(clean_text)
	df = df.dropna().drop_duplicates().reset_index(drop=True)
	df = df[df["text"].astype(str).str.len() > 0].reset_index(drop=True)


	df = _maybe_limit_rows(df, max_rows)
	Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(processed_path, index=False, encoding="utf-8")
	print(f"Processed rows: {len(df)} (saved to {processed_path})")


	train_idx, val_idx, test_idx = _split_indices(len(df), val_ratio, test_ratio, seed=int(cfg.get("training", {}).get("seed", 42)))
	pd.DataFrame({"text": df.iloc[train_idx]["text"]}).to_csv(train_path, index=False, encoding="utf-8")
	pd.DataFrame({"text": df.iloc[val_idx]["text"]}).to_csv(val_path, index=False, encoding="utf-8")
	pd.DataFrame({"text": df.iloc[test_idx]["text"]}).to_csv(test_path, index=False, encoding="utf-8")
	print(f"Splits saved -> train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")


if __name__ == "__main__":
	main()


