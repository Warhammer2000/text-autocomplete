from __future__ import annotations

from typing import Dict, List


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:

	if len(predictions) != len(references):
		raise ValueError("Predictions and references must have the same length")

	try:
		import evaluate  # type: ignore
	except Exception as exc:
		raise RuntimeError("The 'evaluate' package is required for ROUGE computation.") from exc

	metric = evaluate.load("rouge")
	result = metric.compute(
		predictions=predictions,
		references=references,
		use_stemmer=True,
	)
	return {
		"rouge1": float(result.get("rouge1", 0.0)),
		"rouge2": float(result.get("rouge2", 0.0)),
		"rougeL": float(result.get("rougeL", 0.0)),
		"rougeLsum": float(result.get("rougeLsum", 0.0)),
	}


if __name__ == "__main__":
	print(compute_rouge(["hello world"], ["hello world"]))