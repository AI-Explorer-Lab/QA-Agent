from __future__ import annotations

from typing import Any, Dict, List, Sequence


class TransformersCrossEncoderScorer:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 8,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = str(model_name or "BAAI/bge-reranker-base")
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self.local_files_only = bool(local_files_only)
        self.last_error = ""
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def _load(self) -> bool:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return True

        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model.to(self._device)
            self._model.eval()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            self._tokenizer = None
            self._model = None
            self._torch = None
            return False

    def score(self, query: str, texts: Sequence[str]) -> tuple[List[float], Dict[str, Any]]:
        if not texts:
            return [], {"status": "skipped", "reason": "empty_texts"}
        if not self._load():
            return [], {"status": "fallback", "reason": self.last_error or "model_load_failed"}

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        scores: List[float] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[start : start + self.batch_size])
                pairs = [(str(query or ""), str(text or "")) for text in batch_texts]
                encoded = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._torch.no_grad():
                    logits = self._model(**encoded).logits
                if len(logits.shape) == 1:
                    batch_scores = logits
                elif logits.shape[-1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
                scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
            return scores, {
                "status": "applied",
                "model": self.model_name,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "device": self._device,
            }
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            return [], {"status": "fallback", "reason": self.last_error}
