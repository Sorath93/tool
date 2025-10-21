# sentiment_analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import os

try:
    from transformers import pipeline, Pipeline
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers is required. Install with: pip install transformers && pip install 'torch'"
    ) from e

try:
    import emoji
except Exception:  # pragma: no cover
    emoji = None  # demojize becomes a no-op if emoji isn't installed


# --------------------
# Public data models
# --------------------

Label = str  # one of {"NEGATIVE","NEUTRAL","POSITIVE"}


@dataclass(frozen=True)
class SentimentResult:
    score: float  # -1.0 .. +1.0
    label: Label  # NEGATIVE / NEUTRAL / POSITIVE
    confidence: float  # confidence of the predicted label (0..1)
    probabilities: Dict[Label, float]  # per-class probabilities


@dataclass(frozen=True)
class AnalyzerConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: Optional[int] = None  # None:auto-cpu, or 0 for CUDA:0 if installed
    use_demojize: bool = True
    truncation: bool = True
    max_length: int = 256  # social comments are short; keep small for speed
    return_all_scores: bool = True

    @staticmethod
    def from_env() -> "AnalyzerConfig":
        """Allow simple configuration via environment variables without extra deps."""
        return AnalyzerConfig(
            model_name=os.getenv(
                "SA_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ),
            device=(
                int(os.getenv("SA_DEVICE"))
                if os.getenv("SA_DEVICE") is not None
                else None
            ),
            use_demojize=os.getenv("SA_DEMOJIZE", "true").lower() in {"1", "true", "yes"},
            truncation=os.getenv("SA_TRUNCATION", "true").lower() in {"1", "true", "yes"},
            max_length=int(os.getenv("SA_MAX_LENGTH", "256")),
            return_all_scores=True,
        )


# --------------------
# Implementation
# --------------------

class SentimentAnalyzer:
    """
    Hugging Face based sentiment analyzer (free, local).

    - Defaults to the CardiffNLP Twitter RoBERTa model (excellent for short social text).
    - Converts per-class probabilities to a signed score in [-1, +1] via: P(pos) - P(neg).
    - Optional emoji demojize pre-processing so non-textual emojis contribute to signal.

    Usage:
        analyzer = SentimentAnalyzer()  # or SentimentAnalyzer(AnalyzerConfig(...))
        res = analyzer.analyze("I'm crying this is so funny 😂😂")
        print(res.score, res.label)
    """

    _HF_TO_CANONICAL = {
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
        "positive": "POSITIVE",
        # Some models return LABEL_0/1/2; we'll map via id2label when available.
    }

    def __init__(self, config: Optional[AnalyzerConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or AnalyzerConfig.from_env()
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self._pipe: Optional[Pipeline] = None

    # Lazy-load so constructing the object is cheap (handy for tests & CLIs)
    def _ensure_pipeline(self) -> Pipeline:
        if self._pipe is None:
            kwargs = {}
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            self._pipe = pipeline(
                task="sentiment-analysis",
                model=self.config.model_name,
                return_all_scores=self.config.return_all_scores,
                **kwargs,
            )
        return self._pipe

    def _preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("SentimentAnalyzer.analyze expects a string input")
        text = text.strip()
        if self.config.use_demojize and emoji is not None:
            # Turn emojis into words, e.g., "😂" -> ":face_with_tears_of_joy:" which helps the model.
            text = emoji.demojize(text, language="en")
        return text

    def _canonical_labels(self, pipe: Pipeline) -> List[Label]:
        """Return labels in NEGATIVE/NEUTRAL/POSITIVE order if possible."""
        try:
            id2label = getattr(pipe.model.config, "id2label", None)
            if id2label and isinstance(id2label, dict):
                # Normalize to lower for mapping, then upcase canonical
                mapped = []
                for i in range(len(id2label)):
                    raw = str(id2label[i]).lower()
                    mapped.append(self._HF_TO_CANONICAL.get(raw, raw.upper()))
                return mapped
        except Exception:
            pass
        # Fallback common order
        return ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    @staticmethod
    def _signed_score(prob_pos: float, prob_neg: float) -> float:
        # Clamp rounding for stable tests
        s = float(prob_pos) - float(prob_neg)
        return max(-1.0, min(1.0, round(s, 6)))

    def analyze(self, text: str) -> SentimentResult:
        pre = self._preprocess(text)
        if not pre:
            return SentimentResult(
                score=0.0,
                label="NEUTRAL",
                confidence=1.0,
                probabilities={"NEGATIVE": 0.0, "NEUTRAL": 1.0, "POSITIVE": 0.0},
            )

        pipe = self._ensure_pipeline()
        try:
            outputs = pipe(
                pre,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
            )
        except Exception as e:
            self.logger.exception("HF pipeline inference failed")
            raise RuntimeError("Sentiment inference failed") from e

        # When return_all_scores=True, outputs is a list with one element per input
        # Each element is a list of dicts: {"label": <str>, "score": <float>} per class
        if not outputs or not isinstance(outputs, list) or not outputs[0]:
            raise RuntimeError("Unexpected pipeline output structure")

        class_scores: List[Dict[str, float]] = outputs[0]
        labels_order = self._canonical_labels(pipe)

        # Build a dict of probabilities using canonical labels
        probs: Dict[Label, float] = {}
        for entry in class_scores:
            raw = str(entry.get("label", "")).lower()
            prob = float(entry.get("score", 0.0))
            canonical = self._HF_TO_CANONICAL.get(raw, raw.upper())
            probs[canonical] = prob

        # Ensure all expected keys exist
        for k in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
            probs.setdefault(k, 0.0)

        # Determine top label by probability (ties resolve by labels_order priority)
        top_label = max(labels_order, key=lambda k: probs.get(k, 0.0))
        confidence = float(probs.get(top_label, 0.0))

        score = self._signed_score(prob_pos=probs["POSITIVE"], prob_neg=probs["NEGATIVE"])

        return SentimentResult(score=score, label=top_label, confidence=confidence, probabilities=probs)


# --------------------
# Simple CLI for manual checks (optional)
# --------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = AnalyzerConfig.from_env()
    if args.model:
        cfg = AnalyzerConfig(model_name=args.model,
                             device=cfg.device,
                             use_demojize=cfg.use_demojize,
                             truncation=cfg.truncation,
                             max_length=cfg.max_length)

    analyzer = SentimentAnalyzer(cfg)
    res = analyzer.analyze(args.text)
    print(json.dumps({
        "score": res.score,
        "label": res.label,
        "confidence": res.confidence,
        "probabilities": res.probabilities,
    }, indent=2))


# tests/test_sentiment_analyzer.py
import json
import os
import types
import pytest

from sentiment_analyzer import SentimentAnalyzer, AnalyzerConfig, SentimentResult


# --------------------
# Unit tests with a mocked HF pipeline (fast, no model download)
# --------------------
class DummyPipeline:
    class model:
        class config:
            id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def __call__(self, text, truncation=True, max_length=256):
        # Extremely naive mock just for unit tests
        if "error" in text:
            raise RuntimeError("forced error")
        if "😂" in text or ":face_with_tears_of_joy:" in text:
            return [[{"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.10}, {"label": "positive", "score": 0.85}]]
        if "meh" in text:
            return [[{"label": "negative", "score": 0.10}, {"label": "neutral", "score": 0.80}, {"label": "positive", "score": 0.10}]]
        if "bad" in text:
            return [[{"label": "negative", "score": 0.80}, {"label": "neutral", "score": 0.15}, {"label": "positive", "score": 0.05}]]
        # default mildly positive
        return [[{"label": "negative", "score": 0.10}, {"label": "neutral", "score": 0.20}, {"label": "positive", "score": 0.70}]]


def test_empty_returns_neutral(monkeypatch):
    analyzer = SentimentAnalyzer(AnalyzerConfig())
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: DummyPipeline())
    res = analyzer.analyze("")
    assert isinstance(res, SentimentResult)
    assert res.score == 0.0
    assert res.label == "NEUTRAL"
    assert res.probabilities["NEUTRAL"] == 1.0


def test_positive_emoji(monkeypatch):
    analyzer = SentimentAnalyzer(AnalyzerConfig(use_demojize=True))
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: DummyPipeline())
    res = analyzer.analyze("I'm crying this is so funny 😂😂")
    assert res.label == "POSITIVE"
    assert res.score > 0
    assert 0.8 <= res.probabilities["POSITIVE"] <= 0.9


def test_negative_text(monkeypatch):
    analyzer = SentimentAnalyzer(AnalyzerConfig())
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: DummyPipeline())
    res = analyzer.analyze("this is bad")
    assert res.label == "NEGATIVE"
    assert res.score < 0


def test_neutral_text(monkeypatch):
    analyzer = SentimentAnalyzer(AnalyzerConfig())
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: DummyPipeline())
    res = analyzer.analyze("meh")
    assert res.label == "NEUTRAL"


def test_non_string_raises():
    analyzer = SentimentAnalyzer(AnalyzerConfig())
    with pytest.raises(TypeError):
        analyzer.analyze(123)  # type: ignore


def test_pipeline_error_propagates(monkeypatch):
    analyzer = SentimentAnalyzer(AnalyzerConfig())
    monkeypatch.setattr(analyzer, "_ensure_pipeline", lambda: DummyPipeline())
    with pytest.raises(RuntimeError):
        analyzer.analyze("force error please: error")


# --------------------
# Optional slow integration test (downloads the actual model). Marked as slow.
# Run with: pytest -m slow
# --------------------
@pytest.mark.slow
@pytest.mark.integration
def test_integration_cardiffnlp_model():
    cfg = AnalyzerConfig(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
    analyzer = SentimentAnalyzer(cfg)
    res = analyzer.analyze("I love this! 😂")
    assert res.label in {"NEGATIVE", "NEUTRAL", "POSITIVE"}
    assert -1.0 <= res.score <= 1.0



# requirements.txt
# Keep minimal. You must also install a matching CPU/GPU torch wheel for your platform.
transformers>=4.44
emoji>=2.12.1
# Install torch separately (e.g. CPU-only):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu


# README.md (snippet)

## SentimentAnalyzer (free, local; Hugging Face)

**Features**
- Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (great for short-form social text).  
- Maps to the task requirement: returns **score in [-1, +1]** and **label {NEGATIVE, NEUTRAL, POSITIVE}**.
- Optional emoji-to-text via `emoji.demojize` to better capture emoji signal.
- Configurable via env vars: `SA_MODEL`, `SA_DEVICE`, `SA_DEMOJIZE`, `SA_MAX_LENGTH`.

**Install**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# IMPORTANT: install torch for your platform (CPU-only example):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Usage**
```python
from sentiment_analyzer import SentimentAnalyzer
analyzer = SentimentAnalyzer()
print(analyzer.analyze("I'm crying this is so funny 😂😂"))
```

**Run tests**
```bash
pip install pytest
pytest -q                # fast unit tests (mocked)
pytest -q -m slow        # includes HF model integration test (downloads model)
```

**Design notes**
- We compute `score = P(positive) - P(negative)` which is naturally bounded in [-1, +1].
- For models that return `LABEL_0/1/2`, we inspect `model.config.id2label` to map to canonical NEG/NEU/POS.
- If input is empty/whitespace, we return a neutral result with `score=0.0`.
- Exceptions are wrapped in `RuntimeError` with logs, keeping API clean for callers.

