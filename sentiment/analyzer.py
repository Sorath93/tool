import logging
import re
import torch
from typing import Tuple, Literal
from abc import ABC, abstractmethod

Label = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

logger = logging.getLogger(__name__)


# Custom exceptions


class SentimentAnalysisError(Exception):
    """Custom exception for sentiment analysis errors."""


class ModelLoadError(SentimentAnalysisError):
    """Exception raised when the model fails to load."""


# Abstract Base class


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class defining the sentiment analyzer contract.
    This will allow for multiple approaches to be implemented for sentiment analysis.
    """

    @abstractmethod
    def get_sentiment_score(self, text: str) -> float:
        """Return score in between -1.0 and +1.0."""

    @abstractmethod
    def get_sentiment_label(self, text: str) -> Label:
        """Return 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[float, Label]:
        """Get both score and label efficiently in a single call."""
        pass


# Implementation using CardiffNLP Twitter RoBERTa model


class SentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Main sentiment analyzer implementation using CardiffNLP Twitter RoBERTa model.
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAPPING = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        label_mapping: dict = None,
        precision: int = 2,
        token_limit: int = 512,
    ):

        if not isinstance(precision, int) or precision < 0:
            raise ValueError("Precision must be a non-negative integer")

        self.model_name = model_name
        self.precision = precision
        self.token_limit = token_limit
        self.label_mapping = (
            self.LABEL_MAPPING if label_mapping is None else label_mapping
        )

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise ModelLoadError(f"Could not load model '{self.model_name}'") from e

    def _preprocess_text(self, text: str) -> str:
        """
        CardiffNLP Twitter preprocessing pattern:
        - Map @mentions -> '@user'
        - Map URLs -> 'http'
        - Preserve case/hashtags/emojis
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")

        text = re.sub(r"\s+", " ", text).strip()

        # Exact preprocessing from the Hugging Face model card
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)

        return " ".join(new_text)

    def _predict(self, text: str) -> torch.Tensor:
        """
        Get model predictions for text.

        Args:
            text: Input text to analyze

        Returns:
            Softmax probabilities tensor [negative, neutral, positive]
        """
        processed = self._preprocess_text(text)
        inputs = self.tokenizer(
            processed, return_tensors="pt", truncation=True, max_length=self.token_limit
        )
        with torch.no_grad():  # memory efficient
            outputs = self.model(**inputs)
        # Return the probability of the sentiment for each class (negative, neutral, positive)
        return torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()

    def analyze(self, text: str) -> Tuple[float, Label]:
        """
        Primary method to get both score and label.

        Returns:
            Tuple of (score, label)
        """
        probabilities = self._predict(text)

        # Calculate score
        positive_prob = probabilities[2].item()
        negative_prob = probabilities[0].item()
        # Difference allows us to see which one it leans towards (Polarity).
        score = round(positive_prob - negative_prob, self.precision)
        # Other approaches might have been: Weighted sum, Max probability with sign (+, -), Normalized difference

        # Calculate label
        predicted_class = torch.argmax(probabilities).item()
        label = self.label_mapping[predicted_class]

        return score, label

    def get_sentiment_score(self, text: str) -> float:
        """
        Get numerical sentiment score.

        Returns:
            Float between -1.0 (negative) and 1.0 (positive), rounded to configured precision
        """
        score, _ = self.analyze(
            text
        )  # risk of double computation but keeps interface simple. Could be optimized if needed by caching the result in memory.
        return score

    def get_sentiment_label(self, text: str) -> Label:
        """
        Get categorical sentiment label.

        Returns:
            'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
        """
        _, label = self.analyze(text)
        return label


# Factory class for creating sentiment analyzers


class SentimentAnalyzerFactory:
    """
    Makes it easy to switch between different approaches to sentiment analysis.
    """

    _analyzers = {
        "default": SentimentAnalyzer,
        # 'cloud': CloudSentimentAnalyzer,  # Example for future extension
    }

    @classmethod
    def create(cls, analyzer_type: str = "default", **kwargs) -> BaseSentimentAnalyzer:
        """
        Create an instance.

        Args:
            analyzer_type: Type of analyzer ('default', 'cloud', etc.)
            **kwargs: Additional arguments

        Returns:
            Configured sentiment analyzer instance
        """
        if analyzer_type not in cls._analyzers:
            available = ", ".join(cls._analyzers.keys())
            raise ValueError(
                f"Unknown analyzer type '{analyzer_type}'. Available: {available}"
            )

        return cls._analyzers[analyzer_type](**kwargs)

    @classmethod
    def register(cls, name: str, analyzer_class):
        """Register a new analyzer implementation."""
        if not issubclass(analyzer_class, BaseSentimentAnalyzer):
            raise ValueError("Analyzer must inherit from BaseSentimentAnalyzer")
        cls._analyzers[name] = analyzer_class


if __name__ == "__main__":
    # analyzer = SentimentAnalyzer()

    analyzer = SentimentAnalyzerFactory.create("default")

    test_texts = [
        "I love this! 🔥",
        "This is terrible 😞",
        "It's okay I guess",
        "https://example.com @user check this out!!!",
        "good",
        "bad",
        "meh" * 1000,
    ]

    for text in test_texts:
        score, label = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Score: {score}, Label: {label}\n")
