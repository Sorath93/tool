import re
import torch
from typing import Dict, Any, Optional, Tuple, Literal
from abc import ABC, abstractmethod

Label = Literal['POSITIVE', 'NEGATIVE', 'NEUTRAL']


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class defining the sentiment analyzer interface.
    Enables easy swapping between different implementations (Hugging Face, cloud services, etc.)
    """
    
    @abstractmethod
    def get_sentiment_score(self, text: str) -> float:
        """Return score in [-1.0, +1.0]."""

    
    @abstractmethod
    def get_sentiment_label(self, text: str) -> Label:
        """Return 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'."""  
    
    @abstractmethod
    def analyze(self, text: str) -> Tuple[float, Label]:
        """Get both score and label efficiently in a single call."""
        pass


class SentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Main sentiment analyzer implementation using CardiffNLP Twitter RoBERTa model.
    Accepts string input and returns sentiment score (-1 to +1) and label.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", precision=4):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
        self.precision = precision
    
    def _preprocess_text(self, text: str) -> str:
        """
        Light preprocessing for CardiffNLP Twitter sentiment model.
        
        This model was specifically trained on Twitter data with certain preprocessing:
        - URLs and mentions are normalized to <url> and <user> tokens
        - Text case is preserved (model is case-sensitive)
        - Hashtags are kept intact (model uses them for context)
        - Emojis are preserved (tokenizer handles them properly)
        
        Args:
            text: Raw input text to preprocess
            
        Returns:
            Preprocessed text ready for the model
            
        Raises:
            ValueError: If input is not a valid non-empty string
        """
        
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")

        # Normalize URLs and mentions to special tokens the model expects
        text = re.sub(r"http\S+|www\S+", "<url>", text)
        text = re.sub(r"@\w+", "<user>", text)

        # Clean up excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def _predict(self, text: str) -> torch.Tensor:
        """
        Get model predictions for text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Softmax probabilities tensor [negative, neutral, positive]
        """
        processed = self._preprocess_text(text)
        inputs = self.tokenizer(processed, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    
    def analyze(self, text: str) -> Tuple[float, Label]:
        """
        Primary method to get both score and label.
        
        Returns:
            Tuple of (score, label)
        """
        scores = self._predict(text)
        
        # Calculate score
        positive_score = scores[2].item()
        negative_score = scores[0].item()
        score = round(positive_score - negative_score, self.precision)
        
        # Calculate label
        predicted_class = torch.argmax(scores).item()
        label = self.label_mapping[predicted_class]
        
        return score, label
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get numerical sentiment score.
        
        Returns:
            Float between -1.0 (negative) and 1.0 (positive), rounded to configured precision
        """
        score, _ = self.analyze(text)
        return score
    
    def get_sentiment_label(self, text: str) -> Label:
        """
        Get categorical sentiment label.
        
        Returns:
            'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
        """
        _, label = self.analyze(text)
        return label


class SentimentAnalyzerFactory:
    """
    Makes it easy to switch between different implementations.
    """
    
    _analyzers = {
        'default': SentimentAnalyzer,
        # 'cloud': CloudSentimentAnalyzer,  # Example for future extension
    } # why is this a class attribute?
    
    @classmethod
    def create(cls, analyzer_type: str = 'default', **kwargs) -> BaseSentimentAnalyzer:
        """
        Create an instance.
        
        Args:
            analyzer_type: Type of analyzer ('default', 'roberta', etc.)
            **kwargs: Additional arguments for analyzer initialization
            
        Returns:
            Configured sentiment analyzer instance
        """
        if analyzer_type not in cls._analyzers:
            available = ', '.join(cls._analyzers.keys())
            raise ValueError(f"Unknown analyzer type '{analyzer_type}'. Available: {available}")
        
        return cls._analyzers[analyzer_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, analyzer_class):
        """Register a new analyzer implementation."""
        if not issubclass(analyzer_class, BaseSentimentAnalyzer):
            raise ValueError("Analyzer must inherit from BaseSentimentAnalyzer")
        cls._analyzers[name] = analyzer_class


'''
# Example usage and demonstration
if __name__ == "__main__":
    # Method 1: Direct instantiation
    analyzer = SentimentAnalyzer()
    
    # Method 2: Factory (more flexible for future)
    analyzer = SentimentAnalyzerFactory.create('default')
    
    # Test with some examples
    test_texts = [
        "I love this! 🔥",
        "This is terrible 😞", 
        "It's okay I guess",
        "https://example.com @user check this out!"
    ]
    
    for text in test_texts:
        score, label = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Score: {score:.3f}, Label: {label}\n")
'''