"""
Sentiment Analysis Framework

A clean, extensible architecture for sentiment analysis that accepts string input 
and returns sentiment scores (-1 to +1) and labels. Designed with maintainability 
and future-proofing in mind for easy integration of different backends.

Key architectural features:
- Abstract base class for clean interface definition
- Configurable text preprocessing component
- Factory pattern for easy implementation switching
- Proper error handling and validation
- Separation of concerns for maintainability
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""
    pass


class ModelLoadError(SentimentAnalysisError):
    """Raised when model loading fails."""
    pass


class TextPreprocessor:
    """
    Handles text cleaning and normalization with configurable options.
    Separated for reusability across different analyzer implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary with preprocessing options:
                - remove_urls (bool): Remove HTTP/HTTPS URLs
                - remove_mentions (bool): Remove @mentions  
                - remove_hashtag_symbols (bool): Remove # symbols but keep text
                - demojize_emojis (bool): Convert emojis to text descriptions
                - normalize_punctuation (bool): Normalize repeated punctuation
                - to_lowercase (bool): Convert text to lowercase
        """
        default_config = {
            'remove_urls': True,
            'remove_mentions': True,
            'remove_hashtag_symbols': True,
            'demojize_emojis': True,
            'normalize_punctuation': True,
            'to_lowercase': True
        }
        
        self.config = {**default_config, **(config or {})}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate preprocessor configuration."""
        for key, value in self.config.items():
            if not isinstance(value, bool):
                raise ValueError(f"Config option '{key}' must be boolean, got {type(value)}")
    
    def preprocess(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Processed text string
            
        Raises:
            ValueError: If input is not a valid string
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")
        
        processed = text
        
        # Apply configured preprocessing steps
        if self.config['to_lowercase']:
            processed = processed.lower()
        
        if self.config['remove_urls']:
            processed = re.sub(r'https?://\S+|www\.\S+', '', processed)
        
        if self.config['remove_mentions']:
            processed = re.sub(r'@\w+', '', processed)
        
        if self.config['remove_hashtag_symbols']:
            processed = re.sub(r'#(\w+)', r'\1', processed)
        
        if self.config['demojize_emojis']:
            # Note: Actual emoji handling would require emoji library
            # This is a placeholder for the preprocessing step
            pass
        
        if self.config['normalize_punctuation']:
            processed = re.sub(r'([!?.]){2,}', r'\1', processed)
        
        # Always clean up whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        if not processed:
            raise ValueError("Text became empty after preprocessing")
        
        return processed


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class defining the sentiment analyzer interface.
    Enables easy swapping between different implementations (Hugging Face, cloud services, etc.)
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base analyzer.
        
        Args:
            preprocessor: Text preprocessor instance
            config: Configuration dictionary
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_sentiment_score(self, text: str) -> float:
        """
        Get numerical sentiment score.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Float between -1.0 (most negative) and 1.0 (most positive)
        """
        pass
    
    @abstractmethod
    def get_sentiment_label(self, text: str) -> str:
        """
        Get categorical sentiment label.
        
        Args:
            text: Input text to analyze
            
        Returns:
            String: 'positive', 'negative', or 'neutral'
        """
        pass
    
    def analyze(self, text: str) -> Tuple[float, str]:
        """
        Convenience method to get both score and label.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (score, label)
        """
        score = self.get_sentiment_score(text)
        label = self.get_sentiment_label(text)
        return score, label


class SentimentAnalyzerFactory:
    """
    Factory for creating sentiment analyzer instances.
    Provides a clean interface for switching between implementations.
    """
    
    _analyzers = {}  # To be populated with actual implementations
    
    @classmethod
    def register_analyzer(cls, name: str, analyzer_class):
        """
        Register a new analyzer implementation.
        
        Args:
            name: Identifier for the analyzer type
            analyzer_class: Class implementing BaseSentimentAnalyzer
        """
        if not issubclass(analyzer_class, BaseSentimentAnalyzer):
            raise ValueError("Analyzer class must inherit from BaseSentimentAnalyzer")
        cls._analyzers[name.lower()] = analyzer_class
    
    @classmethod
    def create(cls, analyzer_type: str, **kwargs) -> BaseSentimentAnalyzer:
        """
        Create sentiment analyzer instance.
        
        Args:
            analyzer_type: Type of analyzer (registered name)
            **kwargs: Additional arguments for analyzer initialization
            
        Returns:
            Configured sentiment analyzer instance
            
        Raises:
            ValueError: If analyzer type is not supported
        """
        analyzer_type = analyzer_type.lower()
        if analyzer_type not in cls._analyzers:
            available = ', '.join(cls._analyzers.keys()) if cls._analyzers else 'None'
            raise ValueError(f"Unknown analyzer type '{analyzer_type}'. Available: {available}")
        
        return cls._analyzers[analyzer_type](**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available analyzer types."""
        return list(cls._analyzers.keys())


# TODO: Implement concrete analyzer classes
# Example structure for implementations:
#
# class HuggingFaceSentimentAnalyzer(BaseSentimentAnalyzer):
#     """Transformer-based implementation using Hugging Face models."""
#     def get_sentiment_score(self, text: str) -> float:
#         # Implementation here
#         pass
#     
#     def get_sentiment_label(self, text: str) -> str:
#         # Implementation here  
#         pass
#
# class CloudSentimentAnalyzer(BaseSentimentAnalyzer):
#     """Cloud API implementation (Google Cloud, AWS, Azure, etc.)."""
#     def get_sentiment_score(self, text: str) -> float:
#         # Implementation here
#         pass
#     
#     def get_sentiment_label(self, text: str) -> str:
#         # Implementation here
#         pass
#
# # Register implementations
# SentimentAnalyzerFactory.register_analyzer('huggingface', HuggingFaceSentimentAnalyzer)
# SentimentAnalyzerFactory.register_analyzer('cloud', CloudSentimentAnalyzer)


if __name__ == "__main__":
    print("🏗️  Sentiment Analysis Framework")
    print("=" * 40)
    print("Clean architecture ready for implementation!")
    print()
    print("Available components:")
    print("- TextPreprocessor: Configurable text cleaning")
    print("- BaseSentimentAnalyzer: Abstract interface")
    print("- SentimentAnalyzerFactory: Implementation management")
    print()
    print("To add implementations:")
    print("1. Create class inheriting from BaseSentimentAnalyzer")
    print("2. Implement get_sentiment_score() and get_sentiment_label()")
    print("3. Register with SentimentAnalyzerFactory.register_analyzer()")
    print()
    print(f"Currently registered analyzers: {SentimentAnalyzerFactory.get_available_types()}")