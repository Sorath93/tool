from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_scores(sentence: str) -> None:
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print(f"Sentiment Scores: {sentiment_dict}")
    print(f"Negative Sentiment: {sentiment_dict['neg']*100}%")
    print(f"Neutral Sentiment: {sentiment_dict['neu']*100}%")
    print(f"Positive Sentiment: {sentiment_dict['pos']*100}%")
    
    if sentiment_dict['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif sentiment_dict['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")



class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return {"score": round(compound, 4), "label": label}












from typing import Optional, Dict, Any

class SentimentAnalyzerConfig:
    """
    Configuration for SentimentAnalyzer.
    """
    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05, handle_empty_input: bool = True):
        if not (-1.0 <= negative_threshold < positive_threshold <= 1.0):
            raise ValueError("Thresholds must be in range and positive > negative.")
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.handle_empty_input = handle_empty_input


class SentimentAnalyzer:
    """
    Analyzes sentiment of text using VADER.
    Returns a score (-1 to +1) and a label (NEGATIVE, NEUTRAL, POSITIVE).
    """
    def __init__(self, config: Optional[SentimentAnalyzerConfig] = None):
        self.config = config or SentimentAnalyzerConfig()
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VADER: {e}")

    def analyze(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Analyze the sentiment of the input text.
        Returns: dict with 'score' (float) and 'label' (str)
        """
        if text is None or (isinstance(text, str) and not text.strip()):
            if not self.config.handle_empty_input:
                raise ValueError("Input text cannot be None or empty.")
            return {"score": 0.0, "label": "NEUTRAL"}
        if not isinstance(text, str):
            raise TypeError(f"Input must be a string, got {type(text)}")
        try:
            scores = self.analyzer.polarity_scores(text)
            compound = scores["compound"]
            label = self._label_from_score(compound)
            return {"score": round(compound, 4), "label": label}
        except Exception as e:
            raise RuntimeError(f"Sentiment analysis failed: {e}")

    def _label_from_score(self, score: float) -> str:
        if score >= self.config.positive_threshold:
            return "POSITIVE"
        elif score <= self.config.negative_threshold:
            return "NEGATIVE"
        else:
            return "NEUTRAL"