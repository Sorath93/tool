from typing import Dict, List, Tuple
from django.conf import settings
from jobs.models import Comment, SentimentResult
from sentiment.analyzer2 import SentimentAnalyzerFactory


class SentimentAnalysisService:
    """Service class to integrate sentiment analysis with the job system"""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzerFactory.create('default')
    
    def analyze_comment(self, comment: Comment) -> SentimentResult:
        """
        Analyze sentiment for a single comment and create SentimentResult
        
        Args:
            comment: Comment instance to analyze
            
        Returns:
            SentimentResult instance
        """
        # Get sentiment analysis results
        score, label = self.analyzer.analyze(comment.comment_text)
        
        # Get individual scores for all sentiments
        scores = self.analyzer._predict(comment.comment_text)
        negative_score = scores[0].item()
        neutral_score = scores[1].item()
        positive_score = scores[2].item()
        
        # Map label to our model choices
        label_mapping = {
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative',
            'NEUTRAL': 'neutral'
        }
        
        # Create or update SentimentResult
        sentiment_result, created = SentimentResult.objects.get_or_create(
            comment=comment,
            defaults={
                'sentiment_label': label_mapping[label],
                'confidence_score': max(negative_score, neutral_score, positive_score),
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
            }
        )
        
        if not created:
            # Update existing result
            sentiment_result.sentiment_label = label_mapping[label]
            sentiment_result.confidence_score = max(negative_score, neutral_score, positive_score)
            sentiment_result.positive_score = positive_score
            sentiment_result.negative_score = negative_score
            sentiment_result.neutral_score = neutral_score
            sentiment_result.save()
        
        return sentiment_result
    
    def analyze_comments_batch(self, comments: List[Comment]) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple comments in batch
        
        Args:
            comments: List of Comment instances
            
        Returns:
            List of SentimentResult instances
        """
        results = []
        for comment in comments:
            try:
                sentiment_result = self.analyze_comment(comment)
                results.append(sentiment_result)
            except Exception as e:
                print(f"Error analyzing comment {comment.id}: {e}")
                continue
        
        return results
    
    def get_job_sentiment_summary(self, job_id) -> Dict:
        """
        Get sentiment analysis summary for a job
        
        Args:
            job_id: Job UUID
            
        Returns:
            Dictionary with sentiment distribution and statistics
        """
        from jobs.models import Job
        
        try:
            job = Job.objects.get(id=job_id)
        except Job.DoesNotExist:
            return {}
        
        # Get all sentiment results for this job
        sentiment_results = SentimentResult.objects.filter(
            comment__job=job
        ).values_list('sentiment_label', flat=True)
        
        # Count sentiments
        sentiment_counts = {
            'positive': sentiment_results.filter(sentiment_label='positive').count(),
            'negative': sentiment_results.filter(sentiment_label='negative').count(),
            'neutral': sentiment_results.filter(sentiment_label='neutral').count(),
        }
        
        total_analyzed = sum(sentiment_counts.values())
        
        if total_analyzed == 0:
            return {
                'total_analyzed': 0,
                'sentiment_distribution': sentiment_counts,
                'sentiment_percentages': {'positive': 0, 'negative': 0, 'neutral': 0},
                'overall_sentiment': 'neutral'
            }
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: (count / total_analyzed) * 100
            for sentiment, count in sentiment_counts.items()
        }
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        return {
            'total_analyzed': total_analyzed,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'overall_sentiment': max_sentiment,
            'job_id': str(job_id),
            'job_status': job.status
        }
    
    def analyze_text_preview(self, text: str) -> Dict:
        """
        Analyze sentiment for a single text without saving to database
        Useful for testing or preview functionality
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            score, label = self.analyzer.analyze(text)
            scores = self.analyzer._predict(text)
            
            return {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment_label': label.lower(),
                'sentiment_score': score,
                'confidence_scores': {
                    'negative': scores[0].item(),
                    'neutral': scores[1].item(),
                    'positive': scores[2].item()
                }
            }
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0
            }