from django.db import models
import uuid
from django.contrib.postgres.fields import ArrayField


class Job(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    PLATFORM_CHOICES = [
        ('instagram', 'Instagram'),
        ('youtube', 'YouTube'),
        ('tiktok', 'TikTok'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    urls = models.JSONField(help_text="List of URLs to analyze")
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    total_comments = models.IntegerField(default=0)
    processed_comments = models.IntegerField(default=0)
    csv_file_path = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return f"Job {self.id} - {self.platform} - {self.status}"
    
    @property
    def progress_percentage(self):
        if self.total_comments == 0:
            return 0
        return (self.processed_comments / self.total_comments) * 100


class Comment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='comments')
    post_url = models.URLField()
    comment_text = models.TextField()
    author = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    original_comment_id = models.CharField(max_length=255, null=True, blank=True)
    likes_count = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Comment {self.id} - {self.author or 'Anonymous'}"


class SentimentResult(models.Model):
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    comment = models.OneToOneField(Comment, on_delete=models.CASCADE, related_name='sentiment')
    sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    confidence_score = models.FloatField()
    positive_score = models.FloatField(default=0.0)
    negative_score = models.FloatField(default=0.0)
    neutral_score = models.FloatField(default=0.0)
    processed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Sentiment: {self.sentiment_label} ({self.confidence_score:.2f})"
