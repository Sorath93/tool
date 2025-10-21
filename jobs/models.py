from django.db import models

class Job(models.Model):
    STATUS_CHOICES = [
    ('PENDING', 'Pending'),
    ('IN_PROGRESS', 'In Progress'),
    ('COMPLETED', 'Completed'),
    ('FAILED', 'Failed'),
    ]

    id = models.AutoField(primary_key=True)
    post_urls = models.JSONField() # Store list of post URLs
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    message = models.TextField(null=True, blank=True)  # Optional message field, e.g., for error details

class SocialMediaPost(models.Model):

    PLATFORM_CHOICES = [
        ('Instagram', 'Instagram'),
        ('YouTube', 'YouTube'),
        ('TikTok', 'TikTok'),
    ]

    job = models.ForeignKey(Job, related_name='posts', on_delete=models.CASCADE)
    post_url = models.URLField()
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES)

class Comment(models.Model):
    post = models.ForeignKey(SocialMediaPost, related_name='comments', on_delete=models.CASCADE)
    comment_text = models.TextField()

class CommentSentiment(models.Model):
    comment = models.OneToOneField(Comment, on_delete=models.CASCADE)
    score = models.FloatField()
    label = models.CharField(max_length=10)  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'










'''
job = models.ForeignKey(Job, related_name='sentiments', on_delete=models.CASCADE)
post_url = models.URLField()
comment_text = models.TextField()
sentiment_score = models.FloatField()
sentiment_label = models.CharField(max_length=10)  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
'''