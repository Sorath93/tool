from django.contrib import admin
from .models import Job, Comment, SentimentResult


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ['id', 'platform', 'status', 'total_comments', 'processed_comments', 'progress_percentage', 'created_at']
    list_filter = ['platform', 'status', 'created_at']
    search_fields = ['id', 'urls']
    readonly_fields = ['id', 'created_at', 'updated_at', 'progress_percentage']


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['id', 'job', 'author', 'likes_count', 'created_at']
    list_filter = ['job__platform', 'created_at']
    search_fields = ['comment_text', 'author']
    readonly_fields = ['id', 'created_at']


@admin.register(SentimentResult)
class SentimentResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'comment', 'sentiment_label', 'confidence_score', 'processed_at']
    list_filter = ['sentiment_label', 'processed_at']
    readonly_fields = ['id', 'processed_at']
