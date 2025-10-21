from celery import shared_task
from datetime import datetime


@shared_task
def process_job(job_id):
    """
    Simple job processor: scrape comments and analyze sentiment
    """
    try:
        from .models import Job, SocialMediaPost, Comment, CommentSentiment
        from sentiment.analyzer import SentimentAnalyzer
        
        job = Job.objects.get(id=job_id)
        job.status = 'IN_PROGRESS'
        job.message = '' 
        job.completed_at = None
        job.save()
        
        analyzer = SentimentAnalyzer() # REPLACE WITH FACTORY
        
        # Process each URL
        for url in job.post_urls:
            # Create mock post and comments (replace with actual scraping)
            post = SocialMediaPost.objects.create(
                job=job,
                post_url=url,
                platform='TikTok',  # TODO: detect platform from URL
            )
            
            # Mock comments (replace with real scraping)
            mock_comments = [
                "nothing beats a jet2 holiday",
                "FYP getting too local",
                "who else got kidnapped?",
                "Slay queen!",
                "jail",
                "I hate how people have a better use of freewill than me",
                "Love this video!",
                "Drop the skincare routine please!"
            ]
            
            # Create comments and analyze sentiment
            for text in mock_comments:
                comment = Comment.objects.create(
                    post=post,
                    comment_text=text,
                )
                
                # Analyze sentiment immediately
                score, label = analyzer.analyze(comment.text)
                CommentSentiment.objects.create(
                    comment=comment,
                    score=score,
                    label=label
                )

        job.status = 'COMPLETED'
        job.completed_at = datetime.now()
        job.save()
        
    except Exception as e:
        job.status = 'FAILED'
        job.error_message = str(e)
        job.save()
        raise
