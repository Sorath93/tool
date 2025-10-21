from celery import shared_task, group
from django.utils import timezone
from typing import List, Dict
import logging

from jobs.models import Job, Comment, SentimentResult
from jobs.services.scrapers import SocialMediaScraperFactory
from jobs.services.sentiment_service import SentimentAnalysisService

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_social_media_job(self, job_id: str):
    """
    Main task to process a social media sentiment analysis job
    
    Args:
        job_id: UUID string of the job to process
    """
    try:
        job = Job.objects.get(id=job_id)
        logger.info(f"Starting processing for job {job_id}")
        
        # Update job status
        job.status = 'in_progress'
        job.save()
        
        # Process each URL in the job
        all_comment_ids = []
        
        for url in job.urls:
            try:
                # Extract comments from the URL
                comment_ids = extract_comments_from_url.delay(job_id, url).get()
                all_comment_ids.extend(comment_ids)
            except Exception as e:
                logger.error(f"Error processing URL {url} for job {job_id}: {e}")
                continue
        
        # Update total comments count
        job.total_comments = len(all_comment_ids)
        job.save()
        
        if not all_comment_ids:
            job.status = 'completed'
            job.error_message = 'No comments found for any URLs'
            job.completed_at = timezone.now()
            job.save()
            return f"Job {job_id} completed with no comments"
        
        # Process sentiment analysis in batches
        batch_size = 10
        sentiment_tasks = []
        
        for i in range(0, len(all_comment_ids), batch_size):
            batch = all_comment_ids[i:i + batch_size]
            sentiment_tasks.append(analyze_comments_sentiment.s(job_id, batch))
        
        # Execute sentiment analysis tasks
        job_group = group(sentiment_tasks)
        result = job_group.apply_async()
        result.get()  # Wait for all tasks to complete
        
        # Generate CSV export
        generate_job_csv.delay(job_id).get()
        
        # Mark job as completed
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save()
        
        logger.info(f"Job {job_id} completed successfully")
        return f"Job {job_id} completed with {len(all_comment_ids)} comments analyzed"
        
    except Job.DoesNotExist:
        logger.error(f"Job {job_id} not found")
        return f"Job {job_id} not found"
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        
        # Update job status to failed
        try:
            job = Job.objects.get(id=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
        except:
            pass
        
        # Retry the task
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=2)
def extract_comments_from_url(self, job_id: str, url: str) -> List[str]:
    """
    Extract comments from a single social media URL
    
    Args:
        job_id: UUID string of the job
        url: URL to extract comments from
        
    Returns:
        List of comment IDs that were created
    """
    try:
        job = Job.objects.get(id=job_id)
        logger.info(f"Extracting comments from {url} for job {job_id}")
        
        # Get appropriate scraper
        scraper = SocialMediaScraperFactory.get_scraper(url)
        if not scraper:
            logger.warning(f"No scraper available for URL: {url}")
            return []
        
        # Extract comments
        comments_data = scraper.extract_comments(url)
        
        # Save comments to database
        comment_ids = []
        for comment_data in comments_data:
            try:
                comment = Comment.objects.create(
                    job=job,
                    post_url=url,
                    comment_text=comment_data.get('text', ''),
                    author=comment_data.get('author', ''),
                    original_comment_id=comment_data.get('comment_id', ''),
                    likes_count=comment_data.get('likes', 0)
                )
                comment_ids.append(str(comment.id))
            except Exception as e:
                logger.error(f"Error saving comment for job {job_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(comment_ids)} comments from {url}")
        return comment_ids
        
    except Job.DoesNotExist:
        logger.error(f"Job {job_id} not found")
        return []
    
    except Exception as e:
        logger.error(f"Error extracting comments from {url}: {e}")
        raise self.retry(exc=e, countdown=30)


@shared_task(bind=True, max_retries=2)
def analyze_comments_sentiment(self, job_id: str, comment_ids: List[str]):
    """
    Analyze sentiment for a batch of comments
    
    Args:
        job_id: UUID string of the job
        comment_ids: List of comment UUID strings to analyze
    """
    try:
        job = Job.objects.get(id=job_id)
        sentiment_service = SentimentAnalysisService()
        
        logger.info(f"Analyzing sentiment for {len(comment_ids)} comments in job {job_id}")
        
        # Get comments
        comments = Comment.objects.filter(id__in=comment_ids, job=job)
        
        # Analyze sentiment
        results = sentiment_service.analyze_comments_batch(list(comments))
        
        # Update processed count
        job.processed_comments += len(results)
        job.save()
        
        logger.info(f"Completed sentiment analysis for {len(results)} comments")
        return f"Analyzed {len(results)} comments"
        
    except Job.DoesNotExist:
        logger.error(f"Job {job_id} not found")
        return f"Job {job_id} not found"
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment for job {job_id}: {e}")
        raise self.retry(exc=e, countdown=30)


@shared_task(bind=True, max_retries=2)
def generate_job_csv(self, job_id: str):
    """
    Generate CSV export for a completed job
    
    Args:
        job_id: UUID string of the job
    """
    try:
        from jobs.services.csv_service import CSVExportService
        
        job = Job.objects.get(id=job_id)
        logger.info(f"Generating CSV for job {job_id}")
        
        csv_service = CSVExportService()
        csv_path = csv_service.generate_job_csv(job)
        
        # Update job with CSV path
        job.csv_file_path = csv_path
        job.save()
        
        logger.info(f"CSV generated for job {job_id}: {csv_path}")
        return f"CSV generated: {csv_path}"
        
    except Job.DoesNotExist:
        logger.error(f"Job {job_id} not found")
        return f"Job {job_id} not found"
    
    except Exception as e:
        logger.error(f"Error generating CSV for job {job_id}: {e}")
        raise self.retry(exc=e, countdown=30)


@shared_task
def cleanup_old_jobs():
    """
    Periodic task to clean up old completed jobs
    Run this task periodically to prevent database bloat
    """
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=30)  # Keep jobs for 30 days
    
    old_jobs = Job.objects.filter(
        status='completed',
        completed_at__lt=cutoff_date
    )
    
    count = old_jobs.count()
    
    # Delete associated files and records
    for job in old_jobs:
        try:
            # Delete CSV file if exists
            if job.csv_file_path:
                import os
                if os.path.exists(job.csv_file_path):
                    os.remove(job.csv_file_path)
            
            # Delete job (cascade will delete comments and sentiment results)
            job.delete()
            
        except Exception as e:
            logger.error(f"Error cleaning up job {job.id}: {e}")
    
    logger.info(f"Cleaned up {count} old jobs")
    return f"Cleaned up {count} old jobs"


@shared_task
def test_sentiment_analysis(text: str = "This is a test message"):
    """
    Test task for sentiment analysis
    Useful for testing the Celery setup
    """
    try:
        sentiment_service = SentimentAnalysisService()
        result = sentiment_service.analyze_text_preview(text)
        logger.info(f"Test sentiment analysis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in test sentiment analysis: {e}")
        return {"error": str(e)}