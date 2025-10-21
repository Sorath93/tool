from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from .models import Job
from .tasks import process_job


class JobCreateView(APIView):
    """
    Create a new sentiment analysis job
    POST /api/jobs/create/
    """
    def post(self, request, *args, **kwargs):
        # Get URLs from request
        urls = request.data.get('urls', [])
        
        if not urls or not isinstance(urls, list):
            return Response(
                {"error": "Please provide a list of URLs"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        valid_platforms = ['instagram.com', 'youtube.com', 'tiktok.com']
        for url in urls:
            if not any(platform in url.lower() for platform in valid_platforms):
                return Response(
                    {"error": f"Unsupported URL: {url}. Only Instagram, YouTube, and TikTok URLs are supported."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        job = Job.objects.create(post_urls=urls)
     
        process_job.delay(str(job.id))
        
        return Response({
            "job_id": str(job.id),
            "status": job.status,
            "message": "Job created"
        }, status=status.HTTP_202_ACCEPTED)


class JobStatusView(APIView):
    """
    Get job status and progress and other details
    """
    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        
        return Response({
            "job_id": str(job.id),
            "status": job.status,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error_message": job.message if job.status == 'failed' else None
        })


class JobCSVDownloadView(APIView):
    """
    Download job results as CSV
    """
    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        
        if job.status != 'COMPLETED':
            return Response({
                "error": "Results not ready. Job status: " + job.status,

            }, status=status.HTTP_409_CONFLICT)
    
        response = StreamingHttpResponse(
            self.generate_csv_rows(job),
            content_type='text/csv'
        )
        response['Content-Disposition'] = f'attachment; filename="sentiment_analysis_{job_id}.csv"'
        
        return response
    
    def generate_csv_rows(self, job):
        """A method to generate the rows of the CSV file for CommentSentiment entries related to the job."""
        from .models import CommentSentiment
        
        # CSV header row
        yield "post_url,platform,comment_text,sentiment_label,sentiment_score\n"
        
        # Get sentiment results for this job
        sentiments = CommentSentiment.objects.filter(comment__post__job=job)
        
        for sentiment in sentiments:
            # return one CSV row as a string --> memory efficient
            yield f"{sentiment.comment.post.post_url},{sentiment.comment.post.platform},{sentiment.comment.comment_text},{sentiment.label},{sentiment.score}\n" 