from django.urls import path
from . import views

urlpatterns = [
    # Job management endpoints  
    path('api/jobs/create/', views.JobCreateView.as_view(), name='job-create'),
    path('api/jobs/<uuid:job_id>/status/', views.JobStatusView.as_view(), name='job-status'),
    path('api/jobs/<uuid:job_id>/download/', views.JobCSVDownloadView.as_view(), name='job-download'),
]