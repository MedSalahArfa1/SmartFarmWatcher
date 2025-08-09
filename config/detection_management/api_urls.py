# detection_management/api_urls.py - Django REST Framework API URLs

from django.urls import path
from . import api_views

app_name = 'detection_api'

urlpatterns = [
    # Health check
    path('health/', api_views.api_health_check, name='health_check'),
    
    # Dashboard endpoints
    path('dashboard/stats/', api_views.dashboard_stats, name='dashboard_stats'),
    path('dashboard/latest-detection/', api_views.latest_detection, name='latest_detection'),
    path('dashboard/latest-project/', api_views.latest_project, name='latest_project'),
    path('dashboard/recent-detections/', api_views.recent_detections, name='recent_detections'),
    
    # User projects
    path('projects/', api_views.user_projects, name='user_projects'),
    path('projects/<int:project_id>/detections/', api_views.project_detections, name='project_detections'),
    
    # Camera detections
    path('cameras/<int:camera_id>/detections/', api_views.camera_detections, name='camera_detections'),
    
    # Detection endpoints
    path('detections/<int:detection_id>/', api_views.detection_detail, name='detection_detail'),
    path('detections/<int:detection_id>/toggle-false-positive/', api_views.toggle_false_positive, name='toggle_false_positive'),

    path('history/', api_views.detection_history, name='detection_history'),

]

# URLs to add to main urls.py:
# 