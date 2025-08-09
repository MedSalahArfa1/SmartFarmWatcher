# detection_management/urls.py
from django.urls import path
from . import views

app_name = 'detection_management'

urlpatterns = [
    # Main dashboard - shows latest detections from all cameras
    path('', views.detection_dashboard, name='dashboard'),
    
    # API endpoint for cameras to send images
    path('receive-image/', views.receive_image, name='receive_image'),
    
    # View all detections for a specific camera
    path('camera/<int:camera_id>/', views.camera_detections, name='camera_detections'),
    
    # Mark detection as false positive (AJAX)
    path('mark-false-positive/<int:detection_id>/', views.mark_false_positive, name='mark_false_positive'),

    # Detection history and filtering
    path('history/', views.detection_history, name='detection_history'),

    # Detection detail view
    path('<int:detection_id>/', views.detection_detail_view, name='detection_detail'),
]