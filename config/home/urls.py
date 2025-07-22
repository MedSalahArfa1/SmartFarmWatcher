# detection_management/urls.py
from django.urls import path
from . import views

app_name = 'home'

urlpatterns = [
    # Main dashboard - shows latest detections from all cameras
    path('', views.home_view, name='home'),
]