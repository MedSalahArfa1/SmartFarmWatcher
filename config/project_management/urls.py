# project_management/urls.py - UPDATE your existing urls.py

from django.urls import path
from . import views

app_name = 'project_management'

urlpatterns = [
    # Project creation wizard
    path('create/', views.create_project_wizard, name='create_project_wizard'),
    
    # AJAX validation endpoints
    path('validate-boundary-step/', views.validate_boundary_step, name='validate_boundary_step'),
    path('validate-camera-step/', views.validate_camera_step, name='validate_camera_step'),
    
    # Project list and management
    path('', views.project_list, name='project_list'),
    path('project/<slug:slug>/', views.project_detail, name='project_detail'),
    path('project/<slug:slug>/delete/', views.project_delete, name='project_delete'),
    path('project/<slug:slug>/toggle-status/', views.project_status_toggle, name='project_status_toggle'),
    path('project/<slug:slug>/regenerate-code/', views.project_regenerate_code, name='project_regenerate_code'),
]