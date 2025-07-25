from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from project_management.models import Project
from detection_management.models import Detection, Camera

@login_required
def mobile_app_view(request):
    """Dedicated mobile app download page - accessible to all users"""
    
    # Get user's projects for basic stats (optional - can be used for personalization)
    if request.user.user_type == 'supervisor':
        user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    else:
        user_projects = Project.objects.filter(
            user_roles__user=request.user,
            is_active=True
        ).distinct()
    
    # Basic stats (optional)
    project_count = user_projects.count()
    camera_count = Camera.objects.filter(
        project__in=user_projects,
        is_active=True
    ).count()
    
    context = {
        'project_count': project_count,
        'camera_count': camera_count,
        'user_projects': user_projects,
    }
    
    return render(request, 'dashboard/download_mobile_app.html', context)