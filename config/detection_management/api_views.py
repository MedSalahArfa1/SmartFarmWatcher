# detection_management/api_views.py - Updated for your models

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta
from django.shortcuts import get_object_or_404

from project_management.models import Project, Camera, UserProjectRole
from .models import Detection, DetectionType


def get_user_projects(user):
    """Get all projects accessible to the user"""
    return Project.objects.filter(
        Q(created_by=user) | Q(user_roles__user=user),  # Updated to use user_roles
        is_active=True
    ).distinct()


def format_detection_data(detection):
    """Format detection data for API response"""
    return {
        'id': str(detection.id),
        'type': detection.detection_type.name.title(),
        'camera_id': f"Camera {detection.camera.id:02d}",
        'camera_name': detection.camera.description or f"Camera {detection.camera.id}",
        'confidence': float(detection.confidence_score),  # Convert Decimal to float
        'confidence_percentage': round(float(detection.confidence_score) * 100),
        'timestamp': detection.detected_at.isoformat(),
        'status': 'false_positive' if detection.is_false_positive else 'active',
        'location': detection.camera.farm_boundary.description if detection.camera.farm_boundary else None,
        'project_name': detection.camera.project.name,
        'project_id': str(detection.camera.project.id),
        'bounding_boxes': detection.bounding_boxes or [],
        'image_original_url': detection.image_original.url if detection.image_original else None,
        'image_annotated_url': detection.image_annotated.url if detection.image_annotated else None,
        'created_at': detection.detected_at.isoformat(),
        'notes': detection.notes or '',
    }


def format_project_data(project):
    """Format project data for API response"""
    return {
        'id': str(project.id),
        'name': project.name,
        'slug': project.slug,
        'location': project.location_city,  # Updated field name
        'description': project.description or None,
        'is_active': project.is_active,
        'created_at': project.created_at.isoformat(),
        'camera_count': project.get_total_cameras(),
        'detection_count': Detection.objects.filter(camera__project=project).count(),
        'access_code': project.access_code,  # Your model has this field
        'contact_person': project.contact_person,
        'contact_phone': project.contact_phone,
        'total_area_hectares': project.get_total_farm_area_hectares(),
        'farm_boundaries_count': project.get_total_farm_boundaries(),
    }


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    """Get dashboard statistics for the home screen"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get all cameras from user's projects
        user_cameras = Camera.objects.filter(
            project__in=user_projects,
            is_active=True
        )
        
        # Get all detections from user's projects
        all_detections = Detection.objects.filter(camera__project__in=user_projects)
        
        # Calculate time periods
        now = timezone.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        # Get active alerts (recent fire/smoke detections that are not false positives)
        active_alerts = all_detections.filter(
            detected_at__gte=today_start,
            is_false_positive=False,
            detection_type__name__in=['fire', 'smoke']
        ).count()
        
        # Get today's detections
        today_detections = all_detections.filter(
            detected_at__gte=today_start
        ).count()
        
        # Get detection type breakdown
        detection_stats = all_detections.values('detection_type__name').annotate(
            count=Count('id')
        )
        
        detection_breakdown = {}
        for stat in detection_stats:
            detection_breakdown[stat['detection_type__name']] = stat['count']
        
        # Calculate statistics
        stats = {
            'projects': user_projects.count(),
            'cameras': user_cameras.count(),
            'alerts': active_alerts,
            'detections': today_detections,
            'total_detections': all_detections.count(),
            'detection_breakdown': {
                'fire': detection_breakdown.get('fire', 0),
                'smoke': detection_breakdown.get('smoke', 0),
                'person': detection_breakdown.get('person', 0),
                'total': all_detections.count(),
            },
            'time_periods': {
                'today': all_detections.filter(detected_at__gte=today_start).count(),
                'week': all_detections.filter(detected_at__gte=week_start).count(),
                'month': all_detections.filter(detected_at__gte=month_start).count(),
            },
            'false_positives': all_detections.filter(is_false_positive=True).count(),
        }
        
        return Response({
            'success': True,
            'stats': stats,
            'timestamp': now.isoformat()
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def latest_detection(request):
    """Get the latest detection for the home screen"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get latest detection from user's projects
        latest_detection = Detection.objects.filter(
            camera__project__in=user_projects
        ).select_related(
            'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
        ).order_by('-detected_at').first()
        
        if not latest_detection:
            return Response({
                'success': True,
                'detection': None,
                'message': 'No detections found'
            })
        
        detection_data = format_detection_data(latest_detection)
        
        return Response({
            'success': True,
            'detection': detection_data
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def latest_project(request):
    """Get the latest/most recent project for the home screen"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get user's latest project
        latest_project = user_projects.order_by('-created_at').first()
        
        if not latest_project:
            return Response({
                'success': True,
                'project': None,
                'message': 'No projects found'
            })
        
        project_data = format_project_data(latest_project)
        
        return Response({
            'success': True,
            'project': project_data
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_detections(request):
    """Get recent detections for the home screen"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get limit from query params (default 10)
        limit = int(request.GET.get('limit', 10))
        
        # Get recent detections
        recent_detections = Detection.objects.filter(
            camera__project__in=user_projects
        ).select_related(
            'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
        ).order_by('-detected_at')[:limit]
        
        detections_data = [format_detection_data(detection) for detection in recent_detections]
        
        return Response({
            'success': True,
            'detections': detections_data,
            'count': len(detections_data)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def detection_detail(request, detection_id):
    """Get detailed information about a specific detection"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get detection
        detection = get_object_or_404(
            Detection.objects.select_related(
                'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
            ),
            id=detection_id,
            camera__project__in=user_projects
        )
        
        detection_data = format_detection_data(detection)
        
        # Add additional details for detail view
        detection_data.update({
            'camera_details': {
                'id': detection.camera.id,
                'camera_type': detection.camera.camera_type,
                'ip_address': detection.camera.ip_address,
                'port': detection.camera.port,
                'cellular_identifier': detection.camera.cellular_identifier,
                'location': {
                    'latitude': float(detection.camera.location.y) if detection.camera.location else None,
                    'longitude': float(detection.camera.location.x) if detection.camera.location else None,
                } if detection.camera.location else None,
                'description': detection.camera.description,
                'is_active': detection.camera.is_active,
                'farm_boundary_id': detection.camera.farm_boundary.id if detection.camera.farm_boundary else None,
            },
            'project_details': {
                'id': detection.camera.project.id,
                'name': detection.camera.project.name,
                'slug': detection.camera.project.slug,
                'description': detection.camera.project.description,
                'location_city': detection.camera.project.location_city,
                'contact_person': detection.camera.project.contact_person,
                'contact_phone': detection.camera.project.contact_phone,
                'is_active': detection.camera.project.is_active,
                'access_code': detection.camera.project.access_code,
            },
            'farm_boundary_details': {
                'id': detection.camera.farm_boundary.id if detection.camera.farm_boundary else None,
                'description': detection.camera.farm_boundary.description if detection.camera.farm_boundary else None,
                'area_hectares': float(detection.camera.farm_boundary.area_hectares) if detection.camera.farm_boundary and detection.camera.farm_boundary.area_hectares else None,
            } if detection.camera.farm_boundary else None
        })
        
        return Response({
            'success': True,
            'detection': detection_data
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_false_positive(request, detection_id):
    """Toggle false positive status of a detection"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Get detection
        detection = get_object_or_404(
            Detection,
            id=detection_id,
            camera__project__in=user_projects
        )
        
        # Toggle false positive status
        detection.is_false_positive = not detection.is_false_positive
        detection.save()
        
        status_text = "false positive" if detection.is_false_positive else "valid"
        
        return Response({
            'success': True,
            'message': f'Detection marked as {status_text}',
            'is_false_positive': detection.is_false_positive,
            'detection': format_detection_data(detection)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_projects(request):
    """Get all projects for the authenticated user"""
    try:
        user = request.user
        projects = get_user_projects(user)
        
        projects_data = [format_project_data(project) for project in projects]
        
        return Response({
            'success': True,
            'projects': projects_data,
            'count': len(projects_data)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def project_detections(request, project_id):
    """Get all detections for a specific project"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Verify user has access to this project
        project = get_object_or_404(Project, id=project_id, id__in=user_projects.values_list('id', flat=True))
        
        # Get detections for this project
        detections = Detection.objects.filter(
            camera__project=project
        ).select_related(
            'camera', 'detection_type'
        ).order_by('-detected_at')
        
        # Apply pagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        paginated_detections = paginator.paginate_queryset(detections, request)
        
        detections_data = [format_detection_data(detection) for detection in paginated_detections]
        
        return paginator.get_paginated_response({
            'success': True,
            'detections': detections_data,
            'project': format_project_data(project)
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def camera_detections(request, camera_id):
    """Get all detections for a specific camera"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Verify user has access to this camera
        camera = get_object_or_404(
            Camera,
            id=camera_id,
            project__in=user_projects,
            is_active=True
        )
        
        # Get detections for this camera
        detections = Detection.objects.filter(
            camera=camera
        ).select_related('detection_type').order_by('-detected_at')
        
        # Apply pagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        paginated_detections = paginator.paginate_queryset(detections, request)
        
        detections_data = [format_detection_data(detection) for detection in paginated_detections]
        
        camera_data = {
            'id': camera.id,
            'camera_type': camera.camera_type,
            'description': camera.description,
            'project_name': camera.project.name,
            'project_id': camera.project.id,
            'farm_boundary_id': camera.farm_boundary.id if camera.farm_boundary else None,
            'is_active': camera.is_active,
            'location': {
                'latitude': float(camera.location.y) if camera.location else None,
                'longitude': float(camera.location.x) if camera.location else None,
            } if camera.location else None,
            'connection_string': camera.get_connection_string(),
        }
        
        return paginator.get_paginated_response({
            'success': True,
            'detections': detections_data,
            'camera': camera_data
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_auth_test(request):
    """Test authentication endpoint"""
    try:
        user = request.user
        return Response({
            'success': True,
            'message': 'Authentication successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': getattr(user, 'first_name', ''),
                'last_name': getattr(user, 'last_name', ''),
                'full_name': getattr(user, 'get_full_name', lambda: f"{user.username}")(),
            },
            'timestamp': timezone.now().isoformat()
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Health check endpoint (no authentication required)
@api_view(['GET'])
@permission_classes([AllowAny])
def api_health_check(request):
    """Health check endpoint for the API"""
    return Response({
        'success': True,
        'message': 'Detection Management API is healthy',
        'timestamp': timezone.now().isoformat(),
        'version': '1.0.0'
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def detection_history(request):
    """Get paginated detection history with filtering options"""
    try:
        user = request.user
        user_projects = get_user_projects(user)
        
        # Base queryset - detections from user's projects
        detections = Detection.objects.filter(
            camera__project__in=user_projects
        ).select_related(
            'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
        )
        
        # Apply filters based on query parameters
        
        # Filter by detection type
        detection_type = request.GET.get('type')
        if detection_type and detection_type != 'all':
            detections = detections.filter(detection_type__name__iexact=detection_type)
        
        # Filter by status
        status_filter = request.GET.get('status')
        if status_filter and status_filter != 'all':
            if status_filter == 'false_positive':
                detections = detections.filter(is_false_positive=True)
            elif status_filter == 'active':
                detections = detections.filter(is_false_positive=False)
            elif status_filter == 'resolved':
                # You can add logic here if you have a resolved status
                # For now, treating resolved same as active
                detections = detections.filter(is_false_positive=False)
        
        # Filter by time range
        time_range = request.GET.get('time_range')
        if time_range and time_range != 'all':
            now = timezone.now()
            if time_range == 'today':
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                detections = detections.filter(detected_at__gte=start_date)
            elif time_range == 'week':
                start_date = now - timedelta(days=7)
                detections = detections.filter(detected_at__gte=start_date)
            elif time_range == 'month':
                start_date = now - timedelta(days=30)
                detections = detections.filter(detected_at__gte=start_date)
        
        # Search functionality
        search_query = request.GET.get('search')
        if search_query:
            detections = detections.filter(
                Q(camera__description__icontains=search_query) |
                Q(camera__project__name__icontains=search_query) |
                Q(detection_type__name__icontains=search_query) |
                Q(notes__icontains=search_query)
            )
        
        # Order by most recent first
        detections = detections.order_by('-detected_at')
        
        # Apply pagination
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get total count for pagination info
        total_count = detections.count()
        
        # Get paginated results
        paginated_detections = detections[offset:offset + page_size]
        
        # Format detection data
        detections_data = []
        for detection in paginated_detections:
            detection_data = format_detection_data(detection)
            detections_data.append(detection_data)
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        has_next = page < total_pages
        has_previous = page > 1
        
        return Response({
            'success': True,
            'detections': detections_data,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_previous': has_previous,
            },
            'filters_applied': {
                'type': detection_type or 'all',
                'status': status_filter or 'all',
                'time_range': time_range or 'all',
                'search': search_query or '',
            },
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)