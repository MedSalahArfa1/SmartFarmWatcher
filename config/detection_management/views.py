# detection_management/views.py - COMPLETE CORRECTED VERSION

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from project_management.models import Project, Camera
from .models import Detection, DetectionType
import json
import cv2
import numpy as np
from PIL import Image
import torch
import io
import base64
from datetime import datetime
import os
from django.conf import settings

# Load AI models (initialize once)
print("=== LOADING AI MODELS ===")
fire_model = None
person_model = None

try:
    from ultralytics import YOLO
    
    # Check if model files exist
    fire_model_path = 'models/FireShield.pt'
    person_model_path = 'models/yolo11s.pt'
    
    print(f"Checking fire model: {fire_model_path}")
    if os.path.exists(fire_model_path):
        print("✅ FireShield.pt found")
        fire_model = YOLO(fire_model_path)
        print("✅ Fire model loaded successfully with ultralytics YOLO")
    else:
        print("❌ FireShield.pt not found")
    
    print(f"Checking person model: {person_model_path}")
    if os.path.exists(person_model_path):
        print("✅ yolo11s.pt found")
        person_model = YOLO(person_model_path)
        print("✅ Person model loaded successfully with ultralytics YOLO")
    else:
        print("❌ yolo11s.pt not found")
        
except ImportError as e:
    print(f"❌ ultralytics not available: {e}")
    print("Will use dummy detection for testing")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Will use dummy detection for testing")


def process_detection_results(results, detection_type):
    """Process AI model results into standardized format"""
    detections = []
    
    try:
        print(f"Processing {detection_type} results...")
        
        # Handle ultralytics YOLO results
        if hasattr(results, '__iter__'):
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get box coordinates and confidence
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                        
                        print(f"Detection: box={box}, conf={conf}, cls={cls}")
                        
                        if conf > 0.3:  # Lower threshold for testing
                            x1, y1, x2, y2 = box
                            detections.append({
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1),
                                'confidence': float(conf),
                                'class': int(cls)
                            })
        
        print(f"Found {len(detections)} {detection_type} detections above threshold")
        
    except Exception as e:
        print(f"Error processing {detection_type} results: {e}")
        import traceback
        traceback.print_exc()
    
    return detections


def annotate_image(image_array, detections, detection_type):
    """Draw bounding boxes on image"""
    print(f"Annotating image with {len(detections)} {detection_type} detections")
    
    annotated = image_array.copy()
    
    # Color based on detection type
    color = (255, 0, 0) if detection_type == 'fire' else (0, 255, 0)  # Red for fire, Green for person
    
    for detection in detections:
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
        conf = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence text
        label = f"{detection_type}: {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Drew box at ({x1},{y1}) to ({x2},{y2}) with confidence {conf:.2f}")
    
    return annotated


def save_detection(camera, detection_type_name, detections, original_image, annotated_image):
    """Save detection to database"""
    print(f"Saving {detection_type_name} detection to database...")
    
    # Get or create detection type
    detection_type, created = DetectionType.objects.get_or_create(
        name=detection_type_name,
        defaults={'description': f'{detection_type_name.title()} detection'}
    )
    
    if created:
        print(f"Created new detection type: {detection_type_name}")
    
    # Calculate average confidence
    avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
    print(f"Average confidence: {avg_confidence}")
    
    # Create detection record
    detection = Detection(
        camera=camera,
        detection_type=detection_type,
        confidence_score=avg_confidence,
        bounding_boxes=detections
    )
    
    # Save original image
    original_filename = f"camera_{camera.id}_{detection_type_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_original.jpg"
    print(f"Saving original image as: {original_filename}")
    
    detection.image_original.save(
        original_filename,
        original_image,
        save=False
    )
    
    # Save annotated image
    annotated_pil = Image.fromarray(annotated_image)
    annotated_buffer = io.BytesIO()
    annotated_pil.save(annotated_buffer, format='JPEG')
    annotated_buffer.seek(0)
    
    annotated_filename = f"camera_{camera.id}_{detection_type_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.jpg"
    print(f"Saving annotated image as: {annotated_filename}")
    
    detection.image_annotated.save(
        annotated_filename,
        ContentFile(annotated_buffer.getvalue()),
        save=False
    )
    
    detection.save()
    print(f"✅ Detection saved to database with ID: {detection.id}")
    
    return detection


@csrf_exempt
@require_http_methods(["POST"])
def receive_image(request):
    """Receive image from camera and process detections"""
    print("\n=== NEW IMAGE RECEIVED ===")
    
    try:
        # Get camera identifier from request - support both methods
        camera_id = request.POST.get('camera_id')  # Direct ID (backward compatibility)
        ip_port = request.POST.get('ip_port')      # For IP cameras: "192.168.1.100:8080"
        cellular_id = request.POST.get('cellular_identifier')  # For cellular cameras
        
        print(f"Camera ID: {camera_id}")
        print(f"IP:Port: {ip_port}")
        print(f"Cellular ID: {cellular_id}")
        
        camera = None
        
        # Try to find camera by different methods
        if camera_id:
            # Direct camera ID lookup (existing method)
            camera = get_object_or_404(Camera, id=camera_id)
            print(f"Camera found by ID: {camera}")
            
        elif ip_port:
            # IP camera lookup by IP:port combination
            try:
                ip_address, port = ip_port.split(':')
                port = int(port)
                camera = get_object_or_404(
                    Camera, 
                    camera_type='ip',
                    ip_address=ip_address,
                    port=port,
                    is_active=True
                )
                print(f"IP Camera found: {camera} at {ip_address}:{port}")
            except ValueError:
                return JsonResponse({
                    'error': 'Invalid IP:port format. Expected format: "192.168.1.100:8080"'
                }, status=400)
            except Camera.DoesNotExist:
                return JsonResponse({
                    'error': f'No active IP camera found with address {ip_port}'
                }, status=404)
                
        elif cellular_id:
            # Cellular camera lookup by identifier
            try:
                camera = get_object_or_404(
                    Camera,
                    camera_type='cellular',
                    cellular_identifier=cellular_id,
                    is_active=True
                )
                print(f"Cellular Camera found: {camera} with ID {cellular_id}")
            except Camera.DoesNotExist:
                return JsonResponse({
                    'error': f'No active cellular camera found with identifier {cellular_id}'
                }, status=404)
        else:
            return JsonResponse({
                'error': 'Camera identifier required. Provide one of: camera_id, ip_port, or cellular_identifier'
            }, status=400)
        
        # Verify camera is active
        if not camera.is_active:
            return JsonResponse({
                'error': f'Camera {camera.id} is not active'
            }, status=400)
        
        # Get image from request
        image_file = request.FILES.get('image')
        print(f"Image file: {image_file}")
        
        if not image_file:
            return JsonResponse({'error': 'Image file required'}, status=400)
        
        # Convert image to format for AI processing
        print("Converting image...")
        image = Image.open(image_file)
        image_array = np.array(image)
        print(f"Image shape: {image_array.shape}")
        
        detections_created = []
        
        # Process fire detection
        print("\n--- FIRE DETECTION ---")
        if fire_model:
            print("Running fire detection...")
            try:
                # Run inference with ultralytics YOLO
                fire_results = fire_model(image_array, conf=0.3)  # Lower confidence for testing
                print(f"Fire results type: {type(fire_results)}")
                
                fire_detections = process_detection_results(fire_results, 'fire')
                print(f"Processed fire detections: {fire_detections}")
                
                if fire_detections:
                    annotated_image = annotate_image(image_array, fire_detections, 'fire')
                    detection = save_detection(camera, 'fire', fire_detections, image_file, annotated_image)
                    detections_created.append(detection.id)
                    print(f"✅ Fire detection saved with ID: {detection.id}")
                else:
                    print("❌ No fire detected")
                    
            except Exception as e:
                print(f"❌ Error in fire detection: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Fire model not loaded - creating dummy detection for testing")
            # Create dummy fire detection for testing
            dummy_detections = [{
                'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200,
                'width': 100, 'height': 100,
                'confidence': 0.85, 'class': 0
            }]
            annotated_image = annotate_image(image_array, dummy_detections, 'fire')
            detection = save_detection(camera, 'fire', dummy_detections, image_file, annotated_image)
            detections_created.append(detection.id)
            print(f"✅ Dummy fire detection created with ID: {detection.id}")
        
        # Process person detection
        print("\n--- PERSON DETECTION ---")
        if person_model:
            print("Running person detection...")
            try:
                # Run inference with ultralytics YOLO
                person_results = person_model(image_array, conf=0.3)  # Lower confidence for testing
                print(f"Person results type: {type(person_results)}")
                
                person_detections = process_detection_results(person_results, 'person')
                print(f"Processed person detections: {person_detections}")
                
                if person_detections:
                    annotated_image = annotate_image(image_array, person_detections, 'person')
                    detection = save_detection(camera, 'person', person_detections, image_file, annotated_image)
                    detections_created.append(detection.id)
                    print(f"✅ Person detection saved with ID: {detection.id}")
                else:
                    print("❌ No person detected")
                    
            except Exception as e:
                print(f"❌ Error in person detection: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Person model not loaded")
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Camera: {camera}")
        print(f"Total detections created: {len(detections_created)}")
        
        return JsonResponse({
            'success': True,
            'camera_id': camera.id,
            'camera_type': camera.camera_type,
            'detections_created': detections_created,
            'message': f'Processed {len(detections_created)} detections for {camera.get_camera_type_display()}'
        })
        
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def detection_dashboard(request):
    """Display latest detections from all cameras in user's projects"""
    # Get all projects for the current user
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    
    # Get all cameras from user's projects
    user_cameras = Camera.objects.filter(
        project__in=user_projects,
        is_active=True
    ).select_related('project', 'farm_boundary')
    
    # Get latest detection for each camera
    latest_detections = []
    for camera in user_cameras:
        latest_detection = Detection.objects.filter(camera=camera).first()
        if latest_detection:

            # Calculate confidence percentage for template use
            latest_detection.confidence_percentage = latest_detection.confidence_score * 100

            latest_detections.append({
                'camera': camera,
                'detection': latest_detection,
                'project': camera.project
            })
    
    # Get detection statistics
    total_detections = Detection.objects.filter(camera__project__in=user_projects).count()
    fire_detections = Detection.objects.filter(
        camera__project__in=user_projects,
        detection_type__name='fire'
    ).count()
    person_detections = Detection.objects.filter(
        camera__project__in=user_projects,
        detection_type__name='person'
    ).count()
    
    context = {
        'latest_detections': latest_detections,
        'user_projects': user_projects,
        'stats': {
            'total_detections': total_detections,
            'fire_detections': fire_detections,
            'person_detections': person_detections,
            'total_cameras': user_cameras.count()
        }
    }
    
    return render(request, 'detection_management/dashboard.html', context)


@login_required
def camera_detections(request, camera_id):
    """Display all detections for a specific camera"""
    camera = get_object_or_404(Camera, id=camera_id, project__created_by=request.user)
    
    detections = Detection.objects.filter(camera=camera).select_related('detection_type')

    for detection in detections:
        detection.confidence_percentage = detection.confidence_score * 100
    
    context = {
        'camera': camera,
        'detections': detections
    }
    
    return render(request, 'detection_management/camera_detections.html', context)


@login_required
def mark_false_positive(request, detection_id):
    """Mark detection as false positive"""
    if request.method == 'POST':
        detection = get_object_or_404(Detection, id=detection_id, camera__project__created_by=request.user)
        detection.is_false_positive = not detection.is_false_positive
        detection.save()
        
        return JsonResponse({
            'success': True,
            'is_false_positive': detection.is_false_positive
        })
    
    return JsonResponse({'success': False})


from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
from datetime import datetime, timedelta
import json

from .models import Detection, DetectionType, Camera


@login_required
def detection_history(request):
    """Display paginated history of all detections with filtering options"""
    # Get all projects for the current user
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    
    # Get all detections from user's projects
    detections = Detection.objects.filter(
        camera__project__in=user_projects
    ).select_related(
        'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
    ).order_by('-detected_at')
    
    # Apply filters
    search_query = request.GET.get('search', '').strip()
    detection_type = request.GET.get('detection_type', '').strip()
    status = request.GET.get('status', '').strip()
    date_range = request.GET.get('date_range', '').strip()
    
    # Search functionality
    if search_query:
        detections = detections.filter(
            Q(camera__project__name__icontains=search_query) |
            Q(camera__id__icontains=search_query) |
            Q(detection_type__name__icontains=search_query)
        )
    
    # Detection type filter
    if detection_type:
        detections = detections.filter(detection_type__name=detection_type)
    
    # Status filter
    if status == 'valid':
        detections = detections.filter(is_false_positive=False)
    elif status == 'false_positive':
        detections = detections.filter(is_false_positive=True)
    
    # Date range filter
    if date_range:
        now = timezone.now()
        if date_range == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'week':
            start_date = now - timedelta(days=7)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'month':
            start_date = now - timedelta(days=30)
            detections = detections.filter(detected_at__gte=start_date)
    
    # Add confidence percentage for each detection
    for detection in detections:
        detection.confidence_percentage = detection.confidence_score * 100
    
    # Pagination
    paginator = Paginator(detections, 20)  # Show 20 detections per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'detection_type': detection_type,
        'status': status,
        'date_range': date_range,
        'total_detections': detections.count(),
    }
    
    return render(request, 'detection_management/detection_history.html', context)


@login_required
@require_POST
def mark_false_positive(request, detection_id):
    """Toggle false positive status for a detection via AJAX"""
    try:
        # Get all projects for the current user
        user_projects = Project.objects.filter(created_by=request.user, is_active=True)
        
        # Get the detection, ensuring it belongs to user's projects
        detection = get_object_or_404(
            Detection, 
            id=detection_id, 
            camera__project__in=user_projects
        )
        
        # Toggle false positive status
        detection.is_false_positive = not detection.is_false_positive
        detection.save()
        
        status_text = "false positive" if detection.is_false_positive else "valid"
        message = f"Detection marked as {status_text}."
        
        return JsonResponse({
            'success': True,
            'message': message,
            'is_false_positive': detection.is_false_positive
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error updating detection status: {str(e)}'
        }, status=400)


@login_required
def detection_by_camera(request, camera_id):
    """Show all detections for a specific camera"""
    # Get all projects for the current user
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    
    # Get the camera, ensuring it belongs to user's projects
    camera = get_object_or_404(
        Camera, 
        id=camera_id, 
        project__in=user_projects
    )
    
    # Get all detections for this camera
    detections = Detection.objects.filter(
        camera=camera
    ).select_related('detection_type').order_by('-detected_at')
    
    # Apply filters (same as history page)
    search_query = request.GET.get('search', '').strip()
    detection_type = request.GET.get('detection_type', '').strip()
    status = request.GET.get('status', '').strip()
    date_range = request.GET.get('date_range', '').strip()
    
    if search_query:
        detections = detections.filter(
            Q(detection_type__name__icontains=search_query)
        )
    
    if detection_type:
        detections = detections.filter(detection_type__name=detection_type)
    
    if status == 'valid':
        detections = detections.filter(is_false_positive=False)
    elif status == 'false_positive':
        detections = detections.filter(is_false_positive=True)
    
    if date_range:
        now = timezone.now()
        if date_range == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'week':
            start_date = now - timedelta(days=7)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'month':
            start_date = now - timedelta(days=30)
            detections = detections.filter(detected_at__gte=start_date)
    
    # Add confidence percentage for each detection
    for detection in detections:
        detection.confidence_percentage = detection.confidence_score * 100
    
    # Pagination
    paginator = Paginator(detections, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get detection statistics for this camera
    stats = {
        'total_detections': detections.count(),
        'fire_detections': detections.filter(detection_type__name='fire').count(),
        'person_detections': detections.filter(detection_type__name='person').count(),
        'false_positives': detections.filter(is_false_positive=True).count(),
    }
    
    context = {
        'camera': camera,
        'page_obj': page_obj,
        'search_query': search_query,
        'detection_type': detection_type,
        'status': status,
        'date_range': date_range,
        'stats': stats,
    }
    
    return render(request, 'detection_management/camera_detections.html', context)


@login_required
def detection_statistics(request):
    """Show detection statistics and analytics"""
    # Get all projects for the current user
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    
    # Get all detections from user's projects
    all_detections = Detection.objects.filter(
        camera__project__in=user_projects
    ).select_related('camera', 'camera__project', 'detection_type')
    
    # Calculate statistics
    stats = {
        'total_detections': all_detections.count(),
        'fire_detections': all_detections.filter(detection_type__name='fire').count(),
        'person_detections': all_detections.filter(detection_type__name='person').count(),
        'false_positives': all_detections.filter(is_false_positive=True).count(),
        'valid_detections': all_detections.filter(is_false_positive=False).count(),
        'today_detections': all_detections.filter(
            detected_at__gte=timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count(),
        'week_detections': all_detections.filter(
            detected_at__gte=timezone.now() - timedelta(days=7)
        ).count(),
        'month_detections': all_detections.filter(
            detected_at__gte=timezone.now() - timedelta(days=30)
        ).count(),
    }
    
    # Get detections by project
    project_stats = []
    for project in user_projects:
        project_detections = all_detections.filter(camera__project=project)
        project_stats.append({
            'project': project,
            'total': project_detections.count(),
            'fire': project_detections.filter(detection_type__name='fire').count(),
            'person': project_detections.filter(detection_type__name='person').count(),
            'false_positives': project_detections.filter(is_false_positive=True).count(),
        })
    
    # Get recent detections for activity feed
    recent_detections = all_detections.order_by('-detected_at')[:10]
    for detection in recent_detections:
        detection.confidence_percentage = detection.confidence_score * 100
    
    context = {
        'stats': stats,
        'project_stats': project_stats,
        'recent_detections': recent_detections,
    }
    
    return render(request, 'detection_management/detection_statistics.html', context)


@login_required
def export_detections(request):
    """Export detections to CSV"""
    import csv
    from django.http import HttpResponse
    
    # Get all projects for the current user
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    
    # Get detections with same filters as history page
    detections = Detection.objects.filter(
        camera__project__in=user_projects
    ).select_related(
        'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
    ).order_by('-detected_at')
    
    # Apply filters if provided
    search_query = request.GET.get('search', '').strip()
    detection_type = request.GET.get('detection_type', '').strip()
    status = request.GET.get('status', '').strip()
    date_range = request.GET.get('date_range', '').strip()
    
    if search_query:
        detections = detections.filter(
            Q(camera__project__name__icontains=search_query) |
            Q(camera__id__icontains=search_query) |
            Q(detection_type__name__icontains=search_query)
        )
    
    if detection_type:
        detections = detections.filter(detection_type__name=detection_type)
    
    if status == 'valid':
        detections = detections.filter(is_false_positive=False)
    elif status == 'false_positive':
        detections = detections.filter(is_false_positive=True)
    
    if date_range:
        now = timezone.now()
        if date_range == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'week':
            start_date = now - timedelta(days=7)
            detections = detections.filter(detected_at__gte=start_date)
        elif date_range == 'month':
            start_date = now - timedelta(days=30)
            detections = detections.filter(detected_at__gte=start_date)
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="detections_export.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Detection ID',
        'Detection Type',
        'Camera ID',
        'Project Name',
        'Farm Boundary',
        'Confidence Score',
        'Detected At',
        'Status',
        'Image URL'
    ])
    
    for detection in detections:
        writer.writerow([
            detection.id,
            detection.detection_type.name,
            detection.camera.id,
            detection.camera.project.name,
            detection.camera.farm_boundary.id if detection.camera.farm_boundary else '',
            f"{detection.confidence_score * 100:.2f}%",
            detection.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            'False Positive' if detection.is_false_positive else 'Valid',
            detection.image_annotated.url if detection.image_annotated else ''
        ])
    
    return response