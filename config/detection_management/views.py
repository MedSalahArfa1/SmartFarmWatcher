import json
import os
import io
import base64
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image

from django.shortcuts import redirect, render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.paginator import Paginator
from django.db.models import Q
from django.utils import timezone
from django.conf import settings

from project_management.models import Project, Camera, UserProjectRole
from .models import Detection, DetectionType


# Load AI models (initialize once)
print("=== LOADING AI MODELS ===")
fire_model = None
person_model = None

try:
    from ultralytics import YOLO
    
    # Model paths
    fire_model_path = 'ai_models/FireShield.pt'
    person_model_path = 'ai_models/yolo11s.pt'
    
    # Load FireShield model (detects fire and smoke)
    print(f"Checking FireShield model: {fire_model_path}")
    if os.path.exists(fire_model_path):
        print("✅ FireShield.pt found")
        fire_model = YOLO(fire_model_path)
        print("✅ FireShield model loaded successfully (detects fire and smoke)")
    else:
        print("❌ FireShield.pt not found")
    
    # Load person detection model
    print(f"Checking person model: {person_model_path}")
    if os.path.exists(person_model_path):
        print("✅ yolo11s.pt found")
        person_model = YOLO(person_model_path)
        print("✅ Person model loaded successfully")
    else:
        print("❌ yolo11s.pt not found")
        
except ImportError as e:
    print(f"❌ ultralytics not available: {e}")
    print("Will use dummy detection for testing")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Will use dummy detection for testing")


def process_detection_results(results, model_type):
    """Process AI model results into standardized format"""
    detections = []
    
    try:
        print(f"Processing {model_type} results...")
        
        if hasattr(results, '__iter__'):
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                        
                        print(f"Detection: box={box}, conf={conf}, cls={cls}")
                        
                        if conf > 0.3:  # Confidence threshold
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
        
        print(f"Found {len(detections)} {model_type} detections above threshold")
        
    except Exception as e:
        print(f"Error processing {model_type} results: {e}")
        import traceback
        traceback.print_exc()
    
    return detections


def get_detection_type_from_class(class_id, model_type):
    """Map class ID to detection type name"""
    if model_type == 'fire':
        # FireShield model classes: 0=fire, 1=smoke
        class_mapping = {0: 'fire', 1: 'smoke'}
        return class_mapping.get(class_id, 'fire')
    elif model_type == 'person':
        # YOLO person detection: 0=person
        return 'person'
    
    return model_type


def get_detection_color(detection_type):
    """Get color for bounding box based on detection type"""
    colors = {
        'fire': (255, 0, 0),      # Red
        'smoke': (128, 128, 128),   # Gray
        'person': (0, 255, 0)     # Green
    }
    return colors.get(detection_type, (255, 255, 255))


def annotate_image(image_array, detections, detection_type):
    """Draw bounding boxes on image"""
    print(f"Annotating image with {len(detections)} {detection_type} detections")
    
    annotated = image_array.copy()
    color = get_detection_color(detection_type)
    
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f"camera_{camera.id}_{detection_type_name}_{timestamp}_original.jpg"
    print(f"Saving original image as: {original_filename}")
    
    detection.image_original.save(original_filename, original_image, save=False)
    
    # Save annotated image
    annotated_pil = Image.fromarray(annotated_image)
    annotated_buffer = io.BytesIO()
    annotated_pil.save(annotated_buffer, format='JPEG')
    annotated_buffer.seek(0)
    
    annotated_filename = f"camera_{camera.id}_{detection_type_name}_{timestamp}_annotated.jpg"
    print(f"Saving annotated image as: {annotated_filename}")
    
    detection.image_annotated.save(
        annotated_filename,
        ContentFile(annotated_buffer.getvalue()),
        save=False
    )
    
    detection.save()
    print(f"✅ Detection saved to database with ID: {detection.id}")
    
    return detection


def get_camera_by_identifier(camera_id=None, ip_port=None, cellular_id=None):
    """Get camera by different identifier types"""
    camera = None
    
    if camera_id:
        camera = get_object_or_404(Camera, id=camera_id)
        print(f"Camera found by ID: {camera}")
        
    elif ip_port:
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
            raise ValueError('Invalid IP:port format. Expected format: "192.168.1.100:8080"')
        except Camera.DoesNotExist:
            raise Camera.DoesNotExist(f'No active IP camera found with address {ip_port}')
            
    elif cellular_id:
        try:
            camera = get_object_or_404(
                Camera,
                camera_type='cellular',
                cellular_identifier=cellular_id,
                is_active=True
            )
            print(f"Cellular Camera found: {camera} with ID {cellular_id}")
        except Camera.DoesNotExist:
            raise Camera.DoesNotExist(f'No active cellular camera found with identifier {cellular_id}')
    
    return camera


def process_fire_smoke_detection(image_array, camera, image_file):
    """Process fire and smoke detection using FireShield model"""
    detections_created = []
    
    print("\n--- FIRE & SMOKE DETECTION ---")
    if fire_model:
        print("Running FireShield detection...")
        try:
            fire_results = fire_model(image_array, conf=0.3)
            print(f"FireShield results type: {type(fire_results)}")
            
            fire_detections = process_detection_results(fire_results, 'fire')
            print(f"Processed FireShield detections: {fire_detections}")
            
            # Group detections by type (fire vs smoke)
            fire_only = []
            smoke_only = []
            
            for detection in fire_detections:
                detection_type = get_detection_type_from_class(detection['class'], 'fire')
                if detection_type == 'fire':
                    fire_only.append(detection)
                elif detection_type == 'smoke':
                    smoke_only.append(detection)
            
            # Save fire detections
            if fire_only:
                annotated_image = annotate_image(image_array, fire_only, 'fire')
                detection = save_detection(camera, 'fire', fire_only, image_file, annotated_image)
                detections_created.append(detection.id)
                print(f"✅ Fire detection saved with ID: {detection.id}")
            
            # Save smoke detections
            if smoke_only:
                annotated_image = annotate_image(image_array, smoke_only, 'smoke')
                detection = save_detection(camera, 'smoke', smoke_only, image_file, annotated_image)
                detections_created.append(detection.id)
                print(f"✅ Smoke detection saved with ID: {detection.id}")
            
            if not fire_only and not smoke_only:
                print("❌ No fire or smoke detected")
                
        except Exception as e:
            print(f"❌ Error in FireShield detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ FireShield model not loaded - creating dummy detections for testing")
        # Create dummy fire detection
        dummy_fire = [{
            'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200,
            'width': 100, 'height': 100,
            'confidence': 0.85, 'class': 0
        }]
        # Create dummy smoke detection
        dummy_smoke = [{
            'x1': 250, 'y1': 150, 'x2': 350, 'y2': 250,
            'width': 100, 'height': 100,
            'confidence': 0.75, 'class': 1
        }]
        
        annotated_image = annotate_image(image_array, dummy_fire, 'fire')
        detection = save_detection(camera, 'fire', dummy_fire, image_file, annotated_image)
        detections_created.append(detection.id)
        
        annotated_image = annotate_image(image_array, dummy_smoke, 'smoke')
        detection = save_detection(camera, 'smoke', dummy_smoke, image_file, annotated_image)
        detections_created.append(detection.id)
        
        print(f"✅ Dummy fire and smoke detections created")
    
    return detections_created


def process_person_detection(image_array, camera, image_file):
    """Process person detection using YOLO model"""
    detections_created = []
    
    print("\n--- PERSON DETECTION ---")
    if person_model:
        print("Running person detection...")
        try:
            # Run YOLO detection
            person_results = person_model(image_array, conf=0.3)
            print(f"Person results type: {type(person_results)}")
            
            # Process results but FILTER for person class only
            all_detections = process_detection_results(person_results, 'person')
            print(f"All YOLO detections: {len(all_detections)}")
            
            # FILTER FOR PERSON CLASS ONLY (class 0 in COCO dataset)
            person_detections = []
            for detection in all_detections:
                if detection['class'] == 0:  # Person class in COCO is 0
                    person_detections.append(detection)
                else:
                    print(f"Filtered out class {detection['class']} (not person)")
            
            print(f"Filtered person detections: {len(person_detections)}")
            
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
    
    return detections_created


@csrf_exempt
@require_http_methods(["POST"])
def receive_image(request):
    """Receive image from camera and process detections"""
    print("\n=== NEW IMAGE RECEIVED ===")
    
    try:
        # Get camera identifier from request
        camera_id = request.POST.get('camera_id')
        ip_port = request.POST.get('ip_port')
        cellular_id = request.POST.get('cellular_identifier')
        
        print(f"Camera ID: {camera_id}")
        print(f"IP:Port: {ip_port}")
        print(f"Cellular ID: {cellular_id}")
        
        # Validate at least one identifier is provided
        if not any([camera_id, ip_port, cellular_id]):
            return JsonResponse({
                'error': 'Camera identifier required. Provide one of: camera_id, ip_port, or cellular_identifier'
            }, status=400)
        
        # Get camera by identifier
        try:
            camera = get_camera_by_identifier(camera_id, ip_port, cellular_id)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Camera.DoesNotExist as e:
            return JsonResponse({'error': str(e)}, status=404)
        
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

        # Process fire and smoke detection FIRST
        fire_smoke_detections = process_fire_smoke_detection(image_array, camera, image_file)
        detections_created.extend(fire_smoke_detections)

        # Only process person detection if NO fire/smoke was detected
        if len(fire_smoke_detections) == 0:
            print("No fire/smoke detected, proceeding with person detection...")
            person_detections = process_person_detection(image_array, camera, image_file)
            detections_created.extend(person_detections)
        else:
            print(f"Fire/smoke detected ({len(fire_smoke_detections)} detections), skipping person detection for safety")

        print(f"\n=== FINAL RESULT ===")
        print(f"Camera: {camera}")
        print(f"Total detections created: {len(detections_created)}")

        return JsonResponse({
            'success': True,
            'camera_id': camera.id,
            'camera_type': camera.camera_type,
            'detections_created': detections_created,
            'fire_smoke_detected': len(fire_smoke_detections) > 0,
            'person_detection_skipped': len(fire_smoke_detections) > 0,
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
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    user_cameras = Camera.objects.filter(
        project__in=user_projects,
        is_active=True
    ).select_related('project', 'farm_boundary')
    
    # Get latest detection for each camera
    latest_detections = []
    for camera in user_cameras:
        latest_detection = Detection.objects.filter(camera=camera).first()
        if latest_detection:
            latest_detection.confidence_percentage = latest_detection.confidence_score * 100
            latest_detections.append({
                'camera': camera,
                'detection': latest_detection,
                'project': camera.project
            })
    
    # Get detection statistics
    all_detections = Detection.objects.filter(camera__project__in=user_projects)
    stats = {
        'total_detections': all_detections.count(),
        'fire_detections': all_detections.filter(detection_type__name='fire').count(),
        'smoke_detections': all_detections.filter(detection_type__name='smoke').count(),
        'person_detections': all_detections.filter(detection_type__name='person').count(),
        'total_cameras': user_cameras.count()
    }
    
    context = {
        'latest_detections': latest_detections,
        'user_projects': user_projects,
        'stats': stats
    }
    
    return render(request, 'detection_management/latest_detection.html', context)


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
@require_POST
def mark_false_positive(request, detection_id):
    """Toggle false positive status for a detection via AJAX"""
    try:
        user_projects = Project.objects.filter(created_by=request.user, is_active=True)
        detection = get_object_or_404(
            Detection, 
            id=detection_id, 
            camera__project__in=user_projects
        )
        
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


def apply_detection_filters(detections, search_query, detection_type, status, date_range):
    """Apply filters to detection queryset"""
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
    
    return detections


@login_required
def detection_history(request):
    """Display paginated history of all detections with filtering options"""
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    detections = Detection.objects.filter(
        camera__project__in=user_projects
    ).select_related(
        'camera', 'camera__project', 'camera__farm_boundary', 'detection_type'
    ).order_by('-detected_at')
    
    # Get filter parameters
    search_query = request.GET.get('search', '').strip()
    detection_type = request.GET.get('detection_type', '').strip()
    status = request.GET.get('status', '').strip()
    date_range = request.GET.get('date_range', '').strip()
    
    # Apply filters
    detections = apply_detection_filters(detections, search_query, detection_type, status, date_range)
    
    # Add confidence percentage for each detection
    for detection in detections:
        detection.confidence_percentage = detection.confidence_score * 100
    
    # Pagination
    paginator = Paginator(detections, 20)
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
def detection_by_camera(request, camera_id):
    """Show all detections for a specific camera"""
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    camera = get_object_or_404(Camera, id=camera_id, project__in=user_projects)
    
    detections = Detection.objects.filter(
        camera=camera
    ).select_related('detection_type').order_by('-detected_at')
    
    # Get filter parameters
    search_query = request.GET.get('search', '').strip()
    detection_type = request.GET.get('detection_type', '').strip()
    status = request.GET.get('status', '').strip()
    date_range = request.GET.get('date_range', '').strip()
    
    # Apply filters
    detections = apply_detection_filters(detections, search_query, detection_type, status, date_range)
    
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
        'smoke_detections': detections.filter(detection_type__name='smoke').count(),
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
    user_projects = Project.objects.filter(created_by=request.user, is_active=True)
    all_detections = Detection.objects.filter(
        camera__project__in=user_projects
    ).select_related('camera', 'camera__project', 'detection_type')
    
    # Calculate statistics
    now = timezone.now()
    stats = {
        'total_detections': all_detections.count(),
        'fire_detections': all_detections.filter(detection_type__name='fire').count(),
        'smoke_detections': all_detections.filter(detection_type__name='smoke').count(),
        'person_detections': all_detections.filter(detection_type__name='person').count(),
        'false_positives': all_detections.filter(is_false_positive=True).count(),
        'valid_detections': all_detections.filter(is_false_positive=False).count(),
        'today_detections': all_detections.filter(
            detected_at__gte=now.replace(hour=0, minute=0, second=0, microsecond=0)
        ).count(),
        'week_detections': all_detections.filter(
            detected_at__gte=now - timedelta(days=7)
        ).count(),
        'month_detections': all_detections.filter(
            detected_at__gte=now - timedelta(days=30)
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
            'smoke': project_detections.filter(detection_type__name='smoke').count(),
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
def detection_detail_view(request, detection_id):
    """Show detailed view of a specific detection with map"""
    detection = get_object_or_404(Detection, id=detection_id)
    project = detection.camera.project
    
    # Check user access to project
    if not UserProjectRole.objects.filter(
        user=request.user, 
        project=project, 
        is_active=True
    ).exists():
        if project.created_by != request.user:
            return redirect('project_management:project_list')
    
    # Calculate confidence percentage
    detection.confidence_percentage = detection.confidence_score * 100
    
    # Get project farm boundaries for map
    boundaries_data = []
    for boundary in project.farm_boundaries.filter(is_active=True):
        if boundary.boundary:
            boundaries_data.append({
                'id': boundary.id,
                'geometry': json.loads(boundary.boundary.geojson),
                'area_hectares': float(boundary.area_hectares) if boundary.area_hectares else 0,
                'description': boundary.description or '',
                'created_at': boundary.created_at.isoformat()
            })
    
    # Get all project cameras
    cameras_data = []
    for camera in project.cameras.filter(is_active=True):
        if camera.location:
            cameras_data.append({
                'id': camera.id,
                'latitude': float(camera.location.y),
                'longitude': float(camera.location.x),
                'camera_type': camera.camera_type,
                'ip_address': camera.ip_address or '',
                'port': camera.port or '',
                'cellular_identifier': camera.cellular_identifier or '',
                'is_within_boundary': camera.is_within_farm_boundary(),
                'farm_boundary_id': camera.farm_boundary.id if camera.farm_boundary else None,
                'description': camera.description or '',
                'created_at': camera.created_at.isoformat()
            })
    
    # Detection camera data
    detection_camera_data = None
    if detection.camera.location:
        detection_camera_data = {
            'id': detection.camera.id,
            'latitude': float(detection.camera.location.y),
            'longitude': float(detection.camera.location.x),
            'camera_type': detection.camera.camera_type,
            'ip_address': detection.camera.ip_address or '',
            'port': detection.camera.port or '',
            'cellular_identifier': detection.camera.cellular_identifier or '',
            'is_within_boundary': detection.camera.is_within_farm_boundary(),
            'farm_boundary_id': detection.camera.farm_boundary.id if detection.camera.farm_boundary else None,
            'description': detection.camera.description or ''
        }
    
    # Map center - prioritize detection camera location
    map_center = [36.8065, 10.1815]  # Default to Tunis
    if detection.camera.location:
        map_center = [float(detection.camera.location.y), float(detection.camera.location.x)]
    else:
        # Fallback to project center if available
        for boundary in project.farm_boundaries.filter(is_active=True, boundary__isnull=False):
            if boundary.boundary:
                center_point = boundary.get_center_point()
                if center_point:
                    map_center = [float(center_point.y), float(center_point.x)]
                    break
    
    # Get project statistics for context
    project_stats = {
        'total_boundaries': project.get_total_farm_boundaries(),
        'total_cameras': project.get_total_cameras(),
        'total_area': project.get_total_farm_area_hectares(),
    }
    
    context = {
        'detection': detection,
        'project': project,
        'project_stats': project_stats,
        'boundaries_data': json.dumps(boundaries_data),
        'cameras_data': json.dumps(cameras_data),
        'detection_camera_data': json.dumps(detection_camera_data),
        'map_center': json.dumps(map_center)
    }
    
    return render(request, 'detection_management/detection_detail.html', context)