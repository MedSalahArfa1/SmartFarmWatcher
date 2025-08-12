# Django core imports
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.contrib.gis.geos import GEOSGeometry, Point

# Local app imports
from .models import Project, FarmBoundary, Camera
from .forms import ProjectForm

# Python standard library
import json

@login_required
def create_project_wizard(request):
    """
    Single page project creation with 4 steps:
    1. Project Information
    2. Draw Farm Boundaries
    3. Place Cameras
    4. Review & Create
    """
    if request.method == 'POST':
        try:
            return _handle_project_creation(request)
        except Exception as e:
            messages.error(request, f'Error creating project: {str(e)}')
            return render(request, 'project_management/create_project_wizard.html', {
                'form': ProjectForm(),
                'step': 1
            })
    
    # GET request - show the wizard
    form = ProjectForm()
    context = {
        'form': form,
        'step': 1,
        'total_steps': 4
    }
    return render(request, 'project_management/create_project_wizard.html', context)


def _handle_project_creation(request):
    """Handle the actual project creation with all data"""
    
    # Parse the submitted data
    project_data = {
        'name': request.POST.get('project_name'),
        'description': request.POST.get('project_description', ''),
        'location_city': request.POST.get('location_city', ''),
        'contact_person': request.POST.get('contact_person', ''),
        'contact_phone': request.POST.get('contact_phone', ''),
    }
    
    # Parse farm boundaries data
    boundaries_json = request.POST.get('farm_boundaries_data', '[]')
    cameras_json = request.POST.get('cameras_data', '[]')
    
    
    try:
        farm_boundaries_data = json.loads(boundaries_json)
        cameras_data = json.loads(cameras_json)
    except json.JSONDecodeError:
        raise ValueError("Invalid farm boundaries or cameras data")
    
    
    # Validate project data
    form = ProjectForm(project_data)
    if not form.is_valid():
        raise ValueError(f"Project data validation failed: {form.errors}")
    
    # Create everything in a transaction
    with transaction.atomic():
        # Create the project
        project = Project(
            name=project_data['name'],
            description=project_data['description'],
            location_city=project_data['location_city'],
            contact_person=project_data['contact_person'],
            contact_phone=project_data['contact_phone'],
            created_by=request.user
        )
        project.full_clean()
        project.save()
        
        # Create farm boundaries and map them
        boundary_mapping = {}
        created_boundaries = []
        
        for boundary_data in farm_boundaries_data:
            farm_boundary = FarmBoundary(
                project=project,
                description=boundary_data.get('description', '')
            )
            
            # Convert boundary geometry with proper handling
            boundary_geom = boundary_data.get('boundary')
            if boundary_geom:
                try:
                    if isinstance(boundary_geom, str):
                        geom_data = json.loads(boundary_geom)
                    else:
                        geom_data = boundary_geom
                    
                    # Handle different geometry structures
                    geometry_json = None
                    if 'geometry' in geom_data:
                        geometry_json = geom_data['geometry']
                    elif 'type' in geom_data and 'coordinates' in geom_data:
                        geometry_json = geom_data
                    
                    if geometry_json:
                        # Create the geometry object
                        geom = GEOSGeometry(json.dumps(geometry_json))
                        
                        # Convert Polygon to MultiPolygon if needed
                        if geom.geom_type == 'Polygon':
                            # Import MultiPolygon from GEOS
                            from django.contrib.gis.geos import MultiPolygon
                            # Convert single polygon to multipolygon
                            multi_geom = MultiPolygon(geom)
                            farm_boundary.boundary = multi_geom
                        elif geom.geom_type == 'MultiPolygon':
                            farm_boundary.boundary = geom
                        else:
                            continue
                    
                except Exception as e:
                    continue
            
            farm_boundary.full_clean()
            farm_boundary.save()
            
            created_boundaries.append(farm_boundary)
            
            # Map temporary ID to actual boundary
            temp_id = boundary_data.get('temp_id')
            if temp_id:
                boundary_mapping[str(temp_id)] = farm_boundary  # Ensure string key
        
        
        # Create cameras
        cameras_created = 0
        for camera_data in cameras_data:
            
            # Find the corresponding farm boundary
            temp_boundary_id = camera_data.get('farm_boundary_temp_id')
            
            # Try different ways to find the boundary
            farm_boundary = None
            
            if temp_boundary_id:
                # Try exact match first
                farm_boundary = boundary_mapping.get(str(temp_boundary_id))
                if not farm_boundary:
                    # Try as integer
                    farm_boundary = boundary_mapping.get(int(temp_boundary_id) if str(temp_boundary_id).isdigit() else temp_boundary_id)
            
            # If still no boundary found, use the first created boundary as fallback
            if not farm_boundary and created_boundaries:
                farm_boundary = created_boundaries[0]
            
            if not farm_boundary:
                continue
            
            
            camera = Camera(
                project=project,
                farm_boundary=farm_boundary,
                camera_type=camera_data.get('camera_type', 'ip'),
                description=camera_data.get('description', ''),
                ip_address=camera_data.get('ip_address'),
                port=camera_data.get('port'),
                cellular_identifier=camera_data.get('cellular_identifier')
            )
            
            # Set camera location
            location_data = camera_data.get('location')
            if location_data:
                lat = location_data.get('lat')
                lng = location_data.get('lng')
                if lat and lng:
                    try:
                        camera.location = Point(float(lng), float(lat), srid=4326)
                    except (ValueError, TypeError) as e:
                        print(f"Error setting camera location: {e}")
            
            try:
                camera.full_clean()
                camera.save()
                cameras_created += 1
            except Exception as e:
                continue
            
    messages.success(request, f'Project "{project.name}" created successfully with {len(created_boundaries)} boundaries and {cameras_created} cameras!')
    return redirect('project_management:project_detail', slug=project.slug)


@csrf_exempt
@require_http_methods(["POST"])
def validate_boundary_step(request):
    """AJAX endpoint to validate farm boundary data before proceeding to camera step"""
    try:
        boundaries_json = request.POST.get('farm_boundaries_data', '[]')
        boundaries_data = json.loads(boundaries_json)
                
        if not boundaries_data:
            return JsonResponse({
                'valid': False,
                'message': 'At least one farm boundary is required.'
            })
        
        # Validate each boundary
        for i, boundary in enumerate(boundaries_data):
            boundary_geom = boundary.get('boundary')
            if not boundary_geom:
                return JsonResponse({
                    'valid': False,
                    'message': f'Farm boundary {i+1} must have a drawn area.'
                })
            
            # Handle different geometry formats
            try:
                if isinstance(boundary_geom, str):
                    # If it's a string, try to parse as JSON
                    geom_data = json.loads(boundary_geom)
                else:
                    # If it's already a dict/object
                    geom_data = boundary_geom
                                
                # Check if we have a geometry object or need to extract it
                geometry_json = None
                if 'geometry' in geom_data:
                    # Full GeoJSON feature
                    geometry_json = geom_data['geometry']
                elif 'type' in geom_data and 'coordinates' in geom_data:
                    # Just the geometry part
                    geometry_json = geom_data
                else:
                    return JsonResponse({
                        'valid': False,
                        'message': f'Farm boundary {i+1} has invalid geometry structure.'
                    })
                
                # Create the geometry object
                geom = GEOSGeometry(json.dumps(geometry_json))
                
                # Convert to MultiPolygon if it's a Polygon (for consistency with model)
                if geom.geom_type == 'Polygon':
                    from django.contrib.gis.geos import MultiPolygon
                    geom = MultiPolygon(geom)
                
                # Validate the geometry
                if not geom.valid:
                    return JsonResponse({
                        'valid': False,
                        'message': f'Farm boundary {i+1} has invalid geometry.'
                    })
                
                if geom.area == 0:
                    return JsonResponse({
                        'valid': False,
                        'message': f'Farm boundary {i+1} must have a valid area.'
                    })
                                
            except Exception as e:
                return JsonResponse({
                    'valid': False,
                    'message': f'Farm boundary {i+1} has invalid geometry: {str(e)}'
                })
        
        # Check for overlaps (simplified version)
        overlaps = _check_boundary_overlaps(boundaries_data)
        if overlaps:
            return JsonResponse({
                'valid': False,
                'message': f'Farm boundaries overlap: {overlaps}'
            })
        
        return JsonResponse({'valid': True})
        
    except json.JSONDecodeError as e:
        return JsonResponse({
            'valid': False,
            'message': 'Invalid boundary data format.'
        })
    except Exception as e:
        return JsonResponse({
            'valid': False,
            'message': f'Validation error: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def validate_camera_step(request):
    """AJAX endpoint to validate camera data before final review"""
    try:
        boundaries_json = request.POST.get('farm_boundaries_data', '[]')
        cameras_json = request.POST.get('cameras_data', '[]')
        
        boundaries_data = json.loads(boundaries_json)
        cameras_data = json.loads(cameras_json)
        
        if not cameras_data:
            return JsonResponse({
                'valid': False,
                'message': 'At least one camera is required.'
            })
        
        # Create boundary geometries for validation
        boundary_geoms = {}
        for boundary in boundaries_data:
            if boundary.get('temp_id') and boundary.get('boundary'):
                try:
                    boundary_geoms[boundary['temp_id']] = GEOSGeometry(boundary['boundary'])
                except Exception:
                    continue
        
        # Validate each camera
        for i, camera in enumerate(cameras_data):
            camera_type = camera.get('camera_type')
            
            # Validate required fields based on camera type
            if camera_type == 'ip':
                if not camera.get('ip_address'):
                    return JsonResponse({
                        'valid': False,
                        'message': f'Camera {i+1}: IP address is required for IP cameras.'
                    })
                if not camera.get('port'):
                    return JsonResponse({
                        'valid': False,
                        'message': f'Camera {i+1}: Port is required for IP cameras.'
                    })
            elif camera_type == 'cellular':
                if not camera.get('cellular_identifier'):
                    return JsonResponse({
                        'valid': False,
                        'message': f'Camera {i+1}: Cellular identifier is required.'
                    })
            
            # Validate camera location
            location = camera.get('location')
            boundary_temp_id = camera.get('farm_boundary_temp_id')
            
            if not location or not location.get('lat') or not location.get('lng'):
                return JsonResponse({
                    'valid': False,
                    'message': f'Camera {i+1}: Location is required.'
                })
            
            # Check if camera is within its assigned boundary
            if boundary_temp_id in boundary_geoms:
                try:
                    camera_point = Point(float(location['lng']), float(location['lat']), srid=4326)
                    boundary_geom = boundary_geoms[boundary_temp_id]
                    
                    if not boundary_geom.contains(camera_point):
                        return JsonResponse({
                            'valid': False,
                            'message': f'Camera {i+1}: Must be placed within its assigned farm boundary.'
                        })
                except Exception:
                    return JsonResponse({
                        'valid': False,
                        'message': f'Camera {i+1}: Invalid location coordinates.'
                    })
        
        return JsonResponse({'valid': True})
        
    except Exception as e:
        return JsonResponse({
            'valid': False,
            'message': f'Validation error: {str(e)}'
        })


def _check_boundary_overlaps(boundaries_data):
    """Check if any farm boundaries overlap with each other"""
    geometries = []
    
    # Parse all geometries
    for boundary in boundaries_data:
        boundary_geom = boundary.get('boundary')
        if boundary_geom:
            try:
                if isinstance(boundary_geom, str):
                    geom_data = json.loads(boundary_geom)
                else:
                    geom_data = boundary_geom
                
                # Handle different geometry structures
                geometry_json = None
                if 'geometry' in geom_data:
                    geometry_json = geom_data['geometry']
                elif 'type' in geom_data and 'coordinates' in geom_data:
                    geometry_json = geom_data
                else:
                    continue
                
                geom = GEOSGeometry(json.dumps(geometry_json))
                
                # Convert to MultiPolygon if needed for consistency
                if geom.geom_type == 'Polygon':
                    from django.contrib.gis.geos import MultiPolygon
                    geom = MultiPolygon(geom)
                
                geometries.append(geom)
            except Exception as e:
                continue
    
    # Check for overlaps
    overlapping_pairs = []
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            if geometries[i].intersects(geometries[j]):
                overlapping_pairs.append(f"Boundary {i+1} and Boundary {j+1}")
    
    return ", ".join(overlapping_pairs) if overlapping_pairs else None

@login_required
def project_list(request):
    """Display list of projects created by the logged-in user"""
    
    # Get search query if any
    search_query = request.GET.get('search', '')
    
    # Base queryset - projects created by the current user
    projects = Project.objects.filter(created_by=request.user)
    
    # Apply search filter if provided
    if search_query:
        projects = projects.filter(
            Q(name__icontains=search_query) |
            Q(location_city__icontains=search_query) |
            Q(description__icontains=search_query)
        )
    
    # Order by creation date (newest first)
    projects = projects.order_by('-created_at')
    
    # Add project statistics
    projects_with_stats = []
    for project in projects:
        project_data = {
            'project': project,
            'total_boundaries': project.get_total_farm_boundaries(),
            'total_cameras': project.get_total_cameras(),
            'total_area': project.get_total_farm_area_hectares(),
        }
        projects_with_stats.append(project_data)
    
    # Pagination
    paginator = Paginator(projects_with_stats, 10)  # Show 10 projects per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_projects': projects.count(),
    }
    
    return render(request, 'project_management/project_list.html', context)


@login_required
def project_detail(request, slug):
    """Display detailed view of a specific project with map"""
    
    # Get the project, ensuring it belongs to the current user
    project = get_object_or_404(Project, slug=slug, created_by=request.user)
    
    # Get all farm boundaries with their geometries
    farm_boundaries = project.farm_boundaries.filter(is_active=True)
    
    # Get all cameras with their locations
    cameras = project.cameras.filter(is_active=True).select_related('farm_boundary')
    
    # Prepare boundary data for the map
    boundaries_data = []
    for boundary in farm_boundaries:
        if boundary.boundary:
            try:
                # Convert geometry to GeoJSON for JavaScript
                boundary_data = {
                    'id': boundary.id,
                    'description': boundary.description,
                    'area_hectares': float(boundary.area_hectares) if boundary.area_hectares else 0,
                    'geometry': json.loads(boundary.boundary.geojson),
                    'created_at': boundary.created_at.isoformat()
                }
                boundaries_data.append(boundary_data)
            except Exception as e:
                continue
    
    # Prepare camera data for the map
    cameras_data = []
    for camera in cameras:
        if camera.location:
            try:
                camera_data = {
                    'id': camera.id,
                    'farm_boundary_id': camera.farm_boundary.id,
                    'camera_type': camera.camera_type,
                    'description': camera.description,
                    'latitude': camera.location.y,
                    'longitude': camera.location.x,
                    'connection_info': camera.get_connection_string(),
                    'is_within_boundary': camera.is_within_farm_boundary(),
                    'created_at': camera.created_at.isoformat()
                }
                cameras_data.append(camera_data)
            except Exception as e:
                continue
    
    # Calculate map center based on boundaries or cameras
    map_center = [36.8065, 10.1815]  # Default to Tunis
    if boundaries_data:
        # Use centroid of all boundaries
        try:
            combined_boundary = project.get_all_farm_boundaries_combined()
            if combined_boundary:
                centroid = combined_boundary.centroid
                map_center = [centroid.y, centroid.x]
        except Exception:
            pass
    elif cameras_data:
        # Use average of camera locations
        try:
            avg_lat = sum(cam['latitude'] for cam in cameras_data) / len(cameras_data)
            avg_lng = sum(cam['longitude'] for cam in cameras_data) / len(cameras_data)
            map_center = [avg_lat, avg_lng]
        except Exception:
            pass
    
    context = {
        'project': project,
        'farm_boundaries': farm_boundaries,
        'cameras': cameras,
        'boundaries_data': json.dumps(boundaries_data),
        'cameras_data': json.dumps(cameras_data),
        'map_center': json.dumps(map_center),
        'project_stats': {
            'total_boundaries': project.get_total_farm_boundaries(),
            'total_cameras': project.get_total_cameras(),
            'total_area': project.get_total_farm_area_hectares(),
        }
    }
    
    return render(request, 'project_management/project_detail.html', context)


@login_required
def project_delete(request, slug):
    """Delete a project"""
    project = get_object_or_404(Project, slug=slug, created_by=request.user)
    
    if request.method == 'POST':
        project_name = project.name
        project.delete()
        messages.success(request, f'Project "{project_name}" has been deleted successfully.')
        return redirect('project_management:project_list')
    
    return render(request, 'project_management/project_confirm_delete.html', {'project': project})


@login_required
def project_status_toggle(request, slug):
    """Toggle project active status via AJAX"""
    if request.method == 'POST':
        project = get_object_or_404(Project, slug=slug, created_by=request.user)
        project.is_active = not project.is_active
        project.save()
        
        return JsonResponse({
            'success': True,
            'is_active': project.is_active,
            'message': f'Project {"activated" if project.is_active else "deactivated"} successfully.'
        })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})


@login_required
def project_regenerate_code(request, slug):
    """Regenerate project access code via AJAX"""
    if request.method == 'POST':
        project = get_object_or_404(Project, slug=slug, created_by=request.user)
        new_code = project.regenerate_access_code()
        
        return JsonResponse({
            'success': True,
            'new_code': new_code,
            'message': 'Access code regenerated successfully.'
        })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})