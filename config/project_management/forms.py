# project_management/forms.py
from django import forms
from django.contrib.gis import forms as geo_forms
from django.core.exceptions import ValidationError
from django.forms import modelformset_factory, inlineformset_factory
from .models import Project, FarmBoundary, Camera
import json


class ProjectForm(forms.ModelForm):
    """Form for creating/editing project general information"""
    
    class Meta:
        model = Project
        fields = ['name', 'description', 'location_city', 'contact_person', 'contact_phone']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter project name',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe your farm project...'
            }),
            'location_city': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'City where the farm is located'
            }),
            'contact_person': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Primary contact person name'
            }),
            'contact_phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '+1234567890',
                'type': 'tel'
            })
        }
        labels = {
            'name': 'Project Name',
            'description': 'Project Description',
            'location_city': 'Farm Location (City)',
            'contact_person': 'Contact Person',
            'contact_phone': 'Contact Phone'
        }
        help_texts = {
            'name': 'Choose a unique name for your farm project',
            'contact_phone': 'Include country code (e.g., +1234567890)',
            'location_city': 'The main city where your farm is located'
        }
    
    def clean_name(self):
        name = self.cleaned_data.get('name')
        if name and len(name.strip()) < 3:
            raise ValidationError("Project name must be at least 3 characters long.")
        return name.strip() if name else name
    
    def clean_contact_phone(self):
        phone = self.cleaned_data.get('contact_phone')
        if phone and not phone.strip():
            return None
        return phone


class FarmBoundaryForm(forms.ModelForm):
    """Form for creating/editing farm boundaries"""
    
    class Meta:
        model = FarmBoundary
        fields = ['description', 'boundary']
        widgets = {
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Description of this farm area...'
            }),
            'boundary': geo_forms.OSMWidget(attrs={
                'map_width': '100%',
                'map_height': 400,
                'default_zoom': 15
            })
        }
        labels = {
            'description': 'Description',
            'boundary': 'Farm Boundary Area'
        }
        help_texts = {
            'boundary': 'Draw the boundary of this farm area on the map',
            'description': 'Optional description of this farm area'
        }
    
    def __init__(self, *args, **kwargs):
        self.project = kwargs.pop('project', None)
        super().__init__(*args, **kwargs)
    
    def clean_boundary(self):
        boundary = self.cleaned_data.get('boundary')
        if not boundary:
            return boundary
        
        # Validate boundary doesn't overlap with existing boundaries in the same project
        if self.project:
            existing_boundaries = FarmBoundary.objects.filter(
                project=self.project,
                boundary__isnull=False,
                is_active=True
            )
            
            if self.instance.pk:
                existing_boundaries = existing_boundaries.exclude(pk=self.instance.pk)
            
            overlapping_boundaries = []
            for existing_boundary in existing_boundaries:
                if boundary.intersects(existing_boundary.boundary):
                    overlapping_boundaries.append(f"Farm Boundary #{existing_boundary.pk}")
            
            if overlapping_boundaries:
                raise ValidationError(
                    f"Farm boundary overlaps with existing boundaries: {', '.join(overlapping_boundaries)}"
                )
        
        return boundary


class CameraForm(forms.ModelForm):
    """Form for creating/editing cameras"""
    
    class Meta:
        model = Camera
        fields = [
            'farm_boundary', 'camera_type', 'description', 
            'location', 'ip_address', 'port', 'cellular_identifier'
        ]
        widgets = {
            'farm_boundary': forms.Select(attrs={
                'class': 'form-control',
                'required': True
            }),
            'camera_type': forms.Select(attrs={
                'class': 'form-control camera-type-select',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'Camera description...'
            }),
            'location': geo_forms.OSMWidget(attrs={
                'map_width': '100%',
                'map_height': 300,
                'default_zoom': 16
            }),
            'ip_address': forms.TextInput(attrs={
                'class': 'form-control ip-field',
                'placeholder': '192.168.1.100'
            }),
            'port': forms.NumberInput(attrs={
                'class': 'form-control ip-field',
                'placeholder': '554',
                'min': 1,
                'max': 65535
            }),
            'cellular_identifier': forms.TextInput(attrs={
                'class': 'form-control cellular-field',
                'placeholder': 'CAM001234567890'
            })
        }
        labels = {
            'farm_boundary': 'Farm Boundary',
            'camera_type': 'Camera Type',
            'description': 'Description',
            'location': 'Camera Location',
            'ip_address': 'IP Address',
            'port': 'Port',
            'cellular_identifier': 'Cellular Identifier'
        }
        help_texts = {
            'farm_boundary': 'Select which farm boundary this camera belongs to',
            'location': 'Click on the map to place the camera location',
            'ip_address': 'IP address for IP cameras only',
            'port': 'Port number for IP cameras (typically 554 for RTSP)',
            'cellular_identifier': 'Unique identifier for cellular cameras'
        }
    
    def __init__(self, *args, **kwargs):
        self.project = kwargs.pop('project', None)
        super().__init__(*args, **kwargs)
        
        # Filter farm boundaries to only those in the current project
        if self.project:
            self.fields['farm_boundary'].queryset = FarmBoundary.objects.filter(
                project=self.project,
                is_active=True
            ).order_by('created_at')
        
        # Set conditional field requirements
        self.fields['ip_address'].required = False
        self.fields['port'].required = False
        self.fields['cellular_identifier'].required = False
    
    def clean(self):
        cleaned_data = super().clean()
        camera_type = cleaned_data.get('camera_type')
        ip_address = cleaned_data.get('ip_address')
        port = cleaned_data.get('port')
        cellular_identifier = cleaned_data.get('cellular_identifier')
        location = cleaned_data.get('location')
        farm_boundary = cleaned_data.get('farm_boundary')
        
        # Validate based on camera type
        if camera_type == 'ip':
            if not ip_address:
                raise ValidationError({'ip_address': 'IP address is required for IP cameras.'})
            if not port:
                raise ValidationError({'port': 'Port is required for IP cameras.'})
        elif camera_type == 'cellular':
            if not cellular_identifier:
                raise ValidationError({'cellular_identifier': 'Cellular identifier is required for cellular cameras.'})
        
        # Validate camera location is within farm boundary
        if location and farm_boundary and farm_boundary.boundary:
            if not farm_boundary.boundary.contains(location):
                raise ValidationError({
                    'location': f'Camera must be placed within the farm boundary.'
                })
        
        # Validate farm boundary belongs to the project
        if farm_boundary and self.project:
            if farm_boundary.project != self.project:
                raise ValidationError({
                    'farm_boundary': 'Selected farm boundary must belong to the current project.'
                })
        
        return cleaned_data


# Formsets for handling multiple farm boundaries and cameras
FarmBoundaryFormSet = modelformset_factory(
    FarmBoundary,
    form=FarmBoundaryForm,
    fields=['description', 'boundary'],
    extra=1,
    can_delete=True,
    can_delete_extra=True
)

CameraFormSet = modelformset_factory(
    Camera,
    form=CameraForm,
    fields=['farm_boundary', 'camera_type', 'description', 'location', 'ip_address', 'port', 'cellular_identifier'],
    extra=1,
    can_delete=True,
    can_delete_extra=True
)


# Inline formsets for related objects
FarmBoundaryInlineFormSet = inlineformset_factory(
    Project,
    FarmBoundary,
    form=FarmBoundaryForm,
    fields=['description', 'boundary'],
    extra=1,
    can_delete=True,
    can_delete_extra=True
)

CameraInlineFormSet = inlineformset_factory(
    Project,
    Camera,
    form=CameraForm,
    fields=['farm_boundary', 'camera_type', 'description', 'location', 'ip_address', 'port', 'cellular_identifier'],
    extra=1,
    can_delete=True,
    can_delete_extra=True
)


class ProjectCreationForm(forms.Form):
    """Combined form for creating a complete project with boundaries and cameras"""
    
    # Project fields
    project_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter project name',
            'required': True
        }),
        label='Project Name',
        help_text='Choose a unique name for your farm project'
    )
    
    project_description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Describe your farm project...'
        }),
        label='Project Description'
    )
    
    location_city = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'City where the farm is located'
        }),
        label='Farm Location (City)'
    )
    
    contact_person = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Primary contact person name'
        }),
        label='Contact Person'
    )
    
    contact_phone = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '+1234567890',
            'type': 'tel'
        }),
        label='Contact Phone',
        help_text='Include country code (e.g., +1234567890)'
    )
    
    # Farm boundaries data (JSON)
    farm_boundaries_data = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        help_text='Farm boundaries data in JSON format'
    )
    
    # Cameras data (JSON)
    cameras_data = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        help_text='Cameras data in JSON format'
    )
    
    def clean_project_name(self):
        name = self.cleaned_data.get('project_name')
        if name and len(name.strip()) < 3:
            raise ValidationError("Project name must be at least 3 characters long.")
        return name.strip() if name else name
    
    def clean_farm_boundaries_data(self):
        data = self.cleaned_data.get('farm_boundaries_data')
        if not data:
            return []
        
        try:
            boundaries = json.loads(data)
            if not isinstance(boundaries, list):
                raise ValidationError("Farm boundaries data must be a list.")
            return boundaries
        except json.JSONDecodeError:
            raise ValidationError("Invalid farm boundaries data format.")
    
    def clean_cameras_data(self):
        data = self.cleaned_data.get('cameras_data')
        if not data:
            return []
        
        try:
            cameras = json.loads(data)
            if not isinstance(cameras, list):
                raise ValidationError("Cameras data must be a list.")
            return cameras
        except json.JSONDecodeError:
            raise ValidationError("Invalid cameras data format.")
    
    def create_project(self, user):
        """Create project with all related objects"""
        from django.db import transaction
        
        with transaction.atomic():
            # Create project
            project = Project(
                name=self.cleaned_data['project_name'],
                description=self.cleaned_data.get('project_description', ''),
                location_city=self.cleaned_data.get('location_city', ''),
                contact_person=self.cleaned_data.get('contact_person', ''),
                contact_phone=self.cleaned_data.get('contact_phone', ''),
                created_by=user
            )
            project.full_clean()
            project.save()
            
            # Create farm boundaries
            farm_boundaries_data = self.cleaned_data.get('farm_boundaries_data', [])
            boundary_mapping = {}  # Map old IDs to new objects
            
            for boundary_data in farm_boundaries_data:
                farm_boundary = FarmBoundary(
                    description=boundary_data.get('description', ''),
                    project=project
                )
                
                # Handle boundary geometry
                if boundary_data.get('boundary'):
                    from django.contrib.gis.geos import GEOSGeometry
                    farm_boundary.boundary = GEOSGeometry(boundary_data['boundary'])
                
                farm_boundary.full_clean()
                farm_boundary.save()
                
                # Store mapping for camera assignment
                old_id = boundary_data.get('id')
                if old_id:
                    boundary_mapping[str(old_id)] = farm_boundary
            
            # Create cameras
            cameras_data = self.cleaned_data.get('cameras_data', [])
            created_cameras = []
            
            for camera_data in cameras_data:
                # Find the corresponding farm boundary
                farm_boundary_id = camera_data.get('farm_boundary_id')
                farm_boundary = boundary_mapping.get(str(farm_boundary_id))
                
                if not farm_boundary:
                    # Skip cameras without valid farm boundary
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
                
                # Handle camera location
                if camera_data.get('location'):
                    from django.contrib.gis.geos import Point
                    lat = camera_data['location'].get('lat')
                    lng = camera_data['location'].get('lng')
                    if lat and lng:
                        camera.location = Point(float(lng), float(lat), srid=4326)
                
                camera.full_clean()
                camera.save()
                created_cameras.append(camera)
            
            return project, created_cameras