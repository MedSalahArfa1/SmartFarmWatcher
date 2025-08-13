# project_management/models.py
from django.contrib.gis.db import models
from django.core.validators import RegexValidator, MinValueValidator
from django.core.exceptions import ValidationError
from django.utils.text import slugify
from django.utils import timezone
from authentication.models import AppUser
import secrets
import string
import uuid


class Project(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True, editable=False)
    access_code = models.CharField(max_length=12, unique=True, editable=False, db_index=True)
    description = models.TextField(null=True, blank=True)
    location_city = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    contact_person = models.CharField(max_length=100, null=True, blank=True)
    contact_phone = models.CharField(
        max_length=20, 
        null=True, 
        blank=True,
        validators=[RegexValidator(r'^\+?[1-9]\d{1,14}$', 'Enter a valid international phone number.')]
    )
    is_active = models.BooleanField(default=True, db_index=True)
    created_by = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name='created_projects')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = [['name', 'created_by']]  # Unique project names per user
        indexes = [
            models.Index(fields=['created_by', '-created_at']),
            models.Index(fields=['is_active', 'created_at']),
        ]
        ordering = ['-created_at']
    
    def save(self, *args, **kwargs):
        if not self.slug:
            # Generate unique slug across all projects
            base_slug = slugify(self.name)
            slug = base_slug
            counter = 1
            while Project.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
                if counter > 1000:
                    slug = f"{base_slug}-{int(timezone.now().timestamp())}"
                    break
            self.slug = slug
            
        if not self.access_code:
            self.access_code = self.generate_access_code()
            
        super().save(*args, **kwargs)
    
    def generate_access_code(self):
        """Generate a unique 12-character access code"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(12))
            if not Project.objects.filter(access_code=code).exists():
                return code
            attempts += 1
        
        # Fallback: use UUID if random generation fails
        return str(uuid.uuid4()).replace('-', '').upper()[:12]
    
    def regenerate_access_code(self):
        """Regenerate the access code"""
        self.access_code = self.generate_access_code()
        self.save(update_fields=['access_code'])
        return self.access_code
    
    def get_total_cameras(self):
        """Get total number of cameras in this project"""
        return self.cameras.filter(is_active=True).count()
    
    def get_total_farm_area_hectares(self):
        """Calculate total farm area in hectares from all farm boundaries"""
        total_area = 0
        for farm_boundary in self.farm_boundaries.filter(is_active=True):
            total_area += farm_boundary.area_hectares or 0
        return round(total_area, 2)
    
    def get_total_farm_boundaries(self):
        """Get total number of active farm boundaries"""
        return self.farm_boundaries.filter(is_active=True).count()
    
    def get_cameras_by_farm_boundary(self, farm_boundary):
        """Get cameras within a specific farm boundary"""
        if not farm_boundary.boundary:
            return self.cameras.filter(is_active=True, farm_boundary=farm_boundary)
        
        return self.cameras.filter(
            is_active=True,
            farm_boundary=farm_boundary,
            location__within=farm_boundary.boundary
        )
    
    def get_all_farm_boundaries_combined(self):
        """Get combined geometry of all farm boundaries"""
        from django.contrib.gis.geos import MultiPolygon
        
        active_boundaries = self.farm_boundaries.filter(is_active=True, boundary__isnull=False)
        if not active_boundaries.exists():
            return None
        
        combined_boundary = None
        for farm_boundary in active_boundaries:
            if combined_boundary is None:
                combined_boundary = farm_boundary.boundary
            else:
                combined_boundary = combined_boundary.union(farm_boundary.boundary)
        
        return combined_boundary
    
    def __str__(self):
        return f"{self.name} ({self.created_by.username})"


class FarmBoundary(models.Model):
    
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='farm_boundaries')
    description = models.TextField(null=True, blank=True)
    
    # Farm boundary polygon
    boundary = models.MultiPolygonField(null=True, blank=True, srid=4326)
    
    # Auto-calculated from boundary
    area_hectares = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        null=True, 
        blank=True,
        validators=[MinValueValidator(0.01)],
        help_text="Automatically calculated from boundary."
    )
    
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['project', 'is_active']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['created_at']
    
    def save(self, *args, **kwargs):
        # Auto-calculate area from boundary
        if self.boundary:
            self.area_hectares = self.get_area_hectares()
            
        super().save(*args, **kwargs)
    
    def clean(self):
        """Custom validation to prevent farm boundary overlap within same project"""
        errors = {}
        
        # Check for farm boundary overlap within the same project
        if self.boundary:
            is_valid, overlapping_boundaries = self.validate_no_boundary_overlap()
            if not is_valid:
                errors['boundary'] = f"Farm boundary overlaps with existing boundaries: {', '.join([str(b) for b in overlapping_boundaries])}"
        
        if errors:
            raise ValidationError(errors)
    
    def get_area_hectares(self):
        """Calculate farm area in hectares from the polygon boundary"""
        if self.boundary:
            try:
                # Transform to Web Mercator for area calculation
                geom_projected = self.boundary.transform(3857, clone=True)
                area_sqm = geom_projected.area
                return round(area_sqm / 10000, 2)  # Convert to hectares
            except Exception as e:
                print(f"Error calculating farm area: {e}")
                return 0
        return 0
    
    def validate_no_boundary_overlap(self):
        """Check if this farm boundary overlaps with existing boundaries in the same project"""
        if not self.boundary:
            return True, []
        
        # Get existing farm boundaries in the same project (exclude current boundary if updating)
        existing_boundaries = FarmBoundary.objects.filter(
            project=self.project,
            boundary__isnull=False,
            is_active=True
        )
        
        if self.pk:  # If updating existing boundary
            existing_boundaries = existing_boundaries.exclude(pk=self.pk)
        
        overlapping_boundaries = []
        for boundary in existing_boundaries:
            if self.boundary.intersects(boundary.boundary):
                overlapping_boundaries.append(boundary)
        
        return len(overlapping_boundaries) == 0, overlapping_boundaries
    
    def get_cameras_count(self):
        """Get number of cameras in this farm boundary"""
        return self.cameras.filter(is_active=True).count()
    
    def get_cameras_inside_boundary(self):
        """Get all cameras that are inside this farm boundary"""
        if not self.boundary:
            return self.cameras.filter(is_active=True)
        
        return self.cameras.filter(
            is_active=True,
            location__within=self.boundary
        )
    
    def get_center_point(self):
        """Get the center point of the farm boundary"""
        if self.boundary:
            try:
                return self.boundary.centroid
            except Exception:
                return None
        return None
    
    def __str__(self):
        return f"Farm Boundary #{self.pk} ({self.project.name})"


class UserProjectRole(models.Model):
    ROLES = [
        ('supervisor', 'Supervisor'),  # Creates and manages farms
        ('client', 'Client'),          # Views farms via access code
    ]
    
    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name='project_roles')
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='user_roles')
    role = models.CharField(max_length=20, choices=ROLES)
    joined_via_code = models.CharField(max_length=12, null=True, blank=True)  # For clients joining with access code
    assigned_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True, db_index=True)
    
    class Meta:
        unique_together = ['user', 'project']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['project', 'role']),
        ]
        ordering = ['-assigned_at']
    
    def __str__(self):
        return f"{self.user.get_full_name()} - {self.project.name} ({self.role})"


class Camera(models.Model):
    CAMERA_TYPES = [
        ('ip', 'IP Camera'),
        ('cellular', 'Cellular Camera'),
    ]
    
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='cameras')
    farm_boundary = models.ForeignKey(
        FarmBoundary, 
        on_delete=models.CASCADE, 
        related_name='cameras',
        help_text="The farm boundary this camera belongs to"
    )
    
    # Camera type
    camera_type = models.CharField(max_length=20, choices=CAMERA_TYPES, default='ip')
    
    # Camera coordinates (markers on map)
    location = models.PointField(srid=4326, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    
    # IP Camera specific fields
    ip_address = models.GenericIPAddressField(
        null=True, 
        blank=True,
        help_text="IP address for IP cameras"
    )
    port = models.PositiveIntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(1)],
        help_text="Port number for IP cameras (1-65535)"
    )
    
    # Cellular Camera specific fields
    cellular_identifier = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        unique=True,
        help_text="Unique identifier for cellular cameras"
    )

    # Heartbeat Tracking
    heartbeat_check = models.BooleanField(default=False, db_index=True)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['farm_boundary_id', 'created_at']
        indexes = [
            models.Index(fields=['project', 'is_active']),
            models.Index(fields=['farm_boundary', 'is_active']),
            models.Index(fields=['camera_type']),
            models.Index(fields=['created_at']),
        ]
    
    def clean(self):
        """Custom validation"""
        errors = {}
        
        # Ensure camera belongs to a farm boundary in the same project
        if self.farm_boundary and self.project:
            if self.farm_boundary.project != self.project:
                errors['farm_boundary'] = 'Camera must belong to a farm boundary within the same project.'
        
        # Validate camera is within farm boundary
        if self.location and self.farm_boundary and self.farm_boundary.boundary:
            if not self.farm_boundary.boundary.contains(self.location):
                errors['location'] = f'Camera must be placed within the farm boundary.'
        
        # Validate IP camera fields
        if self.camera_type == 'ip':
            if not self.ip_address:
                errors['ip_address'] = 'IP address is required for IP cameras.'
            if not self.port:
                errors['port'] = 'Port is required for IP cameras.'
            if self.port and (self.port < 1 or self.port > 65535):
                errors['port'] = 'Port must be between 1 and 65535.'
        
        # Validate cellular camera fields
        elif self.camera_type == 'cellular':
            if not self.cellular_identifier:
                errors['cellular_identifier'] = 'Cellular identifier is required for cellular cameras.'
        
        if errors:
            raise ValidationError(errors)
    
    def is_within_farm_boundary(self):
        """Check if camera is within its assigned farm boundary"""
        if self.farm_boundary and self.farm_boundary.boundary and self.location:
            try:
                return self.farm_boundary.boundary.contains(self.location)
            except Exception:
                return False
        return False
    
    def get_coordinates(self):
        """Get camera coordinates as tuple (lat, lng)"""
        if self.location:
            return (self.location.y, self.location.x)  # (latitude, longitude)
        return None
    
    def get_distance_from_boundary_center(self):
        """Get distance from farm boundary center in meters"""
        if self.location and self.farm_boundary and self.farm_boundary.boundary:
            try:
                center = self.farm_boundary.boundary.centroid
                # Transform to projected coordinate system for distance calculation
                location_proj = self.location.transform(3857, clone=True)
                center_proj = center.transform(3857, clone=True)
                return round(location_proj.distance(center_proj), 2)
            except Exception:
                return None
        return None
    
    def get_connection_string(self):
        """Get connection string based on camera type"""
        if self.camera_type == 'ip' and self.ip_address and self.port:
            return f"{self.ip_address}:{self.port}"
        elif self.camera_type == 'cellular' and self.cellular_identifier:
            return self.cellular_identifier
        return None
    
    def __str__(self):
        return f"Camera #{self.pk} ({self.get_camera_type_display()}) - Farm Boundary #{self.farm_boundary_id}"