# project_management/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from leaflet.admin import LeafletGeoAdmin
from .models import Project, FarmBoundary, Camera, UserProjectRole


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'location_city', 'contact_person', 'created_by', 
        'total_boundaries', 'total_cameras', 'is_active', 'created_at'
    ]
    list_filter = ['is_active', 'location_city', 'created_at', 'created_by']
    search_fields = ['name', 'location_city', 'contact_person', 'access_code']
    readonly_fields = ['slug', 'access_code', 'created_at', 'updated_at', 'project_summary']
    
    fieldsets = (
        ('Project Information', {
            'fields': ('name', 'slug', 'description', 'access_code')
        }),
        ('Location & Contact', {
            'fields': ('location_city', 'contact_person', 'contact_phone')
        }),
        ('Status & Ownership', {
            'fields': ('is_active', 'created_by')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Summary', {
            'fields': ('project_summary',),
            'classes': ('collapse',)
        })
    )
    
    def total_boundaries(self, obj):
        count = obj.get_total_farm_boundaries()
        url = reverse('admin:project_management_farmboundary_changelist')
        return format_html(
            '<a href="{}?project__id__exact={}">{}</a>',
            url, obj.id, count
        )
    total_boundaries.short_description = 'Boundaries'
    
    def total_cameras(self, obj):
        count = obj.get_total_cameras()
        url = reverse('admin:project_management_camera_changelist')
        return format_html(
            '<a href="{}?project__id__exact={}">{}</a>',
            url, obj.id, count
        )
    total_cameras.short_description = 'Cameras'
    
    def project_summary(self, obj):
        if obj.pk:
            return format_html(
                """
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Project Summary</h4>
                    <p><strong>Access Code:</strong> <code>{}</code></p>
                    <p><strong>Total Farm Area:</strong> {} hectares</p>
                    <p><strong>Farm Boundaries:</strong> {}</p>
                    <p><strong>Total Cameras:</strong> {}</p>
                    <p><strong>Created:</strong> {}</p>
                </div>
                """,
                obj.access_code,
                obj.get_total_farm_area_hectares(),
                obj.get_total_farm_boundaries(),
                obj.get_total_cameras(),
                obj.created_at.strftime('%Y-%m-%d %H:%M')
            )
        return "Save the project first to see summary"
    project_summary.short_description = "Summary"


class CameraInline(admin.TabularInline):
    model = Camera
    extra = 0
    fields = [
        'camera_type', 'location', 'ip_address', 'port', 
        'cellular_identifier', 'description', 'is_active'
    ]
    
    # Use Leaflet widget for location field
    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == 'location':
            from leaflet.forms.widgets import LeafletWidget
            kwargs['widget'] = LeafletWidget()
        return super().formfield_for_dbfield(db_field, request, **kwargs)


@admin.register(FarmBoundary)
class FarmBoundaryAdmin(LeafletGeoAdmin):
    list_display = [
        'id_display', 'project', 'area_hectares', 'cameras_count', 
        'is_active', 'created_at'
    ]
    list_filter = ['is_active', 'project', 'created_at']
    search_fields = ['project__name', 'description']
    readonly_fields = ['area_hectares', 'created_at', 'updated_at', 'boundary_info']
    
    fieldsets = (
        ('Boundary Information', {
            'fields': ('project', 'description')
        }),
        ('Geographic Data', {
            'fields': ('boundary', 'area_hectares')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Boundary Details', {
            'fields': ('boundary_info',),
            'classes': ('collapse',)
        })
    )
    
    inlines = [CameraInline]
    
    # Leaflet map settings
    settings_overrides = {
        'DEFAULT_CENTER': (40.0, -95.0),
        'DEFAULT_ZOOM': 10,
        'MIN_ZOOM': 3,
        'MAX_ZOOM': 20,
        'RESET_VIEW': False,
        'SCALE': 'both',
        'ATTRIBUTION_PREFIX': 'Farm Management',
    }
    
    def id_display(self, obj):
        return f"Farm Boundary #{obj.pk}"
    id_display.short_description = 'Boundary'
    id_display.admin_order_field = 'pk'
    
    def cameras_count(self, obj):
        count = obj.get_cameras_count()
        if count > 0:
            url = reverse('admin:project_management_camera_changelist')
            return format_html(
                '<a href="{}?farm_boundary__id__exact={}">{}</a>',
                url, obj.id, count
            )
        return count
    cameras_count.short_description = 'Cameras'
    
    def boundary_info(self, obj):
        if obj.pk and obj.boundary:
            center = obj.get_center_point()
            return format_html(
                """
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Boundary Information</h4>
                    <p><strong>ID:</strong> Farm Boundary #{}</p>
                    <p><strong>Area:</strong> {} hectares</p>
                    <p><strong>Center Point:</strong> {:.6f}, {:.6f}</p>
                    <p><strong>Cameras Inside:</strong> {}</p>
                    <p><strong>Created:</strong> {}</p>
                </div>
                """,
                obj.pk,
                obj.area_hectares or 0,
                center.y if center else 0,
                center.x if center else 0,
                obj.get_cameras_count(),
                obj.created_at.strftime('%Y-%m-%d %H:%M')
            )
        return "Save the boundary first to see details"
    boundary_info.short_description = "Details"


@admin.register(Camera)
class CameraAdmin(LeafletGeoAdmin):
    list_display = [
        'id_display', 'project', 'farm_boundary_display', 'camera_type', 
        'connection_info', 'is_within_boundary', 'is_active', 'created_at'
    ]
    list_filter = [
        'camera_type', 'is_active', 'project', 'farm_boundary', 'created_at'
    ]
    search_fields = [
        'project__name', 'ip_address', 'cellular_identifier', 'description'
    ]
    readonly_fields = [
        'created_at', 'updated_at', 'camera_details', 
        'coordinates_display'
    ]
    
    fieldsets = (
        ('Camera Information', {
            'fields': ('project', 'farm_boundary', 'description')
        }),
        ('Camera Type & Settings', {
            'fields': ('camera_type', 'ip_address', 'port', 'cellular_identifier')
        }),
        ('Location', {
            'fields': ('location', 'coordinates_display')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
        ('Camera Details', {
            'fields': ('camera_details',),
            'classes': ('collapse',)
        })
    )
    
    # Leaflet map settings for cameras
    settings_overrides = {
        'DEFAULT_CENTER': (40.0, -95.0),
        'DEFAULT_ZOOM': 15,
        'MIN_ZOOM': 3,
        'MAX_ZOOM': 20,
        'RESET_VIEW': False,
        'SCALE': 'both',
        'ATTRIBUTION_PREFIX': 'Farm Management',
    }
    
    def id_display(self, obj):
        return f"Camera #{obj.pk}"
    id_display.short_description = 'Camera'
    id_display.admin_order_field = 'pk'
    
    def farm_boundary_display(self, obj):
        if obj.farm_boundary:
            return f"Farm Boundary #{obj.farm_boundary.pk}"
        return '-'
    farm_boundary_display.short_description = 'Farm Boundary'
    farm_boundary_display.admin_order_field = 'farm_boundary__pk'
    
    def connection_info(self, obj):
        connection = obj.get_connection_string()
        if connection:
            return format_html('<code>{}</code>', connection)
        return '-'
    connection_info.short_description = 'Connection'
    
    def is_within_boundary(self, obj):
        within = obj.is_within_farm_boundary()
        if within:
            return format_html(
                '<span style="color: green;">✓ Inside</span>'
            )
        else:
            return format_html(
                '<span style="color: red;">✗ Outside</span>'
            )
    is_within_boundary.short_description = 'Boundary Check'
    
    def coordinates_display(self, obj):
        coords = obj.get_coordinates()
        if coords:
            return format_html(
                'Lat: {:.6f}, Lng: {:.6f}', 
                coords[0], coords[1]
            )
        return 'No coordinates set'
    coordinates_display.short_description = 'Coordinates'
    
    def camera_details(self, obj):
        if obj.pk:
            coords = obj.get_coordinates()
            distance = obj.get_distance_from_boundary_center()
            
            return format_html(
                """
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Camera Details</h4>
                    <p><strong>ID:</strong> Camera #{}</p>
                    <p><strong>Type:</strong> {}</p>
                    <p><strong>Connection:</strong> <code>{}</code></p>
                    <p><strong>Coordinates:</strong> {}</p>
                    <p><strong>Within Boundary:</strong> {}</p>
                    <p><strong>Distance from Center:</strong> {} meters</p>
                    <p><strong>Created:</strong> {}</p>
                </div>
                """,
                obj.pk,
                obj.get_camera_type_display(),
                obj.get_connection_string() or 'Not configured',
                f'Lat: {coords[0]:.6f}, Lng: {coords[1]:.6f}' if coords else 'Not set',
                '✓ Yes' if obj.is_within_farm_boundary() else '✗ No',
                distance if distance else 'Unknown',
                obj.created_at.strftime('%Y-%m-%d %H:%M')
            )
        return "Save the camera first to see details"
    camera_details.short_description = "Details"


@admin.register(UserProjectRole)
class UserProjectRoleAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'project', 'role', 'joined_via_code', 
        'is_active', 'assigned_at'
    ]
    list_filter = ['role', 'is_active', 'assigned_at']
    search_fields = [
        'user__username', 'user__email', 'user__first_name', 
        'user__last_name', 'project__name'
    ]
    readonly_fields = ['assigned_at']
    
    fieldsets = (
        ('Role Assignment', {
            'fields': ('user', 'project', 'role')
        }),
        ('Access Details', {
            'fields': ('joined_via_code', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('assigned_at',)
        })
    )


# Custom admin site configuration
admin.site.site_header = "Farm Management Admin"
admin.site.site_title = "Farm Management"
admin.site.index_title = "Farm Management Administration"