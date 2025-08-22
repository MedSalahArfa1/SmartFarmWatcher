# detection_management/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from django.db.models import Count, Q
from django.urls import reverse
from django.utils.safestring import mark_safe
import json
from .models import DetectionType, Detection


@admin.register(DetectionType)
class DetectionTypeAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'is_active', 'detection_count', 'created_detections']
    list_filter = ['is_active']
    search_fields = ['name', 'description']
    list_editable = ['is_active']
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.annotate(
            total_detections=Count('detection'),
            recent_detections=Count('detection', filter=Q(detection__detected_at__gte=timezone.now() - timezone.timedelta(days=7)))
        )
    
    def detection_count(self, obj):
        """Show total number of detections for this type"""
        return obj.total_detections
    detection_count.short_description = 'Total Detections'
    detection_count.admin_order_field = 'total_detections'
    
    def created_detections(self, obj):
        """Show number of detections in the last 7 days"""
        return f"{obj.recent_detections} (last 7 days)"
    created_detections.short_description = 'Recent Activity'
    created_detections.admin_order_field = 'recent_detections'


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'camera_info', 'detection_type', 'confidence_display', 
        'detected_at', 'is_false_positive', 'image_preview', 'detection_area'
    ]
    list_filter = [
        'detection_type', 'is_false_positive', 'detected_at', 
        'camera__project', 'camera__camera_type', 'camera__farm_boundary'
    ]
    search_fields = ['camera__id', 'detection_type__name', 'notes', 'camera__description']
    readonly_fields = [
        'detected_at', 'image_preview_large', 'annotated_preview_large', 
        'bounding_boxes_display', 'detection_area_calculated'
    ]
    list_editable = ['is_false_positive']
    date_hierarchy = 'detected_at'
    list_per_page = 25
    
    fieldsets = (
        ('Detection Information', {
            'fields': ('camera', 'detection_type', 'confidence_score', 'detected_at')
        }),
        ('Images', {
            'fields': ('image_original', 'image_annotated', 'image_preview_large', 'annotated_preview_large'),
            'classes': ('wide',)
        }),
        ('Detection Data', {
            'fields': ('bounding_boxes', 'bounding_boxes_display', 'detection_area_calculated'),
            'classes': ('collapse',)
        }),
        ('Review', {
            'fields': ('is_false_positive', 'notes'),
        }),
    )
    
    actions = ['mark_as_false_positive', 'mark_as_valid', 'export_detections']
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.select_related(
            'camera', 'detection_type', 'camera__project', 'camera__farm_boundary'
        )
    
    def camera_info(self, obj):
        """Display camera information with link"""
        camera_url = reverse('admin:project_management_camera_change', args=[obj.camera.id])
        camera_type = obj.camera.get_camera_type_display()
        connection = obj.camera.get_connection_string() or 'No connection'
        
        return format_html(
            '<a href="{}">Camera #{}</a><br>'
            '<small>Type: {} | {}<br>'
            'Project: {} | Farm: #{}</small>',
            camera_url,
            obj.camera.id,
            camera_type,
            connection,
            obj.camera.project.name,
            obj.camera.farm_boundary.id if obj.camera.farm_boundary else 'No Farm'
        )
    camera_info.short_description = 'Camera Info'
    camera_info.admin_order_field = 'camera__id'
    
    def confidence_display(self, obj):
        """Display confidence score with color coding"""
        confidence = float(obj.confidence_score)
        if confidence >= 0.8:
            color = 'green'
        elif confidence >= 0.6:
            color = 'orange'
        else:
            color = 'red'
        
        # Convert to percentage manually to avoid format_html issues
        percentage = f"{confidence * 100:.2f}%"
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            percentage
        )
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence_score'
    
    def image_preview(self, obj):
        """Small image preview for list view"""
        if obj.image_annotated:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px;" />',
                obj.image_annotated.url
            )
        elif obj.image_original:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px;" />',
                obj.image_original.url
            )
        return "No Image"
    image_preview.short_description = 'Preview'
    
    def image_preview_large(self, obj):
        """Large image preview for detail view"""
        if obj.image_original:
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px; border-radius: 8px;" /><br><small>Original Image</small>',
                obj.image_original.url
            )
        return "No original image"
    image_preview_large.short_description = 'Original Image'
    
    def annotated_preview_large(self, obj):
        """Large annotated image preview for detail view"""
        if obj.image_annotated:
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px; border-radius: 8px;" /><br><small>Annotated Image</small>',
                obj.image_annotated.url
            )
        return "No annotated image"
    annotated_preview_large.short_description = 'Annotated Image'
    
    def bounding_boxes_display(self, obj):
        """Format bounding boxes for display"""
        if not obj.bounding_boxes:
            return "No bounding boxes"
        
        try:
            boxes = obj.bounding_boxes if isinstance(obj.bounding_boxes, list) else json.loads(obj.bounding_boxes)
            formatted_boxes = []
            for i, box in enumerate(boxes, 1):
                formatted_boxes.append(
                    f"Box {i}: x={box.get('x', 0)}, y={box.get('y', 0)}, "
                    f"width={box.get('width', 0)}, height={box.get('height', 0)}"
                )
            return format_html('<br>'.join(formatted_boxes))
        except (json.JSONDecodeError, TypeError):
            return "Invalid bounding box data"
    bounding_boxes_display.short_description = 'Bounding Boxes'
    
    def detection_area(self, obj):
        """Display detection area in list view"""
        area = obj.get_detection_area()
        return f"{area:.2f}"
    detection_area.short_description = 'Area'
    
    def detection_area_calculated(self, obj):
        """Display detailed detection area calculation"""
        area = obj.get_detection_area()
        box_count = len(obj.bounding_boxes) if obj.bounding_boxes else 0
        return f"{area:.4f} pixels² ({box_count} bounding boxes)"
    detection_area_calculated.short_description = 'Total Detection Area'
    
    # Admin Actions
    def mark_as_false_positive(self, request, queryset):
        """Mark selected detections as false positives"""
        updated = queryset.update(is_false_positive=True)
        self.message_user(request, f'{updated} detections marked as false positive.')
    mark_as_false_positive.short_description = "Mark selected detections as false positive"
    
    def mark_as_valid(self, request, queryset):
        """Mark selected detections as valid (not false positive)"""
        updated = queryset.update(is_false_positive=False)
        self.message_user(request, f'{updated} detections marked as valid.')
    mark_as_valid.short_description = "Mark selected detections as valid"
    
    def export_detections(self, request, queryset):
        """Export selected detections (placeholder for future implementation)"""
        count = queryset.count()
        self.message_user(request, f'Export functionality for {count} detections will be implemented.')
    export_detections.short_description = "Export selected detections"


# Optional: Custom admin site customization
admin.site.site_header = "Smart Farm Watcher Admin"