from django.db import models
from project_management.models import Camera


class DetectionType(models.Model):
    """Types of detection available"""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']


class Detection(models.Model):
    """Store detection results from cameras"""
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='detections')
    detection_type = models.ForeignKey(DetectionType, on_delete=models.CASCADE)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=4)  # 0.0000 to 1.0000
    bounding_boxes = models.JSONField()  # Store bounding box coordinates
    image_original = models.ImageField(upload_to='detections/original/')
    image_annotated = models.ImageField(upload_to='detections/annotated/')
    detected_at = models.DateTimeField(auto_now_add=True)
    is_false_positive = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-detected_at']
        indexes = [
            models.Index(fields=['camera', '-detected_at']),
            models.Index(fields=['detection_type', '-detected_at']),
        ]
    
    def __str__(self):
        return f"{self.detection_type.name} detection from Camera #{self.camera.id}"
    
    def get_detection_area(self):
        """Calculate total area of all bounding boxes"""
        if not self.bounding_boxes:
            return 0.0
        
        total_area = 0.0
        for box in self.bounding_boxes:
            width = box.get('width', 0)
            height = box.get('height', 0)
            total_area += width * height
        
        return total_area