from rest_framework import serializers
from .models import Notification

class NotificationSerializer(serializers.ModelSerializer):
    detection_id = serializers.IntegerField(source='detection.id', read_only=True)
    detection_type = serializers.CharField(source='detection.detection_type.name', read_only=True)
    camera_id = serializers.IntegerField(source='detection.camera.id', read_only=True)
    
    class Meta:
        model = Notification
        fields = [
            'id', 'notification_type', 'title', 'message', 'is_read', 
            'created_at', 'detection_id', 'detection_type', 'camera_id'
        ]