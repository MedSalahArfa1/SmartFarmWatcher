from django.db import models
from authentication.models import AppUser
from detection_management.models import Detection
from django.utils import timezone

class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('detection', 'New Detection'),
        ('system', 'System Alert'),
    ]
    
    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name='notifications')
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, null=True, blank=True)
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='detection')
    title = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['user', 'is_read']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"
    
class FCMToken(models.Model):
    user = models.OneToOneField(AppUser, on_delete=models.CASCADE, related_name='fcm_token')
    token = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"FCM Token for {self.user.username}"