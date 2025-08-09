from django.db.models.signals import post_save
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from detection_management.models import Detection
from project_management.models import UserProjectRole
from .models import Notification
from .serializers import NotificationSerializer
from .firebase_utils import send_fcm_notification

@receiver(post_save, sender=Detection)
def create_detection_notification(sender, instance, created, **kwargs):
    """
    Single receiver function that handles both WebSocket and FCM notifications
    """
    if created and not instance.is_false_positive:
        print(f"🎯 Detection signal fired for detection ID: {instance.id}")
        
        project = instance.camera.project
        users_to_notify = []
        
        # Get all users with access to this project
        project_users = UserProjectRole.objects.filter(
            project=project, 
            is_active=True
        ).select_related('user')
        
        for project_user in project_users:
            users_to_notify.append(project_user.user)
        
        print(f"📨 Will notify {len(users_to_notify)} users")
        
        # Create notifications and send both WebSocket and FCM for each user
        channel_layer = get_channel_layer()
        
        for user in users_to_notify:
            print(f"📤 Processing notification for user: {user.username}")
            
            # Create single notification in database
            notification = Notification.objects.create(
                user=user,
                detection=instance,
                notification_type='detection',
                title=f'New {instance.detection_type.name.capitalize()} Detection',
                message=f'Camera #{instance.camera.id} detected {instance.detection_type.name} with {instance.confidence_score:.1%} confidence'
            )
            
            print(f"✅ Database notification created: ID {notification.id}")
            
            # Send WebSocket notification (for real-time updates when user is active)
            try:
                notification_data = NotificationSerializer(notification).data
                group_name = f"notifications_{user.id}"
                
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        'type': 'notification_message',
                        'notification': notification_data
                    }
                )
                print(f"✅ WebSocket notification sent to {user.username}")
            except Exception as e:
                print(f"❌ WebSocket notification failed for {user.username}: {e}")
            
            # Send FCM notification (for push notifications when user is not active)
            try:
                send_fcm_notification(
                    user=user,
                    title=f'🚨 New {instance.detection_type.name} Detection',
                    body=f'Camera #{instance.camera.id} detected {instance.detection_type.name}',
                    data={
                        'notification_id': str(notification.id),
                        'detection_id': str(instance.id),
                        'type': 'detection',
                        'camera_id': str(instance.camera.id),
                        'project_id': str(project.id),
                    },
                    #high_priority=True,
                    #heads_up=True  # Show as heads-up notification
                )
                print(f"✅ FCM notification sent to {user.username}")
            except Exception as e:
                print(f"❌ FCM notification failed for {user.username}: {e}")
        
        print(f"🎉 Detection notification processing completed for {len(users_to_notify)} users")