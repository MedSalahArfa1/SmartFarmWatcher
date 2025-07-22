import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import UntypedToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth import get_user_model
from .models import Notification
from .serializers import NotificationSerializer

User = get_user_model()

class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Get token from query string
        token = self.scope['query_string'].decode().split('token=')[-1]
        
        # Authenticate user
        user = await self.get_user_from_token(token)
        if user and user.is_authenticated:
            self.user = user
            self.group_name = f"notifications_{user.id}"
            
            # Join notification group
            await self.channel_layer.group_add(
                self.group_name,
                self.channel_name
            )
            await self.accept()
        else:
            await self.close()
    
    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )
    
    async def receive(self, text_data):
        # Handle mark as read requests
        try:
            data = json.loads(text_data)
            if data.get('action') == 'mark_read':
                notification_id = data.get('notification_id')
                await self.mark_notification_read(notification_id)
        except json.JSONDecodeError:
            pass
    
    async def notification_message(self, event):
        # Send notification to WebSocket
        await self.send(text_data=json.dumps(event['notification']))
    
    @database_sync_to_async
    def get_user_from_token(self, token):
        try:
            UntypedToken(token)
            from rest_framework_simplejwt.authentication import JWTAuthentication
            jwt_auth = JWTAuthentication()
            validated_token = jwt_auth.get_validated_token(token)
            user = jwt_auth.get_user(validated_token)
            return user
        except (InvalidToken, TokenError):
            return AnonymousUser()
    
    @database_sync_to_async
    def mark_notification_read(self, notification_id):
        try:
            notification = Notification.objects.get(id=notification_id, user=self.user)
            notification.is_read = True
            notification.save()
        except Notification.DoesNotExist:
            pass