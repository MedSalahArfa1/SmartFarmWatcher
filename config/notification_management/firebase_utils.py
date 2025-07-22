import firebase_admin
from firebase_admin import credentials, messaging
from django.conf import settings
import json
import os

# Initialize Firebase Admin (add your service account key)
SERVICE_ACCOUNT_PATH = os.path.join(settings.BASE_DIR, 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

def send_fcm_notification(user, title, body, data=None):
    try:
        # Get user's FCM token
        fcm_token = user.fcm_token.token if hasattr(user, 'fcm_token') else None
        
        if fcm_token:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                token=fcm_token,
                android=messaging.AndroidConfig(
                    notification=messaging.AndroidNotification(
                        channel_id='smartfarm_notifications',
                        priority='high',
                        sound='default',
                    )
                ),
            )
            
            response = messaging.send(message)
            print(f"✅ FCM notification sent: {response}")
            return True
            
    except Exception as e:
        print(f"❌ FCM notification failed: {e}")
        return False