from django.urls import path
from .views import NotificationListView, mark_notification_read, mark_all_read, store_fcm_token

urlpatterns = [
    path('notifications/', NotificationListView.as_view(), name='notifications'),
    path('notifications/<int:notification_id>/read/', mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', mark_all_read, name='mark_all_read'),


    path('fcm-token/', store_fcm_token, name='store_fcm_token'),
]