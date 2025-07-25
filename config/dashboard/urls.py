# project_management/urls.py - UPDATE your existing urls.py

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Project creation wizard
    path('download_mobile_app/', views.mobile_app_view, name='download_mobile_app'),
]