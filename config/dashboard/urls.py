# project_management/urls.py - UPDATE your existing urls.py

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('download_mobile_app/', views.mobile_app_view, name='download_mobile_app'),
]