from django.urls import path
from . import views

app_name = 'client_management'

urlpatterns = [
    # Main client list view
    path('', views.ClientListView.as_view(), name='client_list'),
    # Alternative function-based view
    # path('', views.client_list_view, name='client_list'),
    
    # Client detail view
    path('<int:pk>/', views.ClientDetailView.as_view(), name='client_detail'),
    # Alternative function-based view
    # path('<int:client_id>/', views.client_detail_view, name='client_detail'),
    
    # CRUD operations
    path('add/', views.ClientCreateView.as_view(), name='add_client'),
    path('<int:pk>/edit/', views.ClientUpdateView.as_view(), name='edit_client'),
    path('<int:pk>/delete/', views.ClientDeleteView.as_view(), name='delete_client'),
    
    # Alternative function-based CRUD views
    # path('add/', views.add_client_view, name='add_client'),
    # path('<int:client_id>/edit/', views.edit_client_view, name='edit_client'),
    # path('<int:client_id>/delete/', views.delete_client_view, name='delete_client'),
    
    # AJAX endpoints
    path('<int:client_id>/toggle-status/', views.toggle_client_status, name='toggle_client_status'),
    
    # Additional functionality
    path('<int:client_id>/projects/', views.client_projects_view, name='client_projects'),
    path('<int:client_id>/assign-project/', views.assign_project_to_client, name='assign_project'),
]