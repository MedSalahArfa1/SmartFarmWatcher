from django.urls import path
from . import views

urlpatterns = [
    path('', views.LoginView.as_view(), name='home'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('validate-access-code/', views.validate_access_code, name='validate_access_code'),

    path('api/signup/', views.client_signup, name='client_signup'),
    path('api/login/', views.client_login, name='client_login'),
    path('api/logout/', views.client_logout, name='client_logout'),
    path('api/token/refresh/', views.token_refresh, name='token_refresh'),
    path('api/token/verify/', views.verify_token, name='verify_token'),

    # Profile endpoints
    path('api/profile/', views.user_profile, name='user_profile'),
    path('api/profile/update/', views.update_profile, name='update_profile'),
]