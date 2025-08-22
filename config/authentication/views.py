# Django core imports
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views import View
from django.http import JsonResponse
from django.db import transaction

# Django REST Framework imports
from rest_framework import status, serializers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.exceptions import TokenError, InvalidToken

# Local app imports
from .models import AppUser
from .forms import SupervisorSignUpForm, ClientSignUpForm, LoginForm
from .serializers import ClientSignupSerializer, UserProfileSerializer
from project_management.models import Project, UserProjectRole

# Python standard library
import json

class SignUpView(View):
    template_name = 'authentication/signup.html'
    
    def get(self, request):
        supervisor_form = SupervisorSignUpForm()
        client_form = ClientSignUpForm()
        context = {
            'supervisor_form': supervisor_form,
            'client_form': client_form,
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        user_type = request.POST.get('user_type')
        
        if user_type == 'supervisor':
            return self.handle_supervisor_signup(request)
        elif user_type == 'client':
            return self.handle_client_signup(request)
        else:
            messages.error(request, 'Invalid user type selected.')
            return redirect('signup')
    
    def handle_supervisor_signup(self, request):
        form = SupervisorSignUpForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save(commit=False)
                    user.user_type = 'supervisor'
                    user.save()
                    
                    # Assign supervisor role
                    #UserProjectRole.objects.create(
                        #user=user,
                        #role='supervisor',
                    #)
                    
                    login(request, user)
                    messages.success(request, f'Welcome {user.get_full_name()}! Your account has been created successfully.')
                    return redirect('dashboard:dashboard')
            except Exception as e:
                messages.error(request, f'An error occurred during registration: {str(e)}')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
        
        return redirect('signup')
    
    def handle_client_signup(self, request):
        form = ClientSignUpForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save(commit=False)
                    user.user_type = 'client'
                    user.save()
                    
                    # Handle project access code
                    access_code = form.cleaned_data.get('access_code')
                    if access_code:
                        try:
                            project = Project.objects.get(access_code=access_code, is_active=True)
                            UserProjectRole.objects.create(
                                user=user,
                                project=project,
                                role='client',
                                joined_via_code=access_code,
                            )
                        except Project.DoesNotExist:
                            messages.error(request, 'Invalid access code provided.')
                            return redirect('signup')
                    
                    login(request, user)
                    messages.success(request, f'Welcome {user.get_full_name()}! Your account has been created successfully.')
                    return redirect('dashboard:dashboard')
            except Exception as e:
                messages.error(request, f'An error occurred during registration: {str(e)}')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
        
        return redirect('signup')

class LoginView(View):
    template_name = 'authentication/login.html'
    
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('dashboard')
        form = LoginForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.get_full_name()}!')
                next_url = request.GET.get('next', 'dashboard:dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
        
        return render(request, self.template_name, {'form': form})

#@login_required
def logout_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    user_name = request.user.get_full_name()
    logout(request)
    messages.success(request, f'Goodbye {user_name}! You have been logged out successfully.')
    return redirect('login')

# AJAX view for validating access codes
def validate_access_code(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            access_code = data.get('access_code', '').strip().upper()
            
            if not access_code:
                return JsonResponse({'valid': False, 'message': 'Access code is required.'})
            
            try:
                project = Project.objects.get(access_code=access_code, is_active=True)
                return JsonResponse({
                    'valid': True, 
                    'project_name': project.name,
                    'message': f'Valid access code for project: {project.name}'
                })
            except Project.DoesNotExist:
                return JsonResponse({'valid': False, 'message': 'Invalid access code.'})
        
        except json.JSONDecodeError:
            return JsonResponse({'valid': False, 'message': 'Invalid request format.'})
    
    return JsonResponse({'valid': False, 'message': 'Invalid request method.'})





class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom JWT token serializer for client login only"""
    
    def validate(self, attrs):
        # Get the user credentials
        username = attrs.get('username')
        password = attrs.get('password')
        
        # Authenticate user
        user = authenticate(username=username, password=password)
        
        if not user:
            raise serializers.ValidationError("Invalid username or password.")
        
        if not user.is_active:
            raise serializers.ValidationError("User account is disabled.")
        
        # Check if user is a client
        if user.user_type != 'client':
            raise serializers.ValidationError("Only clients can login through this endpoint.")
        
        # Call parent validation to get tokens
        data = super().validate(attrs)
        
        # Add user data to the response
        data['user'] = UserProfileSerializer(user).data
        
        return data

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        
        # Add custom claims
        token['user_type'] = user.user_type
        token['full_name'] = user.get_full_name()
        
        return token


@api_view(['POST'])
@permission_classes([AllowAny])
def client_signup(request):
    """
    Client signup endpoint with JWT token generation
    """
    serializer = ClientSignupSerializer(data=request.data)
    
    if serializer.is_valid():
        user = serializer.save()

        project = Project.objects.get(access_code=serializer.validated_data['access_code'], is_active=True)
        UserProjectRole.objects.create(
            user=user,
            project=project,
            role='client',
            joined_via_code=serializer.validated_data['access_code'],
        )
        
        # Generate JWT tokens for the user
        refresh = RefreshToken.for_user(user)
        
        # Add custom claims
        refresh['user_type'] = user.user_type
        refresh['full_name'] = user.get_full_name()
        
        # Return user data with tokens
        user_serializer = UserProfileSerializer(user)
        
        return Response({
            'message': 'Client account created successfully',
            'user': user_serializer.data,
            'tokens': {
                'access': str(refresh.access_token),
                'refresh': str(refresh)
            }
        }, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def client_login(request):
    """
    Client login endpoint with JWT token generation
    """
    serializer = CustomTokenObtainPairSerializer(data=request.data)
    
    try:
        if serializer.is_valid():
            return Response({
                'message': 'Login successful',
                'user': serializer.validated_data['user'],
                'tokens': {
                    'access': serializer.validated_data['access'],
                    'refresh': serializer.validated_data['refresh']
                }
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except TokenError as e:
        raise InvalidToken(e.args[0])


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def client_logout(request):
    """
    Client logout endpoint - blacklists the refresh token
    """
    try:
        refresh_token = request.data.get('refresh_token')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({
                'message': 'Logout successful'
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': 'Refresh token required'
            }, status=status.HTTP_400_BAD_REQUEST)
    except TokenError:
        return Response({
            'error': 'Invalid token'
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def token_refresh(request):
    """
    Token refresh endpoint
    """
    try:
        refresh_token = request.data.get('refresh')
        if not refresh_token:
            return Response({
                'error': 'Refresh token required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        token = RefreshToken(refresh_token)
        
        # Verify the user is still a client
        user = token.payload.get('user_id')
        user_obj = AppUser.objects.get(id=user)
        
        if user_obj.user_type != 'client':
            return Response({
                'error': 'Invalid user type'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        return Response({
            'access': str(token.access_token),
            'refresh': str(token) if token.get('rotate_refresh_tokens') else refresh_token
        }, status=status.HTTP_200_OK)
        
    except TokenError:
        return Response({
            'error': 'Invalid refresh token'
        }, status=status.HTTP_401_UNAUTHORIZED)
    except AppUser.DoesNotExist:
        return Response({
            'error': 'User not found'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    Get current user profile
    """
    serializer = UserProfileSerializer(request.user)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['PUT', 'PATCH'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """
    Update user profile
    """
    serializer = UserProfileSerializer(
        request.user, 
        data=request.data, 
        partial=True if request.method == 'PATCH' else False
    )
    
    if serializer.is_valid():
        serializer.save()
        return Response({
            'message': 'Profile updated successfully',
            'user': serializer.data
        }, status=status.HTTP_200_OK)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def verify_token(request):
    """
    Verify if the current token is valid
    """
    return Response({
        'message': 'Token is valid',
        'user': UserProfileSerializer(request.user).data
    }, status=status.HTTP_200_OK)