# serializers.py
from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import AppUser
from project_management.models import Project  # Make sure to import Project model

class ClientSignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        min_length=8,
        style={'input_type': 'password'}
    )
    password_confirm = serializers.CharField(
        write_only=True,
        style={'input_type': 'password'}
    )
    access_code = serializers.CharField(
        max_length=12,
        required=True,
        write_only=True,
        style={'input_type': 'text'},
        help_text='Enter the 12-character access code provided by your supervisor'
    )
    
    class Meta:
        model = AppUser
        fields = [
            'username', 'email', 'first_name', 'last_name',
            'phone_number', 'gender', 'profile_picture',
            'access_code', 'password', 'password_confirm'
        ]
        extra_kwargs = {
            'email': {'required': True},
            'first_name': {'required': True},
            'last_name': {'required': True},
        }
    
    def validate_email(self, value):
        if AppUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value
    
    def validate_username(self, value):
        if AppUser.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value
    
    def validate_access_code(self, value):
        """Validate the access code against the Project model"""
        access_code = value.strip().upper()
        if access_code:
            try:
                project = Project.objects.get(access_code=access_code, is_active=True)
                return access_code
            except Project.DoesNotExist:
                raise serializers.ValidationError('Invalid access code. Please check with your supervisor.')
        return access_code
    
    def validate(self, attrs):
        password = attrs.get('password')
        password_confirm = attrs.pop('password_confirm', None)
        
        if password != password_confirm:
            raise serializers.ValidationError("Passwords do not match.")
        
        # Validate password strength
        try:
            validate_password(password)
        except ValidationError as e:
            raise serializers.ValidationError({"password": e.messages})
        
        return attrs
    
    def create(self, validated_data):
        # Get the access code and find the associated project
        access_code = validated_data.pop('access_code', None)
        validated_data.pop('password_confirm', None)
        
        # Set user_type to 'client' for all signups through this serializer
        validated_data['user_type'] = 'client'
        
        # Create user with encrypted password
        user = AppUser.objects.create_user(**validated_data)
        
        # Optional: If you want to associate the user with the project
        # You might need to add a project field to your AppUser model
        # or create a separate relationship
        if access_code:
            try:
                project = Project.objects.get(access_code=access_code, is_active=True)
                # If you have a project field in AppUser model:
                # user.project = project
                # user.save()
                
                # Or if you have a separate UserProject model:
                # UserProject.objects.create(user=user, project=project)
                
            except Project.DoesNotExist:
                # This shouldn't happen due to validation, but just in case
                pass
        
        return user

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile information"""
    class Meta:
        model = AppUser
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name',
            'phone_number', 'gender', 'profile_picture', 'user_type',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user_type', 'created_at', 'updated_at']