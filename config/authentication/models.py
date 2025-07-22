# models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import RegexValidator

class AppUser(AbstractUser):
    USER_TYPES = [
        ('client', 'Client'),
        ('supervisor', 'Supervisor'),
    ]
    
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]
    
    user_type = models.CharField(max_length=20, choices=USER_TYPES)
    phone_number = models.CharField(
        max_length=20, 
        null=True, 
        blank=True,
        validators=[RegexValidator(r'^\+?1?\d{9,15}$', 'Enter a valid phone number.')]
    )
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.get_full_name()} ({self.user_type})"
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip() or self.username