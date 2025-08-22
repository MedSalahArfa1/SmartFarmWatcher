import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class FireRiskPredictor:
    def __init__(self):
        self.api_key = getattr(settings, 'OPENWEATHER_API_KEY', None)
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather_data(self, lat, lon):
        """Get current weather data for coordinates"""
        if not self.api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def calculate_fire_risk(self, lat, lon):
        """Calculate fire risk based on weather data"""
        weather_data = self.get_weather_data(lat, lon)
        
        if not weather_data:
            return {
                'risk_score': 0,
                'risk_level': 'Unknown',
                'temperature': None,
                'humidity': None,
                'wind_speed': None,
                'error': 'Unable to fetch weather data'
            }
        
        try:
            # Extract weather parameters
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            wind_speed = weather_data['wind']['speed']
            
            # Apply fire risk rule (adjust this based on your specific rule)
            risk_score = self.apply_fire_risk_rule(temperature, humidity, wind_speed)
            
            return {
                'risk_score': risk_score,
                'risk_level': self.get_risk_level(risk_score),
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'weather_description': weather_data['weather'][0]['description'].title()
            }
        except KeyError as e:
            logger.error(f"Missing weather data key: {e}")
            return {
                'risk_score': 0,
                'risk_level': 'Unknown',
                'error': 'Invalid weather data format'
            }
    
    def apply_fire_risk_rule(self, temp, humidity, wind_speed):
        """
        Apply fire risk calculation rule
        Example rule - adjust based on your specific requirements:
        - High temperature increases risk
        - Low humidity increases risk  
        - High wind speed increases risk
        """
        # Normalize parameters to 0-100 scale
        temp_score = min(temp * 2, 100) if temp > 0 else 0  # Temp above 50°C = max score
        humidity_score = max(100 - humidity, 0)  # Lower humidity = higher risk
        wind_score = min(wind_speed * 10, 100)  # Wind above 10 m/s = max score
        
        # Weighted average (adjust weights based on your rule)
        risk_score = (temp_score * 0.4) + (humidity_score * 0.4) + (wind_score * 0.2)
        
        return round(risk_score, 1)
    
    def get_risk_level(self, score):
        """Convert numeric score to risk level"""
        if score < 25:
            return "Low"
        elif score < 50:
            return "Medium"
        elif score < 75:
            return "High"
        else:
            return "Extreme"
    
    def get_risk_color(self, risk_level):
        """Get color for risk level"""
        colors = {
            'Low': '#28a745',      # Green
            'Medium': '#ffc107',   # Yellow
            'High': '#fd7e14',     # Orange
            'Extreme': '#dc3545',  # Red
            'Unknown': '#6c757d'   # Gray
        }
        return colors.get(risk_level, '#6c757d')