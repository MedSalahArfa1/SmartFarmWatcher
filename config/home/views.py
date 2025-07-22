from django.shortcuts import render

# Create your views here.
def home_view(request):
    """
    Home view for the application.
    This can be used to display a welcome message or redirect to the dashboard.
    """
    return render(request, 'index.html', {})