from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
import json

User = get_user_model()

class ClientListView(LoginRequiredMixin, ListView):
    """List view for clients with search and filtering capabilities."""
    model = User
    template_name = 'client_management/client_list.html'
    context_object_name = 'page_obj'
    paginate_by = 10
    
    def get_queryset(self):
        """Filter clients based on search query and status."""
        # Only show clients (assuming you have a user_type field)
        queryset = User.objects.filter(user_type='client').select_related().prefetch_related('project_roles__project')
        
        # Search functionality
        search_query = self.request.GET.get('search', '').strip()
        if search_query:
            queryset = queryset.filter(
                Q(first_name__icontains=search_query) |
                Q(last_name__icontains=search_query) |
                Q(username__icontains=search_query) |
                Q(email__icontains=search_query)
            )
        
        # Status filter
        status_filter = self.request.GET.get('status', '').strip()
        if status_filter == 'active':
            queryset = queryset.filter(is_active=True)
        elif status_filter == 'inactive':
            queryset = queryset.filter(is_active=False)
        
        return queryset.order_by('-date_joined')
    
    def get_context_data(self, **kwargs):
        """Add search parameters to context."""
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        context['status_filter'] = self.request.GET.get('status', '')
        return context


@login_required
def client_list_view(request):
    """Function-based view alternative for client list."""
    # Get all clients with related data
    clients = User.objects.filter(user_type='client').select_related().prefetch_related('project_roles__project')
    
    # Search functionality
    search_query = request.GET.get('search', '').strip()
    if search_query:
        clients = clients.filter(
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query) |
            Q(username__icontains=search_query) |
            Q(email__icontains=search_query)
        )
    
    # Status filter
    status_filter = request.GET.get('status', '').strip()
    if status_filter == 'active':
        clients = clients.filter(is_active=True)
    elif status_filter == 'inactive':
        clients = clients.filter(is_active=False)
    
    # Order by most recent
    clients = clients.order_by('-date_joined')
    
    # Pagination
    paginator = Paginator(clients, 10)  # Show 10 clients per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'status_filter': status_filter,
    }
    
    return render(request, 'client_management/client_list.html', context)


class ClientDetailView(LoginRequiredMixin, DetailView):
    """Detail view for individual client."""
    model = User
    template_name = 'client_management/client_detail.html'
    context_object_name = 'client'
    
    def get_queryset(self):
        return User.objects.filter(user_type='client').prefetch_related('project_roles__project')


@login_required
def client_detail_view(request, client_id):
    """Function-based view for client details."""
    client = get_object_or_404(User, id=client_id, user_type='client')
    
    # Get client's projects and roles
    project_roles = client.project_roles.select_related('project').all()
    
    context = {
        'client': client,
        'project_roles': project_roles,
    }
    
    return render(request, 'client_management/client_detail.html', context)


@login_required
@require_POST
def toggle_client_status(request, client_id):
    """Toggle client active/inactive status via AJAX."""
    try:
        client = get_object_or_404(User, id=client_id, user_type='client')
        
        # Parse JSON data
        data = json.loads(request.body)
        new_status = data.get('active', not client.is_active)
        
        # Update status
        client.is_active = new_status
        client.save()
        
        status_text = "activated" if new_status else "deactivated"
        message = f"Client {client.get_full_name() or client.username} has been {status_text}."
        
        return JsonResponse({
            'success': True,
            'message': message,
            'is_active': client.is_active
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error updating client status: {str(e)}'
        }, status=400)


@login_required
def add_client_view(request):
    """View to add a new client."""
    if request.method == 'POST':
        # Handle form submission
        # You'll need to create a form for this
        pass
    
    return render(request, 'client_management/add_client.html')


@login_required
def edit_client_view(request, client_id):
    """View to edit an existing client."""
    client = get_object_or_404(User, id=client_id, user_type='client')
    
    if request.method == 'POST':
        # Handle form submission
        # You'll need to create a form for this
        pass
    
    context = {'client': client}
    return render(request, 'client_management/edit_client.html', context)


@login_required
def delete_client_view(request, client_id):
    """View to delete a client."""
    client = get_object_or_404(User, id=client_id, user_type='client')
    
    if request.method == 'POST':
        client_name = client.get_full_name() or client.username
        client.delete()
        messages.success(request, f'Client {client_name} has been deleted successfully.')
        return redirect('client_management:client_list')
    
    context = {'client': client}
    return render(request, 'client_management/delete_client.html', context)


# Class-based views for CRUD operations
class ClientCreateView(LoginRequiredMixin, CreateView):
    """Create a new client."""
    model = User
    template_name = 'client_management/add_client.html'
    fields = ['username', 'email', 'first_name', 'last_name', 'phone']
    success_url = reverse_lazy('client_management:client_list')
    
    def form_valid(self, form):
        form.instance.user_type = 'client'
        messages.success(self.request, 'Client added successfully!')
        return super().form_valid(form)


class ClientUpdateView(LoginRequiredMixin, UpdateView):
    """Update an existing client."""
    model = User
    template_name = 'client_management/edit_client.html'
    fields = ['username', 'email', 'first_name', 'last_name', 'phone', 'is_active']
    success_url = reverse_lazy('client_management:client_list')
    
    def get_queryset(self):
        return User.objects.filter(user_type='client')
    
    def form_valid(self, form):
        messages.success(self.request, 'Client updated successfully!')
        return super().form_valid(form)


class ClientDeleteView(LoginRequiredMixin, DeleteView):
    """Delete a client."""
    model = User
    template_name = 'client_management/delete_client.html'
    success_url = reverse_lazy('client_management:client_list')
    context_object_name = 'client'
    
    def get_queryset(self):
        return User.objects.filter(user_type='client')
    
    def delete(self, request, *args, **kwargs):
        client = self.get_object()
        client_name = client.get_full_name() or client.username
        response = super().delete(request, *args, **kwargs)
        messages.success(request, f'Client {client_name} has been deleted successfully.')
        return response


# Additional utility views
@login_required
def client_projects_view(request, client_id):
    """View client's assigned projects."""
    client = get_object_or_404(User, id=client_id, user_type='client')
    project_roles = client.project_roles.select_related('project').all()
    
    context = {
        'client': client,
        'project_roles': project_roles,
    }
    
    return render(request, 'client_management/client_projects.html', context)


@login_required
def assign_project_to_client(request, client_id):
    """Assign a project to a client."""
    client = get_object_or_404(User, id=client_id, user_type='client')
    
    if request.method == 'POST':
        # Handle project assignment
        # You'll need to implement this based on your ProjectRole model
        pass
    
    # Get available projects for assignment
    # available_projects = Project.objects.exclude(projectrole__user=client)
    
    context = {
        'client': client,
        # 'available_projects': available_projects,
    }
    
    return render(request, 'client_management/assign_project.html', context)