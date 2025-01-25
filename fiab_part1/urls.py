# fiab_part1/urls.py
# fiab_part1/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Render the front page
    path('compute_reliability/', views.compute_reliability, name='compute_reliability'),  # Handle AJAX requests
]