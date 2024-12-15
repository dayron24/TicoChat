from django.urls import path
from .views import recommendation

urlpatterns = [
    path('recommendation/', recommendation, name='recommendation'),
]
