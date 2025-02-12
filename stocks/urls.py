from django.urls import path
from .views import stock_search

urlpatterns = [
    path('', stock_search, name='stock_search'),
]