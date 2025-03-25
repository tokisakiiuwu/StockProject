from django.urls import path
from .views import stock_search, portfolio, optimize_portfolio  # Add the new view for optimization

urlpatterns = [
    path('', stock_search, name='stock_search'),
    path('portfolio/', portfolio, name='portfolio'),
    path('optimize/', optimize_portfolio, name='optimize_portfolio'),  # Add the new URL for optimization
]
