from django.urls import path

from stocks import views

urlpatterns = [
    path("stocks/", views.get_stock_data, name="get_stock_data"),
]