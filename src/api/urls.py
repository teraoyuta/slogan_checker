from django.urls import path
from . import views

urlpatterns = [
    # path('hello/', SloganAPIView.as_view(), name='helloworld'),
    path('slogan/', views.check_slogan, name='check_slogan')
]
