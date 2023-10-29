from django.urls import path
from . import views

urlpatterns = [
    path('',views.SignupPage,name='signup'),
    path('signup',views.SignupPage,name='signup'),
    path('login',views.LoginPage,name='login'),
    path('index',views.index, name = 'index'),
    path('logout',views.LogoutPage,name='logout'),
    path('predictor', views.predictor, name = 'predictor'),
    path('result',views.formInfo, name = 'result'),
    path('CS', views.strength, name = 'CS'),
    path('cs_result', views.Color_str, name = 'cs_result'),
]



import os
from django.conf.urls.static import static
from django.conf import settings

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

