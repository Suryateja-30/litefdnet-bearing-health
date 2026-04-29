"""Bearing_Fault_Diagnosis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from Mainapp import views as main_views
from Userapp import views as user_views
from Adminapp import views as admin_views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Main
    path('',main_views.index, name ='index'),
    path('contact',main_views.contact, name ='contact'),
    path('register',main_views.UserRegister, name ='register'),
    path('admin',main_views.AdminLogin, name ='admin'),
    path('login',main_views.UserLogin, name ='login'),
    path('about', main_views.about, name = 'about' ),

     # User
    path('userdashboard',user_views.userdashboard,name='userdashboard'),
    path('userlogout', user_views.userlogout, name = 'userlogout'),
    path('predict_drift/',user_views.LiteFDNet_Predict_Form_btn,name='predict_drift'),


      #admin
    path('admin-dashboard', admin_views.admindashboard, name = 'admindashboard'),
    path('admin-graph',admin_views.admingraph, name='admingraph'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),



    path('PlainNet_btn/',admin_views.PlainNet_btn,name='PlainNet_btn'),
   
    path('RF_btn/',admin_views.RF_btn,name='RF_btn'),

    path('LiteFDNet_btn/',admin_views.LiteFDNet_btn,name='LiteFDNet_btn'),
 

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
