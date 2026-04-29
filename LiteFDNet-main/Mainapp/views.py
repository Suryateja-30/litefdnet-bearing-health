from django.shortcuts import render,redirect
import urllib.request
import urllib.parse
from Mainapp.models import *
from Userapp.models import *
from Adminapp.models import *
import random 
import ssl
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings


def index(req):
    return render(req,'main/index.html')
    
def contact(req):
    if req.method == 'POST':
        # Handle form submission here
        full_name = req.POST.get('Name')
        phone_number = req.POST.get('PhoneNumber')
        email = req.POST.get('Email')
        message = req.POST.get('message')

        #Print values for debugging
        print("Full Name:", full_name)
        print("Phone Number:", phone_number)
        print("Email:", email)
        print("Message:", message)

        # For demonstration purposes, let's just display a success message
        Contact.objects.create(Full_name = full_name, Phone_Number = phone_number, Email=email, Message=message)
        print("data inserted ")
        messages.success(req, 'Your message has been submitted successfully.')
        return redirect('contact') 
    return render(req,'main/contact.html')

def UserRegister(req):
    if req.method == 'POST' :
        name = req.POST.get('myName')
        age = req.POST.get('myAge')
        password = req.POST.get('myPwd')
        phone = req.POST.get('myPhone')
        email = req.POST.get('myEmail')
        address = req.POST.get("address")
        image = req.FILES['image']
        print(name,age,password,phone,email,address,image)
        image = req.FILES['image']
        number = random.randint(1000,9999)
        
        print(number)
        # try:
        #     user_data = User.objects.get(Email = email)
        #     if user_data!=None:
        #         messages.warning(req, 'Email was already registered, choose another email..!')
        #         return redirect("register")
        # except:
            # sendSMS(name,number,phone)
        User.objects.create(Full_name = name, Image = image, Age = age, Password = password, Address = address, Email = email, Phone_Number = phone)
        mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {number}'
        print(mail_message)
        # send_mail("User Password", mail_message , settings.EMAIL_HOST_USER, [email])
        req.session['Email'] = email
        messages.success(req, 'Your account was created..')
            # return redirect('otpverify')
    return render(req,'main/register.html')
            
   
def AdminLogin(req):
    admin_name = 'admin'
    admin_pwd = 'admin'
    if req.method == 'POST':
        admin_n = req.POST.get('adminName')
        admin_p = req.POST.get('adminPwd')
        if (admin_n == admin_name and admin_p == admin_pwd):
            messages.success(req, 'You are logged in..')
            return redirect('admindashboard')
        else:
            messages.error(req, 'You are trying to loging with wrong details..')
            return redirect('admin')
    return render(req,'main/admin.html')
    


def UserLogin(req):
    if req.method == 'POST':
        Email = req.POST.get('email')
        Password = req.POST.get('password')
        print( Email,Password)
        
        user_data = User.objects.get(Email = Email)
        print(user_data)
        if user_data.Password == Password:
            # if user_data.User_Status=='accepted':
                req.session['User_id'] = user_data.User_id
                messages.success(req, 'You are logged in..')
                user_data.No_Of_Times_Login += 1
                user_data.save()
                return redirect('userdashboard')
                        
        else:
            # messages.warning(req, 'verifyOTP...!')
            req.session['Email'] = user_data.Email
            # return redirect('otpverify')
    else: 
        messages.error(req, 'incorrect credentials...!')
        return render(req,'main/user.html')
        
    return render(req,'main/user.html')
    
def about(req):
    return render(req,'main/about.html')
