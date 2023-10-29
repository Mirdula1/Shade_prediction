from PIL import Image, ImageCms
from joblib import load
from colormath.color_objects import LabColor
from utils.color_utils import create_color_image, delta_e_cie1976, color, create_abs_coeff, lab_to_rgb
from .forms import *
import cv2
from django.core.files.storage import FileSystemStorage
from skimage.io import imread
from skimage.color import rgb2lab
import numpy as np
import colour
from django.conf import settings
import os
import colorspacious as cs
from .models import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.shortcuts import render,redirect
from django.contrib import messages
from Shade_prediction.settings import BASE_DIR

def SignupPage(request):
    print("Register Page")
    if request.method == "POST":
        username = request.POST['username']
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']
        address = request.POST['address']
        designation = request.POST['designation']
        phone = request.POST['phone']
        email = request.POST['email']
        pwd1 = request.POST['pwd1']
        pwd2 = request.POST['pwd2']
        DOB = request.POST['DOB']
        
        if pwd1 == pwd2:
            print("Pwd same")            
            user_profile = UserProfile(username=username,  firstname =firstname, lastname=lastname,address=address,designation=designation, phone=phone,email=email, pwd1=pwd1, pwd2=pwd2, DOB=DOB)
            user_profile.save()
            messages.success(request, "Successfully registered")
            context = {
                    'user': user_profile
                }
            return render(request, 'home.html',context)
        else:
            print("pwd not same")
            messages.error(request, "Password mismatch")

    return render(request, 'signup.html')

    
def LoginPage(request):
    print("Login Request")
    if request.method == "POST":
        username = request.POST['username']
        pwd1 = request.POST['pwd1']
        
        try:
            user_profile = UserProfile.objects.get(username=username,pwd1=pwd1)
        except UserProfile.DoesNotExist:
            user_profile = None
            messages.error(request, "User not found")
            return render(request, 'login.html')
        
        if user_profile and user_profile.pwd1 == pwd1:
            messages.success(request, "Login successful")
            context = {
                'user': user_profile,
            }
            return render(request, 'home.html',context)    
        else:
            messages.error(request, "Invalid username or password")
            return redirect('login.html')

    
    return render(request, 'login.html')


def LogoutPage(request):
    logout(request)
    print("User Logged out")
    return redirect('signup')

DyeingMethod_encoder = {
    "Directly dyed without mordant": 0,
    "POM with CuSo4": 1,
    "SM with CuSo4": 2
}

Chemical_encoder = {
    "Directly dyed without mordant": 0,
    "Post-Mordanting with copper sulphate": 1,
    "Simultaneous Mordanting with Copper Sulphate": 2
}


Substrate_encoder = {
    "Cotton" :0,
    "Linen" : 1
}

thread_group_encoder = {
"TRP": 40,
"CFP-MEDIUM" : 7,
"CFP-COR-A" : 5,
"BRP_FINE" : 4,
"BRP_COARSE" : 3,
"CFP-COR-D" : 6, 
"DURA_MICRO" : 13, 
"GRAL" : 17,
"IBN_MEDIUM" : 23,
"IBN_CORSER" : 22,
"PPC-TJNM-F" : 28,
"PPC ZHON FINE" : 27,
"TXP_M_THAIMAN" : 44,
"RPC_FINE" : 32,
"IBN FINE" : 19,
"nan" : 45,
"IBN MEDIUM" : 20,
"TXN_TAHILON" : 41,
"MFP": 24, 
"SSA FINE": 36, 
"FENGSH-F": 14, 
"SSP_V_FINE": 39, 
"SSA COR-B": 35, 
"SSP-FENGSHU-VF": 38,  
"TXN_TAHILON_M": 42,
"PBT": 25,
"CFP_FINE_B":9,
"CFP_940X3_UNIFULL": 8, 
"CFP_HYOSUNG_M": 11, 
"DURA MICRO FINE": 12,
"FENGSHU-C": 15,
"312X3_COARSE": 1,
"IBN COARSER": 18,
"FENGSHU-F": 16,
"SSA_FS_FINE": 37, 
"PPC-XPQ-F": 29, 
"152X2_ZONG_MICRO": 0, 
"IBN-G3-HY": 21,
"RPC_FINE_87X2": 33,
"TXP FINE G": 43,
"PPC_FENGSHU": 30,
"PPC 2053 T": 26,
"365X2_PPC_ZHONG": 2,
"SSA COR-A" : 34,
 "PPC_HYOSUNG": 31,
 "CFP_HYJ_FINE": 10
}

Thread_encoder = {
    "Single Fibre" : 1,
    "Coarse" : 0
}

Fastness_encoder = {
"HIGH WASH FAST" : 2,
"CORE DYED" : 0,
"HIGH LIGHT FAST" : 1,
"MULTI-DYE" : 3,
"NON-PREMIUM" : 4,
"NORMAL" : 5
}

Lubricant_encoder = { 
             "L1" : 0,
             "L2" : 1,
             "L3" : 2
}

ransac = load('C:/Users/HP/Pro/savedModels/RANSAC1.joblib')

dtmodel = load('C:/Users/HP/Pro/savedModels/DT.joblib')


@login_required(login_url='login')
def index(request):
    return render(request, 'home.html')

def predictor(request):
    return render(request, 'main.html')

def strength(request):
    return render(request, 'CS.html')
   
def formInfo(request):
    ran_prediction = None
    delta_e = None
    image1 = None
    image2 = None
    
    if request.method == 'POST':
        if 'Li' in request.POST and 'Ai' in request.POST and 'Bi' in request.POST:
            # Form for entering L, A, and B values
            Li = float(request.POST['Li'])
            Ai = float(request.POST['Ai'])
            Bi = float(request.POST['Bi'])
            R,G,B = lab_to_rgb(Li, Ai, Bi)
            Concentration = float(request.POST.get('Concentration'))
            pH = float(request.POST.get('pH'))
            Temp = int(request.POST.get('Temp'))
            WaterBathRatio = float(request.POST.get('WaterBathRatio'))
            DyeingMethod = request.POST.get('DyeingMethod')
            Duration = int(request.POST.get('Duration'))
            #Substrate = request.POST.get('Substrate')
            Thread = request.POST.get('Thread')
            Thickness = float(request.POST.get('Thickness'))
            thread_group = request.POST.get('thread_group')
            Abs_coeff = create_abs_coeff(Thickness, R,G,B)
            
            # Substrate_encoder.get(Substrate),
            #Substrate_encoder.get(Substrate, 0),Thread_encoder.get(Thread),
            ran_prediction = ransac.predict([[Li, Ai, Bi, Concentration, pH, Temp, WaterBathRatio,DyeingMethod_encoder.get(DyeingMethod),Duration
                           ,Thread_encoder.get(Thread), Thickness, thread_group_encoder.get(thread_group),Abs_coeff]])
            
            ran_prediction = ran_prediction.tolist()
            
            Lf, Af, Bf = ran_prediction[0]
            
            r,g,b = lab_to_rgb(Li, Ai, Bi)
           
            rf,gf,bf = lab_to_rgb(Lf,Af,Bf)
            
            
            print("Input Shade:")
            print("RGB:", r,g,b)
            print("Li:", Li)
            print("Ai:", Ai)
            print("Bi:", Bi)
            
            print("Final Shade:")
            print("RGB:", rf,gf,bf)
            print("Lf:", Lf)
            print("Af:", Af)
            print("Bf:", Bf)
            
            request.session['ran_prediction'] = ran_prediction
            
            color2 = LabColor(Lf, Af, Bf)
            color1 = LabColor(Li, Ai, Bi)
        
            delta_e = delta_e_cie1976(color1, color2)

            #Substrate=Substrate,
            shade_prediction = ShadePrediction(li=Li, ai=Ai, bi=Bi, Concentration=Concentration, pH=pH,Temp=Temp, WaterBathRatio = WaterBathRatio , DyeingMethod= DyeingMethod,Duration=Duration,
                             Thread=Thread,Thickness=Thickness, thread_group=thread_group,Abs_coeff =Abs_coeff, Lf=Lf,Af=Af,Bf=Bf,delta_e = delta_e)
                             
            shade_prediction.save()
            
            #hex_code2 = lab_to_hex(Lf, Af, Bf)
            #hex_code1 = lab_to_hex(Li, Ai, Bi)

            image1_path = 'Shade/static/images/color_images1.jpg'
            image2_path = 'Shade/static/images/color_images2.jpg'
            image1 = create_color_image(int(r),int(g),int(b), image1_path)
            #image1 = create_color_image(hex_code1, image1_path)
            image2 = create_color_image(int(rf),int(gf),int(bf), image2_path)
            #image2 = create_color_image(hex_code2, image2_path)
            return render(request, 'result.html', {'result': ran_prediction, 'delta_e': delta_e, 'image1': image1, 'image2': image2,'Li': Li, 'Ai': Ai, 'Bi': Bi,'Lf': Lf, 'Af': Af, 'Bf': Bf})
         
           
        else:
            # Form for uploading an image
            form = ImageUploadForm(request.POST, request.FILES)

            if form.is_valid():
                uploaded_image = request.FILES['image']
                fs = FileSystemStorage()
                dynamic_filename = uploaded_image.name
                image_path = os.path.join(settings.MEDIA_ROOT, dynamic_filename)
                fs.save(image_path, uploaded_image)

                rgba_image = imread(image_path)
                
                rgb_image = rgba_image[:, :, :3]

                lab_image = rgb2lab(rgb_image)

                L, A, B = lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]

                Li = np.mean(L)
                Ai = np.mean(A)
                Bi = np.mean(B)
                #hex_code1 = lab_to_hex(Li, Ai, Bi)
                
                r,g,b = lab_to_rgb(Li,Ai,Bi)
                
                #hex_code1 = "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
                #print("Hex code1:", hex_code1)
              
                print("Input Shade LAB Values:")
                print("RGB:", r,g,b)
                print("Li:", Li)
                print("Ai:", Ai)
                print("Bi:", Bi)

                R,G,B = lab_to_rgb(Li, Ai, Bi)
                Concentration = float(request.POST.get('Concentration'))
                pH = float(request.POST.get('pH'))
                Temp = int(request.POST.get('Temp'))
                WaterBathRatio = float(request.POST.get('WaterBathRatio'))
                DyeingMethod = request.POST.get('DyeingMethod')
                Duration = int(request.POST.get('Duration'))
                #Substrate = request.POST.get('Substrate')
                Thread = request.POST.get('Thread')
                Thickness = float(request.POST.get('Thickness'))
                thread_group = request.POST.get('thread_group')
                Abs_coeff = create_abs_coeff(Thickness, R,G,B)
                #Substrate_encoder.get(Substrate,0),   Substrate_encoder.get(Substrate),
                ran_prediction = ransac.predict([[Li, Ai, Bi, Concentration, pH, Temp, WaterBathRatio,DyeingMethod_encoder.get(DyeingMethod),Duration, 
                                                  Thread_encoder.get(Thread),Thickness, thread_group_encoder.get(thread_group),Abs_coeff]])
                
                ran_prediction = ran_prediction.tolist()
                
                Lf, Af, Bf = ran_prediction[0]
                
                rf,gf,bf = lab_to_rgb(Lf,Af,Bf)
                
                #hex_code2 = "#{:02X}{:02X}{:02X}".format(int(rf), int(gf), int(bf))
                #print("Hex code2:", hex_code2)
                
                print("Final Shade LAB Values:")
                print("RGB:", rf,gf,bf)
                print("Lf:", Lf)
                print("Af:", Af)
                print("Bf:", Bf)
                
                #hex_code2 = lab_to_hex(Lf, Af, Bf)
                print("Final Shade",ran_prediction)
                
                #Substrate=Substrate,
                color2 = LabColor(Lf, Af, Bf)
                color1 = LabColor(Li, Ai, Bi)
        
                delta_e = delta_e_cie1976(color1, color2)
                
                #Substrate=Substrate,
                shade_prediction = ShadePrediction(li=Li, ai=Ai, bi=Bi, Concentration=Concentration, pH=pH,Temp=Temp, WaterBathRatio = WaterBathRatio , DyeingMethod= DyeingMethod,Duration=Duration, 
                             Thread=Thread,Thickness=Thickness, thread_group=thread_group,Abs_coeff =Abs_coeff, Lf=Lf,Af=Af,Bf=Bf,delta_e = delta_e)
                shade_prediction.save()
                
                print(image_path)
                image1_path = 'Shade/static/images/color_images1.jpg'
                image2_path = 'Shade/static/images/color_images2.jpg'
                image1 = create_color_image(int(r),int(g),int(b), image1_path)
                #image1 = create_color_image(hex_code1, image1_path)
                image2 = create_color_image(int(rf),int(gf),int(bf), image2_path)
                #return redirect('cs_result', Lf=Lf, Af=Af, Bf=Bf)
                
                return render(request, 'result.html', {'result': ran_prediction, 'delta_e': delta_e, 'image1': image1, 'image2': image2,'Li': Li, 'Ai': Ai, 'Bi': Bi,'Lf':Lf,'Af':Af,'Bf':Bf})
         
    else:
        form = ImageUploadForm()
 
    return render(request, 'main.html', {'form': form})
    
    
def Color_str(request):
    if request.method == 'POST':
        pH = float(request.POST.get('pH'))
        Temp = int(request.POST.get('Temp'))
        #Substrate = request.POST.get('Substrate ')
        thread_group = request.POST.get('thread_group')
        Thread = request.POST.get('Thread')
        Thickness = float(request.POST.get('Thickness'))
        Chemical = request.POST.get('Chemical')
        Chemical_Conc = float(request.POST.get('Chemical_Conc'))
        D_Duration = int(request.POST.get('D_Duration'))
        Fastness_Type = request.POST.get('Fastness_Type')
        Washings = int(request.POST.get('Washings'))
        Lubricant =  request.POST.get('Lubricant')
        image2_path = 'Shade/static/images/color_images2.jpg'
        rgba_image = imread(image2_path)
        rgb_image = rgba_image[:, :, :3]
        lab_image = rgb2lab(rgb_image)
        L, A, B = lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]
        Lf = np.mean(L)
        Af = np.mean(A)
        Bf = np.mean(B)
        #username = request.POST.get('username')
        #print("session",Lf)
        #Substrate_encoder.get(Substrate),
        S_pred = dtmodel.predict([[float(Lf), float(Af), float(Bf), float(pH), int(Temp), Thread_encoder.get(Thread),thread_group_encoder.get(thread_group), float(Thickness),
                              int(D_Duration),Fastness_encoder.get(Fastness_Type), int(Washings),Chemical_encoder.get(Chemical),float(Chemical_Conc),Lubricant_encoder.get(Lubricant)]])
         
        cs = S_pred[0]
        CS = max(0, min(cs * 100, 100))

        #user_profile ,Substrate=Substrate
        dye_quality_prediction = DyeQualityPrediction(Lf=Lf, Af=Af, Bf=Bf, pH=pH, Temp=Temp, Thickness=Thickness, Thread=Thread, thread_group=thread_group,
            D_Duration=D_Duration, Fastness_Type=Fastness_Type, Washings=Washings, Chemical=Chemical,
            Chemical_Conc=Chemical_Conc, Lubricant=Lubricant, CS = CS)
        dye_quality_prediction.save()
        
        return render(request, 'cs_result.html', {'CS': CS })

    return render(request, 'CS.html')



        
    
    