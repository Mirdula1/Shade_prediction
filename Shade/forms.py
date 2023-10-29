from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm

class ImageUploadForm(forms.Form):
    captured_image = forms.ImageField(required=False)
    
    
    