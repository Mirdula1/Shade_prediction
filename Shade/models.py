from django.db import models
from django.db.models.signals import post_save

class UserProfile(models.Model):
    username = models.CharField(primary_key=True, max_length=10,unique=True)
    firstname = models.CharField(max_length= 10)
    lastname = models.CharField(max_length= 10)
    address = models.CharField(max_length = 100)
    designation = models.CharField(max_length = 10)
    phone = models.CharField(max_length =  10,unique=True)
    email = models.CharField(max_length = 20,unique=True)
    pwd1 = models.CharField(max_length = 10)
    pwd2 = models.CharField(max_length = 10)
    DOB = models.DateField()
    #image = models.ImageField(upload_to='profile_image', blank=True)

    def __str__(self):
        return self.username
#user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='shade_predictions')


    
class ShadePrediction(models.Model):
    
    li = models.FloatField()
    ai = models.FloatField()
    bi = models.FloatField()
    Concentration =models.FloatField()
    pH = models.FloatField()
    Temp = models.IntegerField()
    WaterBathRatio = models.FloatField()
    DyeingMethod = models.TextField()
    Duration = models.IntegerField()
    Substrate = models.TextField()
    Thread = models.TextField()
    Thickness = models.FloatField()
    thread_group = models.TextField()
    Abs_coeff =  models.FloatField()
    Lf = models.FloatField()
    Af = models.FloatField()
    Bf = models.FloatField()
    delta_e = models.FloatField()
    
#user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='dye_quality_predictions')

class DyeQualityPrediction(models.Model):
    
    Lf = models.FloatField()
    Af = models.FloatField()
    Bf = models.FloatField()
    pH = models.FloatField()
    Temp = models.IntegerField()
    Substrate = models.TextField()
    Thread = models.TextField()
    Thickness =models.FloatField()  
    thread_group = models.TextField()
    D_Duration = models.IntegerField()
    Fastness_Type = models.TextField()
    Washings = models.IntegerField()
    Chemical = models.TextField()
    Chemical_Conc = models.FloatField()
    Lubricant = models.TextField()
    CS =  models.IntegerField()

    
   

    