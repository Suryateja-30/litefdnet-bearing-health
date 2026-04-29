
# Create your models here.
from django.db import models
from Mainapp.models import *


        
class Dataset(models.Model):
   Data_id = models.AutoField(primary_key=True)
   Image = models.ImageField(upload_to='media/') 
   class Meta:
        db_table = "upload" 


        


