from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'

        
class RF(models.Model):
    name = models.CharField(max_length=200)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    def __str__(self):
        return self.name  
    
    class Meta:
        db_table = 'RF'

class PlainNet(models.Model):
    name = models.CharField(max_length=200)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    def __str__(self):
        return self.name   

    class Meta:
        db_table = 'PlainNet'  


class LiteFDNet(models.Model):
    name = models.CharField(max_length=200)  # Name of the model (e.g., Ensemble Model (GB + AdaBoost))
    accuracy = models.FloatField()  # Accuracy score of the model
    precision = models.FloatField()  # Precision score of the model
    recall = models.FloatField()  # Recall score of the model
    f1_score = models.FloatField()  # F1 score of the model

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'LiteFDNet'




