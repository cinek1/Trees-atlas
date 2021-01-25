from django.db import models

class Leaf(models.Model): 
    id_user = models.IntegerField()
    name = models.CharField(max_length=50, default="") 
    leaf_image_url = models.ImageField(upload_to='images/') 
    analyze = models.BooleanField(default=False) 
    prediction = models.IntegerField(default=0)
    url = models.CharField(max_length=100, default="")


class Tree(models.Model):
   name = models.CharField(max_length=50, default="") 
   url = models.CharField(max_length=100) 
   leaf_image_url = models.ImageField(upload_to='siteImages/')
   tree_image_url = models.ImageField(upload_to='siteImages/')