from django import forms 
from .models import *
from django.contrib.auth.forms import UserCreationForm
  
class LeafForm(forms.ModelForm): 
  
    class Meta: 
        model = Leaf 
        fields = ['leaf_image_url'] 


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        fields = UserCreationForm.Meta.fields + ("email",)