from django.contrib import admin
from django.urls import path, include
from django.conf.urls import  url
from .views import *


from .views import *
urlpatterns = [
   path('', index, name='index'),
   path('index', index, name='index'),
   path('add_picture', add_picture, name = 'add_picture'), 
   path('info_tree', info_tree, name = 'info_tree'),
   path('your_trees', display_images, name = 'your_trees'),
   path('tree_summary', tree_summary, name = 'tree_summary'),
   path('analyze_picture', analyze_picture, name = 'analyze_picture'),
   path('delete_picture', delete_picture, name = 'delete_picture'),
   url(r"^accounts/", include("django.contrib.auth.urls")),
   url(r"^register/", register, name="register"),
]