from django.shortcuts import render, redirect
from .forms import *
from django.http import HttpResponse 
from .models import Tree
from django.contrib.auth import login
from django.shortcuts import redirect, render
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from .utils.wikipedia import get_wikipedia_content
from .utils.recognizeLeaf import recognize_leaf
from .model_repo import *

def index(request):
    return render(request, "index.html")

@login_required()
def display_images(request): 
    current_user = request.user.id 
    leafs = Leaf.objects.filter(id_user=current_user) 
    form = LeafForm() 
    return render(request, 'user-trees.html', 
                {'Leafs' : leafs, "form" : form})

@login_required()
def add_picture(request): 
    current_user = request.user
    if request.method == 'POST': 
        form = LeafForm(request.POST, request.FILES) 
        if form.is_valid():             
            stock = form.save(commit=False)
            stock.id_user = current_user.id
            stock.save()
        leafs = Leaf.objects.filter(id_user=current_user.id) 
        form = LeafForm() 
        return render(request, 'user-trees.html', 
            {'Leafs' : leafs, "form" : form})

def info_tree(request):
    trees = Tree.objects.all()
    return render(request, 'avaible-trees.html', {'Trees' : trees})

def tree_summary(request):
    id = request.POST.get('mytextbox')
    tree =  Tree.objects.all().filter(name=id) 
    content = get_wikipedia_content(tree[0].url)
    return render(request, 'summary-of-tree.html', {'content': content, 'tree': tree[0]})

def register(request):
    if request.method == "GET":
        return render(
            request, "users/register.html",
            {"form": CustomUserCreationForm}
        )
    elif request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("index")
        

@login_required()
def analyze_picture(request):
    url = str(request.POST.get("url")) 
    name, pred = recognize_leaf(url)
    leaf = Leaf.objects.filter(leaf_image_url=url).first()
    leaf.name = name
    leaf.prediction = int(pred)
    leaf.analyze = True
    leaf.save()
    tree = Tree.objects.filter(name=name) 
    content = get_wikipedia_content(tree[0].url)
    return render(request, 'summary-of-tree.html', {'content': content, 'tree': tree[0]})
    
@login_required()
def delete_picture(request):
    url = str(request.POST.get("url")) 
    Leaf.objects.filter(leaf_image_url=url).delete()
    return display_images(request)