import cv2
import os
import ntpath
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL
import torch.utils.data as data
import TreeAtlas.settings as st
classes_dict = {'Acer': 0, 'Actindia': 1, 'Ailanthus': 2, 'Alnus': 3, 'Amorpha': 4, 'Berberis': 5, 'Betula': 6, 'Buddleja': 7, 'Buxus': 8, 'Caragana': 9, 'Carpinus': 10, 'Carya': 11, 'Castanea': 12, 'Catalpa': 13, 'Ceeltis': 14, 'Cercidiphyllum': 15, 'Clemantis': 16, 'Colutea': 17, 'Cornus': 18, 'Juglans': 19, 'Kerria': 20, 'Koelreuteria': 21, 'Kolkwitzia': 22, 'Laburnunum': 23, 'Lavandula': 24, 'Ledum': 25, 'Ligustrum': 26, 'Liliodendron': 27, 'Liquidambar': 28, 'Lonicera': 29, 'Loranthus': 30, 'Lycium': 31, 'Maclura': 32, 'Mahonia': 33, 'Mespilus': 34, 'Morus': 35, 'Nerium': 36, 'Parthenocissus': 37, 'Paulownia': 38, 'Phellodendron': 39, 'Philadelphus': 40, 'Physocarpus': 41, 'Platanus': 42, 'Populus': 43, 'Prunus': 44, 'Pterocarya': 45, 'Pyracantha': 46, 'Quercus': 47, 'Rhamus': 48, 'Rhus': 49, 'Ribes': 50, 'Robinia': 51, 'Salix': 52, 'Sambucus': 53, 'Sorbaria': 54, 'Sorbus': 55, 'Sorphora': 56, 'Spiraela': 57, 'Staphylea': 58, 'Syhorimcarpos': 59, 'Syringa': 60, 'Wisteria': 61, 'Zelkova': 62}
classes = list(classes_dict.keys()) 

loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])



def image_loader(image_name):
    image = PIL.Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image 

def recognize_leaf(path):
    model = torch.load(os.path.join(st.STATICFILES_DIRS[0], "entire_model.pt"), map_location='cpu')
    model.eval()
    print(st.MEDIA_ROOT)
    print( path.replace('/', '\\'))
    image = image_loader(str(st.BASE_DIR) + '\\media\\'+ path.replace('/', '\\'))
    output = model.forward(image)
    output = torch.exp(output)
    probs, class_id = output.topk(1, dim=1)
    return classes[class_id.item()], probs.item() * 100