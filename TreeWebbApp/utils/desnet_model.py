import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import PIL 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


#Define global variables
BATCH_SIZE = 36
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "./RecognitionLeafe/train/"
TEST_DATA_PATH = "./RecognitionLeafe/test/"
LABELS = 63
EPOCHS = 25 
#Transform functions
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

TRANSFORM_IMG_AUG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip()
    ])
#Loader to data
train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_aug = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG_AUG)
train_data = train_data + train_data_aug
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
#Load pretrained model densenet161
model = models.densenet161(pretrained=True)
#Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False


#Get input size to linear layers
classifier_input = model.classifier.in_features

#Craete classffier for linear layer
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, LABELS),
                           nn.LogSoftmax(dim=1))

#Replace  classifier
model.classifier = classifier


#Train loop
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    acc_epoch = []
    loss_epoch = [] 
    for epoch in range(EPOCHS):
        acc = 0
        los = 0 
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            los = los + loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            acc = acc + (correct / total)
            
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item()}, Accuracy: { (correct / total) * 100}%')
        acc_epoch.append(acc/len(train_loader))
        loss_epoch.append(los/len(train_loader))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy of the model on the {len(test_data)} test images: {(correct / total) * 100}%')


    torch.save(model, "entire_model.pt")

    x = loss_list
    y = [i/len(train_loader) for i in range(len(acc_list))] 
    plt.plot(y, x)
    plt.title('Loss') 
    plt.savefig('loss.pdf')
    plt.clf() 

    x = acc_list
    y = [i/len(train_loader) for i in range(len(acc_list))] 
    plt.plot(y, x)
    plt.title('Acc') 
    plt.savefig('acc.pdf')
    plt.clf() 

    x = acc_epoch
    y = [i + 1 for i in range(EPOCHS)] 
    plt.plot(y, x, marker='o', markerfacecolor='blue', markersize=12)
    plt.title('Acc epoch') 
    plt.savefig('accEpoch.pdf')
    plt.clf() 

    x = loss_epoch
    y = [i + 1 for i in range(EPOCHS)] 
    plt.plot(y, x)
    plt.title('Loss epoch') 
    plt.savefig('LossEpoch.pdf')
