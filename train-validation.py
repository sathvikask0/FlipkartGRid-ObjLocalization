import numpy as np

import torch.nn as nn
import torch, os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models 
import torch.optim as optim
import torch.backends.cudnn as cudnn
 
from tqdm import tqdm
   
import csv
import os
from PIL import Image

L = []      
with open('training.csv') as FILE:
    rows = csv.reader(FILE, delimiter=',')
    start = 0 
    Xc = float(224/640)
    Yc = float(224/480)
    for row in rows:
        if( start == 0 ):
            continue
        start = 1
        l = [Xc*float(row[1]),Xc*float(row[2]),Yc*float(row[3]),Yc*float(row[4])]
        L.append(l)
                     
DATA = np.array(L)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,transform=None):
        self.data = DATA
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = Image.open(os.path.join('data/{}.png'.format(idx)))
        if self.transform:
            sample = self.transform(img)
        return (sample,self.data[idx])
        
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
Transform = transforms.Compose([transforms.ToTensor(),normalize])
data = CustomDataset(transform = Transform)
trainset , validationset = torch.utils.data.random_split(data,(10000,4000))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(validationset, batch_size=32,shuffle=True, num_workers=4)


def train(model,epoch, trainloader, optimizer, LossFun ):
    losses = 0.0
    model.train()
    for data in tqdm(trainloader,desc='Train epoc: %d'%(epoch+1)):
        context , target = data
        target = target.float()
        if torch.cuda.is_available():
            context, target = context.cuda(), target.cuda()
        # zeroing the gradients of parameters
        optimizer.zero_grad()
        outputs = model(context)
        loss = LossFun(outputs,target)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses/len(trainloader)

def test(model,testloader,LossFun):

    running_loss = 0.0
    iou = 0.0
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            context , target = data
            target = target.float()
            if torch.cuda.is_available():
                context, target = context.cuda(), target.cuda()

            outputs = model(context)
            loss = LossFun(outputs, target)
            running_loss += loss.item()
            iou += IOU(target,outputs)
    return (iou / (len(testloader)),running_loss / (len(testloader)))

def IOU(labels,outputs):
    x1 = torch.max(labels[:,0],outputs[:,0])
    x2 = torch.min(labels[:,1],outputs[:,1])
    y1 = torch.max(labels[:,2],outputs[:,2])
    y2 = torch.min(labels[:,3],outputs[:,3])
    a = torch.abs((x2-x1)*(y2-y1))
    u = torch.abs(labels[:,1]-labels[:,0])*torch.abs(labels[:,3]-labels[:,2])+torch.abs(outputs[:,1]-outputs[:,0])*torch.abs(outputs[:,3]-outputs[:,2])
    iou = a/(u-a)
    return torch.sum(iou)/32


model = models.resnet50()


inFeatures = model.fc.in_features
model.fc = nn.Linear(inFeatures,4)

for param in model.parameters():
    param.requires_grad = True

model.load_state_dict(torch.load('./modelsc/model-55.pth'))

         
learningRate = 0.001
epochs = 40
Lambda = 0.0005
step_size = 90
LossFun = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    cudnn.benchmark = True
    LossFun = nn.MSELoss().cuda()
    
optimizer  = optim.Adam(model.parameters(),weight_decay = Lambda,lr = learningRate)




Losses = [] 
for epoch in range(epochs):
    l = train(model,epoch,trainloader,optimizer,LossFun)
    iou, testLoss = test(model,testloader,LossFun)
    Losses += [l]
    print( 'Train loss = %f'%l)
    print( 'Test loss = %f'%testLoss)
    print( 'IOU = %f'%iou)




