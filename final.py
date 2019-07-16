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

    
DATA = np.arange(24045)#test data set length = 24045


#Creation of customDataset! Combining Images with the bounding box.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,transform=None):
        self.data = DATA
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        im = Image.open(os.path.join('test/{}.png'.format(idx)))
        if self.transform:
            sample = self.transform(im)
        return sample

#normalization and loading the data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
Transform = transforms.Compose([transforms.ToTensor(),normalize])
data = CustomDataset(transform = Transform)
testloader = torch.utils.data.DataLoader(data, batch_size=1,shuffle=False, num_workers=1)

def test(model,testloader):
    model.eval()
    O = []
    with torch.no_grad():
        for data in tqdm(testloader):
            images = data         
            if torch.cuda.is_available():
                images = images.cuda()

            outputs = model(images)
            O += [np.array(outputs)]
            
    return O

model = models.resnet50()


inFeatures = model.fc.in_features
model.fc = nn.Linear(inFeatures,4)

for param in model.parameters():
    param.requires_grad = True

model.load_state_dict(torch.load('./modelsc/model-55.pth'))

if torch.cuda.is_available():
    model = model.cuda()

O = test(model , testloader )

L = []      
with open('test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0 
    for row in csv_reader:
        if( i != 0 ):
            L.append(row[0])
        i = i+1  

i = 0 

Xc = float(224/640)
Yc = float(224/480)

with open('finalc3.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['image_name','x1','x2','y1','y2'])
    for j in np.arange(len(O)):
    	for k in np.arange(len(O[0])):
    		employee_writer.writerow([L[i],O[j][k][0]/Xc, O[j][k][1]/Xc,O[j][k][2]/Yc,O[j][k][3]/Yc])
    		i = i + 1
