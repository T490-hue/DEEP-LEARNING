import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.nn as nn 
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class BrainTumorDataset(Dataset): # we inherit torch Dataset because it will be compatible with pytorch utilities especially DataLoader

    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.classes=['glioma','meningioma','notumor','pituitary']
        self.class_idx={cls:idx for idx,cls in enumerate(self.classes)}
        self.images=[]
        self.labels=[]


        for class_name in self.classes:
            class_path=os.path.join(root_dir,class_name)
            for img_name in os.listdir(class_path):
                img_path=os.path.join(class_path,img_name)

                self.images.append(img_path)
                self.labels.append(self.class_idx[class_name])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):

        img_path=self.images[idx]
        label=self.labels[idx]
        image=Image.open(img_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        
        return image,label
    
class BrainTumourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*32*32,512)
        self.fc2=nn.Linear(512,4)
        self.dropout=nn.Dropout(0.5)



    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=x.view(-1,64*32*32)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.fc2(x)
        return x
def train_epoch(model,train_loader,criterion,optimizer,device):
    model.train()
    running_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        #When you calculate the loss using a loss function like nn.CrossEntropyLoss(), it returns a tensor.
        #.item() converts this single-value tensor into a Python scalar (e.g., a float).

        running_loss+=loss.item()
    return running_loss/len(train_loader)


def evaluate(model,test_loader,device):
    all_preds=[]
    all_labels=[]

    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        output=model(images)
        _,predicted=torch.max(output,1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_preds,all_labels),all_preds,all_labels
def main():


    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 20
    batch_size = 32
    learning_rate = 0.0001
    transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225]) 
        
    ])

    train_dataset=BrainTumorDataset(root_dir='/home/teena/Documents/BrainTumour_CNN/Resized_Dataset/Training',transform=transform)
    test_dataset=BrainTumorDataset(root_dir='/home/teena/Documents/BrainTumour_CNN/Resized_Dataset/Testing',transform=transform)


    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

    model=BrainTumourCNN().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    best_accuracy=0.0
    for epoch in range(num_epochs):
        train_loss=train_epoch(model,train_loader,criterion,optimizer,device)
        accuracy,pred,label=evaluate(model,test_loader,device)

        print(f'epoch:{epoch}')
        print(f'train loss:{train_loss:.4f}')
        print(f'accuracy:{accuracy:.4f}')

        if (accuracy>best_accuracy):
            best_accuracy=accuracy
            torch.save(model.state_dict(),'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    final_accuracy,final_preds,final_labels=evaluate(model,test_loader,device)
    print('\nFinal Model Performance:')
    print(f'Best Test Accuracy: {final_accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(final_labels, final_preds, 
                              target_names=['glioma', 'meningioma', 'pituitary', 'notumor']))


if __name__=='__main__':
    main()