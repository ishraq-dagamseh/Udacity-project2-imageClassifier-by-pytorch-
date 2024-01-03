import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


#####arguments commmands#############
import argparse
def get_input_args():
    ## Argument 1: that's a path to a folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = 'paind-project/flowers', 
                    help = 'path to the folder of flower images')
    parser.add_argument('--traindir', type=str, default='paind-project/flowers/train',help='trainImages Folder')
    parser.add_argument('--valdir', type=str, default='paind-project/flowers/valid',help='validationImages Folder')
    parser.add_argument('--testdir', type=str, default='paind-project/flowers/test',help='testImages Folder')
    ## Argument 2: that's a CNN model arch
    parser.add_argument('--arch', type=str, default = 'vgg16', 
                    help = ' a VGG CNN model arch') 
    parser.add_argument('--learning_rate', default='0.001',type=float, help='learning rate')     
      
    parser.add_argument('--save_dir', help='save_directory')
    parser.add_argument('--hidden_units',default=4000,type=int,  help='number of hidden neurons')
    parser.add_argument('--epochs', default=15, help='number of epochs', type=int)
    parser.parse_args('--gpu', help='gpu available for training', action='store_true',default='False')
    parser.add_argument('--checkpoint', type=str, default='',help='Save trained model to file')
 
    return  parser.parse_args()

#######load data############
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
validationt_transforms =  transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

testing_transforms =  transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder

train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms )
test_image_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms )
val_image_datasets = datasets.ImageFolder(valid_dir, transform=validationt_transforms) 
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=128)
valloader = torch.utils.data.DataLoader(val_image_datasets, batch_size=128)

class_to_idx = train_image_datasets.class_to_idx

###################
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
##################
#
'''
def build_model(arch='vgg16', num_hidden_neurons=4000,learning_rate=0.001 ):
    #model=models.vgg16(pretrained=True)
    device =(torch.device("cuda") if torch.cuda.is_available else "cpu")
    
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_feature
    
       
                                       
# Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
#->vgg9216->alexnet
    model.classifier = nn.Sequential(nn.Linear(input_size,num_hidden_neurons ),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),#0.5
                                 #nn.Linear(256, 128),
                                 #nn.ReLU(),
                                 nn.Linear(num_hidden_neurons, 102),# this from model the previous layer of last one  classifier
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
#.CrossEntropyLoss()
# Only train the classifier parameters, feature parameters are frozen
#optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001)#, lr=0.01)0.003
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)#, weight_decay=0.0001)
#optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)#, momentum=0.9, weight_decay=0.0001)
# L2 regularization 
#optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, weight_decay=0.01)
    model.to(device);
#model

'''
#class train():
##################
def build_model(hidden_units,class_to_idx, arch='vgg16',learning_rate=0.001, gpu=True):
    if arch == 'densenet121':
         model = models.densenet121(pretrained=True)
         input_size = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_feature
    for param in model.parameters():
        param.requires_grad = False
#->vgg9216->alexnet
    model.classifier = nn.Sequential(nn.Linear(25088,hidden_units ),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),#0.5
                                 #nn.Linear(256, 128),
                                 #nn.ReLU(),
                                 nn.Linear(hidden_units , 102),# this from model the previous layer of last one  classifier
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)#, weight_decay=0.0001)
    model.class_to_idx = class_to_idx
    return model, optimizer, criterion
#####################
def train(trainloader,valloader ,hidden_units,class_to_idx, arch,learning_rate, gpu=True, checkpoint='',epochs=15):
    model, optimizer, criterion = build_model(hidden_units,class_to_idx,  arch, learning_rate)
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
        
    else:
        device = torch.device("cpu")
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 20#50
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
          # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                         f"Train loss: {running_loss/len(trainloader):.3f}.. "
                         f"val loss: {val_loss/len(valloader):.3f}.. "
                         f"val accuracy: {accuracy/len(valloader):.3f}")
                # TODO: Do validation on the test set-
                
########################                
def testing(testloader,model):
    images, labels = next(iter(testloader))
    images, labels = images.cuda(), labels.cuda() 
# Get the class probabilities
    ps = torch.exp(model.forward(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
    print(ps.shape)
    top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
    print(top_class[:10,:])
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'TestAccuracy: {accuracy.item()*100}%')
##################                
def save_checkpoint(arch, learning_rate, hidden_units, epochs, save_path, optimizer,criterion,class_to_idx):
    
    # TODO: Save the checkpoint 
    train_image_datasets.class_to_idx
    model.class_to_idx = train_image_datasets.class_to_idx
    #optimizer.state_dict
    checkpoint = {'model': arch,
              'learning_rate': learning_rate,
              'hidden_units': hidden_units,
              'epochs': epochs, 
              'optimizer': optimizer.state_dict(), 
              'state_dict': model.state_dict(),         
              'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')

                
                
#main()
'''def main():
    args=get_input_args()
    train(trainloader,valloader ,epochs,hidden_units,class_to_idx, arch,learning_rate, gpu=True, checkpoint='')
    save_checkpoint(model,args.learning_rate,args.hidden_units,
                   args.epochs,args.checkpoint,optimizer,criterion,class_to_idx)
        
if __name__ == "__main__":
    main()   '''     