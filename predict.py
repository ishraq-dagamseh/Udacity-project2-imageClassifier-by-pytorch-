import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
def get_input_args():
    ## Argument 1: that's a path to a folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to the image to test')

    parser.add_argument('--top_k', type = int, default = '5', 
                    help = 'number of top classes')                                                      
    parser.add_argument('--category_names', default = 'paind-project/cat_to_name.json', 
                    help = ' a jason file for category names') 
    parser.add_argument('--learning_rate', default='0.001',type=float, help='learning rate')     
      
    parser.add_argument('--hidden_units',default=4000,type=int,  help='number of hidden neurons')
    parser.add_argument('--epochs', default=15, help='number of epochs', type=int)
    parser.parse_args('--gpu', action='store_true',default='False', help='gpu available for training')

    return  parser.parse_args()  


def load_checkpoint(checkpoint):   
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']  
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    optimizer = checkpoint['optimizer']
    epochs= checkpoint['epochs']
       
    for param in model.parameters(): 
        param.requires_grad = False 
    return model, checkpoint['class_to_idx']
#model, class_to_idx = load_checkpoint('checkpoint.pt')

'''def load_model(checkpoint,,arch,hidden_units):
    device = torch.device("cuda")

    state_dict = torch.load(checkpoint,)
    #print(state_dict.keys())
    
    class_to_idx = checkpoint_state['class_to_idx']
    model, optimizer, criterion = build_model(hidden_units, class_to_idx,arch)
    #model.to(device)
    model.load_state_dict(state_dict)
    return model'''

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size =256,256
    img=Image.open(image)
    img=img.resize(size)
    im_crop = img.crop((20,20,244,244))
    np_image = np.array(im_crop)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ((np_image/255) -  mean)/std    
    image = np.transpose(image, (2, 0, 1))
    return  image
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path,labels,checkpoint, arch, hidden_units, gpu=True, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file 
    #img=Image.open(image_path)
    model, optimizer, criterion = load_checkpoint(checkpoint)
    img = process_image(image_path)# using process func
  
    # Convert 2D image to 1D vector    
    if gpu:
        Input = torch.FloatTensor(img).cuda()
    else:
        Input = torch.FloatTensor(img)

    model.eval() 
    image = Input.unsqueeze_(0)
    model.to(device) 
    logps = model(image)
    ps = torch.exp(logps)
    top_p, top_classes = torch.topk(ps, topk)
    '''indexOfclass=[]
    for x in model.class_to_idx:
            Cofind=model.class_to_idx[x]
            indexOfclass.append(Cofind)'''
    class_index = {}
    for k, v in model.class_to_idx.items():
        class_index[v] = k

    #np_top_classes = top_classes[0].numpy()
   
    pred_classes = []
    
    for i in top_classes.cpu().numpy()[0]:
        pred_classes.append(class_index[i])
        
    return top_p.cpu().detach().numpy()[0], pred_classes  