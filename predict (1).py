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
def get_input_args1():
    ## Argument 1: that's a path to a folder
    #parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, help='path to the image in testing')
    parser.add_argument('--topk', type=int, help='return number of the top classes',default=5)
    parser.add_argument('--checkpoint', type=str, help='Saved Checkpoint') 
    parser.add_argument('--gpu', default='False',action='store_true', help='dvice use to predict')
    parser.add_argument('--epoch', type=int, help='amount of times to train model')
    parser.add_argument('--labels', type=str, help=' label names file ',default='aipnd-project/cat_to_name.json')
    # arch and hidden units of checkpoint added per review
    parser.add_argument('--arch', type=str, default='vgg16', help='chosen model')
    parser.add_argument('--hidden_units', type=int, default=4000, help='hidden units for the model')

    return parser.parse_args()
######################
def load_checkpoint(filepath):   
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    optimizer = checkpoint['optimizer'] 
    #epochs = checkpoint['epochs']  
    for param in model.parameters(): 
        param.requires_grad = False 
    return model, checkpoint['class_to_idx']
#model, class_to_idx = load_checkpoint('checkpoint.pth')
#print(model)

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
    img=Image.open(image).convert('RGB')
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
##################
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   
    
    # TODO: Implement the code to predict the class from an image file 
    
    img=Image.open(image_path)
    img = process_image(img)# using process func
  
    # Convert 2D image to 1D vector    
    img = np.expand_dims(img, 0) 
    img = torch.from_numpy(img) 
    model.eval() 
    inputs =img.to(device)
    model.to(device)
    logits = model(inputs) 
    #ps = F.softmax(logits,dim=1)
    topk = logits.cpu().topk(topk)        
    return (e.data.numpy().squeeze().tolist() for e in topk)


def predict(image_path, model, train_data, topk):

    img = process_image(image_path)

   # Convert np_img to PT tensor and send to GPU
    #img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.from_numpy(img).float().to(device)

   # Unsqueeze to get shape of tensor from [Ch, H, W] to [Batch, Ch, H, W]
    img = img.unsqueeze(0)
    # Run the model to predict
    output = model(img)
    probs = torch.exp(output)

    # Pick out the topk from all classes 
    top_probs, top_classes = probs.topk(topk)
    # Convert to list on CPU without grads
    top_probs = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_classes = top_classes.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Invert the class_to_idx dict to a idx_to_class dict
    idx_to_class = {value: key for key, value in train_data.class_to_idx.items()}
    topclass_names = {idx_to_class[index] for index in top_classes}
    return top_probs, topclass_names



    
#################
def main():
    args = get_input_args1()
    
    if torch.cuda.is_available():#and args.gpu
        print("Using GPU.")
        device = torch.device('cuda')
    else:
        print("Using CPU.")
        device = torch.device('cpu')
        
    if(args.checkpoint,args.image):
        #top_p, top_classes = predict(args.image, args.checkpoint, args.topk,args.labels,args.gpu)
        probs, classes = predict(args.image, args.checkpoint, args.topk,train_data)
        print(probs)
        print(classes)
if __name__ == "__main__":
    main()
