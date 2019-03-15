# -*- coding: utf-8 -*-


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models


from Helper import load_image, get_images, im_convert


"""
Neural Style Transfer with Deep Convolutional Neural Networks:

We’ll *recreate* a style transfer method that is outlined in the paper,
[Image Style Transfer Using Convolutional Neural Networks, by Gatys]
(https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
in PyTorch. This is one of the first method to do style transfer. 

In this paper, style transfer uses the features found in the 19-layer VGG Network,
which is comprised of a series of convolutional and pooling layers,
and a few fully-connected layers.
The convolutional layers are named by stack and their order in the stack.
Conv_1_1 is the first convolutional layer that an image is passed through, in the first stack.
Conv_2_1 is the first convolutional layer in the second stack.The deepest convolutional layer in the network is conv_5_4.

**Note: This is a general code you can change the model, weights and the selected layers
"""

<<<<<<< HEAD:Style_Transfer.py
=======
# Import resources

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models



def load_image(img_path, max_size=400, shape=None):
    """
    Loading image:

    `load_image` is a helper function.
    It will load the image from `img_path` path and then preprocess the image.
    We process the image because the model accepts a specific format for its input.

    This function will do certain steps:

    1- Load the image from `img_path` path.
       We will use `Image.open` from Pillow library. After that, we convert it to `RGB` photo using `convert`. 
    2- We put a threshold for the size of the image using `max_size` function argument.
       Big images will slow processing. So if our image size is bigger than the `max_size` then we will crop it.
    3- We will use `transform` object to preprocess the image. it will do three operations.
       First, it will resize the image size to `size`.
       Then, it will convert this image into a tensor.
       Finally, it will normalize the produced tensor (The first parameter is a tuple of means for every channel and the second parameter is the corresponding standard deviations.).
    4- Finally, we will discard the alpha channel and add one dimension for the batch number.
    """
    
    image = Image.open(img_path).convert('RGB')
    
    # Large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # Discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def get_images(content_image_path, style_image_path):
    """
    This function will load content and style images.
    Make sure to pass a correct path for your image.
    Because the two images can be different in their size, we will resize the style image to match the content image.
    """
    
    content = load_image(content_image_path).to(device)
    
    # Resize style to match content.
    style = load_image(style_image_path, shape=content.shape[-2:]).to(device)

    return content, style



def im_convert(tensor):
    """ 
    This is a helper function.
    It will un-normalize an image and convert it from a Tensor image to a NumPy image for displaying it.
    """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image




>>>>>>> opengenus/master:Neural_Style_Transfer.py
def get_features(image, model, layers=None):
    """ 
    This function will return the features maps produced by the layers in `layers` dictionary.
    These feature maps are used to calculate the losses.
    The layers of the model are indexed using numbers.
    `layers` is a dictionary that maps between the index of a given layer and its name.
    Here we will use the layers that are mentioned in the paper.
    Then we feedforward the image through the layers.
    We store the feature maps of every layer in `layers` dictionary in a new dictionary `features`.
    Then we return this dictionary.
    """
    
    # The layers that are mentioned in Gatys et al (2016) paper
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2',
                 '28' : 'conv5_1'}

    features = {}
    x = image
    
    # `model._modules` is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """
    The Gram matrix
    The output of every convolutional layer is a Tensor
    with dimensions associated with the batch_size, a depth, d and some height and width (h, w).
    
    The Gram matrix of a convolutional layer can be calculated as follows:
    1- Get the depth, height, and width of a tensor using `tensor.shape`. 
    2- Reshape that tensor so that the spatial dimensions are flattened.
       We wil use `view` function to reshape the Tensor. 
    3- Calculate the gram matrix by multiplying the reshaped tensor by it's transpose.
    `torch.mm` is used to multiply the two tensors.
    """
  
    # Reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(tensor.shape[1], tensor.shape[2]*tensor.shape[3])
    
    # Calculate the gram matrix
    gram = torch.mm(tensor,tensor.t())
    
    return gram


def get_content_and_style_features(content, style, model, layers=None):
    """
    Getting features for style and content images: 
    Now we will extract features from our images and calculate the Gram matrix for each layer in our style representation.
    Also, we create a copy from our content image and assign it to our target image.
    """
    
    # Get content and style features
    content_features = get_features(content, model, layers)
    style_features = get_features(style, model, layers)

    # Calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    return content_features, style_features, style_grams





def style_transfer(content, style, model, steps, content_weight, style_weight, style_weights, layers=None):
    """
    This function will do the style transfer. It will go through many steps:
    
    1- Getting content features and style features. Then we initalize our target image.

    2- Determine our hyperparameters. Our optimizer will be `Adam`.
    Then we will determine the number of steps (how many iterations we update the target image).

    3- Getting the target features using `get_features`. 

    4- Calculate the content loss.
    which is defined as the mean square difference between the target and content features at layer `conv4_2`.

    5- Calculate the style loss. We iterate through each style layer.
    Then, we get the corresponding target features from `target features`.
    After that, we calculate the Gram matrix for the target features.
    The style loss for one layer is the mean square difference between the Gram matrices.
    Finally, we add the loss of the current layer to the total loss. We normalize the loss by dividing by (d X w X h). 

    6- Calculate the total loss. The total loss will be the weighted sum of the content and style losses. 

    7- Update the target image by back-propagating the loss `total_loss.backward()` and doing one optimizer step.

    8- Display the loss and the intermediate images every `show_every` steps.
    """
    
    # For displaying the target image, intermittently
    show_every = 400

    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initalizing our target image
    target = content.clone().requires_grad_(True).to(device)

    # Getting features
    content_features, style_features, style_grams = get_content_and_style_features(content, style, model, layers)

<<<<<<< HEAD:Style_Transfer.py
    # ten precent of steps
    ten_percent = int(0.1*steps)

    # iteration hyperparameters
=======
    # Iteration hyperparameters
>>>>>>> opengenus/master:Neural_Style_Transfer.py
    optimizer = optim.Adam([target], lr=0.003)

    print("Processing: ")

    for ii in range(1, steps+1):
        
        # Get the features from your target image    
        target_features = get_features(target, model)
        
        # Calculate the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # Initialize the style loss to 0
        style_loss = 0
        
        # Iterate through each style layer and add to the style loss
        for layer in style_weights:
          
            # Get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            
            # Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)
            
            # Get the "style" style representation
            style_gram = style_grams[layer]
            
            # Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer]* torch.mean((target_gram - style_gram)**2)
            
            # Add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
            
        # Calculate the *total* loss
        total_loss = content_weight*content_loss + style_weight*style_loss

        # Update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # show percentage 
        if ii%ten_percent == 0: 
            print("{}%".format(int(ii*100.0/steps)))
        
<<<<<<< HEAD:Style_Transfer.py
        # display intermediate images and print the loss
        '''
=======
        # Display intermediate images and print the loss
>>>>>>> opengenus/master:Neural_Style_Transfer.py
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()
        '''
            
    return target 

<<<<<<< HEAD:Style_Transfer.py
            


=======
"""
********************************************************************************************
Here we you must load your model. In this example, we are using VGG19 model.

Any model has two parts.The first part consists of convolutional layers
and the second part is two fully connected layers and it acts as a classifier.
Here we just want to get the feature maps produced by the convolutional layers so we will load the first part only.

The model will be loaded from torchvision pre-trained models.
Also, we will freeze the parameters of the convolutional layers
because we don't want them to be changed and we are only changing the target image.
*********************************************************************************************
"""

# Loading the first part
vgg = models.vgg19(pretrained=True).features

# Freezing the parameters
for param in vgg.parameters():
    param.requires_grad_(False)
    
# Move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

'''
******************************************
Here we load our content and style images.
******************************************
'''
content, style = get_images(content_image_path="content.jpg", style_image_path="style.jpg")

"""
*********************************************************************************************************
Weights:  
We have two kinds of weights:
1- Style representation weights.
You can put weights for the layers that are used to calculate the Gram matrix. 

2- Style and content loss weights.
We use the value that are mentioned in the paper. This ratio will affect the stylization of the final image.
It's recommended that you leave the content_weight = 1 and set the style_weight to achieve the ratio you want.
*************************************************************************************************************
"""

# Weights for each style layer 
# Weighting earlier layers more will result in *larger* style artifacts
# Notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

content_weight = 1  # alpha
style_weight = 1e6  # beta

"""
****************************************************************
The defalut value for layers will be used if we don't pass the last parameter
layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2',
                 '28' : 'conv5_1'}
In order to change it you must define your layers dictionary and then pass it as a final parameter. 
**************************************************************                 
"""
steps = 2000

target = style_transfer(content, style, vgg, steps, content_weight, style_weight, style_weights)

# Display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
>>>>>>> opengenus/master:Neural_Style_Transfer.py

