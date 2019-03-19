
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

from Helper import load_image, get_images, im_convert
from Style_Transfer import style_transfer

import os
import argparse 

def is_valid_directory(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg

def get_paths(): 

    parser = argparse.ArgumentParser(
        description='Style Transfer. You need to pass two directorys for the content and style images. ')

    parser.add_argument(
        'content_image_dir',
        help='Content image directory.',
        type=lambda x: is_valid_directory(parser, x))

    parser.add_argument(
        'style_image_dir',
        help='Style image directory.',
        type=lambda x: is_valid_directory(parser, x))
    
    args = parser.parse_args()

    return args.content_image_dir, args.style_image_dir

if __name__ == "__main__":
    # loading the first part
    vgg = models.vgg19(pretrained=True).features

    # freezing the parameters
    for param in vgg.parameters():
        param.requires_grad_(False)
        
    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.to(device)

    content_image_path, style_image_path = get_paths()

    content, style = get_images(content_image_path, style_image_path)

    # weights for each style layer 
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    style_weights = {'conv1_1': 1.,
                    'conv2_1': 0.8,
                    'conv3_1': 0.5,
                    'conv4_1': 0.3,
                    'conv5_1': 0.1}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    steps = 3000

    target = style_transfer(content, style, vgg, steps, content_weight, style_weight, style_weights)

    plt.imsave("result.png", im_convert(target))

    # display content and final, target image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax2.imshow(im_convert(target))
    plt.show() 