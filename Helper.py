
from PIL import Image
import numpy as np

import torch 
from torchvision import transforms

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
    
    # large images will slow down processing
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

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def get_images(content_image_path, style_image_path):
    """
    This function will load content and style images.
    Make sure to pass a correct path for your image.
    Because the two images can be different in their size, we will resize the style image to match the content image.
    """
    # move to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

