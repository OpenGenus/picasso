# Picasso
Style transfer is a technique in which we compose two images together, a content image, a style image, and produce a third image that will look like the content image and the style of it will be like the style image. This is an implementation for style transfer using PyTorch. The method used to implement this project is outlined in the paper, Image Style Transfer Using Convolutional Neural Networks, by Gatys el. (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## Prerequisites
* Python 3 
* PyTorch
* Torchvision
* Pillow 
* Matplotlib
* Numpy 
* argparse

## Usage 
To use this project for stylizing your own images you must follow these steps: 

* Clone the repositry: 
```
git clone https://github.com/OpenGenus/picasso
```
* Run this command: 
```
python CLI_interface.py content_image_path style_image_path
```

Where: 

`content_image_path` is the absoulte path of your content image. 

`style_image_path` is the absoulte path of your style image. 

The result will be saved to the current directory. The name of the image will be "result.png".  

## Using Google Colaboratory
You need powerful GPUs for fast execution. fortunately, Google Colab provides free GPUs. To run this project in Google colab you must follow these steps: 
* Go to https://colab.research.google.com. 
* Create new python 3 notebook. 
* Clone the repositry and change the directory by running the following commands: 
```
! git clone https://github.com/OpenGenus/picasso
! cd picasso
```
* Go to files and then upload your images (the small arrow in the left). 
* Copy this script to any cell in the notebook and run it: 
```
! python CLI_interface.py ../content_image ../style_image

from google.colab import files
files.download("result.png")
```

Where: 

```content_image``` is the name of your content image including extension (e.g content.jpg). 

```style_image``` is the name of your style image including extension (e.g style.jpg).

After that result image will start downloading.
