import torch
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib as mpl
import helpers

def main():
    image_array = helpers.get_image("/Users/tunger/neural_graphics/images/gradient.png")
    print(helpers.sample(0.8, 0.2, image_array))
    helpers.compress(image_array)


main()
