import torch
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib as mpl
import helpers

def main():
    image_array = helpers.get_image("/Users/tunger/neural_graphics/Neural_Graphics_A1/images/test_HF_img.png")
    image_array = image_array[:, :, :3]
    print(helpers.sample(0.8, 0.2, image_array))
    print(image_array.shape)

    encoding = helpers.S3TC(image_array)

    decodedImg = helpers.decode_S3TC(encoding, image_array.shape)

    helpers.save_image(decodedImg, "/Users/tunger/neural_graphics/Neural_Graphics_A1/images/test_HF_img_S3TC.png")

    print(f"{helpers.psnr(decodedImg, image_array):.2f} dB")
main()
