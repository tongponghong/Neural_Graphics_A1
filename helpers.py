import torch
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib as mpl
import math
import struct

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_image(image_path):
    return np.asarray(Image.open(image_path))

def save_image(numpy_array, out_path):
    Image.fromarray(numpy_array).save(out_path)
    return 
    
def sample(u, v, image_array):
    image_w = image_array.shape[1]
    image_h = image_array.shape[0]
    print(image_w)
    print(image_h)

    u_scaled = image_w * u
    v_scaled = image_h * (1.0 - v)

    u_prime = math.floor(u_scaled - 0.5)
    v_prime = math.floor(v_scaled - 0.5)

    c0x, c0y = (np.clip(u_prime, 0, image_w - 1), 
                np.clip(v_prime, 0, image_h - 1))
    
    c1x, c1y = (np.clip(u_prime + 1, 0, image_w - 1), 
                np.clip(v_prime + 1, 0, image_h - 1))

    s = (u_scaled - 0.5) - u_prime
    t = (v_scaled - 0.5) - v_prime

    tex_c00 = image_array[c0y][c0x]
    tex_c01 = image_array[c1y][c0x]
    tex_c10 = image_array[c0y][c1x]
    tex_c11 = image_array[c1y][c1x]

    return (1-s)*(1-t)*tex_c00 + s*(1-t)*tex_c10 + (1-s)*t*tex_c01 + s*t*tex_c11

# linearizes the color!
def convert_to_565(color):
    levels = (31, 63, 31)                # 5, 6, 5 bits for R, G, B
    q      = np.rint(color * levels / 255)           # integer 565 code
    q /= 255

    c_packed = np.int16(q[0] << 11 | q[1] << 5 | q[2])

    return c_packed, color

def process_s3tc_tile(tile):
    # expects a 4x4x3 array 

    mean_color = np.mean(tile, axis=(0, 1))
    tilepix = tile.reshape(-1, 3)

    cov = np.cov(tilepix, rowvar=False)

    eigval, eigvec = np.linalg.eigh(cov)
    # eigenvector at max eigenvalue
    principal_axis = eigvec[:, -1]

    norm_paxis = np.dot(principal_axis, principal_axis)

    projections = (tilepix @ principal_axis) - (mean_color @ principal_axis)
    
    minproj = np.min(projections)
    maxproj = np.max(projections)

    c0 = mean_color + minproj * principal_axis
    c1 = mean_color + maxproj * principal_axis
    c0 = np.clip(c0, 0, 255)
    c1 = np.clip(c1, 0, 255)

    c0_packed, c0_original = convert_to_565(c0)
    c1_packed, c1_original = convert_to_565(c1)

    if (c0_packed > c1_packed):
        c2 = 2/3 * c0_original + 1/3 * c1_original
        c3 = 1/3 * c0_original + 2/3 * c1_original

    else:
        c2 = 1/2 * c0_original + 1/2 * c1_original
        c3 = np.zeros(3)

    colors = np.stack([c0, c1, c2, c3])
    norms = ((tilepix[:, None, :] - colors[None, :, :])**2).sum(axis = -1)
    indices = norms.argmin(axis = 1)

    return c0_packed, c1_packed, indices

def get_tiles(texture: np.ndarray):
    tileh = 4
    tilew = 4
    imgh_over_tileh = texture.shape[0] // tileh
    imgw_over_tilew = texture.shape[1] // tilew
    channels = 3

    tiled_image_array = texture.reshape(imgh_over_tileh, tileh, imgw_over_tilew, tilew, channels)
    tiled_image_array = tiled_image_array.swapaxes(1, 2)

    return tiled_image_array

def S3TC(texture: np.ndarray):
    tiled_image = get_tiles(texture)
    outblocks = []

    for j in range(tiled_image.shape[0]):
        for i in range(tiled_image.shape[1]):
            c0_packed, c1_packed, indices = process_s3tc_tile(tiled_image[j][i])

            idx_bits = 0
            for shift, i in enumerate(indices):
                idx_bits |= (i & 0b11) << (shift * 2)

            outblocks.append(struct.pack('<HHI', c0_packed, c1_packed, idx_bits))

    
def decode_S3TC(original_texture: np.ndarray):
    

    # def find_index(texel, color_vals):
#     norms = [np.dot(texel - c, texel - c) for c in color_vals]

#     return np.argmin(norms)
# make note in cv assingment that i am reusing code


    






