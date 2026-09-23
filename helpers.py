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

def c565_to_rgb(color):
    # https://stackoverflow.com/questions/2442576/how-does-one-convert-16-bit-rgb565-to-24-bit-rgb888
    r5 = (color & 0xF800) >> 11
    newR = (r5 * 527 + 23) >> 6

    g6 = (color & 0x07E0) >> 5
    newG = (g6 * 259 + 33) >> 6

    b5 = (color & 0x001F)
    newB = (b5 * 527 + 23) >> 6

    return newR, newG, newB

def convert_to_565(color):
    levels = (31, 63, 31)                # 5, 6, 5 bits for R, G, B
    q      = np.rint(np.clip(color, 0, 255) * levels / 255).astype(np.int64)         # integer 565 code
    # q /= 255

    c_packed = np.uint16((int(q[0]) << 11) | (int(q[1]) << 5) | int(q[2]))

    return c_packed, np.array(c565_to_rgb(int(c_packed)), dtype = np.float64)


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

    shifts = np.arange(16, dtype = np.uint32) * 2
    
    for j in range(tiled_image.shape[0]):
        for i in range(tiled_image.shape[1]):
            c0_packed, c1_packed, indices = process_s3tc_tile(tiled_image[j][i])

           
            idx_bits = int(np.bitwise_or.reduce((indices.ravel().astype(np.uint32) & 0b11) << shifts))

            outblocks.append(struct.pack('<HHI', c0_packed, c1_packed, idx_bits))

    return outblocks

    
def decode_S3TC(encodedBlocks, imageShape):  
    bytes = b''.join(encodedBlocks)
    codeStruct = np.dtype([('c0', '<u2'), ('c1', '<u2'), ('idx', '<u4')])
    blocks = np.frombuffer(bytes, codeStruct)

    numBlocks = len(blocks)
    c0 = blocks['c0']
    c1 = blocks['c1']
    ind = blocks['idx']

    c0r5 = (c0 & 0xF800) >> 11
    c0R = (c0r5 * 527 + 23) >> 6

    c0g6 = (c0 & 0x07E0) >> 5
    c0G = (c0g6 * 259 + 33) >> 6

    c0b5 = (c0 & 0x001F)
    c0B = (c0b5 * 527 + 23) >> 6

    c1r5 = (c1 & 0xF800) >> 11
    c1R = (c1r5 * 527 + 23) >> 6

    c1g6 = (c1 & 0x07E0) >> 5
    c1G = (c1g6 * 259 + 33) >> 6

    c1b5 = (c1 & 0x001F)
    c1B = (c1b5 * 527 + 23) >> 6

    c0_RGB8 = np.stack([c0R, c0G, c0B], axis=-1)
    c1_RGB8 = np.stack([c1R, c1G, c1B], axis=-1)

    
        # c2 = 2/3 * c0_RGB8 + 1/3 * c1_RGB8
        # c3 = 1/3 * c0_RGB8 + 2/3 * c1_RGB8

        # c2 = 1/2 * c0_RGB8 + 1/2 * c1_RGB8
        # c3 = np.zeros(3)

    isGt = (c0 > c1)[:, None]

    c2 = np.where(isGt, 2/3 * c0_RGB8 + 1/3 * c1_RGB8, 1/2 * c0_RGB8 + 1/2 * c1_RGB8)
    c3 = np.where(isGt, 1/3 * c0_RGB8 + 2/3 * c1_RGB8, np.zeros_like(c0_RGB8))

    colorPal = np.stack([c0_RGB8, c1_RGB8, c2, c3], axis = 1)
    
    shifts = np.arange(0, 32, 2, dtype = np.uint32)
    indices = (ind[:, None] >> shifts) & 0b11

    decompressed_tex = np.take_along_axis(colorPal, indices[:, :, None], axis=1)

    grid_h = imageShape[0] // 4
    grid_w = imageShape[1] // 4

    tiles = decompressed_tex.reshape(grid_h, grid_w, 4, 4, 3)

    outImage = tiles.swapaxes(1, 2,).reshape(imageShape)


    return np.clip(outImage, 0, 255).astype(np.uint8)


def psnr(reconstructed, original, max_val = 255.0):
    x = np.asarray(original, dtype=np.float64)
    x_hat = np.asarray(reconstructed, dtype=np.float64)

    if x.shape != x_hat.shape:
        raise ValueError("ur shapes are wrong nerd")

    mse = np.mean((x - x_hat) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(max_val) - 10 * np.log10(mse)

        

        
        










