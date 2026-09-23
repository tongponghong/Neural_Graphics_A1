import numpy as np
from PIL import Image
import matplotlib as mpl
import helpers

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.io as io
import trainer 
import glob

def main_loop(config):
    # ------------------- S3TC -------------------------------
    image_array = helpers.get_image("/Users/tunger/neural_graphics/Neural_Graphics_A1/images/test_HF_img.png")
    image_array = image_array[:, :, :3]
    print(helpers.sample(0.8, 0.2, image_array))
    print(image_array.shape)

    encoding = helpers.S3TC(image_array)

    decodedImg = helpers.decode_S3TC(encoding, image_array.shape)

    helpers.save_image(decodedImg, "/Users/tunger/neural_graphics/Neural_Graphics_A1/images/test_HF_img_S3TC.png")

    print(f"{helpers.psnr(decodedImg, image_array):.2f} dB")

    # -------------------- NEURAL PART -------------------
    print(f"reading texture from {config['texture']}")
    print(f"resolutions: {config['resolutions']} | feat_dim: {config['feat_dim']}")
    image = io.read_image(config['texture'], mode="RGB")
    image = image.permute(1, 2, 0)
    if (image.dtype == torch.uint16):
        image = image / 65535.0
    else:
        image = image / 255.0

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    image_height = image.shape[0]
    image_width = image.shape[1]
    total_pixels = image_height*image_width
    pixel_centers_x = (torch.arange(image_width) + 0.5)/image_width
    pixel_centers_y = (torch.arange(image_height) + 0.5)/image_height
    coords = torch.cartesian_prod(pixel_centers_y, pixel_centers_x)
    target = torch.flatten(image, start_dim=0, end_dim=1)

    device = get_device()
    print("device : " + device)

    model = trainer.NeuralTexture().to(device)

    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    criterion = nn.MSELoss()

    model.train()
    PSNRs = []
    for step in range(config['epochs']):
        sample_indices = torch.randperm(total_pixels)[:config['batch_size']]
        sample_coords = torch.index_select(coords, 0, sample_indices)
        sample_targets = torch.index_select(target, 0, sample_indices)
        sample_coords = sample_coords.to(device)
        sample_targets = sample_targets.to(device)

        optimizer.zero_grad()
        sample_predictions = model(sample_coords)
        loss = criterion(sample_predictions, sample_targets)
        psnr = -10 * torch.log10(loss)
        if (step % 200 == 0): PSNRs.append(round(psnr.item(), 3))
        if (step % 400 == 0): print("running epoch " + str(step) + "/" + str(config['epochs']))
        loss.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient mean: {param.grad.abs().mean().item()}")
        #     else:
        #         print(f"❌ {name} has NO GRADIENT!")

    coords = coords.to(device)
    target = target.to(device)
    model.eval()
    with torch.no_grad():
        output = model(coords)
        loss = criterion(output, target)
    final_psnr = -10 * torch.log10(loss.to('cpu')).item()
    output = output.to('cpu')
    output = torch.unflatten(output, dim=0, sizes = (image_height, image_width))
    output =  output.permute(2, 0, 1)
    output = torch.clip(output, 0.0, 1.0)
    output = (output*255).to(torch.uint8)
    io.write_png(output, config['image_out'], compression_level = 0)
    print(f"wrote output texture to {config['image_out']}")

    PSNRs.append(round(psnr.item(), 3))
    print("PSNRs (taken every 200 epochs):")
    print(PSNRs)
    print(f"Final PSNR: {final_psnr}")

    raw_bytes = image_width*image_height*3
    print(f"Raw bytes: {raw_bytes}")
    num_grid_params = sum(p.numel() for p in model.grid.parameters())
    num_mlp_params = sum(p.numel() for p in model.mlp.parameters())
    if(config['quantize']):
        neural_bytes = num_grid_params + num_mlp_params * 4
    else:
        neural_bytes = num_grid_params * 4 + num_mlp_params * 4
    print(f"Neural bytes: {neural_bytes}")
    print(f"Compression ratio: {neural_bytes/raw_bytes}")

def main():
    original_img_paths = glob.glob("/Users/tunger/neural_graphics/Neural_Graphics_A1/images/original_images/*.png")
    newSettings = [
        {'resolutions': (16, 32, 64, 128), 'feat_dim': 4},
        {'resolutions': (16, 32, 64), 'feat_dim': 2},
        {'resolutions': (64), 'feat_dim': 2},
    ]
    
    for img_path in original_img_paths:
        for setting in newSettings:
            currConfig = {**trainer.base_config(), **setting}
            main_loop()



main()


