import numpy as np
from PIL import Image
import matplotlib as mpl
import helpers

from matplotlib import pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.io as io
import trainer 

from pathlib import Path
import glob

def size_vs_PSNR(image_name, runs, out_path):
    runs = sorted(runs, key = lambda r: r["neural_bytes"])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    xs = [r["neural_bytes"] / 1024 for r in runs]
    ys = [r["neural_psnr"] for r in runs]

    ax.plot(xs, ys, marker='o', color='tab:blue', label='Neural (float32)')

    for (r, x, y) in zip(runs, xs, ys):
        ax.annotate(f"{r['label']}\n{r['raw_bytes'] / r['neural_bytes']:.0f}x",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize = 7,
                    ha = "center",
                    color = 'tab:blue')

    q = [r for r in runs if r["quantized_psnr"] is not None]

    if q:
        ax.plot([r["quantized_bytes"] / 1024 for r in q],
                [r["quantized_psnr"] for r in q],
                marker = "s",
                linestyle="--",
                color = "tab:green",
                label = "Neural (8-bit grid)")

    s = runs[0]
    sx, sy = s["s3tc_bytes"] / 1024, s["s3tc_psnr"]
    ax.plot(sx, sy, 
            marker="*", markersize=18, 
            linestyle="none",
            color="tab:red", 
            label="S3TC / DXT1")
    
    ax.annotate("6x", (sx, sy),
                textcoords="offset points", 
                xytext=(0, -18),
                fontsize=8, ha="center", color="tab:red")

    ax.set_xscale("log")
    ax.set_xlabel("Stored size (KB, log scale)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{image_name}: size vs. reconstruction quality")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main_loop(config):
    # ------------------- S3TC -------------------------------
    image_array = io.read_image(config['texture']).numpy().transpose(1, 2, 0)
    print(config['texture'])
    print(image_array.shape)
    image_array = image_array[:, :, :3]
    print(helpers.sample(0.8, 0.2, image_array))
    print(image_array.shape)

    encoding = helpers.S3TC(image_array)

    decodedImg = helpers.decode_S3TC(encoding, image_array.shape)

    
    outPathParts = list(config['image_out'].parts)

    # im so sorry to whoever needs to read this syntax
    outPathParts[-1] = outPathParts[-1][:outPathParts[-1].find('_')] + "_S3TC_processed.png"
    outPathParts[-2] = "S3TC_output"

    helpers.save_image(decodedImg, Path(*outPathParts))

    s3tc_psnr = helpers.psnr(decodedImg, image_array)
    print(f"{s3tc_psnr:.2f} dB")

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
    #print("device : " + device)

    model = trainer.NeuralTexture(config).to(device)

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
        #if (step % 400 == 0): print("running epoch " + str(step) + "/" + str(config['epochs']))
        loss.backward()
        optimizer.step()

    coords = coords.to(device)
    target = target.to(device)
    print("PSNRs (taken every 200 epochs):")
    print(PSNRs)

    model.eval()
    with torch.no_grad():
        output = model(coords)
        loss = criterion(output, target)

    quantized_psnr = None
    if (config['quantize']): 
        trainer.quantize_model(model)
        with torch.no_grad():
            quantized_output = model(coords)
            quantized_loss = criterion(quantized_output, target)
        quantized_psnr = -10 * torch.log10(quantized_loss.to('cpu')).item()
    final_psnr = -10 * torch.log10(loss.to('cpu')).item()

    output = output.to('cpu')
    output = torch.unflatten(output, dim=0, sizes = (image_height, image_width))
    output =  output.permute(2, 0, 1)
    output = torch.clip(output, 0.0, 1.0)
    output = (output*255).to(torch.uint8)
    io.write_png(output, config['image_out'], compression_level = 0)
    print(f"wrote output texture to {config['image_out']}")
    #print(f"Quantizatoin: {config['quantize']}")

    print(f"Final PSNR: {final_psnr}")
    if (config['quantize']): print(f"Quantized PSNR: {quantized_psnr}")

    raw_bytes = image_width*image_height*3
    num_grid_params = sum(p.numel() for p in model.grid.parameters())
    num_mlp_params = sum(p.numel() for p in model.mlp.parameters())
    neural_bytes = num_grid_params * 4 + num_mlp_params * 4
    quantized_bytes = num_grid_params + num_mlp_params * 4
    print(f"Compression ratio: {neural_bytes/raw_bytes}")
    print(f"Quantized compression ratio: {quantized_bytes/raw_bytes}")



    # --------------- MAKE GRAPHS -------------------
    outPath = config['image_out']
    outPathParts = list(outPath.parts)

    outPathParts[-2] = "plots"
    outPathParts[-1] = outPath.name[:-4] + "_graph.png"

    x = np.arange(1, len(PSNRs) + 1)
    fig1 = plt.figure()
    plt.plot(x, PSNRs, marker = 'o')
    plt.title(outPath.name[:-4] + " PSNRs")
    plt.xlabel("Epochs (x200)")
    plt.ylabel("PSNR (dB)")

    fig1.savefig(Path(*outPathParts))
    plt.close(fig1)

    return {
        "label": f"R={len(config['resolutions'])}, F={config['feat_dim']}",
        "raw_bytes": raw_bytes,
        "neural_bytes": neural_bytes,
        "neural_psnr": final_psnr,
        "quantized_bytes": quantized_bytes,
        "quantized_psnr": quantized_psnr,
        "s3tc_bytes": image_width * image_height // 2,
        "s3tc_psnr": s3tc_psnr,
    }

def main():
    original_img_paths = glob.glob("/Users/tunger/neural_graphics/Neural_Graphics_A1/images/original_images/*.png")


    for img_path in original_img_paths:
        currPath = Path(img_path)
        pathParts = list(currPath.parts)

        pathParts[-2] = "neural_output"

        name = currPath.name[:-4] 
        outImageName = name + "_neural.png"
        pathParts[-1] = outImageName

        print(f"image {name} and new out image {outImageName}")

        outPath = Path(*pathParts)

        newSettings = [
            {'texture': img_path, 'resolutions': [16, 32, 64, 128], 'feat_dim': 4, 'image_out': outPath},
            {'texture': img_path, 'resolutions': [16, 32, 64], 'feat_dim': 2, 'image_out': outPath},
            {'texture': img_path, 'resolutions': [64], 'feat_dim': 2, 'image_out': outPath},
        ]

        runs = []
        for setting in newSettings:
            setting['image_out'] = outPath.with_name(outPath.name[:-4] + "_" + str(setting['resolutions']) + ".png")
            currConfig = {**trainer.base_config(), **setting}
            runs.append(main_loop(config=currConfig))

        size_vs_PSNR(name, runs, currPath.parent.parent / "plots" / f"{name}_size_vs_psnr.png")


main()



