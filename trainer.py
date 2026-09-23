import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.io as io
import sys

config = {
    'texture': "images/test_HF_img.png", 
    'resolutions': (16, 32, 64, 128), #sizes of feature grids
    'feat_dim': 2, #number of features per grid
    'batch_size': 16384,
    'lr': 0.01,
    'epochs': 2000, 
    'image_out': "images/Experiment_1_srgb_neural.png", #NOT YET IMPLEMENTED
    'quantize': True #NOT YET IMPLEMENTED
    # Include other parameters as needed.
}

class FeatureGrid(nn.Module):
    def __init__(self, resolutions, feat_dim):
        super().__init__()
        self.grids = nn.ParameterList([nn.Parameter(torch.ones(1, feat_dim, R, R)) 
                                       for R in resolutions])
        self.out_dim = feat_dim * len(resolutions)
    def forward(self, uv):
        result_list = []
        uv_scaled = uv * 2 - 1 #(N, 2)
        uv_scaled_4d = uv_scaled[None, None, :, :] #(1, 1, N, 2) = (N, H_out, W_out, 2)
        for feature_grid in self.grids: #(1, feat_dim, R, R) = (N, C, H_in, W_in)
            samples = F.grid_sample(feature_grid, uv_scaled_4d, mode="bilinear", 
                                             padding_mode="border", align_corners=False) 
                                             #(1, feat_dim, 1, N) = (N, C, H_out, W_out)
            samples = samples.squeeze(2).permute(0, 2, 1).squeeze(0) #(N, feat_dim)
            result_list.append(samples)
        result = torch.cat(result_list, dim=1)#(N, feat_dim x L)
        return result 
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class NeuralTexture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grid = FeatureGrid(cfg["resolutions"], cfg["feat_dim"])
        self.mlp = ColorMLP(self.grid.out_dim)
    def forward(self, uv):
        return self.mlp(self.grid(uv))

def quantize_uint8(x):
    lo, hi = torch.min(x), torch.max(x)
    scale = (hi-lo)/255
    q = torch.round((x-lo)/scale)
    x_hat = lo+q*scale
    return q, lo, scale, x_hat

def quantize_model(model, quantize_mlp=False):
    for grid in model.grid.grids:
        q, lo, scale, x_hat = quantize_uint8(grid.data)
        with torch.no_grad():
            grid.data.copy_(x_hat)

image_height = image.shape[0]
image_width = image.shape[1]
total_pixels = image_height*image_width
pixel_centers_x = (torch.arange(image_width) + 0.5)/image_width
pixel_centers_y = (torch.arange(image_height) + 0.5)/image_height
coords = torch.cartesian_prod(pixel_centers_y, pixel_centers_x)
target = torch.flatten(image, start_dim=0, end_dim=1)

device = get_device()
#print("device : " + device)

model = NeuralTexture().to(device)

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
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} gradient mean: {param.grad.abs().mean().item()}")
    #     else:
    #         print(f"❌ {name} has NO GRADIENT!")

coords = coords.to(device)
target = target.to(device)
print("PSNRs (taken every 200 epochs):")
print(PSNRs)

model.eval()
with torch.no_grad():
    output = model(coords)
    loss = criterion(output, target)
if (config['quantize']): 
    quantize_model(model)
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
