import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.io as io
import sys

config = {
    'texture': "images/gradient.png", 
    'resolutions': (16, 32, 64, 128), #sizes of feature grids
    'feat_dim': 2, #number of features per grid
    'batch_size': 16384,
    'lr': 0.01,
    'epochs': 2000, 
    'quantize': False #NOT YET IMPLEMENTED
    # Include other parameters as needed.
}

image = io.read_image(config['texture'], mode="RGB")
image = image.permute(1, 2, 0)
image = image / 255.0

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=config["resolutions"], feat_dim = config['feat_dim']):
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
    def __init__(self):
        super().__init__()
        self.grid = FeatureGrid()
        self.mlp = ColorMLP(self.grid.out_dim)
    def forward(self, uv):
        return self.mlp(self.grid(uv))

image_height = image.shape[0]
image_width = image.shape[1]
total_pixels = image_height*image_width
pixel_centers_x = (torch.arange(image_width) + 0.5)/image_width
pixel_centers_y = (torch.arange(image_height) + 0.5)/image_height
coords = torch.cartesian_prod(pixel_centers_y, pixel_centers_x)
target = torch.flatten(image, start_dim=0, end_dim=1)

device = get_device()
print("device : " + device)

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
    if (step % 400 == 0): print("running epoch " + str(step) + "/" + str(config['epochs']))
    loss.backward()
    optimizer.step()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} gradient mean: {param.grad.abs().mean().item()}")
    #     else:
    #         print(f"❌ {name} has NO GRADIENT!")
PSNRs.append(round(psnr.item(), 3))
print("PSNRs (taken every 200 epochs):")
print(PSNRs)