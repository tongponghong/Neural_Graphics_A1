# Neural Graphics A1 — Texture Compression
<!-- 
Two ways of compressing an RGB texture, compared on the same size-vs-quality axes:

1. **S3TC / DXT1** — classic fixed-rate block compression, 6:1 by construction.
2. **Neural texture** — a multi-resolution feature grid plus a small MLP, trained
   per image to reproduce the texture from UV coordinates.

Both are scored with PSNR against the original image and plotted together, so
the trade-off reads two ways: at a matched size, which method looks better, and
at a matched quality, which method is smaller.

--- -->
## Files

| File | Contents |
| --- | --- |
| `helpers.py` | Bilinear sampling, RGB565 packing, S3TC encode/decode, PSNR |
| `trainer.py` | `FeatureGrid`, `ColorMLP`, `NeuralTexture`, 8-bit grid quantization, `base_config()` |
| `main.py` | Per-image pipeline (`main_loop`), sweep driver (`main`), size-vs-PSNR plot (`size_vs_PSNR`) |
| `images/` | Inputs and every generated output (see below) |

## Directory layout

```
images/
  original_images/   # input PNGs (width and height must be multiples of 4)
  S3TC_output/       # <name>_S3TC_processed.png  — DXT1 reconstruction
  neural_output/     # <name>_neural_[res].png    — neural reconstruction per config
  plots/             # <name>_neural_[res]_graph.png — training PSNR curve
                     # <name>_size_vs_psnr.png       — per-image comparison plot
```

All output paths are derived from the input path, so these folders must exist
before running.

## Running

```bash
python main.py
```

`main()` runs at import time and globs an **absolute** path
(`/Users/tunger/neural_graphics/Neural_Graphics_A1/images/original_images/*.png`);
change it to run on another machine. The script picks CUDA, then MPS, then CPU.

**Dependencies:** PyTorch, torchvision, NumPy, Pillow, Matplotlib.

---

## Part 1 — S3TC / DXT1 (`helpers.py`)

**Sampling** 

`sample(u, v, image_array)` is a bilinear lookup with half-texel
centering and clamp-to-edge borders. `v` is bottom-up, so it is flipped before
indexing the top-down array.

**Tiling** 

`get_tiles` reshapes an `(H, W, 3)` image into
`(H/4, W/4, 4, 4, 3)` with one reshape and one `swapaxes`.

**Encoding a block** 

(`process_s3tc_tile`):
1. Compute the mean and 3×3 covariance of the 16 texels.
2. Take the eigenvector of the largest eigenvalue (`np.linalg.eigh`) as the
   principal color axis.
3. Project the texels onto it; the min and max projections give the two
   endpoint colors.
4. Quantize both endpoints to RGB565.
5. Build the 4-entry palette and give each texel the index of its nearest entry
   (squared Euclidean distance).

<!-- `S3TC` packs each block as `struct.pack('<HHI', c0, c1, idx_bits)` — 8 bytes:
two little-endian 565 endpoints, then sixteen 2-bit indices with texel `(x, y)`
at bits `2*(4y+x) .. 2*(4y+x)+1`. -->

**Decoding** 

`decode_S3TC` is vectorized over all blocks: it parses the byte
stream with a structured dtype, expands 565 → RGB888, builds both palette
variants and picks per block with `np.where(c0 > c1, …)`, extracts indices with a
broadcast shift, and gathers colors with `np.take_along_axis`.

**Size** 

8 bytes per 4×4 block:

```
s3tc_bytes = W * H / 2          # 6:1 against 24-bit RGB
```

---

## Part 2 — Neural texture (`trainer.py`)

**Model**  

`FeatureGrid` holds one learnable grid of shape `(1, feat_dim, R, R)`
per entry in `resolutions`. UVs are sampled from every grid with
`F.grid_sample` (bilinear, border padding) and concatenated, giving
`feat_dim × len(resolutions)` features per point. `ColorMLP` maps those to RGB
through two 64-unit ReLU layers and a sigmoid.

**Training** 

(`main_loop`): Coordinates are pixel centers in `[0, 1]`. Each step
draws `batch_size` random pixels, computes MSE against the target, and steps
Adam. Training PSNR is logged every 200 steps and saved as a curve.

**Quantization** 

`quantize_model` applies uniform affine 8-bit quantization to
each feature grid and writes the dequantized values back in place, so the
quantized PSNR reflects the rounding error while the model still runs in float.
The MLP stays float32.

**Size**

```
raw_bytes       = W * H * 3
neural_bytes    = (grid_params + mlp_params) * 4     # all float32
quantized_bytes = grid_params + mlp_params * 4       # 8-bit grids, float32 MLP
```

**Configuration**

```python
currConfig = {**trainer.base_config(), **setting}
```

Base config:
| Key | Default |
| --- | --- |
| `resolutions` | `(16, 32, 64, 128)` |
| `feat_dim` | `2` |
| `batch_size` | `16384` |
| `lr` | `0.01` |
| `epochs` | `2000` |
| `quantize` | `True` |

For different possible model sizes:
| `resolutions` | `feat_dim` |
| --- | --- |
| `[16, 32, 64, 128]` | 4 |
| `[16, 32, 64]` | 2 |
| `[64]` | 2 |

---

## Output

**Per run** 

Reconstructs texture, a training-PSNR curve, and console output
with the S3TC PSNR, final and quantized neural PSNR, and compression ratios.

**Per image** 

`plots/<name>_size_vs_psnr.png` (from `size_vs_PSNR`), with a
log-scale size axis:

- blue line — neural configs at float32, each point labelled with its
  compression factor
- green dashed line — the same configs with 8-bit grids
- red star — DXT1 at its fixed 6:1 point
<!-- 
--- -->

<!-- ## Known limitations

- **DXT1 mode isn't chosen by error.** For the same endpoint pair, `(hi, lo)`
  selects the 4-color ramp and `(lo, hi)` the 3-color + black ramp.
  `process_s3tc_tile` takes whichever order the eigenvector's arbitrary sign
  produced instead of scoring both. It also matches texels against the
  pre-quantization endpoints rather than the 565-dequantized ones the decoder
  uses, so indices aren't optimal for the reconstructed palette.
- **The two PSNRs aren't measured the same way.** S3TC uses `helpers.psnr` on
  uint8 arrays; the neural number is `-10·log10(mse)` on float predictions
  before clipping and casting to uint8. For a strict comparison, measure both
  on the final uint8 images.
- **`FeatureGrid` initializes to `torch.ones`,** so every UV initially maps to
  the same feature. Training breaks the symmetry, but small random init would
  converge faster.
- **`get_tiles` needs H and W divisible by 4** and assumes 3 channels. Alpha is
  sliced off in `main_loop`; other sizes would need padding.
- **`c565_to_rgb` uses a rounding approximation** (`(v*527+23)>>6`,
  `(v*259+33)>>6`) instead of bit replication. Encoder and decoder agree, but
  the output isn't guaranteed bit-exact with a hardware decoder.
- **Housekeeping:** `main()` has no `if __name__ == "__main__"` guard,
  `sample()` prints debug output, and the `NOT YET IMPLEMENTED` comments in
  `base_config()` are stale (both `image_out` and `quantize` are used). -->
