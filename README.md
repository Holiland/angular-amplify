
# MultiDiffusion with Tiled VAE

English｜[中文](README_CN.md)

This repository contains two scripts that enable **ultra-large image generation**.

- The MultiDiffusion comes from existing work. Please refer to their paper and GitHub page [MultiDiffusion](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/multidiffusion.github.io)
- The Tiled VAE is my original algorithm, which is **very powerful** in VRAM saving

## Update on 2023.3.7

- Added Fast Mode for Tiled VAE, which increase the speed by 5X and eliminated the need for extra RAM.
- Now you can use 16GB GPU for 8K images, and the encoding/decoding process will be around 25 seconds. For 4k images, the process completes almost instantly.
- If you encountered VAE NaN or black image output:
  - Use the OpenAI provided 840000 VAE weights. This usually solves the problem.
  - Use --no-half-vae on startup is also effective.

## MultiDiffusion

Note: [The latest sampler by Google](https://energy-based-model.github.io/reduce-reuse-recycle/) seems to achieve **theoretically** better results in local control than MultiDiffusion. We are investigating their differences and may provide an implementation in this repo.

****

**Fast ultra-large images refinement (img2img)**

- **MultiDiffusion is especially good at adding details to upscaled images.**
  - **Faster than highres.fix** with proper params
  - Much finer results than SD Upscaler & Ultimate SD Upscaler
- **How to use:**
  - **The checkpoint is crucial**. 
    - MultiDiffusion works very similar to highres.fix, so it highly relies on your checkpoint.
    - A checkpoint that good at drawing details (e.g., trained on high resolution images) can add amazing details to your image.
    - Some friends have found that using a **full checkpoint** instead of a pruned one yields much finer results.
  - **Don't include any concrete objects in your positive prompts.**  Otherwise the results get ruined.
      - Just use something like "highres, masterpiece, best quality, ultra-detailed unity 8k wallpaper, extremely clear".
  - You don't need too large tile size, large overlap and many denoising steps, or it can be slow.
    - Latent tile size=64 - 96, Overlap=32 - 48, and steps=20 - 25 are recommended. **If you find seams, please increase overlap.**
  - **CFG scale can significantly affect the details**, together with a proper sampler.
    - A large CFG scale (e.g., 14) gives you much more details. For samplers,I personally prefer Euler a and DPM++ SDE Karras.
  - You can control how much you want to change the original image with **denoising strength from 0.1 - 0.6**.
  - If your results are still not as satisfying as mine, [see our discussions here.](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/3)

- Example: 1024 * 800 -> 4096 * 3200 image, denoise=0.4, steps=20, Sampler=DPM++ SDE Karras, Upscaler=RealESRGAN++, Negative Prompts=EasyNegative
  - Before: 
  - ![lowres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/lowres.jpg?raw=true)
  - After: 4x upscale.
  - 1min12s on NVIDIA Testla V100. (If 2x, it completes in 10s)
  - ![highres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/highres.jpeg?raw=true)

****
