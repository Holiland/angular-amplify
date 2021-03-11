
# MultiDiffusion with Tiled VAE

English｜[中文](README_CN.md)

This repository contains two scripts that enable **ultra-large image generation**.

- The MultiDiffusion comes from existing work. Please refer to their paper and GitHub page [MultiDiffusion](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/multidiffusion.github.io)
- The Tiled VAE is my original algorithm, which is **very powerful** in VRAM saving

## Update on 2023.3.7

- Added Fast Mode for Tiled VAE, which increase the speed by 5X and eliminated the need for extra RAM.
- Now you can use 16GB GPU for 8K images, and the encoding/decoding process will be around 25 seconds. For 4k images, the process completes almost instantly.