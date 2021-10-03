
# ------------------------------------------------------------------------
#
#   MultiDiffusion for Automatic1111 WebUI
#
#   Introducing a revolutionary large image drawing method - MultiDiffusion!
#
#   Techniques is not originally proposed by me, please refer to
#   Original Project: https://multidiffusion.github.io
#
#   The script contains a few optimizations including:
#       - symmetric tiling bboxes
#       - cached tiling weights
#       - batched denoising
#       - prompt control for each tile (in progress)
#
# ------------------------------------------------------------------------
#
#   This script hooks into the original sampler and decomposes the latent
#   image, sampled separately and run weighted average to merge them back.
#
#   Advantages:
#   - Allows for super large resolutions (2k~8k) for both txt2img and img2img.
#   - The merged output is completely seamless without any post-processing.
#   - Training free. No need to train a new model, and you can control the
#       text prompt for each tile.
#
#   Drawbacks:
#   - Depending on your parameter settings, the process can be very slow,
#       especially when overlap is relatively large.
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes UNet.
#
#   How it works (insanely simple!)
#   1) The latent image x_t is split into tiles
#   2) The tiles are denoised by original sampler to get x_t-1
#   3) The tiles are added together, but divided by how many times each pixel
#       is added.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-03
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------


import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

from modules import sd_samplers, images, devices, shared, scripts, prompt_parser, sd_samplers_common
from modules.shared import opts, state
from modules.sd_samplers_kdiffusion import CFGDenoiserParams


class MultiDiffusionDelegate(object):
    """
    Hijack the original sampler into MultiDiffusion samplers
    """

    def __init__(self, sampler, sampler_name, steps, 
                 w, h, tile_w=64, tile_h=64, overlap=32, tile_batch_size=1, 
                 tile_prompt=False, prompt=[], neg_prompt=[], 
                 controlnet_script=None, control_tensor_cpu=False):

        self.steps = steps
        # record the steps for progress bar
        # hook the sampler
        self.is_kdiff = sampler_name not in ['DDIM', 'PLMS', 'UniPC']
        if self.is_kdiff:
            self.sampler = sampler.model_wrap_cfg
            if tile_prompt:
                self.sampler_func = self.sampler.forward
                self.sampler.forward = self.kdiff_tile_prompt
                raise NotImplementedError("Tile prompt is not supported yet")
            else:
                # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
                # Otherwise, the masked-redraw will break due to the init_latent
                self.sampler_func = self.sampler.inner_model.forward
                self.sampler.inner_model.forward = self.kdiff_repeat
        else:
            self.sampler = sampler
            if tile_prompt:
                raise NotImplementedError("Tile prompt is not supported yet")
            else:
                self.sampler_func = sampler.orig_p_sample_ddim
                self.sampler.orig_p_sample_ddim = self.ddim_repeat

        # initialize the tile bboxes and weights
        self.w, self.h = w//8, h//8
        if tile_w > self.w:
            tile_w = self.w
        if tile_h > self.h:
            tile_h = self.h
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        if overlap < 0:
            overlap = 0
        self.tile_w = tile_w
        self.tile_h = tile_h
        bboxes, weights = self.split_views(tile_w, tile_h, overlap)
        self.batched_bboxes = []
        self.batched_conds = []
        self.batched_unconds = []
        self.num_batches = math.ceil(len(bboxes) / tile_batch_size)
        optimal_batch_size = math.ceil(len(bboxes) / self.num_batches)
        self.tile_batch_size = optimal_batch_size
        self.tile_prompt = tile_prompt