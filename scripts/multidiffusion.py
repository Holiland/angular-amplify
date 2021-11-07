
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
        for i in range(self.num_batches):
            start = i * tile_batch_size
            end = min((i + 1) * tile_batch_size, len(bboxes))
            self.batched_bboxes.append(bboxes[start:end])
            # TODO: deal with per tile prompt
            if tile_prompt:
                self.batched_conds.append(prompt_parser.get_multicond_learned_conditioning(
                    shared.sd_model, prompt[start:end], self.steps))
                self.batched_unconds.append(prompt_parser.get_learned_conditioning(
                    shared.sd_model, neg_prompt[start:end], self.steps))

        # Avoid the overhead of creating a new tensor for each batch
        # And avoid the overhead of weight summing
        self.weights = weights.unsqueeze(0).unsqueeze(0)
        self.x_buffer = None
        # For ddim sampler we need to cache the pred_x0
        self.x_buffer_pred = None
        self.pbar = None

        # For controlnet
        self.controlnet_script = controlnet_script
        self.control_tensor_batch = None
        self.control_params = None
        self.control_tensor_cpu = control_tensor_cpu

    @staticmethod
    def splitable(w, h, tile_w, tile_h, overlap):
        w, h = w//8, h//8
        min_tile_size = min(tile_w, tile_h)
        if overlap >= min_tile_size:
            overlap = min_tile_size - 4
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)
        return cols > 1 or rows > 1

    def split_views(self, tile_w, tile_h, overlap):
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        w, h = self.w, self.h
        cols = math.ceil((w - overlap) / non_overlap_width)
        rows = math.ceil((h - overlap) / non_overlap_height)

        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

        bbox = []
        count = torch.zeros((h, w), device=devices.device)
        for row in range(rows):
            y = int(row * dy)
            if y + tile_h >= h:
                y = h - tile_h
            for col in range(cols):
                x = int(col * dx)
                if x + tile_w >= w:
                    x = w - tile_w
                bbox.append((row, col, [x, y, x + tile_w, y + tile_h]))
                count[y:y+tile_h, x:x+tile_w] += 1
        return bbox, count

    def repeat_con_dict(self, cond_input, bboxes):
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = cond_input['c_concat'][0]
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for _, _, bbox in bboxes:
                image_cond_list.append(
                    image_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat(
                (len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}

    def kdiff_repeat(self, x_in, sigma_in, cond):
        def func(x_tile, bboxes):
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_con_dict(cond, bboxes)
            x_tile_out = self.sampler_func(
                x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out