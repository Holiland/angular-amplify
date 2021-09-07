
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