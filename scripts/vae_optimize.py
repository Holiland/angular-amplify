
# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - Giant RAM needed. To store the intermediate results for a 4096x4096
#       images, you need 32 GB RAM it consumes ~20GB); for 8192x8192
#       you need 128 GB RAM machine (it consumes ~100 GB)
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE