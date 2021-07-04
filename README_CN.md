
# MultiDiffusion + Tiled VAE

[English](readme.md) | 中文

本存储库包含两个脚本，使用 [MultiDiffusion](multidiffusion.github.io) 和Tiled VAE（**原创方法**）处理**超大图片**。

- 第一个是前人已有的优秀工作。请参考他们的论文和网页。
- 第二个是我的原创算法。尽管原理上很简单，但非常强力，**让6G显存跑高清大图成为可能**

##  2023.3.7 更新

- Tiled VAE 添加了快速模式，**提升了五倍以上的速度并不再有额外的内存负担**
- 现在在16GB设备上只需要25秒左右就能编码+解码8K大图。4K图片时间几乎是立即完成。
- **如果您遇到VAE NaN 或者输出图像纯黑色：**
  - 使用官方提供的840000 VAE权重常常可以解决问题
  - 并且可以使用 --no-half-vae 禁用半精度 VAE

## MultiDiffusion

****

**快速超大图像细化（img2img）**

- **MultiDiffusion 特别擅长于大图像添加细节。**
  - **速度比Highres快一倍**，只要参数调整合适
  - 参数合适时，比SD Upscaler和Ultimate Upscaler产生更多的细节
- 食用提示：
  - **Checkpoint非常关键**
    - MultiDiffusion工作原理和普通highres.fix很相似，不过它是一块一块地重绘。因此checkpoint很重要
    - 一个好的checkpoint（例如在大图上训练的）可以为你的图像增加精致的细节
    - 一些朋友发现使用完整的checkpoint而不是pruned（修剪版）会产生更好的结果。推荐尝试。
  - **请不要使用含有具体物体的正面prompt**, 否则结果会被毁坏
    - 可以用类似这样的：masterpiece, best quality, highres, extremely clear, ultra-detailed unity 8k wallpaper
  - 你不需要太大的Tile尺寸否则结果会不精细，也不需要大量的步数，overlap也不宜过大，否则速度将会很慢。
    - Tile size=64 - 96, overlap=32 - 48，20 - 25步通常足够. 如果结果中出现缝隙再调大overlap。
  - **更高的CFG Scale（提示强度）可以显著地使图像更尖锐并添加更多细节**。需要配合合适的采样器。
    - 比如CFG=14，sampler=DPM++ SDE Karras或者Eular a
  - 你可以通过去噪强度0.1-0.6控制修改的幅度。越低越接近原图，越高差异越大。
  - 如果您的结果仍然不如我的那样细致，可以[参考我们的一些讨论](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/issues/3)
- 示例：
  - 参数：masterpiece, best quality, highres, extremely detailed, clear background, 去噪=0.4，步数=20，采样器=DPM++ SDE Karras，放大器=RealESRGAN, Tile size=96, Overlap=48, Tile batch size=8.
  - 处理前
  - ![lowres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/lowres.jpg?raw=true)
  - 处理后：4x放大，NVIDIA Tesla V100,
    - 总耗时 1分55秒，其中30秒用于VAE编解码。
    - 如果是2x放大仅需20秒
  - ![highres](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/highres.jpeg?raw=true)

****

**宽图像生成（txt2img）**

- **MultiDiffusion适合生成宽图像**，例如韩国偶像团体大合照（雾）
- txt2img 全景生成，与 MultiDiffusion 中提到的相同。
  - 目前所有局部区域共享相同的prompt。
  - **因此，请使用简单的正prompt以获得良好的结果**，否则结果将很差。
  - 我们正在加急处理矩形和细粒度prompt控制。
- 示例 - mastepiece, best quality, highres, city skyline, night

- ![panorama](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/city_panorama.jpeg?raw=true)

****

**与 ControlNet 配合**，产生具有受控内容的宽图像。

- 目前，虽然您不能使用复杂的prompt，但可以使用 ControlNet 完全控制内容。
- Canny edge似乎是最好用的，因为它提供足够的局部控制。
- 示例：22020 x 1080 超宽图像转换 - 清明上河图
  - Masterpiece, best quality, highres, ultra detailed 8k unity wallpaper, bird's-eye view, trees, ancient architectures, stones, farms, crowd, pedestrians