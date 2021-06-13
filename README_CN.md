
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