---
title: 'My First Blog：凌晨折腾github，torch和ffmpeg'
date: 2024-09-26
permalink: /posts/2024/09/my-first-blog/
tags:
  - cool posts
  - category1
---

## Hello World And Hello GitHub!

首先搞半天终于有了自己的github blog，大概会不定时记一些踩坑心得，毕竟作为电脑苦手在各种配环境/配系统/python库等等这个那个的坑踩了无数。

**作为一个AI研究者，我实在没有精力去研究透彻每一个工具背后的逻辑，让一个算是搞数学的去完全弄懂ubuntu的各种抽象特性和python的各种抽象环境属实强人所难**，毕竟没人理解计算机系统，我也没参与pytorch开发，更不会写前端后端和小程序，只好按着最傻瓜的理解，跟着各种教程去用它最直接的功能。

然而这些工具有时候并不会如你心意，这种时候往往只能找遍网上教程挨个试一遍死马当活马医，在这个过程中花了无数的时间交学费。

这个博客大概就是记一下这些坑，目的一个是降低自己下次遇到这种破事的时间成本，第二个是也供大家参考~

请电脑高手们绕道，这是给我这种连安装deb包都需要查查dpkg命令怎么用，配环境只会`sudo apt-get install 和 pip install`的人看的。

## 三个坑，加班三小时

首先是github的主页，我是网上fork别人的模版，刚好他写了博客功能，我还没搞懂这玩意具体怎么操作，毕竟我不会写网页，等我什么时候弄明白了就把Talks Teaching Portfolio什么的都删掉，Guide我也不打算做，Pub里面我暂时没改，有空再说。
写博客比较简单，_post文件夹里面塞博客文件就好了，格式是markdown，非常友好。前面的metadata照抄填空就行。
但建议不要手贱去按他说的把.config.yml里面的future改成false，它会导致date为未来的所有blog不可见，然而抽象的是我并不知道它是按什么时区算的，反正9.26日凌晨写的在页面上没出现，我还以为是哪里格式出问题查了半天。

第二个是torch的向量type问题，这个大概是老生常谈了，不过也没有什么很好的讲的清楚的。首先如果你用to(device)把tensor转到GPU上了，它的type就变成了torch.cuda.xxx（你可以试试在GPU上和在CPU上输出它的type，是不一样的，所以不要转GPU以后随便再改它dtype什么什么）。
然后是一个很奇怪的观察，torch在CUDA上的uint（8,16,32..）类型居然连加法都不支持（？）你要是给他做加法它会报一个`RuntimeError: "ufunc_add_CUDA" not implemented for 'UInt16'`，总之就是CUDA上uint是不能运算的。
这听起来有点反直觉，但你就只能用int或者float，double，这让我一度以为安装pytorch又出了什么问题，然而没有。
jupyter爽是爽，但是如果运行出错中途退出显存并不会释放，得重启Jupyter内核才行。

最后一个是ffmpeg，本意是想用opencv的videowriter输出视频，但是一直报一个错
```
OpenCV: FFMPEG: tag 0x34363248/'H264' is not supported with codec id 27 and format 'mp4 / MP4 (MPEG-4 Part 14)'
```
查了一下，是opencv的videowriter编码器的问题，我之前的解码器一直都写着
```
cv2.VideoWriter_fourcc(*'H264')
```
本来按照GPT的写法，参数是'mp4v'，后来网上不知道在哪看到说用H264或者X264
然后就报错了
接下来我忘掉了这回事，网上基本上所有的blog都在说，直接安装的ffmpeg会因为一些许可证原因用不了libx264，就是H.264编码的库。得手动下载ffmpeg源码然后配置libx264库，用gcc编译安装
然后我就花了一个多小时在配这玩意，首先找一个容易上手的简单教程就不容易了
好不容易装上了一个，发现还是报这个错
这个时候大概会怀疑安装是否正确什么之类的？总之就是重装，各种配置试了半天，最后找到一篇解决的办法是——"mp4格式输出参数fourcc里面参数得写mp4v"
改了，跑了，有结果了，但至今我也不知道用apt-get安装的ffmpeg能不能用libx264，要是换回那个版本不能用我还得装一次，试错成本看起来有点高，暂时不打算做这个实验。
之前win下用的ffmpeg一直没出过这么多破事（）

## 今天你早睡了吗？

总之，如你所见，这个博客充满了破防、骂街、暴论等元素，以后的博客风格根据作者心态而定，请读者们注意分辨并提取有价值信息。

最后，配环境和debug本质上没什么区别，有时候实在弄不出来了就去睡觉，第二天早上突然试一种方法，它就成功了。所以还是先睡觉吧，晚安。
