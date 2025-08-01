---
title: '扩散概率模型 Diffusion model'
date: 2025-07-29
permalink: /posts/2025/07/Diffusion/
tags:
  - learning note
---

---

# 从不同角度（去噪、变分推断、Score matching）理解扩散模型 Diffusion Model 
学习资料：
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Estimation of Non-Normalized Statistical Modelsby Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
- [由浅入深了解Diffusion Models](https://zhuanlan.zhihu.com/p/525106459)
- [深度推导DDPM数学原理](https://zhuanlan.zhihu.com/p/656757576)
- [论文学习笔记SDE：从EBM到Score-Matching，随机微分方程视角下的Diffusion大一统](https://zhuanlan.zhihu.com/p/702574549)
- [【学习笔记】深度生成模型（七）：能量函数模型EBM、Contrastive Divergence、朗之万动力学](https://zhuanlan.zhihu.com/p/743549038)
- [【中字】扩散模型背后的无名英雄：一门用概率解决几乎一切问题的学问](https://www.bilibili.com/video/BV1tXgQzNEAJ)
- [扩散模型与能量模型，Score-Matching和SDE，ODE的关系](https://zhuanlan.zhihu.com/p/576779879)

博客快写完了，发现有人已经做了类似的讨论
- [大一统视角理解扩散模型Understanding Diffusion Models: A Unified Perspective 阅读笔记](https://zhuanlan.zhihu.com/p/558937247)

## 1. DDPM

扩散概率模型实际上2015年就被提出来了。正向过程表示为一个不断添加高斯噪声的马尔可夫链，反向过程则是一个参数化的去噪过程。

整体的思路其实与VAE有共通之处——隐空间建模成了标准正态分布，而采用迭代的马尔可夫链取代了VAE的Encoder-Decoder架构，

本文将从两个角度理解扩散模型：一方面将你能搜索到的最常见的加噪-去噪过程的推导用简单的方式[推导一遍](#11-前向过程)，另一方面则会从[最原始的扩散模型](#13-最原始的扩散模型)出发，通过类似推导VAE的方式，找寻它背后的动机和根本原理。

论文原文提到
> 扩散模型的特定参数化方式在训练过程中揭示了与多噪声水平下score matching的等价性，而在采样过程中则与退火朗之万动力学等价。这些概念都是什么，暂且按下不表。这篇文章从原理层面按照个人理解的思路推导一下DDPM以及其改进。

如果想先了解这些概念，可以先看[相关内容](#3-相关内容)。


### 1.1 前向过程

令观测到的数据为 $x_0$ ，前向过程是一个迭代的马尔可夫链，逐步添加噪声。具体来说，设t为时间步，$\beta_t$ 为噪声系数，则

$$
q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

利用重参数化，可以写成以下形式
（关于什么是重参数化，参考上一篇讲VAE的文章）

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$


设 $\alpha_t=1-\beta_t $，则

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t\\
=\sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t} \epsilon_t + \sqrt{\alpha_t (1-\alpha_{t-1})} \epsilon_{t-1}\\
$$

由于两个服从正态分布的随机变量的线性组合仍然服从正态分布，即

$$
    \mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$
并且
$$
a \cdot \mathcal{N}(\mu, \sigma^2) = \mathcal{N}(a\mu, a^2\sigma^2)
$$

因此两项正态分布随机变量可以合并，均值仍然为0，方差为前面系数的平方和，可以重参数化为。

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\epsilon
$$
以此类推，设

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

可以很方便的写出通项

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$
即
$$
q(x_t \vert x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

直观上来说，$\beta_t$随着$t$的增大，逐渐从接近0到接近1，意味着噪声越来越大。这意味着
$$
\lim\limits_{t \to \infty} \bar{\alpha}_t =0\\
\lim\limits_{t \to \infty} q(x_t \vert x_0) = \mathcal{N}(x_t; 0, I)
$$

在实际操作中，$t$的上界取一个例如1000左右的值，$\alpha_t$则取一个线性或cosine衰减的序列。

### 1.2 反向过程（采样）

有了后验分布，我们尝试写出反向的采样过程 $p(x_{t-1} \vert x_t)$

利用贝叶斯公式

$$
p(x_{t-1} \vert x_t) = \frac{p(x_t \vert x_{t-1}) p(x_{t-1})}{p(x_t)}
$$

后验 $p(x_t\vert x_{t-1})$ 使用1.1中的（近似）后验（为什么是近似，在1.4节中会说明）$q$ 代替。但是每一轮隐变量的边缘分布 $p(x_t)$ 都无法根据已有条件计算。

直观上理解，如果不知道最终的采样方向，每一步的隐变量分布都是未知的——不知道该往什么方向去噪。

扩散模型通过祖先采样（即引入祖先$x_0$）来解决，假如添加一个条件$x_0$（后面会说明这一假设的合理性），则

$$
p(x_{t-1} \vert x_t, x_0) = \frac{q(x_t \vert x_{t-1}, x_0) p(x_{t-1} \vert x_0)p(x_0)}{p(x_t \vert x_0)p(x_0)} = \frac{q(x_t \vert x_{t-1}) q(x_{t-1} \vert x_0)}{q(x_t \vert x_0)}
$$

注意到$q(x_t \vert x_{t-1})$与条件 $x_0$ 无关。而原本无法计算的隐变量边缘分布也可以通过近似后验$q$来描述了。

考虑$p(x_{t-1}\vert x_t, x_0)$概率密度函数的形式，右边是正态分布概率密度的乘积，不考虑归一化常数的情况下，就变成指数的加减法，因此

$$
p(x_{t-1} \vert x_t, x_0) \propto \exp\left(-\frac{1}{2} \left({(x_t-\sqrt\alpha_t x_{t-1})^2 \over 1-\alpha_t} + {(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}} x_0)^2 \over 1-\bar{\alpha}_{t-1}} - {(x_t-\sqrt{\bar{\alpha}_t} x_0)^2 \over 1-\bar{\alpha}_t}\right) \right)
$$

可以将只与 $x_t,x_0$ 相关而与 $x_{t-1}$ 无关的项视为常数，再进行配方，一定可以写成如下形式（推导过程略）

$$
p(x_{t-1} \vert x_t, x_0) \propto \exp\left(-\frac{1}{2} \left({(x_{t-1} - \mu_{t}(x_0,x_t))^2 \over \sigma_{t}^2(x_0,x_t)}\right)\right)
$$

其中**均值和方差**可以写成
$$
\mu_{t}(x_0,x_t) = {\sqrt \alpha_t(1-\bar\alpha_{t-1})\over 1-\bar\alpha_t}x_t + {\sqrt{\bar{\alpha}_{t-1}}\beta_t \over 1 - \bar{\alpha}_t} x_0\\
\sigma_{t}^2(x_0,x_t) =\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t = \sigma_t^2
$$

方差与$x_0,x_t$均无关

因此已知$x_t,x_0$前提下，这也是一个正态分布。

等等。这里仍然有$x_0$，如果都知道$x_0$了，还采什么样呢？还需要进一步修改。

回想起前面的前向过程，$x_t$可以用重参数化写成$x_0$与服从标准正态分布的噪声$\epsilon$的线性组合。反过来也是一样的，即

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon\right)
$$

代入前面均值和方差的表达式，整理可得

$$
\mu_t(x_t)= {1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon\right)
$$

$\epsilon$ 是正向加噪得到 $x_t$ 时从标准正态分布中采样的噪声。在去噪时要找到对应的那个噪声，而不是随便再采一个。

最直接简单的思路就是：用神经网络来预测这个逐像素的噪声，输入是$x_t$和时间步$t$，即

$$
\epsilon \approx \epsilon_\theta(x_t, t)
$$

代入即可得到$x_{t-1}$服从的正态分布的均值，对其采样即可得到$x_{t-1}$。再利用重参数化技巧使得采样过程可微，就有

$$
x_{t-1} = {1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\frac{(1 - \bar{\alpha}_{t-1})\beta_t}{1 - \bar{\alpha}_t}} z, \quad z \sim \mathcal{N}(0, I)
$$

如果你不关心更深层的动机和理论原理，而只是想简单的使用它，可以跳到[训练](#15-训练)部分。

但是在这里，我们仍然需要回答一个问题：

- 既然已经直接预测噪声$\epsilon$了，为什么不能直接通过公式算出$x_0$？还要大费周章一步步迭代采样？

预测服从高斯分布的噪声，自然使用的是L2损失。

问题实际上是：噪声的最大似然，是否等价于生成$x_0$的最大似然？直接从公式算出的$x_0$，和逐步采样出的$x_0$，区别在哪？

**后面的推导，我们将推倒之前理解它的方式，从定义出发，重新审视扩散模型。**

### 1.3 以变分推断的视角重新审视扩散模型

扩散概率模型，是一种隐变量模型。如果VAE是直接将观测数据和隐变量使用参数化的Encoder-Decoder架构联系起来，那么扩散模型则固定了观测数据和隐变量之间关系的形式——一个马尔可夫链，除了$x_0$以外，其余的$x_t$都可以视为隐变量。


下面的讨论中，我们确认几个概念：
- $x_0$ 是观测数据
- $x_1\cdots x_T$是隐变量
- $x_0\to x_T$的过程，即后验分布$p(x_{t+1}\vert x_t)$，称为**前向过程**。
- $x_T\to x_0$的过程，即条件似然$p(x_t\vert x_{t+1})$，称为**逆向过程**或者**生成过程**。

边缘似然函数可以写成

$$
p_\theta(x_0) = \int p_\theta(x_0, x_1, \ldots, x_T) dx_1 \ldots dx_T = \int p_\theta(x_{0:T}) dx_{1:T}
$$

在这个模型下，观测数据是通过一个马尔可夫链生成的，即逆向过程：

$$
p_\theta(x_{t_0}\vert x_{T}) = p_\theta(x_T)\prod_{t=t_0+1}^T p_\theta(x_{t-1} \vert x_t)
$$

其中 $x_T\sim\ \mathcal{N}(0,I)$。并且转移是一个高斯分布，即

$$
p_\theta(x_{t-1} \vert x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma^2_\theta(x_t,t))
$$

这些高斯分布参数是未知的，我们想要优化 $\theta$，使得边缘似然尽可能大。然而边缘似然函数是不可计算的，因为多个隐变量的高维积分算不出来。
$$
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T} = \log \int p_\theta(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \vert x_t) dx_{1:T}
$$

真实的前向过程也算不出来

$$
p_\theta(x_t \vert x_0) = \frac{p_\theta(x_0 \vert x_t) p_\theta(x_t)}{p_\theta(x_0)}
$$

采用变分推断，引入变分分布 $q(x_t \vert x_0)$（**对前向过程进行近似**）

与VAE中使用神经网络参数化的变分分布不同的是，扩散模型的变分分布被定义为一个具有高斯分布形式转移的马尔可夫链，具体形式已经在1.1中详细描述过了。

优化目标转变为最大化证据下界ELBO：(具体为什么同样参考上一篇VAE的文章，**优化ELBO意味着联合优化生成过程的参数$\theta$和变分分布q（使得近似前向和真实前向更接近）**)

$$
\mathcal{L} = \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log {p_{\theta}(x_0 \vert x_{1:T})p_\theta(x_{1:T})\over q(x_{1:T}\vert x_0)}\right] =  \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log p_\theta(x_T) + \sum\limits_{t=1}^T \log {p_\theta(x_{t-1} \vert x_t)\over q(x_{t} \vert x_{t-1})}  \right]
$$

这里是将联合概率利用马尔可夫链的特性展开。

使用贝叶斯公式（对前向近似，可以推出这个对应近似出来的逆向是什么），同时和之前的技巧一样，引入$x_0$，有

$$
{1 \over q(x_t\vert x_{t-1})} = {q(x_{t-1}\vert x_0)q(x_0) \over q(x_{t-1}\vert x_t,x_0)q(x_t\vert x_0)q(x_0)}
$$

$q(x_0)$ 可以消去，而求和式可以裂项消去 $q(x_{t-1}\vert x_0)\over q(x_t\vert x_0)$

因此ELBO可以写成（注意$q(x_0\vert x_0)=1$）

$$
= \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log {p_\theta(x_T)\over q(x_T\vert x_0)} + \sum\limits_{t=2}^T \log {p_\theta(x_{t-1} \vert x_t)\over q(x_{t-1} \vert x_{t},x_0)} + \log p_\theta(x_0\vert x_1)\right]
$$

期望下分布部分只需要考虑相关的变量，与内部不相关的边缘积出来都是1。取负把对数内分子分母颠倒以后，可以发现构成KL散度的形式，即

$$
=-\left[D_{KL}(q(x_T\vert x_0)\vert\vert p_\theta(x_T)) + \sum\limits_{t=2}^T D_{KL}(q(x_{t-1}\vert x_t,x_0)\vert\vert p_\theta(x_{t-1}\vert x_t)) -\mathbb E_{q(x_{1:T}\vert x_0)}\log p_\theta(x_0\vert x_1)\right]
$$

上面是论文的公式。实际上在裂项的时候，可以把最右边的一项与中间的项合并，得到

$$
= \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log {p_\theta(x_T)\over q(x_T\vert x_0)} + \sum\limits_{t=1}^T \log {p_\theta(x_{t-1} \vert x_t)\over q(x_{t-1} \vert x_{t},x_0)}\right]
$$

实际上最后一项在计算KL散度的时候就没了，因此可以直接改成求和下界从1开始。

就变成最小化（取负号最大化）

$$
D_{KL}(q(x_T\vert x_0)\vert\vert p_\theta(x_T)) + \sum\limits_{t=1}^T D_{KL}(q(x_{t-1}\vert x_t,x_0)\vert\vert p_\theta(x_{t-1}\vert x_t))
$$

注意到这里的KL散度发生了奇妙的变化。在先前，最大化ELBO等价于最小化变分分布与后验分布（**即近似前向和真实前向**）的KL散度。而在这里，需要最小化的是**近似逆向和真实逆向**的KL散度，加上前面一个什么东西。
- 可以细品与VAE推出的损失函数的区别。但是共同点很明显，思路都是最小化变分与后验KL—>最大化ELBO->最小化另一个好算的KL加上什么东西。

到了这里就比较好算了。前面那一项实际上与 $\theta$ 无关（因为 $q(x_T\vert x_0)$ 是固定的，而 $p_\theta (x_T)$ 是个标准正态），因此可以忽略。

第二项前半部分，近似在1.2中已经推导过一遍了，是一个均值与 $x_0,x_t$ 相关的正态分布。后半部分则是最开始定义的真实逆向过程，也是正态分布，但参数未知。

再简单回顾一下，近似逆向的过程是

$$
q(x_{t-1} \vert x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_t(x_0,x_t), \sigma_t^2 I)
$$

$$
\mu_{t}(x_0,x_t) = {\sqrt \alpha_t(1-\bar\alpha_{t-1})\over 1-\bar\alpha_t}x_t + {\sqrt{\bar{\alpha}_{t-1}}\beta_t \over 1 - \bar{\alpha}_t} x_0\\
\sigma_{t}^2 =\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t
$$

在近似过程中，我们推出了$x_0$和$x_t$有明确的关系

$$
    x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt {(1 - \bar{\alpha}_t)}\epsilon ,\quad \epsilon \sim \mathcal {N}(0,I)\\
    x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon\right)
$$

所以近似逆向过程中，均值可以只与$x_t$相关

$$
\mu_t(x_t) = {1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon\right)
$$

而方差与$x$无关，那么$\Sigma^2_\theta$完全可以就取$\sigma^2_tI$。高斯分布的KL散度是有封闭形式的，而在方差一致的条件下，最小化KL散度等价于最小化均值的平方差，即

$$
    \arg\min_\theta D_{KL}(q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t)) = \arg\min_\theta\frac{1}{2\sigma^2_t}\vert\vert\mu_t(x_t) - \mu_\theta(x_t,t)\vert\vert_2^2
$$


那么整个算法就呼之欲出了，对于真实的未知参数反向过程，利用参数化的神经网络预测均值，使得在训练过程中均值尽可能接近近似逆向算出的均值。

而预测均值又可以利用上面的公式改为预测噪声 $\epsilon$，即最小化 $\vert\vert \epsilon_\theta(x_t,t) - \epsilon \vert\vert_2^2$。

这样我们就训练出了真实的逆向过程 $p_\theta(x_{t-1}\vert x_t)$ 的参数。已知了高斯分布的均值和方差，就可以用重参数化采样了

$$
p_\theta(x_{t-1}\vert x_t) =\mathcal {N}\left(x_{t-1};\mu_\theta(x_t,t), \sigma^2_t I\right) \\
x_{t-1}={1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\frac{(1 - \bar{\alpha}_{t-1})\beta_t}{1 - \bar{\alpha}_t}z, \quad z \sim \mathcal{N}(0, I)
$$

现在大概能够回答[1.2节的末尾](#13-重新审视扩散模型)的那个问题了——我们训练的网络到底是个什么？训出来的是真实的参数化生成过程。引入变分分布只是希望使得参数化生成过程变得可优化的同时尽可能近似真实的算不出来的后验分布。也就是说，我们从头到尾都不知道真实情况的后验分布（正向过程）到底长成什么样，正向加噪只是变分近似。

而如果在采样过程中，预测出噪声就直接用加噪公式反推出$x_0$，这并不是真实的生成过程，而只是近似的变分加噪的逆向过程。真实的生成过程是被定义为参数化的高斯转移，只是利用重参数化技巧后，体现为“去噪”。

### 1.4 训练与生成

有了前面的推导，训练过程其实非常简单。
- 从数据集中采样$x_0$
- 从1到T均匀采样$t$
- 采样噪声$\epsilon \sim \mathcal{N}(0, I)$
- 计算加噪后的$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
- 计算网络输出的噪声 $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
- 计算损失 $\mathcal{L} = \frac{1}{2} \mathbb{E}_{t} \left[ \left\| \hat{\epsilon} - \epsilon \right\|^2 \right]$

生成过程也非常简单，从标准正态分布中采样$x_T$，然后从$t=T$到$t=0$迭代采样
$$
x_{t-1} = {1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\frac{(1 - \bar{\alpha}_{t-1})\beta_t}{1 - \bar{\alpha}_t}} z, \quad z \sim \mathcal{N}(0, I)
$$

## 2. 从另一个角度看DDPM

### 2.1 基于能量的模型 Energy-based Models

传统的前向模型可以视为一个参数化的显式函数$y=y_\theta(x)$，其中x是模型输入，y是模型的输出。

而能量模型通过一个隐函数来定义能量函数$E_\theta(x,y)$，通过最小化能量函数来将前向估计问题转变为优化问题，即

$$
y(x)=\arg\min\limits_{x} E_\theta(x,y)
$$


> 这并不是什么新的发明，而是LeCun在理论上进行统一的一次尝试。（把生成模型和判别模型统一起来）

如果能量函数是可微的，我们可以尝试使用梯度下降等优化方法来优化出$y$。

相比于确定的前向模型，概率模型输出的不是一个确定的值，而是建模了条件分布 $p_\theta(y \vert x)$），它可以视为能量模型的一个特例——通过玻尔兹曼分布，能量模型能够转化为概率模型。玻尔兹曼分布用来描述空间中某种势能下的粒子分布：

$$
p(x) = \propto \exp(-E(x))
$$

这里x可以看作在空间中的位置，能量低的区域，粒子分布密度高。应用到能量模型上，就是把y,x视为同一片数据空间中的不同维度。联合概率分布可以写成

$$
p_\theta(x,y) = {\exp(-E_\theta(x,y))\over \int \exp(-E_\theta(x',y')) dx'dy'}
$$

能量模型中，x,y其实并没有什么本质的差别，都只不过是空间中的某个/某些维度。我们需要哪个就提取哪个。如果把$X=(x,y)$视为观测数据，那么它描述了一个生成模型。如果把$x$视为输入，$y$视为输出，那么它描述了一个判别模型，条件概率可以写成

$$
p_\theta(y \vert x) ={\exp(-E_\theta(x,y))\over \int \exp(-E_\theta(x,y')) dy'}
$$

前面VAE和Diffusion的讨论中我们都提到过，分母归一化常数需要在高维空间中积分，往往是不可计算的。能量模型想做的就是绕开这个归一化常数。

我们在这里主要关注它是如何描述生成模型的（方便起见，在后面我们将统一$x=(x,y)$代表整个样本）。这就涉及到两个问题：
- 假如有一个样本集合$D=\{(x_i)\}$，如何训练能量模型的参数$\theta$？
- 如果把能量模型看作一个生成模型，怎么从其中按照概率分布采样？

对于第一个问题，自然想到的是最大似然估计。对数边缘似然函数可以写成

$$
\log p_\theta(x) =  - E_\theta(x) - \log \int \exp(-E_\theta(x')) dx' =- E_\theta(x) - \log Z_\theta
$$

VAE和Diffusion中解决这个问题的方式是引入变分分布。这里则简单粗暴一些，考虑直接计算对数似然的梯度（最大化要梯度上升，假设概率分布是可微的）：

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta \\
= -\nabla_\theta E_\theta(x) - \frac{\nabla_\theta Z_\theta}{Z_\theta}\\
= -\nabla_\theta E_\theta(x) - \int {\exp (-E_\theta(x'))\over Z_\theta}(-\nabla_\theta E_\theta(x'))  dx'\\
= -\nabla_\theta E_\theta(x) - \int p_\theta(x')(-\nabla_\theta E_\theta(x')) dx'\\
= -\nabla_\theta E_\theta(x) - \mathbb{E}_{p_\theta(x')}\left[-\nabla_\theta E_\theta(x')\right]
$$

成功地把难以计算的归一化常数变成了能量函数梯度的一个期望。

> 插一句嘴，把概率取对数的好处除了把乘法变加法便于计算外，在梯度下降中还有很好的性质。概率函数取值范围必然在[0,1]之间，概率密度也一定是非负的，那么很有可能在某些区域密度值接近0时，梯度非常的小，直接梯度上升/下降半天都走不出来。
> 但是$\nabla \log f = \nabla f/f$，相当于把f的梯度值按照f的真实值归一化了。避免了这一问题

先前有过很多相关的工作研究如何计算梯度。Hinton提出的CD算法（Contrastive Divergence）是其中一个比较有名的算法。它的核心思想是通过蒙特卡洛方法近似期望。在 $p_\theta(x')$ 中用Gibbs方法采样出一个$x_{sample}$，从而

$$
\nabla_\theta \log p_\theta(x) \approx -\nabla_\theta E_\theta(x) + \nabla_\theta E_\theta(x_{sample})
$$

就可以进行梯度上升了。

### 2.2 Langevin采样

第二个问题是，我们如何从能量模型中采样？或者说，如何在某个概率分布$p_\theta(x)$上采样？

朗之万采样（Langevin sampling），也叫朗之万动力学，提供了一种只需要知道概率分布对数密度的梯度就可以采样的方式。关于朗之万动力学的直观理解可以参考[这个视频](https://www.bilibili.com/video/BV1tXgQzNEAJ)。证明可以在别的地方找到。

简单来说，朗之万采样是这样进行的：从任意分布初始化$x_0$，然后迭代更新直到收敛（或者更新一个比较大的步数，例如1000次）：

$$
x_{t+1} = x_t - {\sigma^2\over 2}\nabla_x \log p_\theta(x_t) + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
x_{\infty}\to x^* \sim p_\theta(x)
$$

可以把这个过程看作是在能量模型的潜在空间中进行的随机梯度下降。如果没有后面的噪声项，最终会下降到分布的峰值。参数$ \sigma $可以类比为下降的“学习率”。

这个添加了噪声项的梯度过程似乎有些熟悉：
- 随机梯度下降
- 扩散模型的反向采样过程
- ...

在上面的过程中，我们只需要知道 $\nabla_x \log p_\theta(x)$ 就可以采样了，而

$$
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x) - \nabla_x \log Z_\theta = -\nabla_x E_\theta(x)
$$

后面的等号是因为$Z_\theta$是一个与$x$无关的常数，梯度为0。因此我们只需要能量函数关于$x$的负梯度。如果用神经网络来拟合能量函数，神经网络是可微的，我们可以直接反向传播计算对于能量对于输入的梯度。

在实际使用中，某些分布可能大部分梯度都集中在某些区域，而初始化的$x_0$可能会处在一个梯度很小，远离峰值的位置。因此我们往往不一定会使用实际分布，而是近似一个性质比较好的分布，想要分布尽量平滑一些。

### 2.3 Score Matching

回到刚才最大似然的过程。总觉得训练估计 $\theta$ 的时候，每次都得采样一个$x_{sample}$，有点麻烦（实际上单纯的EBM效果也并不好），最后训出来的能量网络，我们也不需要它的值，而是需要它关于输入的梯度。能不能直接绕过能量函数，直接用网络拟合这个梯度呢？

这就是Score Matching的思路。它的核心思想是直接最小化对数似然的梯度与真实数据分布的梯度之间的差异。这个梯度被命名为得分函数（score function）即

$$
s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

类似于变分推断，我们想要尽可能让这个得分函数与真实数据分布的得分函数接近。它通过最小化均方差损失来实现。

$$
\mathbb{E}_{x \sim p_{data}(x)}\left[\vert\vert s_\theta(x) - \nabla_x \log p_{data}(x)\vert\vert_2 ^2\right] = D_{F}(p_{data}(x) \vert\vert p_\theta(x))
$$

这被称为费舍尔散度（Fisher Divergence）。与KL散度不同，它从梯度的角度衡量两个分布之间的差异。

很明显 $D_{F}(p_{data}(x) \vert\vert p_\theta(x)) \geq 0$，而[原论文](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)（Theorem 2）证明了当且仅当 $p_\theta = p_{data}$ 时取到0。

尽管如此，我们并不清楚真实数据对数似然的梯度。一种方案是用核函数估计，即把真实分布视为每个样本点的核函数（例如高斯核）的加权和，这里不再赘述。

尝试进一步改写 $D_F$ 。

$$
D_{F}(p_{data}(x) \vert\vert p_\theta(x)) = \int \left(\vert\vert s_\theta(x) \vert\vert_2^2 + \vert\vert \nabla_x \log p_{data}(x)\vert\vert_2 ^2 -2s_\theta(x) \odot \nabla_x \log p_{data}(x)\right) dx
$$

这里要明确的是，得分是能量函数的负梯度，是与 $x$ 相同维数的向量，$\odot$ 表示点积。

$\vert\vert \nabla_x \log p_{data}(x)\vert\vert ^2$ 项与 $\theta$无关，优化时可以删去。

对于点积部分，继续拆解

$$
s_\theta(x) \odot \nabla_x \log p_{data}(x) = s_\theta(x) \odot {\nabla_x p_{data}(x) \over p_{data}(x)}
$$

因此这部分积分可以写成
$$
\int 2p_{data}(x) s_\theta(x) \odot \nabla_x \log p_{data}(x) dx = 2\int s_\theta(x) \odot \nabla_x p_{data}(x) dx
$$

这里使用分部积分，只不过是多维版本，需要逐维度计算，形式是类似的。严格证明在原论文中有阐述。

$$
\int s_{\theta,i}(x)\nabla_{x_i} p_{data}(x) dx = \int \left(\int \nabla _{x_i} (s_{\theta,i}(x)p_{data}(x)) dx_i\right)dx_0\cdots dx_{i-1}dx_{i+1} dx_n - \int p_{data}(x)\nabla_{x_i} s_{\theta,i}(x) dx
$$

$x$ 的积分范围是整个空间（负无穷到正无穷），原论文对概率分布做了基本的假设，当$x$的任意一维趋于$+\infty$或$-\infty$时，$s_\theta(x)\odot p_{data}(x)\to 0$（这符合我们的直觉）。因此对其梯度积分结果是0。而后一个积分项又变成了期望的形式。

所以最终我们有
$$
D_F(p_{data}(x) \vert\vert p_\theta(x)) = \mathbb{E}_{x \sim p_{data}(x)}\left[\vert\vert s_\theta(x) \vert\vert_2^2 + 2\sum_i\nabla_{x_i} s_{\theta,i}(x)\right]
$$

得分的梯度实际上是$p$的二阶导，实际上是一个Hessian矩阵，而在这里我们只关心对角线的和（矩阵的迹）

$$
=\mathbb{E}_{x \sim p_{data}(x)}\left[\vert\vert s_\theta(x) \vert\vert_2^2 + 2\sum_i{\partial^2 \log p_{data}(x)\over \partial x_i^2}\right]
$$

真实数据实际上就符合$p_{data}$分布，样本点本身可以视为一个采样。然而这个二阶导计算起来非常不方便。

还有一些工作尝试通过其他方式来估计得分函数，例如Song Yang的Slice Score Matching。

### 2.4 在Score Matching中引入噪声

之前提过，在数据密度低的区域，直接估计得分函数是非常困难的（要么是0取对数无意义，要么几乎没有梯度，要么低密度区域缺乏监督信号）。Denoising Score Matching（DSM）提出了一种新的方法来解决这个问题——向数据分布中添加高斯噪声。

一方面，高斯分布在整个空间都有值，填补了数据分布的空白区域；另一方面，添加噪声后，数据分布变得平滑（扩展了那些峰值区域），使得低密度区域也能够做监督。

>另一个问题仍然存在：我们如何为扰动过程选择合适的噪声大小？较大的噪声显然可以覆盖更多的低密度区域，以获得更好的分数估计，但它过度损坏了数据，并使其与原始分布发生了重大变化。另一方面，较小的噪音会减少原始数据分布的损坏，但并不能像我们希望的那样覆盖低密度区域。

先前已经有了一些基本的思路，而根据Generative Modeling by Estimating Gradients of the Data Distribution，解决方法是采用了多个不同大小高斯分布噪声，把这些噪声分别加到数据分布上。

具体来说，有一列均值为0，各向同性且独立的高斯分布

$$
\{\mathcal {N}(0, \sigma_i^2 I),i=1,\cdots,T\}
$$

对应于一列加噪后的数据分布

$$
p_{\sigma_i}(x) = \int p_{data}(x') \mathcal{N}(x; x', \sigma_i^2 I) dx'
$$

采样过程可以写成

$$
x' \sim p_{data}(x'), \quad x' = x + \sigma_i z, \quad z \sim \mathcal{N}(0, I)
$$

而此时我们需要对所有的加噪后的分布估计得分，输入除了包括$x$，还包括噪声的大小/下标。目标是最小化所有噪声水平下的Fisher散度总和

$$
\sum\limits_{i}\lambda(i) D_{F}(p_{\sigma_i}(x) \vert\vert p_\theta(x))=\sum\limits_{i}\lambda(i) \mathbb{E}_{x \sim p_{\sigma_i}(x)}\left[\vert\vert s_\theta(x, i)- \nabla_x \log p_{\sigma_i}(x)\vert\vert_2^2\right]
$$

权重 $\lambda(i)$ 通常被定为 $\sigma_i^2$

然而我们尽管能够从 $p_{\sigma_i}(x)$ 中获得样本，但求不出它的表达式。

类似Diffusion反向过程，这里也可以使用祖先采样，把直接从 $p_{\sigma}$ 中采样拆分成两步，引入一个 $x \sim p_{data}(x)$ 作为条件，期望可以拆分

$$
p_{\sigma_i}(x) = \int p_{data}(x') p_{\sigma_i}(x \vert x') dx'
$$

那么有

$$
\mathbb{E}_{p_{\sigma_i}(x)}\left[\vert\vert s_\theta(x, i) - \nabla_x \log p_{\sigma_i}(x)\vert\vert_2^2\right] \\
= \int p_{\sigma_i}(x) \vert\vert s_\theta(x, i) - \nabla_x \log p_{\sigma_i}(x)\vert\vert_2^2 dx \\
= \int\int p_{data}(x)p_{\sigma_i}(x'\vert x) \mathbb{E}_{ p_{\sigma_i}(x'\vert x)}\left[\vert\vert s_\theta(x', i) - \nabla_{x'} \log p_{\sigma_i}(x'\vert x)\vert\vert_2^2\right] dx'dx \\
=\mathbb {E}_{p_{data}(x)}\left[\mathbb{E}_{ p_{\sigma_i}(x'\vert x)}\left[\vert\vert s_\theta(x', i) - \nabla_{x'} \log p_{\sigma_i}(x'\vert x)\vert\vert_2^2\right]\right]
$$

而条件分布就是高斯分布，梯度是可以直接求出来的

$$
\nabla_{x'} \log p_{\sigma_i}(x'\vert x) = -\frac{(x' - x)}{\sigma_i^2}
$$

所以最终优化的目标函数就是

$$
\sum\limits_{i}\sigma_i^2 \mathbb{E}_{p_{data}(x)}\left[\mathbb{E}_{ p_{\sigma_i}(x'\vert x)}\left[\vert\vert s_\theta(x', i) + \frac{(x' - x)}{\sigma_i^2}\vert\vert_2^2\right]\right]
$$

现在样本集中采样，再从对应的噪声分布中采样$x'$，就可以计算损失了。

训练估计除了得分函数后，我们还需要使用这个多噪声的模型来生成数据，生成数据仍然使用了Langevin采样。

$$
x_{i+1} = x_i - {\sigma^2\over 2}s_\theta(x_i, i) + \sigma_i z, \quad z \sim \mathcal{N}(0, I)
$$

在先前的Langvin采样中，没有对噪声的规模（方差）做出规定。在这里，采样所添加的噪声的规模与对应下标的噪声分布一致。便于对应，我们将下标反向，令初始结果为$x_T$,$x_0$为最终结果，就得到了生成数据时的迭代过程。

$$
x_{i-1} = x_i - {\sigma_i^2\over 2}s_\theta(x_i, i) + \sigma_i z, \quad z \sim \mathcal{N}(0, I)
$$

### 2.4 一个理解DDPM新的角度

细心一点不难发现，这个式子和DDPM生成时从高斯分布中采样的过程非常相似：

$$
x_{t-1} = {1\over \sqrt{\alpha_t}}\left(x_t + {\beta_t\over \sqrt {1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\frac{(1 - \bar{\alpha}_{t-1})\beta_t}{1 - \bar{\alpha}_t}} z, \quad z \sim \mathcal{N}(0, I)
$$

本质上都是Langevin采样。而观察二者的训练目标，DDPM的训练目标写成期望形式的话有

$$
\theta^* = \arg\min_\theta \mathbb {E}_{t,x_0,\epsilon }\vert\vert \epsilon - \epsilon_\theta(x_t,t)\vert\vert_2^2\\
= \arg\min_\theta \mathbb {E}_{t,x_0,\epsilon }\left[\vert\vert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar \alpha_t}\epsilon,t)\vert\vert_2^2\right]\\
$$

而在前面，我们推导过加了祖先采样后，条件分布就是高斯分布，它的梯度是
$$
\nabla_x \log \mathcal N(x;\mu,\sigma I)= -{x-\mu\over \sigma^2}
$$

多噪声Score Matching的训练目标是

$$
\theta^* = \arg\min_\theta \sum\limits_{i}\sigma_i^2 \mathbb{E}_{p_{data}(x)}\left[\mathbb{E}_{ p_{\sigma_i}(x'\vert x)}\left[\vert\vert s_\theta(x', i) + \frac{(x' - x)}{\sigma_i^2}\vert\vert_2^2\right]\right]
$$

注意到这里$x'$也是通过$x$加噪产生的，二者相减就是噪声$\sigma_iz$

$$
= \arg\min_\theta \mathbb{E}_{i,p_{data}(x),z}\left[\sigma_i^2\vert\vert s_\theta(x', i) + {z\over \sigma_i}\vert\vert_2^2\right] \\= \arg\min_\theta \mathbb{E}_{i,p_{data}(x),z}\left[\vert\vert z - (-\sigma_is_\theta(x', i))\vert\vert_2^2\right]
$$

不能说是非常相像，只能说是完全一致。

至此，我们将DDPM从单纯的加噪去噪、变分推断、基于score matching的Langvin采样三种视角统一了起来。

关于DDPM的后续改进，这篇已经太长了，留个坑放在下一篇blog中。
