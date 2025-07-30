---
title: '扩散概率模型 Diffusion model'
date: 2025-07-29
permalink: /posts/2025/07/Diffusion/
tags:
  - learning note
---

## 扩散模型
学习资料：
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [由浅入深了解Diffusion Models](https://zhuanlan.zhihu.com/p/525106459)
- [深度推导DDPM数学原理](https://zhuanlan.zhihu.com/p/656757576)
### 1. DDPM

扩散概率模型实际上2015年就被提出来了。正向过程表示为一个不断添加高斯噪声的马尔可夫链，反向过程则是一个参数化的去噪过程。

整体的思路其实与VAE有共通之处——隐空间建模成了标准正态分布，而采用迭代的马尔可夫链取代了VAE的Encoder-Decoder架构，

本文将从两个角度理解扩散模型：一方面将你能搜索到的最常见的加噪-去噪过程的推导用简单的方式[推导一遍](#11-前向过程)，另一方面则会从[最原始的扩散模型](#13-最原始的扩散模型)出发，通过类似推导VAE的方式，找寻它背后的动机和根本原理。

论文原文提到
> 扩散模型的特定参数化方式在训练过程中揭示了与多噪声水平下score matching的等价性，而在采样过程中则与退火朗之万动力学等价。这些概念都是什么，暂且按下不表。这篇文章从原理层面按照个人理解的思路推导一下DDPM以及其改进。

如果想先了解这些概念，可以先看[相关内容](#3-相关内容)。


#### 1.1 前向过程

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

#### 1.2 反向过程（采样）

有了后验分布，我们尝试写出反向的采样过程 $p(x_{t-1} \vert x_t)$

利用贝叶斯公式

$$
p(x_{t-1} \vert x_t) = \frac{p(x_t \vert x_{t-1}) p(x_{t-1})}{p(x_t)}
$$

后验 $p(x_t\vert x_{t-1})$ 使用1.1中的（近似）后验（为什么是近似，在1.4节中会说明）$q$ 代替。但是每一轮隐变量的边缘分布 $p(x_t)$ 都无法根据已有条件计算。

直观上理解，如果不知道最终的采样方向，每一步的隐变量分布都是未知的——不知道该往什么方向去噪。

扩散模型通过引入一个辅助变量来解决，假如添加一个条件$x_0$（后面会说明这一假设的合理性），则

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

#### 1.3 重新审视扩散模型

扩散概率模型，是一种隐变量模型。如果VAE是直接将观测数据和隐变量使用参数化的Encoder-Decoder架构联系起来，那么扩散模型则固定了观测数据和隐变量之间关系的形式——一个马尔可夫链，除了$x_0$以外，其余的$x_t$都可以视为隐变量。

边缘似然函数可以写成

$$
p_\theta(x_0) = \int p_\theta(x_0, x_1, \ldots, x_T) dx_1 \ldots dx_T = \int p_\theta(x_{0:T}) dx_{1:T}
$$

在这个模型下，观测数据是通过一个马尔可夫链生成的（被称为逆向过程）：

$$
p_\theta(x_{t_0}\vert x_{T}) = p_\theta(x_T)\prod_{t=t_0+1}^T p_\theta(x_{t-1} \vert x_t)
$$

其中 $x_T\sim\ \mathcal{N}(0,I)$。并且转移是一个高斯分布，即

$$
p_\theta(x_{t-1} \vert x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma^2_\theta(x_t,t))
$$

逆向转移的高斯分布参数是未知的，我们想要优化 $\theta$，使得边缘似然尽可能大。然而边缘似然函数是不可计算的，因为多个隐变量的高维积分算不出来。
$$
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T} = \log \int p_\theta(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \vert x_t) dx_{1:T}
$$

真实后验分布也算不出来

$$
p_\theta(x_t \vert x_0) = \frac{p_\theta(x_0 \vert x_t) p_\theta(x_t)}{p_\theta(x_0)}
$$

经典的问题。我们使用变分推断，引入变分分布 $q(x_t \vert x_0)$

与VAE中使用神经网络参数化的变分分布不同的是，扩散模型的变分分布被定义为一个具有高斯分布形式转移的马尔可夫链，具体形式已经在1.1中详细描述过了。

优化目标转变为最大化证据下界ELBO：(具体为什么同样参考上一篇VAE的文章，**优化ELBO意味着联合优化生成过程的参数$\theta$和变分分布q**)

$$
\mathcal{L} = \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log {p_{\theta}(x_0 \vert x_{1:T})p_\theta(x_{1:T})\over q(x_{1:T}\vert x_0)}\right] =  \mathbb E_{q(x_{1:T}\vert x_0)}\left[\log p_\theta(x_T) + \sum\limits_{t=1}^T \log {p_\theta(x_{t-1} \vert x_t)\over q(x_{t} \vert x_{t-1})}  \right]
$$

这里是将联合概率利用马尔可夫链的特性展开。

使用贝叶斯公式，同时和之前的技巧一样，引入$x_0$，有

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

注意到这里的KL散度发生了奇妙的变化。在先前，最大化ELBO等价于最小化变分分布与后验分布的KL散度。而在这里，需要最小化的是变分近似的后验分布的逆向过程，与定义的参数未知的符合高斯分布逆向过程（生成分布）的的KL散度，加上前面一个什么东西
- 可以细品与VAE推出的损失函数的区别。但是共同点很明显，思路都是最小化变分与后验KL—>最大化ELBO->最小化另一个好算的KL加上什么东西。

到了这里就比较好算了。前面那一项实际上与$\theta$无关（因为$q(x_T\vert x_0)$是固定的，而$p_\theta (x_T)$是个标准正态），因此可以忽略。

第二项前半部分，变分分布$q$的逆向过程在1.2中已经推导过一遍了，是一个均值与$x_0,x_t$相关的正态分布。后半部分则是最开始定义的真实逆向过程，也是正态分布，但参数未知。

再简单回顾一下，变分的逆向过程是

$$
q(x_{t-1} \vert x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_t(x_0,x_t), \sigma_t^2 I)
$$

$$
\mu_{t}(x_0,x_t) = {\sqrt \alpha_t(1-\bar\alpha_{t-1})\over 1-\bar\alpha_t}x_t + {\sqrt{\bar{\alpha}_{t-1}}\beta_t \over 1 - \bar{\alpha}_t} x_0\\
\sigma_{t}^2 =\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t
$$

停在这一步，先不要把$x_0$消掉。既然方差与$x$无关，那么$\Sigma^2_\theta$完全可以就取$\sigma^2_tI$。那么剩下的就是均值。高斯分布的KL散度是有封闭形式的，而在方差一致的条件下，最终结果是最小化KL散度等价于最小化均值的平方差，即

$$
    D_{KL}(q(x_{t-1} \vert x_t, x_0) \vert\vert p_\theta(x_{t-1} \vert x_t)) = \frac{1}{2\sigma^2_t}\left(\mu_t(x_0,x_t) - \mu_\theta(x_t,t)\right)^2 +C
$$

其中C是与$\theta$无关的常数。

那么整个算法就呼之欲出了，利用参数化的神经网络预测均值，使得在训练过程中均值尽可能接近近似后验的逆向过程算出的那个值（训练集的$x_0$是已知的），而方差则固定为$\sigma^2_tI$。

预测均值可以利用重参数化技巧改为预测噪声，方法就是1.2中提到的代入$x_0$

现在大概能够回答[1.2节的末尾](#13-重新审视扩散模型)的那个问题了——我们训练的网络到底是个什么？训出来的是真实的参数化生成过程。引入变分分布只是希望使得参数化生成过程变得可优化的同时尽可能近似真实的算不出来的后验分布。也就是说，我们从头到尾都不知道真实情况的后验分布（正向过程）到底长成什么样，正向加噪只是变分近似。

而如果在采样过程中，预测出噪声就直接用加噪公式反推出$x_0$，这并不是真实的生成过程，而只是近似的变分加噪的逆向过程。真实的生成过程是被定义为参数化的高斯转移，只是利用重参数化技巧后，体现为“去噪”。

#### 1.4 训练



### 2. DDPM 的改进

#### 2.1 DDIM

#### 2.2 LDM

#### 2.3 Stable Diffusion



### 3. 相关内容

#### 3.1 Energy-based Models

#### 3.2 Score-based Generative Models

#### 3.3 Score Matching
