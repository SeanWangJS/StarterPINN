# 对数极坐标 Mixed-PINN 联合模型

## 1. 核心设计哲学 (Core Philosophy)

* **空间折叠 (Spatial Folding)**：将复杂的带孔极坐标系 $(r, \theta)$ 映射为纯矩形的计算域 $(s, \theta)$。在计算域中进行均匀采样，映射回物理空间后，自然就变成了在孔边呈现指数级密集的“完美自适应采样”。
* **算子解耦 (Operator Decoupling)**：抛弃 Navier-Cauchy 位移方程，直接把极坐标系的平衡方程（1阶）和本构方程（1阶）作为并列的损失函数。

---

## 2. 空间重映射与数学引擎 (Mathematical Engine)

首先，定义对数极坐标变换：


$$s = \ln\left(\frac{r}{a}\right) \implies r = a e^s$$

根据链式法则，关于 $r$ 的偏导数转化为关于 $s$ 的偏导数：


$$r \frac{\partial}{\partial r} = \frac{\partial}{\partial s}$$

在这个变换下，我们来看令人惊叹的“奇点消除”过程：

### 2.1 降阶后的动量平衡方程（网络必须满足的第一个规律）

原极坐标系下，径向和切向的二维力学平衡方程含有 $\frac{1}{r}$：

1. $\frac{\partial \sigma_{rr}}{\partial r} + \frac{1}{r}\frac{\partial \tau_{r\theta}}{\partial \theta} + \frac{\sigma_{rr} - \sigma_{\theta\theta}}{r} = 0$
2. $\frac{\partial \tau_{r\theta}}{\partial r} + \frac{1}{r}\frac{\partial \sigma_{\theta\theta}}{\partial \theta} + \frac{2\tau_{r\theta}}{r} = 0$

将两边同乘 $r$ 并代入 $r \frac{\partial}{\partial r} = \frac{\partial}{\partial s}$，我们得到了**无分母、无奇点、纯一阶**的优雅方程组：

* **PDE Loss 1 (径向平衡)**：

$$\frac{\partial \sigma_{rr}}{\partial s} + \frac{\partial \tau_{r\theta}}{\partial \theta} + \sigma_{rr} - \sigma_{\theta\theta} = 0$$


* **PDE Loss 2 (切向平衡)**：

$$\frac{\partial \tau_{r\theta}}{\partial s} + \frac{\partial \sigma_{\theta\theta}}{\partial \theta} + 2\tau_{r\theta} = 0$$



### 2.2 降阶后的本构方程（网络必须满足的第二个规律）

我们强制网络输出的应力 $\boldsymbol{\sigma}$ 与位移 $\mathbf{u}$ 的一阶导数之间满足胡克定律（以平面应力为例）：

* **PDE Loss 3 (径向本构)**：

$$a e^s \frac{1-\nu^2}{E} \sigma_{rr} - \left( \frac{\partial u_r}{\partial s} + \nu \left( \frac{\partial u_\theta}{\partial \theta} + u_r \right) \right) = 0$$


* **PDE Loss 4 (切向本构)**：

$$a e^s \frac{1-\nu^2}{E} \sigma_{\theta\theta} - \left( \frac{\partial u_\theta}{\partial \theta} + u_r + \nu \frac{\partial u_r}{\partial s} \right) = 0$$


* **PDE Loss 5 (剪切本构)**：

$$a e^s \frac{2(1+\nu)}{E} \tau_{r\theta} - \left( \frac{\partial u_r}{\partial \theta} + \frac{\partial u_\theta}{\partial s} - u_\theta \right) = 0$$



---

## 3. 网络拓扑与数据流设计 (Network Topology)

在这个联合架构下，神经网络的形态发生了本质变化：

* **输入层 (2 维)**：计算域坐标 $(s, \theta)$。
* $s \in [0, s_{max}]$（其中 $s_{max} = \ln(W/a)$，即孔口处 $s=0$，远场为 $s_{max}$）。
* $\theta \in [0, \pi/2]$（四分之一对称域）。


* **隐藏层**：4-5 层，每层 64-128 个神经元，激活函数使用 `Tanh`。
* **输出层 (5 维)**：直接输出位移与应力张量的无量纲化值。
* $\hat{\mathbf{y}} = [\bar{u}_r, \bar{u}_\theta, \bar{\sigma}_{rr}, \bar{\sigma}_{\theta\theta}, \bar{\tau}_{r\theta}]$


* **物理还原层 (反归一化)**：在计算 Loss 之前，乘以相应的特征缩放系数，例如 $\sigma_{rr} = \bar{\sigma}_{rr} \times \sigma_0$。

---

## 4. 边界条件重塑：将最难的边界变成最简单的直线

在 $(x, y)$ 笛卡尔空间中，孔是一个曲线边界，法向量随位置变化，极难写约束。但在 $(s, \theta)$ 计算空间中，**整个圆孔被“压伸”成了一条直线边界 $s=0$**。

边界条件极其清晰：

* **孔口自由面边界 (Hole BC)**：仅在 $s=0$ 线上：

$$Loss_{Hole} = \sum (\bar{\sigma}_{rr}^2 + \bar{\tau}_{r\theta}^2)$$


* **远场拉伸边界 (Far-field BC)**：在 $s = s_{max}$ 线上，将其转换为极坐标下的解析远场力进行约束。
* **对称轴边界 (Symmetry BC)**：
* 在 $\theta = 0$ (右侧轴) 线上：$u_\theta = 0, \tau_{r\theta} = 0$
* 在 $\theta = \pi/2$ (上方轴) 线上：$u_\theta = 0, \tau_{r\theta} = 0$



---

## 5. 实施此方案的工程收益总结

1. **彻底废弃复杂的自适应采样（Adaptive Sampling）**：你只需要在 $s \in [0, \ln(10)]$ 和 $\theta \in [0, \pi/2]$ 的矩形网格内**进行最无脑的均匀采样**（Uniform Sampling）。因为 $s$ 域的均匀网格在反向映射回 $r$ 域时，会自动变成 $r = a e^s$ 的指数级加密网格，完美覆盖应力集中区。
2. **根除频谱偏置（Spectral Bias）**：网络不再需要拟合陡峭的应力峰值。因为在 $s$ 空间看过去，剧烈的应力爬升被 $\ln$ 函数拉长成了一段极其平缓的坡度。神经网络能够非常轻松地用低频信号拟合出这条曲线。
3. **计算图大幅提速**：Mixed-PINN 只需要调用一次 `autograd.grad`（计算一阶导），而不需要二次回传（Hessian 矩阵）。这会使你的 PyTorch/JAX 代码训练速度提升 2~3 倍，且显存占用显著下降。