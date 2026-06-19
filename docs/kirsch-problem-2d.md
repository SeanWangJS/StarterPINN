# 基于 PINN 的二维 Kirsch 问题（带孔平板拉伸）求解系统设计文档

## 1. 系统定义与物理模型 (Domain & Physics)

本系统旨在利用物理信息神经网络（PINN）求解受单向拉伸的无限大带孔平板的位移场与应力场。为节省算力并消除刚体位移，工程学上通常利用对称性，仅取四分之一平板（右上象限）进行建模。

**1.1 几何与物理参数：**

* **计算域**：$\Omega = \{(x,y) | 0 \le x \le W, 0 \le y \le H, x^2 + y^2 \ge a^2\}$
* $W, H$：远场截断宽度与高度（通常取 $W, H \ge 10a$ 以模拟无限大边界）
* $a$：中心圆孔半径
* **物理常数**：杨氏模量 $E$，泊松比 $\nu$，远场拉应力 $\sigma_0$。
* **系统状态（网络输出）**：二维位移场 $\mathbf{u} = [u(x,y), v(x,y)]^T$。

**1.2 控制方程 (Navier-Cauchy 平面应力方程)：**
在稳态无体力的情况下，二维线弹性动量平衡方程为：

$$\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y} = 0, \quad \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} = 0$$

其中应力由应变（位移的一阶导数）通过本构方程定义：

$$\sigma_{xx} = \frac{E}{1-\nu^2} \left( \frac{\partial u}{\partial x} + \nu \frac{\partial v}{\partial y} \right), \quad \sigma_{yy} = \frac{E}{1-\nu^2} \left( \frac{\partial v}{\partial y} + \nu \frac{\partial u}{\partial x} \right), \quad \tau_{xy} = \frac{E}{2(1+\nu)} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)$$

**1.3 边界条件 (Boundary Conditions, BCs)：**

1. **左边界（对称轴 $x=0$）**：$u = 0, \tau_{xy} = 0$
2. **下边界（对称轴 $y=0$）**：$v = 0, \tau_{xy} = 0$
3. **右边界（远场拉伸 $x=W$）**：$\sigma_{xx} = \sigma_0, \tau_{xy} = 0$
4. **上边界（远场自由 $y=H$）**：$\sigma_{yy} = 0, \tau_{xy} = 0$
5. **圆孔边界（面力自由 $r=a$）**：边界法向面力与切向面力均为 0。
   * 法向量 $\mathbf{n} = (n_x, n_y) = (-\frac{x}{a}, -\frac{y}{a})$ *(注意1/4孔的法向量指向圆心)*
   * $T_x = \sigma_{xx}n_x + \tau_{xy}n_y = 0$
   * $T_y = \tau_{xy}n_x + \sigma_{yy}n_y = 0$

---

## 2. 核心架构设计 (Architecture Design)

与一维系统相比，二维 PINN 的架构需升级为**多输入多输出（MIMO）**模型，并引入**自适应采样器**。

### 模块 A：CSG 几何引擎与自适应采样器 (Geometry & Adaptive Sampler)

* **实现逻辑**：
  1. **构造实体几何 (CSG)**：计算域为 $\Omega = R \setminus C$。在代码实现中，可以使用随机点生成后，过滤掉 $x^2 + y^2 < a^2$ 的点。
  2. **基础采样**：使用拉丁超立方抽样（LHS）在 $\Omega$ 内生成配点。
  3. **基于残差的自适应重采样 (RAR - Residual-based Adaptive Refinement)**：
     *由于孔口存在极大的应力集中，均匀采样会导致孔边物理定律未能被充分学习。*
     *代码实现策略*：每 1000 个 Epoch，在域内随机撒 10 万个测试点，计算每个点的 $Loss_{PDE}$，选取残差最大的前 1000 个点，将其永久加入到后续的训练集 `x_f, y_f` 中。

### 模块 B：网络拓扑映射 (Network Topology)

* **输入层**：维度 2，坐标 $(x, y)$。建议将 $x, y$ 拆分为两个独立的 `[N, 1]` 张量输入，方便后续对各自独立求偏导。
* **隐藏层**：建议 4-6 层，每层 50-100 个神经元。
* **激活函数**：推荐使用 `Tanh` 或 `SiLU`（Swish）。

### 模块 C：张量微分与残差计算器 (Autograd Evaluator)

* **二维导数代码实现要点**：
  由于需要计算多项偏导数，极易写错。推荐将位移输出 $u, v$ 与坐标 $x, y$ 彻底分离计算，防止梯度图纠缠不清。
  ```python
  # PyTorch 实现二维应变的核心示例
  u, v = model(x, y)
  
  # u 对 x 和 y 的导数
  du = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
  du_dx, du_dy = du[0], du[1]
  
  # v 对 x 和 y 的导数
  dv = torch.autograd.grad(v, [x, y], grad_outputs=torch.ones_like(v), create_graph=True)
  dv_dx, dv_dy = dv[0], dv[1]
  
  # 接着根据本构方程计算 sigma_xx, sigma_yy, tau_xy
  # 然后再次调用 autograd.grad 对应力求偏导以计算 PDE 残差
  ```

---

## 3. 损失函数装配 (Loss Function Assembly)

$$Loss_{Total} = w_{pde} \cdot Loss_{PDE} + w_{sym} \cdot Loss_{Symmetry} + w_{far} \cdot Loss_{FarField} + w_{hole} \cdot Loss_{Hole}$$

*(注：由于 $Loss_{Hole}$ 涉及导数且在几何边界上，往往是最难收敛的。)*

**动态权重策略（开发避坑）**：
不同 Loss 项的量级和梯度下降速度差异极大。在代码实现中，不建议手动死磕常数权重。推荐一种最易落地的自适应权重法——**将权重设为可学习的参数 (Uncertainty Weighting)**：
```python
# 将损失权重定义为网络可训练参数，使用对数屏障防止负数
w_pde = nn.Parameter(torch.tensor(0.0))
w_bc = nn.Parameter(torch.tensor(0.0))
# 实际使用的权重为 exp(-w)，并在 Loss 中加上 w 正则项
loss_total = torch.exp(-w_pde) * loss_pde + w_pde + torch.exp(-w_bc) * loss_bc + w_bc
```

---
