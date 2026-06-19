# 基于 PINN 的一维静力学方程求解系统设计文档

## 1. 业务背景与数学模型

本系统旨在利用物理信息神经网络（PINN）求解一维均质弹性杆的静态拉伸问题。系统将摒弃传统的网格划分与矩阵组装，通过神经网络的自动微分（Autograd）能力逼近物理方程的解析解。

**物理场定义：**

* **计算域**：空间坐标 $x \in [0, L]$
* **待求场**：位移场 $u(x)$
* **已知常数**：杨氏模量 $E$，杆长 $L$，端点拉应力 $\sigma_0$

**控制方程（PDE）：**

$$E \frac{d^2 u}{dx^2} = 0$$

**边界条件（BCs）：**

* **左端固定（Dirichlet条件）**：$u(0) = 0$
* **右端受拉（Neumann条件）**：$E \frac{du(L)}{dx}\Big|_{x=L} = \sigma_0$

---

## 2. 系统架构概览

从代码实现的角度，建议将系统拆解为四个解耦的核心模块：**采样器（Sampler）**、**网络模型（Network）**、**物理损失评估器（Physics Evaluator）**、**优化求解器（Optimizer）**。

这种模块化设计符合面向对象（OOP）原则，便于代码的单元测试，并为后期无缝扩展到二维、三维或多场耦合问题奠定基础。

---

## 3. 核心模块详细设计

### 模块 A：空间采样器 (Geometry & Sampler)

* **职责**：在计算域和边界上生成训练数据（即配点）。这是无网格法的核心体现。
* **数据结构**：
  * `x_f` (Tensor, shape `[N_f, 1]`): 内部配点集。使用拉丁超立方抽样（LHS）或均匀采样在 $[0, L]$ 内生成 $N_f$ 个点。
  * `x_bc_left` (Tensor, shape `[1, 1]`): 左边界点，值为 $0$。
  * `x_bc_right` (Tensor, shape `[1, 1]`): 右边界点，值为 $L$。
* **开发避坑指南**：
  * **形状对齐**：必须确保所有 Tensor 是二维的 `[N, 1]` 而非一维 `[N]`，否则在与网络输出计算 MSE 时会引发 PyTorch 隐式广播（Broadcasting）错误，导致 Loss 计算完全错误。
  * **梯度追踪**：所有输入网络的张量必须开启梯度追踪（`x.requires_grad_(True)`），这是计算 PDE 残差的前提。

### 模块 B：神经网络拟合器 (Neural Network)

* **职责**：作为位移 $u(x)$ 的全局解析函数代理。
* **网络结构**：标准多层感知机（MLP）。
  * 输入层：维度 1（特征 $x$）。
  * 隐藏层：建议 3 到 4 层，每层 20-50 个神经元。
  * 输出层：维度 1（预测位移 $u_{pred}$）。
* **激活函数**：**严禁使用 ReLU**。由于本问题 PDE 涉及二阶导数，ReLU 的二阶导数处处为 0，会导致 $Loss_{PDE}$ 梯度消失，网络无法学习物理定律。必须使用二次连续可导的激活函数，如 `Tanh`、`Sigmoid` 或 `Swish`（首选 `Tanh`）。

### 模块 C：物理损失评估器 (Physics Evaluator)

* **职责**：接收网络预测值，调用自动微分计算各阶导数，并组装全局 Loss。
* **核心算法**（PyTorch 关键代码示例）：
  1. **计算二阶导与 PDE Loss**：
     ```python
     # 必须设置 create_graph=True 才能计算高阶导数
     u_pred = model(x_f)
     du_dx = torch.autograd.grad(u_pred, x_f, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
     d2u_dx2 = torch.autograd.grad(du_dx, x_f, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
     loss_pde = torch.mean((E * d2u_dx2)**2)
     ```
  2. **边界 Loss 计算**：
     ```python
     # Dirichlet BC: u(0) = 0
     loss_bc_left = torch.mean((model(x_bc_left) - 0)**2)
     
     # Neumann BC: E * u'(L) = sigma_0
     u_right = model(x_bc_right)
     du_dx_right = torch.autograd.grad(u_right, x_bc_right, grad_outputs=torch.ones_like(u_right), create_graph=True)[0]
     loss_bc_right = torch.mean((E * du_dx_right - sigma_0)**2)
     ```
  3. **加权求和**：
     `loss_total = w_pde * loss_pde + w_bc1 * loss_bc_left + w_bc2 * loss_bc_right`
* **开发避坑指南（量纲与归一化）**：在实际工程中，杨氏模量 $E$ 通常极大（如 $2 \times 10^{11}$ Pa），而位移极小。这会导致 `loss_pde` 与 `loss_bc` 量级相差悬殊，网络极难收敛。**强烈建议**在送入网络前对物理量进行无量纲化（归一化），或为不同 Loss 项设置合理的动态权重。

### 模块 D：混合优化策略 (Hybrid Optimizer)

* **职责**：驱动网络参数更新，寻找非凸优化空间的全局最优解。
* **执行策略**：由于物理机理的引入，Loss 空间极其复杂。建议采用两阶段优化：
  * **Phase 1 (Global Search)**：使用 `Adam` 优化器，学习率设为 $1e^{-3}$，迭代几千次。此阶段目标是让模型快速拟合边界条件并进入低 Loss 区域。
  * **Phase 2 (Local Fine-tuning)**：切换为 `L-BFGS`（拟牛顿法）。PINN 对精度要求较高，L-BFGS 利用海森矩阵近似进行强悍的局部搜索，能将物理残差压榨到机器精度级别。

---

## 4. 主循环执行流程 (Main Loop)

```python
# 伪代码执行流
1. 初始化物理参数 (E, L, sigma_0)，建议进行无量纲化处理
2. 实例化 Sampler, 抽取全批量配点 (x_f, x_bc_left, x_bc_right)
3. 实例化 MLP 模型
4. 配置 Adam 优化器

# 第一阶段：Adam 训练
5. For step in range(max_steps_adam):
       optimizer_adam.zero_grad()
       Loss_Total = Evaluator(model, x_f, x_bc_left, x_bc_right)
       Loss_Total.backward()
       optimizer_adam.step()

# 第二阶段：L-BFGS 训练
6. 配置 L-BFGS 优化器
7. 定义闭包函数 (closure):
       def closure():
           optimizer_lbfgs.zero_grad()
           Loss_Total = Evaluator(model, x_f, x_bc_left, x_bc_right)
           Loss_Total.backward()
           return Loss_Total
8. optimizer_lbfgs.step(closure)

# 测试与可视化阶段
9. 切换为评估模式 model.eval()，在 torch.no_grad() 上下文中推理
10. 将连续域离散化输入模型预测位移，绘制 u(x) 曲线
11. 解析解对比：u_exact = (sigma_0 / E) * x，计算相对误差 L2-Error。
```

## 5. 技术栈与框架建议

* **开发语言**：Python 3.8+
* **核心框架**：PyTorch
* **科学计算与可视化**：NumPy, Matplotlib