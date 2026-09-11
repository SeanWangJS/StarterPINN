# Griffith 2D：理论驱动的 PINN/DEM 相场断裂实现

`griffith2d` 是二维 Griffith 相场断裂的新实现。项目以
[理论推导](../tutorials/pinn-dem-phase-field-derivation.md)为规范来源，先建立可验证的基础计算核心，
再通过外围实验探索网络架构、离散方式和求解策略。

旧目录 [`../griffith`](../griffith) 仅作为历史基线和经验记录；新实现不直接复制其中的训练流程或
实验代码。旧实现中的方案只有在理论依据明确、接口职责清楚且通过独立验证后，才进入本项目。

## 设计目标

1. **理论决定核心。** 物理公式必须能够追溯到理论文档，不由某次训练现象反向定义。
2. **物理与数值分离。** 本构、能量和不可逆性不依赖网络宽度、优化器或采样方案。
3. **验证先于耦合。** 先验证张量、能量和单场问题，再训练完整 Griffith 算例。
4. **实验位于外围。** 临时方案可以依赖核心，核心不能依赖任何实验。
5. **失败必须可定位。** 分别观察模型误差、离散误差和优化误差，不用单一总 loss 代替诊断。
6. **配置不是物理实现。** 配置只选择已有模型和算法，不复制公式或隐藏新的物理假设。

## 目录结构

```text
griffith2d/
├── README.md
├── configs/
│   └── baselines/            # 通过验证的基线配置
├── src/
│   └── griffith2d/
│       ├── physics/          # 连续模型和能量定义
│       ├── discretization/   # 网络试探空间、自动微分、积分与采样
│       ├── solvers/          # 单场及耦合求解流程
│       └── diagnostics/      # 物理量、误差和收敛指标
├── tests/
│   ├── unit/                 # 公式级与局部数值测试
│   ├── benchmarks/           # 有解析解或可信参考解的算例
│   └── integration/          # 跨模块和完整求解流程测试
├── experiments/
│   ├── configs/              # 尚在评估的实验配置
│   ├── runners/              # 临时实验入口
│   └── analysis/             # 实验比较与结果分析
├── scripts/                  # 稳定、可复用的训练与评估入口
└── artifacts/
    ├── runs/                 # 日志、指标和配置快照
    ├── figures/              # 生成的图像
    └── checkpoints/          # 模型权重与求解状态
```

当前已完成阶段 1 的物理核心，包括材料参数、小应变、平面应变三维体积—偏应变分解、
Amor 能量分解、二次退化函数、退化弹性能和解析应力。实现状态与验收证据见
[`../docs/plans/001-griffith2d-physics-core.md`](../docs/plans/001-griffith2d-physics-core.md)。

阶段 2 也已完成：当前具有确定性 Gauss–Legendre 区间、线段和矩形求积，固定 `tanh`
位移网络，精确对称边界试探解，逐点空间 Jacobian，纯弹性总势能组装，以及带解析场、
反力、牵引和平衡残差诊断的二维 DEM 基准。验收记录见
[`../docs/plans/002-griffith2d-quadrature-elastic-dem.md`](../docs/plans/002-griffith2d-quadrature-elastic-dem.md)。

阶段 3 已完成：当前具有 AT2 裂纹能、显式历史场尺度与纯更新函数、相场一阶/二阶自动微分、
预裂纹硬试探解、独立相场 DEM，以及有限区间预裂纹和均匀历史场两个解析基准。历史场方法在
本项目中明确作为冻结驱动力代理，不声称等价于严格相场不可逆约束。验收记录见
[`../docs/plans/003-griffith2d-at2-phase-field.md`](../docs/plans/003-griffith2d-at2-phase-field.md)。

阶段 4 已完成：当前具有冻结均匀相场退化弹性 DEM、独立多根标量参考与低损伤分支延拓、
候选/已提交历史状态、事务式载荷步回滚，以及覆盖单调加载、卸载、峰值内重载和超峰值重载的
固定准静态路径。该阶段只验证空间均匀历史场代理的耦合接口，不构成二维裂纹扩展证据。验收记录见
[`../docs/plans/004-griffith2d-staggered-coupling.md`](../docs/plans/004-griffith2d-staggered-coupling.md)。

## 核心模块

### `src/griffith2d/physics`

连续物理模型的唯一实现位置，预计包括：

- 小变形运动学与应变计算；
- 各向同性线弹性参数；
- Amor 拉压能量分解；
- 相场退化函数；
- AT2 裂纹表面能；
- 内部能、外力势与总势能；
- 历史场或其他不可逆性约束；
- 一致的无量纲化；
- 边界和初始裂纹的物理定义。

该模块只表达“求解什么”，不得导入网络、优化器、训练循环或实验配置。

### `src/griffith2d/discretization`

负责把连续场和积分离散为可计算对象，预计包括：

- 位移场与相场的神经网络表示；
- 本质边界条件的试探解构造；
- 自动微分算子；
- 确定性求积与随机采样；
- 积分点、权重和区域划分；
- 网络输出到物理场的映射。

该模块可以依赖 `physics`，但不能修改物理公式来适配某个网络或采样器。

### `src/griffith2d/solvers`

组织求解算法，预计包括：

- 固定相场下的位移求解；
- 给定驱动力下的相场求解；
- 交替最小化；
- 联合优化的对照实现；
- 载荷步推进、收敛判据和状态恢复；
- 优化器调度与检查点管理。

求解器组合 `physics` 与 `discretization`，但不重复本构、能量或积分公式。

### `src/griffith2d/diagnostics`

集中定义训练和验证所需的可观测量，预计包括：

- 弹性能、断裂能、外力势和总势能；
- 平衡残差和边界误差；
- 不可逆性违反量；
- 位移、应力和相场剖面误差；
- 梯度范数、积分收敛和载荷步收敛指标；
- Griffith 解析基准比较所需的数据。

诊断代码只读取状态，不应改变训练过程或模型参数。

## 依赖方向

核心依赖必须保持单向：

```mermaid
flowchart LR
    T["理论文档"] --> P["physics"]
    P --> D["discretization"]
    P --> S["solvers"]
    D --> S
    P --> G["diagnostics"]
    D --> G
    S --> G
    P --> B["tests / benchmarks"]
    D --> B
    S --> B
    G --> B
    P --> E["experiments"]
    D --> E
    S --> E
    G --> E
```

禁止以下反向依赖：

- `physics` 依赖 `discretization`、`solvers` 或 `experiments`；
- 核心模块读取某个实验目录的配置；
- `scripts` 或 `experiments` 内重新实现能量、本构和边界公式；
- 为了降低训练 loss 而在实验中静默改变物理模型。

## 验证顺序

完整耦合训练不是第一个里程碑。实现应按以下顺序逐级通过验证：

1. **张量与能量单元测试**
   - 刚体平移和小刚体转动不产生小应变；
   - 应力等于能量对应变的导数；
   - 均匀应变状态与解析能量一致；
   - 拉压分解和退化函数满足极限情况。
2. **数值积分测试**
   - 常数和低阶多项式积分达到预期精度；
   - 分区积分权重、面积和量纲一致；
   - 随积分辨率增加，能量稳定收敛。
3. **纯弹性基准**
   - 无裂纹问题恢复解析位移和应力；
   - 平衡残差、边界误差与能量误差同时报告。
4. **固定相场的位移问题**
   - 在已知退化场下验证位移平衡和应力响应。
5. **给定驱动力的相场问题**
   - 验证一维 AT2 裂纹剖面、长度尺度和不可逆性。
6. **位移—相场耦合问题**
   - 验证交替迭代、载荷步和能量演化。
7. **二维 Griffith 基准**
   - 最后比较解析基准、有限域修正及收敛行为。

只有前一级通过后，后一级的失败才具有可解释性。

## 实验约定

每个实验在运行前必须写明：

- **问题**：当前哪个可观测现象没有达到基线；
- **假设**：为什么所提修改可能解决该问题；
- **自变量**：本次唯一主动改变的因素；
- **控制量**：必须保持不变的物理、离散和训练设置；
- **否证判据**：什么结果意味着假设不成立；
- **比较指标**：不能只报告总 loss；
- **随机性**：种子数量及汇总方式；
- **退出条件**：实验何时停止，以及代码何时删除或晋升。

建议每个实验使用独立、带日期或主题名称的配置。运行结果写入 `artifacts/runs`，图像写入
`artifacts/figures`，权重写入 `artifacts/checkpoints`。任何生成文件都不能成为核心模块的隐式输入。

实验结论有三种去向：

1. 被否证：记录结论后删除临时实现；
2. 证据不足：保留在 `experiments`，不得进入基线；
3. 多基准验证通过：提炼通用接口、补齐测试和文档后晋升到核心模块。

## 基线配置与脚本

`configs/baselines` 只保存通过对应验证层级的稳定配置。配置应包含完整参数并可独立复现，
不得依赖代码中的隐式默认值。

`scripts` 只放稳定入口，例如基准测试、正式训练、评估和绘图。一次性的参数扫描、诊断探针和
消融入口应留在 `experiments/runners`，避免稳定入口随实验数量膨胀。

## 实施阶段

**阶段 1 至阶段 3**已完成。完整测试命令为：

```powershell
Set-Location griffith2d
C:\Users\seanwang\miniforge3\envs\general\python.exe -m pytest -q
```

安装本地包后，可运行稳定的纯弹性 DEM 基准：

```powershell
python -m pip install -e .
griffith2d-elastic-tension --config configs/baselines/elastic_tension.toml
```

两个独立相场基准使用同一个稳定入口和各自的完整配置：

```powershell
griffith2d-phase-field --config configs/baselines/phase_field_precrack.toml
griffith2d-phase-field --config configs/baselines/phase_field_uniform_history.toml
```

阶段 4 的固定相场退化弹性和均匀交替耦合基准使用统一的具名 JSON 入口：

```powershell
griffith2d-coupling --config configs/baselines/degraded_elastic_tension.toml
griffith2d-coupling --config configs/baselines/staggered_uniform_tension.toml
```

后续依次进行：

1. 阶段 5：建立二维 Griffith 基线；
2. 阶段 6：在外围开展网络、积分、优化和约束方案实验。

任何阶段如果无法通过本层验证，应停留在该层定位问题，不提前用更复杂的网络或优化策略掩盖失败。
