# 从强形式 PINN 到深度能量法：相场断裂的推导教程

## 1. 目的与范围


> **配套导读：目的与范围：先看清整件事的结构**
>
> ### 1.1 我们究竟在求什么
>
> 二维相场断裂问题有两个未知场：
>
> - 位移场 $\boldsymbol u(\boldsymbol x)$：描述材料点如何移动；
> - 相场 $\phi(\boldsymbol x)$：描述裂纹在哪里以及裂纹带有多宽。
>
> 它们不是两个互不相关的回归目标。位移产生应变，应变产生弹性能；弹性能驱动相场
> 增长；相场又通过降低局部刚度改变位移场。最短的因果链是
>
> $$
> \boldsymbol u
> \longrightarrow \boldsymbol\varepsilon
> \longrightarrow \psi^+
> \longrightarrow \phi
> \longrightarrow g(\phi)
> \longrightarrow \boldsymbol\sigma
> \longrightarrow \boldsymbol u.
> $$
>
> 这形成一个非线性闭环。即使完整材料是线弹性的，耦合后的断裂问题也不是一个普通
> 线性边值问题。
>
> ### 1.2 三个必须分开的层次
>
> 阅读推导时，最好始终区分以下三个层次。
>
> #### 层次 A：连续物理模型
>
> 它包括平衡方程、本构关系、边界条件、断裂表面能和不可逆约束。这个层次与使用有限元、
> 谱方法还是神经网络无关。
>
> #### 层次 B：数学表述
>
> 同一个连续问题可能写成：
>
> - 强形式：逐点满足微分方程；
> - 弱形式：对所有测试函数满足积分恒等式；
> - 能量形式：某个泛函的一阶变分为零或达到极小值；
> - 变分不等式：由于不可逆约束，只允许单向变化。
>
> 这些形式有适用条件，不能仅因为公式看起来相似就互换。
>
> #### 层次 C：数值近似和训练
>
> 神经网络只是在选择一个参数化试探空间：
>
> $$
> \boldsymbol u(\boldsymbol x)\approx\boldsymbol u_\theta(\boldsymbol x),
> \qquad
> \phi(\boldsymbol x)\approx\phi_\eta(\boldsymbol x).
> $$
>
> 配点、求积、自动微分、Adam、L-BFGS、边界罚项都属于离散和优化层。训练失败不一定
> 表示物理模型错误；训练成功、loss 很小也不保证求解了正确的连续问题。
>
> ### 1.3 从强形式到 DEM 的真正动机
>
> 位移平衡方程包含 $\nabla\cdot\boldsymbol\sigma$。若应力由
> $\nabla\boldsymbol u_\theta$ 计算，强形式需要位移网络关于空间坐标的二阶导数。
> 能量形式只需计算应变能，因此停留在一阶导数。
>
> 这不是简单的“把 PDE loss 换成 energy loss”，而是：
>
> 1. 证明应力来自能量密度；
> 2. 用虚功原理建立强形式和弱形式的联系；
> 3. 证明弱形式是总势能的一阶变分；
> 4. 再把函数空间限制到神经网络表示的子空间中。
>
> ### 1.4 准静态断裂还多出一个“时间”维度
>
> 这里的时间通常是载荷参数，而不是包含惯性项的真实动力学时间。一个典型载荷步为：
>
> 1. 增加载荷或边界位移；
> 2. 在当前相场下求位移平衡；
> 3. 更新损伤驱动力或历史场；
> 4. 求解新的相场；
> 5. 检查 $\phi_n\ge\phi_{n-1}$；
> 6. 在位移和相场之间迭代至当前载荷步收敛；
> 7. 进入下一载荷步。
>
> 因此，单次优化一个网络和模拟完整裂纹扩展过程是两件不同的事。



本教程从连续介质力学、变分法和断裂力学的标准结果出发，建立如下理论链条：

1. 用强形式平衡方程训练位移网络的 PINN；
2. 通过虚功原理改写成弱形式，并得到仅需一阶空间导数的深度能量法（DEM）；
3. 在位移问题上耦合 AT2 相场断裂模型、Amor 拉压分解和历史场不可逆性；
4. 给出二维 Griffith 中心裂纹的建模、解析基准与数值验证要求。

文档遵循“**文献 → 理论 → 实现**”的依赖方向：所有概念和方程应由教材或学术论文
支撑，而不以任何特定代码为依据。代码应当另行证明自己实现了这里的方程。推导采用
小变形、准静态、各向同性线弹性；主要依据连续体与有限元教材 [1]、强形式 PINN
[2]、Deep Ritz/DEM [3,4]、变分相场断裂文献 [6-15] 和断裂力学教材 [17]。

为避免概念混淆，本文使用“强形式”表示点态 PDE 残差，“弱形式”表示对测试函数
成立的积分恒等式，“能量形式”表示弱形式还可以由势能的一阶变分得到。弱形式未必
总能写成最小势能问题；只有保守、自伴或具有相应势函数的系统才可以这样做 [1,3]。

## 2. 问题、符号与边界划分


> **配套导读：问题、符号与边界：方程是如何“闭合”的**
>
> ### 2.1 位移为什么足以描述小变形
>
> 设参考位置为 $\boldsymbol x$，变形后位置为
>
> $$
> \boldsymbol y(\boldsymbol x)=\boldsymbol x+\boldsymbol u(\boldsymbol x).
> $$
>
> 完整有限变形理论使用变形梯度
>
> $$
> \boldsymbol F=\frac{\partial\boldsymbol y}{\partial\boldsymbol x}
> =\boldsymbol I+\nabla\boldsymbol u.
> $$
>
> 其中 $\boldsymbol I$ 是二阶单位张量，表示未发生变形时的恒等映射。在二维问题中，
>
> $$
> \boldsymbol I=
> \begin{bmatrix}
> 1 & 0\\
> 0 & 1
> \end{bmatrix}.
> $$
>
> 小变形假设意味着位移梯度足够小，可以忽略二次项。此时真正引起长度变化的是
> $\nabla\boldsymbol u$ 的对称部分：
>
> $$
> \boldsymbol\varepsilon
> =\operatorname{sym}\nabla\boldsymbol u
> =\frac12(\nabla\boldsymbol u+\nabla\boldsymbol u^T).
> $$
>
> 反对称部分对应局部刚体转动，不应产生弹性能。这就是能量依赖
> $\boldsymbol\varepsilon$ 而非整个 $\nabla\boldsymbol u$ 的原因。
>
> ### 2.2 工程剪应变和张量剪应变
>
> 张量分量为
>
> $$
> \varepsilon_{xy}=\frac12(u_{,y}+v_{,x}),
> $$
>
> 工程剪应变则是
>
> $$
> \gamma_{xy}=2\varepsilon_{xy}.
> $$
>
> 两种记号都能使用，但本构矩阵中的剪切系数不同。神经网络实现中最常见的隐蔽错误之一，
> 就是能量公式使用张量剪应变，而 Voigt 向量却装入工程剪应变，导致剪切能相差两倍或
> 四倍。
>
> ### 2.3 平衡方程从哪里来
>
> 对任意小体积 $V$，线动量守恒为
>
> $$
> \int_{\partial V}\boldsymbol\sigma\boldsymbol n\,d\Gamma
> +\int_V\boldsymbol b\,d\Omega
> =\int_V\rho\boldsymbol a\,d\Omega.
> $$
>
> 使用散度定理并取准静态条件 $\boldsymbol a=\boldsymbol0$，得到
>
> $$
> \nabla\cdot\boldsymbol\sigma+\boldsymbol b=\boldsymbol0.
> $$
>
> 因此：
>
> - $\boldsymbol\sigma\boldsymbol n$ 是面力，量纲是应力；
> - $\boldsymbol b$ 是单位体积的体力，量纲是应力/长度；
> - $\nabla\cdot\boldsymbol\sigma$ 与 $\boldsymbol b$ 量纲相同。
>
> 量纲检查可以快速发现 PINN 残差或无量纲化中的漏项。
>
> ### 2.4 为什么边界必须分成两类
>
> 二阶位移方程不能同时在同一段边界上任意指定完整位移和完整牵引。典型划分为
>
> $$
> \boldsymbol u=\bar{\boldsymbol u}\quad\text{on }\Gamma_u,
> \qquad
> \boldsymbol\sigma\boldsymbol n=\bar{\boldsymbol t}\quad\text{on }\Gamma_t.
> $$
>
> Dirichlet 条件决定允许搜索哪些位移场，因此称为本质条件；Neumann 条件在分部积分
> 后自然出现，因此称为自然条件。
>
> 还必须消除刚体模态。二维问题若只有力边界而没有足够位移约束，且外载满足零合力、
> 零合矩，总势能沿刚体平移或转动方向不变，解只在相差一个刚体运动的意义下唯一；
> 若外载不自平衡，则静力平衡解本身不存在。前一种情况在训练中表现为网络输出整体
> 漂移、Hessian 奇异或优化器不断在平坦方向移动。
>
> ### 2.5 为什么需要函数空间
>
> 能量包含 $\nabla\boldsymbol u$ 和 $\nabla\phi$ 的平方，因此至少要求
>
> $$
> \boldsymbol u\in(H^1(\Omega))^2,\qquad
> \phi\in H^1(\Omega).
> $$
>
> $H^1$ 的含义不是函数必须处处光滑，而是函数及其一阶弱导数平方可积。有限元的分片
> 线性函数在单元边界处导数不连续，却仍属于 $H^1$。这说明弱形式能够容纳比经典强解
> 更低的正则性。
>
> 神经网络使用 $\tanh$ 时通常非常光滑，但“更光滑”并不自动等于“更适合裂纹”。
> 裂纹尖端和宽度为 $O(l_0)$ 的相场带具有强局部变化，过度平滑的网络可能产生明显的
> 谱偏置。



设物体占据二维区域 $\Omega \subset \mathbb{R}^2$，边界拆分为

$$
\partial\Omega = \Gamma_u \cup \Gamma_t,
\qquad
\Gamma_u \cap \Gamma_t = \varnothing.
$$

其中 $\Gamma_u$ 是给定位移的 Dirichlet 边界，$\Gamma_t$ 是给定面力的 Neumann
边界。这里的不交应理解为除去角点等零测集后的边界划分；更严格地，可以对各边界片
的闭包进行描述。未知位移为

$$
\boldsymbol{u}(\boldsymbol{x}) = [u(\boldsymbol{x}), v(\boldsymbol{x})]^T.
$$

小应变张量为

$$
\boldsymbol{\varepsilon}(\boldsymbol{u})
= \frac{1}{2}\left(\nabla\boldsymbol{u}
+ \nabla\boldsymbol{u}^{T}\right),
$$

即

$$
\varepsilon_{xx}=u_{,x},
\qquad
\varepsilon_{yy}=v_{,y},
\qquad
\varepsilon_{xy}=\frac{1}{2}(u_{,y}+v_{,x}).
$$

本节的核心控制关系是平衡方程，以及分别定义在 $\Gamma_u$ 和 $\Gamma_t$ 上的两类
边界条件：位移边界条件与力边界条件。它们分别为

$$
\nabla\cdot\boldsymbol{\sigma}+\boldsymbol{b}=\boldsymbol{0}
\quad \text{in } \Omega. \tag{1}
$$

$$
\boldsymbol{u}=\bar{\boldsymbol{u}}
\quad \text{on } \Gamma_u. \tag{2}
$$

$$
\boldsymbol{\sigma}\boldsymbol{n}=\bar{\boldsymbol{t}}
\quad \text{on } \Gamma_t. \tag{3}
$$

在变分术语中，位移条件是**本质边界条件**，应限制试探函数空间；牵引条件是
**自然边界条件**，由分部积分产生的边界项进入弱形式 [1]。零牵引只是
$\bar{\boldsymbol t}=\boldsymbol 0$ 的特例。

## 3. 线弹性与相场退化本构


> **配套导读：线弹性与退化本构：从能量到应力**
>
> ### 3.1 为什么先写能量而不是先写应力
>
> DEM 并不是把任意一个力学量积分后当作损失函数，而是用神经网络近似**最小势能原理**
> 中的试探位移。因此，在讨论应力公式之前，需要先确认当前位移问题是否具有势能结构。
> 这里先固定相场 $\phi$，只考察位移场 $\boldsymbol u$ 的平衡。
>
> 对标准的最小势能型 DEM，材料内部储存的弹性能写为
>
> $$
> U[\boldsymbol u;\phi]
> =\int_\Omega
> \psi(\boldsymbol\varepsilon(\boldsymbol u),\phi)\,\mathrm d\Omega,
> $$
>
> 其中 $\psi$ 是单位体积的标量应变能密度。位移发生一个任意容许变化
> $\delta\boldsymbol u$ 时，应变变化为
>
> $$
> \delta\boldsymbol\varepsilon
> =\operatorname{sym}\nabla\delta\boldsymbol u.
> $$
>
> 对内部能量取一阶变分，可得
>
> $$
> \delta U
> =\int_\Omega
> \frac{\partial\psi}{\partial\boldsymbol\varepsilon}
> :\delta\boldsymbol\varepsilon\,\mathrm d\Omega.
> $$
>
> 另一方面，连续介质力学中的内虚功为
>
> $$
> \delta W_{\mathrm{int}}
> =\int_\Omega
> \boldsymbol\sigma:\delta\boldsymbol\varepsilon\,\mathrm d\Omega.
> $$
>
> 要让“能量的一阶变分”和“应力所做的内虚功”表示同一个物理量，两式的被积函数必须
> 一致，因此要求
>
> $$
> \boldsymbol\sigma=\frac{\partial\psi}{\partial\boldsymbol\varepsilon}.
> $$
>
> 再减去体力和边界面力的外力势，可构造总势能
>
> $$
> \Pi[\boldsymbol u;\phi]
> =U[\boldsymbol u;\phi]
> -\int_\Omega\boldsymbol b\cdot\boldsymbol u\,\mathrm d\Omega
> -\int_{\Gamma_t}\bar{\boldsymbol t}\cdot\boldsymbol u\,\mathrm d\Gamma.
> $$
>
> 平衡条件 $\delta\Pi=0$ 随后给出位移问题的弱形式；对内虚功分部积分，又能恢复正文
> 式 (1) 的平衡方程和式 (3) 的力边界条件。第 5、6 节将完整推导这一步。由此可见，
> 这里先写能量不是表达习惯，而是为了保证 DEM 最小化的标量泛函与原力学问题等价。
>
> 从几何上看，$\boldsymbol\sigma=\partial\psi/\partial\boldsymbol\varepsilon$ 类似于
> “保守向量场”：应力是应变空间中某个标量势函数的梯度，材料沿不同应变路径到达同一
> 状态时储存的弹性能相同。在线弹性中，若
> $\boldsymbol\sigma=\mathbb C:\boldsymbol\varepsilon$ 且弹性张量具有主对称性，便可取
>
> $$
> \psi=\frac12\boldsymbol\varepsilon:
> \mathbb C:\boldsymbol\varepsilon.
> $$
>
> 反之，若人为指定的应力映射不满足相应的可积条件，例如切线刚度缺少主对称性，就未必
> 存在单一的 $\psi$。塑性、摩擦和某些路径相关耗散模型也不能直接套用上述总势能；它们
> 通常需要增量势、耗散势、变分不等式或残差型方法。因此，“必须存在标量能量密度”只
> 针对这里采用的**保守弹性、最小势能型 DEM**，并不是所有力学问题使用神经网络求解
> 的普遍前提。
>
> ### 3.2 各向同性线弹性能量
>
> 三维小应变各向同性材料的能量可写成
>
> $$
> \psi_0
> =\frac12K(\operatorname{tr}\boldsymbol\varepsilon)^2
> +\mu\,\boldsymbol e:\boldsymbol e,
> $$
>
> 其中
>
> $$
> \boldsymbol e
> =\boldsymbol\varepsilon
> -\frac13\operatorname{tr}(\boldsymbol\varepsilon)\boldsymbol I.
> $$
>
> 第一项描述体积变化，第二项描述形状变化。体积模量 $K$ 控制压缩性，剪切模量
> $\mu$ 控制畸变。对近不可压材料 $\nu\to0.5$，$K/\mu$ 很大，会造成严重条件数问题；
> 这对有限元和神经优化都不友好。
>
> ### 3.3 平面应变为什么仍然使用三维分解
>
> 平面应变规定
>
> $$
> \varepsilon_{zz}=\varepsilon_{xz}=\varepsilon_{yz}=0,
> $$
>
> 但并不规定 $\sigma_{zz}=0$。由于泊松效应，约束面外应变会产生面外应力。因此体积
> 应变仍是三维迹，只是此时
>
> $$
> \operatorname{tr}\boldsymbol\varepsilon
> =\varepsilon_{xx}+\varepsilon_{yy}.
> $$
>
> 偏应变必须包含
>
> $$
> e_{zz}=-\frac13(\varepsilon_{xx}+\varepsilon_{yy}).
> $$
>
> 若直接把二维矩阵的迹减去二分之一而不是三分之一，得到的是二维代数意义的无迹分解，
> 并不是三维材料在平面应变下的体积—偏应变分解。
>
> ### 3.4 为什么断裂模型要区分活性能和非活性能
>
> 若把全部弹性能乘以 $g(\phi)$，材料在压缩下也可以通过产生损伤来释放能量，裂纹面
> 还可能互相穿透。Amor 分解写成
>
> $$
> \psi^+
> =\frac12K\langle\operatorname{tr}\boldsymbol\varepsilon\rangle_+^2
> +\mu\boldsymbol e:\boldsymbol e,
> $$
>
> $$
> \psi^-
> =\frac12K\langle\operatorname{tr}\boldsymbol\varepsilon\rangle_-^2.
> $$
>
> 只退化 $\psi^+$：
>
> $$
> \psi=g(\phi)\psi^++\psi^-.
> $$
>
> 可以用三个状态理解它：
>
> 1. 静水拉伸：只有正体积能且可退化，能够驱动断裂；
> 2. 静水压缩：只有 $\psi^-$，不通过损伤释放；
> 3. 单轴压缩或剪切：仍有偏应变能，因此 $\psi^+>0$，仍可能驱动损伤。
>
> 这里的静水状态指三维应变球形部分，此时 $\boldsymbol e=\boldsymbol0$。平面应变下
> 相等的两个面内正应变仍会产生 $e_{zz}$，不能直接当作真正的三维静水状态。
>
> 所以 Amor 分解不是“所有压缩都不损伤”，也不是严格接触模型。它只是用一个相对简单
> 的体积—偏应变规则缓解受压裂纹的非物理行为。
>
> ### 3.5 退化函数在做什么
>
> 常用形式
>
> $$
> g(\phi)=(1-\phi)^2+k
> $$
>
> 使活性应力近似按 $(1-\phi)^2$ 衰减。$k$ 是残余刚度：
>
> - $k=0$ 时完全损伤区失去椭圆性，位移可能不唯一；
> - $k>0$ 时数值问题更稳定，但裂纹带仍传递微小应力；
> - $k$ 太大会显著抬高结构残余承载力；
> - 采用上述形式时 $g(0)=1+k$，完好刚度被略微放大。
>
> 若改成归一化形式
>
> $$
> g(\phi)=(1-k)(1-\phi)^2+k,
> $$
>
> 则 $g(0)=1$。但不能只修改能量而忘记相场方程中的 $g'(\phi)$。
>
> ### 3.6 从能量求应力的实现检查
>
> 最可靠的单元测试不是手工比较很长的应力公式，而是验证有限差分：
>
> $$
> \frac{\psi(\boldsymbol\varepsilon+h\boldsymbol E^{ij},\phi)
> -\psi(\boldsymbol\varepsilon-h\boldsymbol E^{ij},\phi)}{2h}
> \approx \sigma_{ij}.
> $$
>
> 这里对角方向取
> $\boldsymbol E^{ii}=\boldsymbol e_i\otimes\boldsymbol e_i$；非对角方向取对称基底
>
> $$
> \boldsymbol E^{ij}
> =\frac12(\boldsymbol e_i\otimes\boldsymbol e_j
> +\boldsymbol e_j\otimes\boldsymbol e_i),\qquad i\ne j.
> $$
>
> 这样方向导数等于 $\sigma_{ij}$。若改用工程剪应变 $\gamma_{ij}=2\varepsilon_{ij}$，
> 必须同步使用与它功共轭的剪应力约定。
>
> 还可以检查自动微分得到的切线
> $\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon$ 是否对称。测试点应覆盖
> 体积拉伸、体积压缩、纯剪切以及接近
> $\operatorname{tr}\boldsymbol\varepsilon=0$ 的切换区域。



### 3.1 完整材料的线弹性

在无损伤情形，线弹性应力来自应变能密度 $\psi_0$：

$$
\boldsymbol{\sigma}
= \frac{\partial\psi_0}{\partial\boldsymbol{\varepsilon}}.
$$

对于三维各向同性材料，体积模量和剪切模量为

$$
K=\frac{E}{3(1-2\nu)},
\qquad
\mu=\frac{E}{2(1+\nu)}.
$$

在平面应变条件下仍采用三维体积-偏应变分解，即
$\varepsilon_{zz}=\varepsilon_{xz}=\varepsilon_{yz}=0$，但一般
$\sigma_{zz}\ne 0$。二维能量按单位厚度计算。令

$$
\varepsilon_{\mathrm{vol}}=\varepsilon_{xx}+\varepsilon_{yy},
\qquad
\varepsilon_{\mathrm{vol}}^+=\max(\varepsilon_{\mathrm{vol}},0),
\qquad
\varepsilon_{\mathrm{vol}}^-=\min(\varepsilon_{\mathrm{vol}},0),
$$

偏应变分量为

$$
e_{xx}=\varepsilon_{xx}-\frac{1}{3}\varepsilon_{\mathrm{vol}},
\quad
e_{yy}=\varepsilon_{yy}-\frac{1}{3}\varepsilon_{\mathrm{vol}},
\quad
e_{zz}=-\frac{1}{3}\varepsilon_{\mathrm{vol}},
\quad
e_{xy}=\varepsilon_{xy}.
$$

于是

$$
\operatorname{tr}(\boldsymbol{e}^2)
=e_{xx}^2+e_{yy}^2+e_{zz}^2+2e_{xy}^2.
$$

### 3.2 Amor 拉压分解与退化函数

令相场变量 $\phi\in[0,1]$，其中 $\phi=0$ 表示完好材料，$\phi=1$ 表示名义上的
完全断裂状态。常用的带残余刚度二次退化函数为

$$
g(\phi)=(1-\phi)^2+k.
$$

其中 $0<k\ll1$ 用于避免刚度矩阵或能量问题退化。该写法满足
$g(0)=1+k$、$g(1)=k$，所以完好刚度会被略微放大，而完全损伤区仍保留微小刚度。
若要求 $g(0)=1$ 精确成立，可以改用
$g(\phi)=(1-k)(1-\phi)^2+k$，但其后相场方程中的 $g'(\phi)$ 也必须同步修改。

Amor 分裂定义可退化的活性能和不可退化的非活性能：

$$
\psi^+(\boldsymbol{\varepsilon})
=\frac12K(\varepsilon_{\mathrm{vol}}^+)^2
+\mu\operatorname{tr}(\boldsymbol{e}^2),
$$

$$
\psi^-(\boldsymbol{\varepsilon})
=\frac12K(\varepsilon_{\mathrm{vol}}^-)^2.
$$

相场退化后的弹性能密度为

$$
\psi(\boldsymbol{\varepsilon},\phi)
=g(\phi)\psi^+(\boldsymbol{\varepsilon})
+\psi^-(\boldsymbol{\varepsilon}).
$$

对它关于应变求导，得到应力：

$$
\sigma_{xx}
=g(\phi)\left[K\varepsilon_{\mathrm{vol}}^+ +2\mu e_{xx}\right]
+K\varepsilon_{\mathrm{vol}}^-,
$$

$$
\sigma_{yy}
=g(\phi)\left[K\varepsilon_{\mathrm{vol}}^+ +2\mu e_{yy}\right]
+K\varepsilon_{\mathrm{vol}}^-,
\qquad
\tau_{xy}=g(\phi)2\mu\varepsilon_{xy}.
$$

这就是 Amor 等提出的体积-偏应变分裂 [10]，也可见相场断裂综述 [13]。这里的
$\psi^+$ 不宜简单理解为只在拉伸下出现的能量：它包含全部偏应变能，因此纯剪切乃至
单轴压缩都可能产生非零损伤驱动力。该分裂保护的是静水压缩部分，并不等价于完整的
裂纹面接触模型。

关键条件是
$\boldsymbol\sigma=\partial\psi/\partial\boldsymbol\varepsilon$：只有满足这一条件，
下面的位移弱形式才是同一个势能的一阶变分。若先人为指定应力方程，再找不到对应的
$\psi$，便不能假定最小化某个代理能量会恢复原方程。

## 4. 强形式 PINN：直接最小化平衡残差


> **配套导读：强形式 PINN：把微分方程直接变成监督信号**
>
> ### 4.1 强形式 PINN 的计算图
>
> 位移网络的计算链为
>
> $$
> \boldsymbol x
> \xrightarrow{\mathrm{NN}}
> \boldsymbol u_\theta
> \xrightarrow{\nabla_{\boldsymbol x}}
> \boldsymbol\varepsilon_\theta
> \xrightarrow{\text{constitutive}}
> \boldsymbol\sigma_\theta
> \xrightarrow{\nabla_{\boldsymbol x}\cdot}
> \boldsymbol r_\theta.
> $$
>
> 训练最小化残差平方：
>
> $$
> \mathcal L_{\rm eq}
> =\frac1N\sum_i
> \|\nabla\cdot\boldsymbol\sigma_\theta(\boldsymbol x_i)
> +\boldsymbol b(\boldsymbol x_i)\|^2.
> $$
>
> 它可以理解成没有标签的回归：PDE 在每个配点上生成目标值零。
>
> ### 4.2 为什么需要二阶空间导数
>
> 应变含位移一阶导数，应力散度再求一次导数，所以残差含位移二阶导数。训练参数梯度
> 还需要对这个计算图反向传播。这并不等于对参数求“四阶导数”，但计算图会更深、
> 显存更大，激活函数的高阶光滑性也更加重要。
>
> ReLU 网络在每个线性片内部二阶导数为零，因此不适合直接表示一般二阶强形式残差；
> $\tanh$、sine、softplus 等光滑激活更常见。另一方面，光滑激活又可能难以表示裂纹
> 附近的尖锐梯度，这构成表示能力与求导要求之间的张力。
>
> ### 4.3 为什么小残差仍可能对应错误解
>
> 常见原因包括：
>
> - 配点没有覆盖裂纹尖端或窄相场带；
> - PDE 残差与边界 loss 尺度严重失衡；
> - 网络偏向低频函数，在大部分区域残差很小但局部误差很大；
> - 二阶导数数值噪声大；
> - 强形式方程存在多个解，而训练找到的不是物理演化路径对应的解。
>
> 平均残差会稀释局部误差。除全局均方值外，还应查看最大残差、分区残差和空间分布。
>
> ### 4.4 软边界和硬边界
>
> 软边界使用罚项：
>
> $$
> \mathcal L
> =w_{\rm pde}\mathcal L_{\rm eq}
> +w_{\rm bc}\mathcal L_{\rm bc}.
> $$
>
> 它实现简单，但解依赖权重。硬边界使用
>
> $$
> \boldsymbol u_\theta
> =\boldsymbol u_D+B\boldsymbol N_\theta,
> $$
>
> 其中 $B=0$ 于 Dirichlet 边界。硬约束减少了优化器的任务，但 $B$ 必须：
>
> - 在所有对应边界上准确为零；
> - 不在域内产生不必要的高梯度；
> - 保留足够表达能力；
> - 对强形式具有足够光滑性。
>
> 复杂几何中构造这样的 $B$ 本身可能比求解 PDE 更困难。



### 4.1 网络试探解

用位移网络 $\boldsymbol{u}_\theta(\boldsymbol{x})$ 近似连续位移场。自动微分给出
一阶导数以形成应变，再给出应力；对应力再求一次空间导数即可得到强形式残差：

$$
r_x(\boldsymbol{x};\theta)
=\sigma_{xx,x}+\tau_{xy,y}+b_x,
$$

$$
r_y(\boldsymbol{x};\theta)
=\tau_{xy,x}+\sigma_{yy,y}+b_y.
$$

按照 Raissi 等的残差型 PINN 框架 [2]，在内部配点
$\{\boldsymbol{x}_i\}_{i=1}^{N_\Omega}$ 上可定义平衡损失

$$
\mathcal{L}_{\mathrm{eq}}
=\frac{1}{N_\Omega}\sum_{i=1}^{N_\Omega}
\left[r_x(\boldsymbol{x}_i)^2+r_y(\boldsymbol{x}_i)^2\right].
$$

其导数链为

$$
\boldsymbol{u}_\theta
\xrightarrow{\nabla}\boldsymbol{\varepsilon}
\xrightarrow{\text{constitutive law}}\boldsymbol{\sigma}
\xrightarrow{\nabla\cdot}\boldsymbol{r}.
$$

因而，若应力由位移梯度计算，训练位移网络时必须对空间坐标进行二阶自动微分。
Deep Ritz/DEM 文献将降低这一求导阶数列为能量形式的主要计算优势 [3,4]。

### 4.2 强形式的边界损失

若不使用硬约束，上下加载边界采用平方惩罚，例如上边界的竖向位移损失为

$$
\mathcal{L}_{\mathrm{top}}
=\frac{1}{N_t}\sum_{j=1}^{N_t}
\left(v_\theta(\boldsymbol{x}_j)-\bar v\right)^2.
$$

其余本质边界和强形式中显式检查的零牵引边界以同类项加入。总损失可概念化为

$$
\mathcal{L}_{\mathrm{strong}}
=w_{\mathrm{pde}}\mathcal{L}_{\mathrm{eq}}
+w_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}.
$$

Dirichlet 条件也可通过试探函数构造精确满足。一般写成 [16]

$$
\boldsymbol u_\theta(\boldsymbol x)
=\boldsymbol u_D(\boldsymbol x)
+B(\boldsymbol x)\boldsymbol N_\theta(\boldsymbol x),
$$

其中 $\boldsymbol u_D=\bar{\boldsymbol u}$ 且 $B=0$ 在 $\Gamma_u$ 上。这样
$\boldsymbol u_\theta$ 对任意网络参数都属于运动学可行空间。$B$ 的正则性必须满足
控制方程或能量所需的 Sobolev 正则性；具体形式属于几何与边界条件的实现问题，不是
PINN 理论本身的一部分。

## 5. 从平衡方程到弱形式


> **配套导读：从平衡方程到弱形式：分部积分改变了什么**
>
> ### 5.1 测试函数不是额外的物理未知量
>
> 取虚位移 $\delta\boldsymbol u$，它表示任意允许的微小位移变化，并满足
>
> $$
> \delta\boldsymbol u=\boldsymbol0\quad\text{on }\Gamma_u.
> $$
>
> 将平衡方程与它做内积并积分：
>
> $$
> \int_\Omega
> \delta u_i(\sigma_{ij,j}+b_i)\,d\Omega=0.
> $$
>
> 对第一项分部积分：
>
> $$
> \int_\Omega\delta u_i\sigma_{ij,j}\,d\Omega
> =\int_{\partial\Omega}\delta u_i\sigma_{ij}n_j\,d\Gamma
> -\int_\Omega\delta u_{i,j}\sigma_{ij}\,d\Omega.
> $$
>
> 由于应力对称，
>
> $$
> \delta u_{i,j}\sigma_{ij}
> =\boldsymbol\varepsilon(\delta\boldsymbol u):\boldsymbol\sigma.
> $$
>
> 于是得到内虚功等于外虚功。
>
> ### 5.2 分部积分带来的两个结果
>
> 第一，应力的空间导数被转移到测试函数上，位移只需要一阶弱导数。第二，边界项
>
> $$
> \int_{\partial\Omega}
> \delta\boldsymbol u\cdot\boldsymbol\sigma\boldsymbol n\,d\Gamma
> $$
>
> 自动暴露了牵引边界条件。这就是“自然边界”的数学来源，而不是 DEM 自动猜出了边界。
>
> ### 5.3 弱形式不等于积分残差平方
>
> 以下两个对象不同：
>
> $$
> \int_\Omega r(\boldsymbol x)^2\,d\Omega,
> $$
>
> 和
>
> $$
> \int_\Omega w(\boldsymbol x)r(\boldsymbol x)\,d\Omega
> =0\quad\text{for all }w.
> $$
>
> 前者是最小二乘目标；后者是弱方程。Galerkin、Petrov–Galerkin、VPINN 和最小二乘
> 方法对测试空间的选择不同。DEM 又更特殊：它不显式选取一组测试网络，而是利用能量
> 的一阶变分隐式产生测试方向。



强形式要求点态满足 $\nabla\cdot\boldsymbol{\sigma}=0$。弱形式改为：选任意满足
齐次 Dirichlet 条件的虚位移 $\delta\boldsymbol{u}$，使

$$
\delta\boldsymbol{u}=\boldsymbol{0}\quad\text{on }\Gamma_u,
$$

并令平衡方程乘以它后在区域积分：

$$
\int_\Omega
\delta\boldsymbol{u}\cdot(\nabla\cdot\boldsymbol{\sigma}+\boldsymbol{b})\,d\Omega=0.
$$

对应力散度项分部积分，得到

$$
\int_\Omega
\boldsymbol{\varepsilon}(\delta\boldsymbol{u}):\boldsymbol{\sigma}\,d\Omega
=\int_\Omega\delta\boldsymbol{u}\cdot\boldsymbol{b}\,d\Omega
+\int_{\Gamma_t}\delta\boldsymbol{u}\cdot\bar{\boldsymbol{t}}\,d\Gamma.
$$

这就是虚功原理 [1]。它只要求 $\boldsymbol{u}\in(H^1(\Omega))^d$，因而可以容纳
比强形式更低的正则性；对神经网络而言，能避免显式构造应力的空间散度。

请注意一个常见误解：弱形式并不等于“把残差平方后积分”。弱形式要求对所有允许的
虚位移成立。在线弹性或存在势能的本构中，它等价于总势能的驻值条件，这才给出 DEM
可以直接最小化的标量目标 [3,4]。

## 6. 从弱形式到 DEM：总势能驻值


> **配套导读：从弱形式到 DEM：为什么一个标量能量就够了**
>
> ### 6.1 总势能的三部分
>
> 位移问题的总势能为
>
> $$
> \Pi[\boldsymbol u;\phi]
> =U[\boldsymbol u;\phi]-W_b[\boldsymbol u]-W_t[\boldsymbol u],
> $$
>
> 其中
>
> $$
> U=\int_\Omega\psi\,d\Omega
> $$
>
> 是内能，体力和给定牵引的势分别是
>
> $$
> W_b=\int_\Omega\boldsymbol b\cdot\boldsymbol u\,d\Omega,
> \qquad
> W_t=\int_{\Gamma_t}\bar{\boldsymbol t}\cdot\boldsymbol u\,d\Gamma.
> $$
>
> 位移控制边界不是外功积分，而是可行空间的一部分。把同一个位移载荷既硬编码进
> 试探函数、又作为外力功加入，会重复施加载荷。
>
> ### 6.2 驻值和极小值不是同义词
>
> $\delta\Pi=0$ 只表示驻点。若二阶变分正定，它才是稳定极小值。固定相场且残余刚度
> 为正时，位移子问题通常具有稳定性；完整位移—相场能量则是非凸的，可能存在多个
> 局部极小值和能垒。
>
> 这解释了为什么：
>
> - 不同随机初始化可能给出不同裂纹路径；
> - 一次性联合优化和交替最小化可能得到不同解；
> - 优化器跳过能垒可能产生过早开裂；
> - loss 更低不一定意味着沿准静态路径更物理。
>
> ### 6.3 网络参数化后发生了什么
>
> 把无限维函数空间限制到神经网络族：
>
> $$
> \theta^*=\arg\min_\theta\Pi[\boldsymbol u_\theta;\phi].
> $$
>
> 参数梯度可写成
>
> $$
> \frac{\partial\Pi}{\partial\theta}
> =\int_\Omega
> \boldsymbol\sigma:
> \boldsymbol\varepsilon\left(
> \frac{\partial\boldsymbol u_\theta}{\partial\theta}
> \right)d\Omega
> -\frac{\partial W_{\rm ext}}{\partial\theta}.
> $$
>
> 从变分角度看，
> $\partial\boldsymbol u_\theta/\partial\theta$ 就是网络参数变化诱导出的测试方向。
> 网络只能检验自己切空间中的方向，所以参数梯度为零不等于连续泛函对所有允许方向的
> 变分都为零。增加网络容量、改变特征映射或局部加密，实质上是在扩大可检验方向。
>
> ### 6.4 求积是 DEM 的“离散物理”
>
> 若使用确定性求积，
>
> $$
> \Pi_\theta\approx\sum_iw_i\psi(\boldsymbol x_i)-W_{\rm ext}.
> $$
>
> 若均匀随机采样，则
>
> $$
> \int_\Omega\psi\,d\Omega
> \approx\frac{|\Omega|}{N}\sum_i\psi(\boldsymbol x_i).
> $$
>
> 若采样密度为 $p(\boldsymbol x)$，应使用重要性权重
>
> $$
> \frac1N\sum_i\frac{\psi(\boldsymbol x_i)}{p(\boldsymbol x_i)}.
> $$
>
> 仅在裂纹尖端增加样本、仍对所有样本做普通平均，会改变被最小化的能量，而不只是提高
> 精度。机器学习中的数据分布在这里直接定义了离散积分测度。
>
> ### 6.5 为什么 energy loss 通常不趋近零
>
> 平衡状态满足能量梯度为零，而不是能量值为零。在线弹性力控制问题中，平衡点总势能
> 甚至常为负。不能用“energy loss 是否接近零”判断收敛，应检查：
>
> - 参数梯度或优化增量；
> - 平衡残差；
> - 反力；
> - 位移和相场迭代差；
> - 能量随载荷步的变化。
>
> ### 6.6 log-energy 的作用边界
>
> 若 $\mathcal E_\theta>0$，
>
> $$
> \nabla_\theta\log\mathcal E_\theta
> =\frac{\nabla_\theta\mathcal E_\theta}{\mathcal E_\theta}.
> $$
>
> 它不改变单独能量项的驻点，只改变梯度尺度。加入 BC 罚项、正则项或其他 loss 后，
> 只对其中一项取对数会改变相对权重。若总势能含外力势并可能为负，不能直接取对数。



### 6.1 总势能泛函

对给定的相场 $\phi$，定义位移问题的总势能

$$
\Pi[\boldsymbol{u};\phi]
=\int_\Omega\psi(\boldsymbol{\varepsilon}(\boldsymbol{u}),\phi)\,d\Omega
-\int_\Omega\boldsymbol{b}\cdot\boldsymbol{u}\,d\Omega
-\int_{\Gamma_t}\bar{\boldsymbol{t}}\cdot\boldsymbol{u}\,d\Gamma.
$$

当应力满足 $\boldsymbol{\sigma}=\partial\psi/\partial\boldsymbol{\varepsilon}$ 时，
对任意允许的虚位移有

$$
\delta\Pi
=\int_\Omega
\boldsymbol{\sigma}:\boldsymbol{\varepsilon}(\delta\boldsymbol{u})\,d\Omega
-\int_\Omega\boldsymbol{b}\cdot\delta\boldsymbol{u}\,d\Omega
-\int_{\Gamma_t}\bar{\boldsymbol{t}}\cdot\delta\boldsymbol{u}\,d\Gamma.
$$

因此，$\delta\Pi=0$ 恰好恢复上一节的虚功原理。对固定 $\phi$ 且能量具有稳定
极小值的位移子问题，求解可以写成

$$
\boldsymbol{u}^* = \arg\min_{\boldsymbol{u}\in\mathcal{U}_{\bar u}}
\Pi[\boldsymbol{u};\phi].
$$

其中 $\mathcal{U}_{\bar u}$ 表示满足 Dirichlet 条件的可行函数空间。采用 Amor 分裂
后，本构会随 $\operatorname{tr}\boldsymbol\varepsilon$ 的符号切换，因而这里通常是
分段线性而非严格意义上的单一线性弹性问题。

### 6.2 Deep Ritz/DEM 离散化

用网络 $\boldsymbol{u}_\theta$ 限制试探空间，并以积分点和权重近似域积分：

$$
\Pi_\theta
\approx
\sum_{i=1}^N w_i\,
\psi\left(\boldsymbol{\varepsilon}(\boldsymbol{u}_\theta(\boldsymbol{x}_i)),
\phi(\boldsymbol{x}_i)\right)
-\text{external work}.
$$

Deep Ritz 方法以神经网络替代传统有限维试探空间 [3]；在计算力学中，以总势能
训练网络的同类方法通常称为 Deep Energy Method [4]。若积分点由非均匀密度
$p(\boldsymbol{x})$ 独立采样，域积分的无偏重要性采样估计为

$$
\widehat\Pi_\theta
=\frac1N\sum_{i=1}^N
\frac{\psi(\boldsymbol{x}_i)}{p(\boldsymbol{x}_i)}.
$$

若使用确定性求积，则直接写为

$$
\widehat\Pi_\theta
=\sum_{i=1}^N w_i\psi(\boldsymbol x_i)-\widehat W_{\mathrm{ext}}.
$$

两种形式必须逼近同一个连续积分，不能因为程序接口使用 `sum` 或 `mean` 而改变物理
能量的尺度。求积误差会直接改变神经网络看到的能量景观；Berrone 等系统讨论了
求积规则和测试函数对变分 PINN 的影响 [5]，Manav 等也将梯度计算与求积视为相场
断裂 DRM 的关键组成 [15]。

DEM 的位移损失是

$$
\mathcal{L}_{\mathrm{DEM}}
=\Pi_\theta+w_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}.
$$

若使用软边界罚项，$w_{\mathrm{bc}}$ 还必须补偿势能与位移均方误差之间的量纲和离散
归一化差异，不能仅凭两个 loss 的数值大小选取。

自然边界条件已经出现在变分推导的边界项里。因此，对零牵引的外部自由边，以及在
几何上显式建模为内部边界的狭缝，DEM 不需要额外的牵引残差罚项；只有尚未硬编码的
Dirichlet 约束仍需显式施加。需要区别的是，AT2 模型中的弥散裂纹是域内损伤带，
并不存在一个自动生成的内部几何边界。其应力释放来自 $g(\phi)$ 的退化，而且在
$k>0$ 时并不严格为零。

### 6.3 为什么 DEM 仅需一阶导数

DEM 的链条止于应变：

$$
\boldsymbol{u}_\theta
\xrightarrow{\nabla}\boldsymbol{\varepsilon}
\xrightarrow{}\psi
\xrightarrow{\text{quadrature}}\Pi_\theta.
$$

相比强形式少了 $\nabla\cdot\boldsymbol{\sigma}$，所以训练位移网络时不需要二阶空间
导数。这是 DEM 降低空间求导阶数的数学来源；实际速度是否更快还取决于域积分、采样
和优化成本。它也不自动保证更高精度，积分误差、边界处理和网络试探空间的表示误差
仍会决定最终误差。

### 6.4 可选的 log-energy 重标度

Manav 等在其相场断裂 DRM 中采用 [15]

$$
\mathcal{L}_{\mathrm{energy}}=\log(\mathcal E_\theta).
$$

该文在硬位移控制、无体力和无面力的设置下，由非负的弹性、断裂和不可逆罚能组成
$\mathcal E_\theta$。还需避免把该方法描述成纯自动微分、无网格 DRM：其网络输出
节点场值，单元内场梯度由有限元形函数计算，积分采用单元 Gauss 求积。

更一般地，训练实现可能使用 $\log(\Pi_\theta+\epsilon)$。只要
$\Pi_\theta+\epsilon>0$，对能量项本身有

$$
\nabla_\theta\log(\Pi_\theta+\epsilon)
=\frac{\nabla_\theta\Pi_\theta}{\Pi_\theta+\epsilon}.
$$

因此，在 $\Pi_\theta+\epsilon>0$ 且只优化该项时，它不改变驻点，却会重标度梯度。
这不是变分原理的新组成，而是训练时可选的单调优化变换。若总势能因外力势而可能为负，
则对数并不直接有定义；若同时存在软边界罚项，它还会改变各项相对权重。因此必须把
“物理泛函”和“用于训练的单调变换”分开陈述。

## 7. 加入 AT2 相场断裂


> **配套导读：加入 AT2 相场断裂：把未知裂纹变成未知场**
>
> ### 7.1 Griffith 模型为什么难以直接计算
>
> 经典 Griffith 能量可概念化为
>
> $$
> \mathcal E(\boldsymbol u,\Gamma)
> =\int_{\Omega\setminus\Gamma}\psi_0\,d\Omega
> +G_c|\Gamma|-W_{\rm ext}.
> $$
>
> 未知量不仅有位移，还有裂纹集合 $\Gamma$。裂纹可以萌生、延伸、分叉和合并，拓扑
> 变化使直接优化极其困难。
>
> 相场方法用连续函数 $\phi$ 代替显式裂纹集合，把“寻找裂纹几何”变成“寻找一个窄
> 损伤带”。
>
> ### 7.2 AT2 裂纹密度的两个竞争项
>
> AT2 裂纹表面密度为
>
> $$
> \gamma(\phi,\nabla\phi)
> =\frac{\phi^2}{2l_0}
> +\frac{l_0}{2}|\nabla\phi|^2.
> $$
>
> 第一项惩罚大范围损伤，倾向于让 $\phi$ 回到零；第二项惩罚过快变化，倾向于把裂纹
> 扩散开。两者竞争后形成宽度为 $O(l_0)$ 的过渡带。
>
> 对一条无限长直裂纹，在法向坐标 $s$ 上的局部一维最优剖面是
>
> $$
> \phi(s)=e^{-|s|/l_0}.
> $$
>
> 代入可得单位裂纹长度对应的归一化表面能：
>
> $$
> \int_{-\infty}^{\infty}
> \left(
> \frac{\phi^2}{2l_0}
> +\frac{l_0}{2}(\phi')^2
> \right)ds=1.
> $$
>
> 因此乘以 $G_c$ 后恢复每单位新裂纹面积所需的能量 $G_c$。这也是预裂纹常使用指数
> 相场剖面的原因。
>
> ### 7.3 $l_0$ 既是物理正则化也是数值分辨率
>
> $l_0$ 越小，模型越接近 sharp crack，但相场梯度越陡，积分点或单元必须更密，网络
> 也更难表示。不能在固定离散分辨率下不断减小 $l_0$，然后把不收敛归因于优化器。
>
> 应进行至少两层收敛：
>
> 1. 固定 $l_0$，令数值尺度 $h/l_0$ 收敛；
> 2. 在每个 $l_0$ 都充分解析后，再研究 $l_0/a\to0$。
>
> ### 7.4 完整耦合能量如何驱动裂纹
>
> 耦合能量为
>
> $$
> \mathcal E[\boldsymbol u,\phi]
> =\int_\Omega
> \left[
> g(\phi)\psi^+(\boldsymbol\varepsilon)
> +\psi^-(\boldsymbol\varepsilon)
> +G_c\gamma(\phi,\nabla\phi)
> \right]d\Omega-W_{\rm ext}.
> $$
>
> 增大 $\phi$ 会降低活性弹性能，却会增加裂纹表面能。裂纹在释放的弹性能足以补偿
> 新增断裂能时变得有利。这正是 Griffith 能量竞争在弥散表示中的版本。
>
> AT2 与 AT1 还有一个重要差别：理想均匀 AT2 模型通常没有严格的初始弹性阈值，
> 非零载荷下就可能出现很小的弥散损伤；AT1 具有更明确的起裂阈值。对预制 Griffith
> 裂纹，重点通常是裂纹扩展而非无缺陷材料萌生，但解释载荷—位移曲线时仍应知道这一区别。
>
> ### 7.5 为什么必须施加不可逆性
>
> 若每个载荷步都无约束最小化当前能量，卸载后弹性能驱动力减小，最优 $\phi$ 可能下降，
> 即裂纹愈合。脆性断裂模型要求
>
> $$
> \phi_n\ge\phi_{n-1}.
> $$
>
> 这把普通 Euler–Lagrange 等式变成变分不等式。类比带约束优化：
>
> - $\phi-\phi_{n-1}\ge0$ 是可行性；
> - 相场能量梯度给出损伤判据；
> - 两者满足 Kuhn–Tucker 互补条件；
> - 未继续损伤处约束活跃，继续损伤处能量梯度达到临界状态。
>
> ### 7.6 三种不可逆处理不能混为一谈
>
> #### 直接约束或主动集
>
> 最接近原始增量变分问题，能够明确检查可行性和互补条件，但神经网络参数与点值约束
> 之间的映射复杂。
>
> #### 罚函数或增广拉格朗日
>
> 加入
>
> $$
> \frac{\gamma_{\rm ir}}2
> \int_\Omega\langle\phi-\phi_{n-1}\rangle_-^2\,d\Omega.
> $$
>
> 它保留可微标量目标，适合 DRM。罚参数太小会愈合，太大会造成病态；增广拉格朗日能
> 减轻单纯大罚参数的问题。
>
> #### 历史场
>
> 定义
>
> $$
> \mathcal H_n(\boldsymbol x)
> =\max(\mathcal H_{n-1}(\boldsymbol x),\psi_n^+(\boldsymbol x)).
> $$
>
> 然后冻结 $\mathcal H_n$ 求相场。它简单稳健，但不再是原始联合不可逆最小化的严格
> 等价形式。历史场保存的是过去最大驱动力，而不是直接保存可行集。
>
> ### 7.7 历史场相场方程如何读
>
> 冻结历史场后的方程是
>
> $$
> \frac{G_c}{l_0}\phi
> -G_cl_0\Delta\phi
> -2(1-\phi)\mathcal H=0.
> $$
>
> 三项分别代表：
>
> - $\frac{G_c}{l_0}\phi$：局部恢复到完好状态的代价；
> - $-G_cl_0\Delta\phi$：控制空间平滑和裂纹带宽度；
> - $-2(1-\phi)\mathcal H$：历史拉伸能提供的损伤驱动力。
>
> 整理后可看出局部反应系数：
>
> $$
> -G_cl_0\Delta\phi
> +\left(\frac{G_c}{l_0}+2\mathcal H\right)\phi
> =2\mathcal H.
> $$
>
> 固定 $\mathcal H$ 时这是线性椭圆方程；与位移和历史场更新组合后，整体过程仍是
> 非线性的。
>
> ### 7.8 三种神经求解路线
>
> 1. 位移和相场都用强形式 PINN：两个网络都涉及二阶空间导数；
> 2. 固定历史场的分块 DEM：位移和相场分别最小化各自能量，只需一阶导数；
> 3. 联合 DRM：一个或两个网络直接最小化带不可逆处理的完整能量。
>
> 分块法更容易诊断每个子问题；联合法形式紧凑，但非凸优化更难。对研究算法机制，
> 建议先建立可靠的分块基线，再尝试联合训练。



### 7.1 AT2 裂纹表面能

Francfort 与 Marigo 将 Griffith 断裂写成自由不连续集上的全局能量最小化问题 [6]；
Bourdin 等利用 Ambrosio-Tortorelli 型正则化把裂纹集合近似为连续相场 [7,8]。
在本文采用的“$\phi=0$ 完好、$\phi=1$ 断裂”约定下，AT2 裂纹密度为

$$
\gamma(\phi,\nabla\phi)
=\frac{1}{2l_0}\phi^2+\frac{l_0}{2}|\nabla\phi|^2.
$$

对完整耦合问题，形式上的总能量为

$$
\mathcal{E}[\boldsymbol{u},\phi]
=\int_\Omega
\left[
g(\phi)\psi^+(\boldsymbol{\varepsilon})
+\psi^-(\boldsymbol{\varepsilon})
+G_c\gamma(\phi,\nabla\phi)
\right]d\Omega
-W_{\mathrm{ext}}.
$$

这是常见的二次梯度损伤形式 [9,13]。若只对当前总能量作无约束最小化，卸载时
$\phi$ 可能减小，即出现裂纹愈合；所以准静态演化还必须包含不可逆约束。

### 7.2 严格的不可逆增量问题

在载荷步 $n$，严格的增量变分问题是 [8,15]

$$
(\boldsymbol u_n,\phi_n)
\in\operatorname*{argmin}_{\boldsymbol u\in\mathcal U_n,\,
\phi\in\mathcal D_{n-1}}\mathcal E_n[\boldsymbol u,\phi],
$$

其中

$$
\mathcal D_{n-1}
=\{\phi\in H^1(\Omega):\phi\ge\phi_{n-1}\ \text{a.e. in }\Omega\}.
$$

该不等式约束导致变分不等式和 Kuhn-Tucker 互补条件，而不是处处成立的单个等式。
若数值表示不能自动维持 $\phi\in[0,1]$，还应把上下界加入可行集或通过有界输出映射
施加。若 $\phi_n$ 是极小解，则对任意 $\eta\in\mathcal D_{n-1}$ 有

$$
\delta_\phi\mathcal E_n[\boldsymbol u_n,\phi_n](\eta-\phi_n)\ge0.
$$

其自然边界也一般表现为与不可逆约束相容的互补条件，而不是在整个边界上无条件取
$\nabla\phi\cdot\boldsymbol n=0$。

这才是“裂纹不可愈合”的原始数学表述。可用主动集、投影、罚函数或增广拉格朗日法
处理；Gerasimov 与 De Lorenzis 给出了罚参数分析 [14]，Manav 等采用能量罚项以保留
可供 DRM 最小化的整体变分结构 [15]。

### 7.3 历史场近似及其边界

Miehe 等提出的常用历史场方法 [11,12] 定义

$$
\mathcal H(\boldsymbol x,t)
=\max_{\tau\le t}\psi^+(\boldsymbol\varepsilon(\boldsymbol u(\boldsymbol x,\tau))).
$$

冻结 $\mathcal H$ 后，可定义相场子问题的代理泛函

$$
\mathcal E_\phi[\phi;\mathcal H]
=\int_\Omega\left[
(1-\phi)^2\mathcal H
+G_c\left(\frac{\phi^2}{2l_0}
+\frac{l_0}{2}|\nabla\phi|^2\right)
\right]d\Omega.
$$

严格从 $g(\phi)\mathcal H$ 出发还会有 $k\mathcal H$；冻结 $\mathcal H$ 后它与
$\phi$ 无关，因此不改变原始代理泛函的驻点。若再对能量施加对数等非线性变换，
这个常数会改变梯度的整体缩放，省略时应显式说明。

令其一阶变分为零，并采用自然边界
$\nabla\phi\cdot\boldsymbol n=0$，得到

$$
\frac{G_c}{l_0}\phi
-G_cl_0\Delta\phi
-2(1-\phi)\mathcal{H}=0.
$$

这个等式是**冻结历史场后的代理子问题**，不是原始不可逆联合最小化问题的严格
Euler-Lagrange 等价式。历史场法工程上稳健且广泛使用，但可能牺牲严格的全局
变分结构 [13,15]。因此，采用历史场、罚函数或直接变分不等式是三种不同的建模/
离散选择，不能在推导中互相替代而不加说明。

### 7.4 强形式 PINN、分块 DEM 与联合 DRM

相场方程可以像位移方程一样有不同的神经离散方式：

- 对上式残差做配点平方，得到相场的强形式 PINN；它需要 $\Delta\phi$，即网络输出的
   二阶空间导数。
- 固定 $\boldsymbol u$ 或 $\mathcal H$，直接最小化 $\mathcal E_\phi$，得到相场子问题
   的能量形式，只需要 $\nabla\phi$。
- 对带不可逆罚项的完整 $\mathcal E(\boldsymbol u,\phi)$ 直接联合最小化，得到
   Manav 等采用的联合 DRM [15]。
- 交替最小化则依次固定一个场，求另一个场；它与一次性联合优化具有不同的数值轨迹，
   对非凸问题也可能到达不同的局部极小值 [13]。

## 8. 一致的无量纲化


> **配套导读：无量纲化：不仅是物理简化，也是优化预条件**
>
> ### 8.1 为什么神经网络尤其需要无量纲化
>
> 若坐标量级是 $10^{-3}$ m，应力是 $10^{11}$ Pa，断裂能是 $10^2$ J/m²，直接送入
> 网络和 loss 会导致激活饱和、梯度尺度悬殊和优化器步长失配。
>
> 取长度尺度 $L$ 和应力尺度 $E$：
>
> $$
> \boldsymbol x^*=\frac{\boldsymbol x}{L},
> \qquad
> \boldsymbol u^*=\frac{\boldsymbol u}{L},
> \qquad
> \boldsymbol\sigma^*=\frac{\boldsymbol\sigma}{E}.
> $$
>
> 此时应变不变，因为
>
> $$
> \nabla_{\boldsymbol x}\boldsymbol u
> =\nabla_{\boldsymbol x^*}\boldsymbol u^*.
> $$
>
> ### 8.2 为什么出现 $\kappa=G_c/(EL)$
>
> 弹性能密度尺度是 $E$，裂纹密度尺度是 $1/L$，所以断裂能体密度尺度是 $G_c/L$。
> 两者之比为
>
> $$
> \kappa=\frac{G_c}{EL}.
> $$
>
> 它表示在结构尺度 $L$ 上，创造裂纹的代价相对于弹性储能的大小。无量纲后真正控制
> 问题的不是 $E$、$G_c$ 和 $L$ 各自的绝对值，而是它们组成的无量纲组合。
>
> ### 8.3 历史场的两种常用尺度
>
> 若按弹性模量缩放：
>
> $$
> \mathcal H_E^*=\frac{\mathcal H}{E}.
> $$
>
> 若按特征断裂能密度缩放：
>
> $$
> \overline{\mathcal H}
> =\frac{\mathcal H}{G_c/l_0}
> =\frac{\mathcal H l_0}{G_c}.
> $$
>
> 二者关系为
>
> $$
> \overline{\mathcal H}
> =l_0^*\widehat{\mathcal H},
> \qquad
> \widehat{\mathcal H}=\frac{\mathcal H L}{G_c}.
> $$
>
> 实现中最危险的情况不是选择了不同尺度，而是输入按一种尺度归一化，方程系数却来自
> 另一种尺度。
>
> ### 8.4 loss 权重也必须跟着尺度走
>
> 能量是体积分，位移 BC loss 常是边界点上的位移均方值，二者量纲和离散归一化不同。
> 即使所有物理变量已无量纲化，`sum` 与 `mean` 的选择仍会让权重依赖点数。
>
> 建议先定义每个 loss 所逼近的连续量，再推导离散系数，而不是训练后根据曲线大小手工
> 反复调权。



取特征长度 $L$、应力尺度 $E$，定义

$$
x^*=x/L,
\qquad
\boldsymbol u^*=\boldsymbol u/L,
\qquad
\boldsymbol\sigma^*=\boldsymbol\sigma/E,
\qquad
l_0^*=l_0/L.
$$

应变在该选择下不变。再定义

$$
K^*=K/E,
\qquad
\mu^*=\mu/E,
\qquad
\kappa=\frac{G_c}{EL},
\qquad
\boldsymbol b^*=\frac{L\boldsymbol b}{E},
\qquad
\bar{\boldsymbol t}^*=\frac{\bar{\boldsymbol t}}{E}.
$$

以 $EL^d$ 归一化 $d$ 维总能量，得到

$$
\mathcal E^*
=\int_{\Omega^*}\left[
g(\phi)\psi^{+*}+\psi^{-*}
+\kappa\left(
\frac{\phi^2}{2l_0^*}
+\frac{l_0^*}{2}|\nabla^*\phi|^2
\right)\right]d\Omega^*-W_{\mathrm{ext}}^*.
$$

令 $\mathcal H_E^*=\mathcal H/E$，历史场相场方程变为

$$
\frac{\kappa}{l_0^*}\phi
-\kappa l_0^*\Delta^*\phi
-2(1-\phi)\mathcal H_E^*=0.
$$

等价地，令 $\widehat{\mathcal H}=\mathcal H L/G_c=\mathcal H_E^*/\kappa$，可写成

$$
\frac{\phi}{l_0^*}-l_0^*\Delta^*\phi
-2(1-\phi)\widehat{\mathcal H}=0.
$$

这里必须统一选择历史场尺度。若改为

$$
\overline{\mathcal H}
=\frac{\mathcal H}{G_c/l_0}
=\frac{\mathcal H l_0}{G_c}
=l_0^*\widehat{\mathcal H},
$$

则相场方程等价地写成

$$
\phi-(l_0^*)^2\Delta^*\phi
-2(1-\phi)\overline{\mathcal H}=0.
$$

Manav 等给出了另一套以断裂能和临界应变为核心的无量纲化 [15]；不同方案都可以
成立，但同一推导内必须保持长度、能量和历史场尺度自洽。

## 9. 二维 Griffith 中心裂纹算例


> **配套导读：二维 Griffith 中心裂纹：把所有知识点串成一个算例**
>
> ### 9.1 解析基准告诉我们什么
>
> 无限板中心裂纹长度为 $2a$，远场 I 型应力为 $\sigma_\infty$：
>
> $$
> K_I=\sigma_\infty\sqrt{\pi a}.
> $$
>
> 能量释放率为
>
> $$
> G=\frac{K_I^2}{E'},
> \qquad
> E'=
> \begin{cases}
> E,&\text{平面应力},\\
> E/(1-\nu^2),&\text{平面应变}.
> \end{cases}
> $$
>
> 令 $G=G_c$ 得
>
> $$
> \sigma_c=\sqrt{\frac{E'G_c}{\pi a}}.
> $$
>
> 这提供的是 sharp-crack、无限域极限下的物理参考，不是有限相场计算必须逐位匹配的
> 标签。
>
> ### 9.2 有限板为什么不能直接套无限板公式
>
> 若板宽与裂纹长度同量级，边界会改变尖端场，通常写成
>
> $$
> K_I
> =F(a/W,H/W,\text{边界与加载方式})
> \sigma_\infty\sqrt{\pi a},
> $$
>
> 其中 $F$ 是几何与加载修正因子；只有在特定的无限长有限宽板基准中，才常简化写成
> $F(a/W)$。位移加载还会使名义应力随结构刚度退化而变化。比较临界载荷前必须明确：
>
> - 使用远场应力、名义应力还是边界反力；
> - 裂纹长度 $a$ 的定义；
> - 平面应力还是平面应变；
> - 有限尺寸修正；
> - 相场预裂纹的有效长度。
>
> ### 9.3 初始裂纹不是一条“画出来的线”
>
> 若计算域没有真的切开，$\Gamma_0$ 只是用于定义初始相场的几何对象。常用初始化为
>
> $$
> \phi_0(\boldsymbol x)
> \approx e^{-\operatorname{dist}(\boldsymbol x,\Gamma_0)/l_0}.
> $$
>
> 还需要确保后续不可逆算法保留它。若只初始化网络输出，但第一步优化没有
> $\phi\ge\phi_0$、不可逆罚项，或与预裂纹相容的初始历史场 $\mathcal H_0$，
> 优化器仍可能立即把裂纹修复。普通的零初始历史场不会自动记住 $\phi_0$。
>
> 另一种做法是把狭缝从几何域中删除。此时狭缝表面是显式零牵引边界；若同时又在同一
> 位置放置完整相场裂纹能，需要检查是否重复表示和重复计入断裂能。
>
> ### 9.4 一个推荐的基线流程
>
> 1. 先在固定 $\phi_0$ 下求线弹性位移，验证边界、反力和裂纹尖端场；
> 2. 用高分辨率有限元或可信文献数据建立参考；
> 3. 固定历史场，单独验证相场强形式与能量形式一致；
> 4. 实现小载荷增量的交替求解；
> 5. 检查每个载荷步的内迭代收敛和不可逆性；
> 6. 做求积或采样密度关于 $l_0$ 的收敛；
> 7. 再比较强 PINN、DEM、联合 DRM 或新的训练技巧。
>
> 不要同时引入新网络结构、新采样策略、新无量纲化和新不可逆算法，否则实验结果无法
> 归因。
>
> ### 9.5 应该画哪些图
>
> 仅展示最终相场云图不足以验证方法。至少建议包括：
>
> - 载荷—位移或反力—位移曲线；
> - 不同载荷步的 $\phi$ 等值图；
> - 裂纹中心线上的相场剖面；
> - 弹性能、断裂能、罚能和外功；
> - 平衡残差空间分布；
> - $\min(\phi_n-\phi_{n-1})$ 或不可逆违约积分；
> - 沿多个路径计算的 $J$ 积分；
> - 随 $h/l_0$、点数和随机种子的误差统计。
>
> 若只有 loss 曲线，无法区分优化收敛、积分偏差和物理解错误。



### 9.1 几何、载荷与平面假设

作为本文推导的最小验证问题，可取宽度 $2W$、高度 $2H$ 的矩形板

$$
\Omega=(-W,W)\times(-H,H),
$$

其中初始中心裂纹为

$$
\Gamma_0=\{(x,0):|x|\le a\}.
$$

对 I 型张开裂纹，可以施加远场竖向拉应力 $\sigma_\infty$；在 DEM 中更常采用
上下边界的对称位移控制
$v(x,H)=\bar u$、$v(x,-H)=-\bar u$，并释放横向泊松收缩。还需通过一个最小约束
消除整体横向刚体平移，例如固定一个对称点的 $u$，但不能额外约束本应自由的横向
变形。

二维模型必须明确采用平面应力还是平面应变。对应的有效模量是

$$
E'=
\begin{cases}
E, & \text{平面应力},\\[2mm]
\dfrac{E}{1-\nu^2}, & \text{平面应变}.
\end{cases}
$$

本文前述 Amor 公式采用平面应变；若实验采用薄板平面应力，不能只替换一个弹性矩阵，
还应保证能量分解、面外应变消元和临界能量释放率的定义一致。

### 9.2 预裂纹的相场表示

弥散相场模型不会仅凭几何线段自动产生预裂纹。常见处理包括：

1. 在 $\Gamma_0$ 上施加 $\phi=1$，并求最小裂纹表面能延拓；
2. 采用局部一维最优剖面的初始化
   $\phi_0(\boldsymbol x)\approx
   \exp[-\operatorname{dist}(\boldsymbol x,\Gamma_0)/l_0]$；
3. 通过初始历史场或不可逆约束生成并保持同一预裂纹。

这些方法在有限 $l_0$ 和有限计算域中并不严格等价。无论采用哪一种，都应把
$\phi_0$ 纳入首个载荷步的不可逆约束，避免训练初期把预裂纹愈合掉。若几何中已经
显式挖去一条狭缝，则还必须说明它与弥散相场预裂纹是否重复计入裂纹表面能。

### 9.3 Griffith 解析基准与适用范围

对于无限大板中的长度为 $2a$ 的中心裂纹 [17]，

$$
K_I=\sigma_\infty\sqrt{\pi a},
\qquad
G=\frac{K_I^2}{E'}.
$$

经典 Griffith 条件 $G=G_c$ 给出

$$
\sigma_c=\sqrt{\frac{E'G_c}{\pi a}}.
$$

该公式是 sharp-crack、无限域和线弹性条件下的基准。有限宽度板需要引入几何修正
因子；AT2 模型在有限 $l_0/a$、有限 $k$ 和弥散预裂纹下的临界载荷也不会与
$\sigma_c$ 严格相等。因此，不应把单次训练所得峰值载荷与上式的偏差全部归因于
PINN 或 DEM，而应分别检查有限尺寸、相场正则化、积分和优化误差。

### 9.4 建议的收敛与一致性检查

Griffith 算例至少应报告：

- 载荷—位移曲线，以及弹性能、裂纹表面能和外功随载荷步的变化；
- $\phi_n\ge\phi_{n-1}$ 的离散违约量和损伤变量的上下界；
- 位移平衡残差、反力平衡以及自由外边界上的牵引误差；
- $J$ 积分或应力强度因子，并检查积分路径是否避开弥散损伤带；
- 随 $l_0/a$、积分点间距或单元尺度 $h/l_0$、网络容量和随机初始化的收敛性；
- 强形式 PINN、分块 DEM 或有限元参考解之间的场量与能量比较，而不只比较最终
  loss。

积分点必须充分解析宽度为 $O(l_0)$ 的相场过渡带。对非均匀加密采样，还应使用与
目标积分一致的求积权重，否则裂纹附近增加采样点会人为提高该区域在总能量中的权重。

## 10. 需要保持清醒的结论


> **配套导读：需要保持清醒的结论：一张心智地图**
>
> 1. 平衡方程来自动量守恒，本构关系来自材料能量，二者不是同一层次。
> 2. 强形式、弱形式和能量形式是连续问题的不同数学表述，不是三个任意 loss。
> 3. DEM 能降低空间求导阶数，但把困难转移到了积分、可行空间和非凸优化。
> 4. Amor 分解只保护静水压缩，不等价于完整裂纹接触。
> 5. AT2 用宽度 $O(l_0)$ 的损伤带逼近裂纹，必须在数值上解析该长度尺度。
> 6. 历史场、不可逆罚项和直接变分不等式是不同模型/算法选择。
> 7. 强形式驻点与能量极小值只在适当稳定性条件下对应。
> 8. 神经网络参数梯度为零只说明在网络可表示的方向上驻值。
> 9. 非均匀采样若没有正确权重，会改变离散能量而非单纯提高精度。
> 10. Griffith 解析式是极限基准；有限域、有限 $l_0$ 和残余刚度都会产生系统偏差。



1. **对固定相场的保守位移问题，强形式的稳定弱解和 DEM 的能量极小解在本构、
   边界与积分一致时应一致。** 对完整非凸断裂问题，强形式驻点不必是局部或全局能量
   极小值。实际训练差异还来自试探空间、积分、优化和边界施加。
2. **DEM 的“弱形式”优势是降低位移子问题的空间求导阶数。** 它不是免除采样、积分
   或边界处理的捷径。
3. **历史场法是不可逆性的实用近似，不等于原始约束最小化。** 冻结历史场后存在
   相场子问题的代理泛函，但不应据此宣称整个演化仍是同一个全局能量的严格最小化。
4. **势能值并非强形式残差。** 能量在平衡点通常不趋于零，二者不能按数值大小直接
   比较；方法验证应回到场误差、守恒/平衡、边界条件和能量一致性。

## 11. 后续可从推导直接得到的检查


> **配套导读：从理论直接得到的实现与排错清单**
>
> ### 11.1 本构层
>
> - 用有限差分验证 $\boldsymbol\sigma=\partial\psi/\partial\boldsymbol\varepsilon$；
> - 检查工程剪应变与张量剪应变约定；
> - 分别测试体积拉伸、压缩、纯剪切和单轴状态；
> - 检查 $g(0)$、$g(1)$ 与预期一致；
> - 明确平面应力或平面应变。
>
> ### 11.2 边界层
>
> - 检查是否消除了刚体模态；
> - 检查位移载荷是否被重复计入外功；
> - 区分显式狭缝边界与弥散裂纹带；
> - 对自由边直接评估 $\boldsymbol\sigma\boldsymbol n$；
> - 对硬约束随机采样边界点做机器精度检查。
>
> ### 11.3 积分层
>
> - 用常数函数验证权重和等于域面积；
> - 用线性或多项式函数验证求积阶数；
> - 非均匀采样时检查重要性权重；
> - 分别增加全局点数和裂纹带内点数；
> - 检查 `sum`、`mean` 与物理能量尺度是否一致。
>
> ### 11.4 优化层
>
> - 同时记录各 loss 值和参数梯度范数；
> - 对多个随机种子报告均值和离散程度；
> - 区分 Adam 预训练与 L-BFGS 精修阶段；
> - 检查联合训练是否跨越了不应跨越的能垒；
> - 不以 energy 值接近零作为停止标准。
>
> ### 11.5 演化层
>
> - 每步检查 $\phi_n\ge\phi_{n-1}$；
> - 记录内迭代位移差、相场差和能量差；
> - 做载荷步长收敛；
> - 检查卸载时裂纹是否保持；
> - 区分当前驱动力 $\psi_n^+$ 与历史驱动力 $\mathcal H_n$。



在提出新训练技巧前，可先用本教程的方程进行低成本预筛：

- 验证候选本构能否写成单一能量密度，并检查
  $\boldsymbol{\sigma}=\partial\psi/\partial\boldsymbol{\varepsilon}$；否则不能直接
  用 DEM。
- 对非均匀积分点验证权重是否使离散和逼近同一个域积分；否则“能量下降”并不表示
  连续势能下降。
- 对每个 loss 项比较参数梯度范数，而不仅是比较 loss 数值；尤其是
  $\nabla_\theta\log(\Pi+\epsilon)$ 与 BC 罚项梯度的相对尺度。
- 若使用历史场，明确它是代理子问题；若要求严格变分一致，应使用不可逆约束、罚项
  或增广拉格朗日形式，并验证离散后仍满足相应能量不等式。

## 12. 参考文献


> **配套导读：如何使用参考文献**
>
> 本节编号沿用主教程的参考文献列表。
>
> 不必按参考文献列表从头到尾阅读，可以按问题选择：
>
> - 想理解强形式 PINN：读 Raissi 等 [2]；
> - 想理解“神经网络为何能直接最小化泛函”：读 Deep Ritz [3]；
> - 想看计算力学中的 DEM：读 Samaniego 等 [4]；
> - 想理解 Griffith 变分框架和相场正则化：读 [6-9]；
> - 想理解 Amor 体积—偏应变分解：读 [10]；
> - 想理解历史场和多场算法：读 [11,12]；
> - 想比较不同相场断裂模型和分解：读综述 [13]；
> - 想理解不可逆罚参数：读 [14]；
> - 想研究相场断裂 DRM 的网络、梯度和求积实现：读 [15]；
> - 想系统复习 SIF、能量释放率和 $J$ 积分：读 Anderson [17]。
>
> 建议始终把文献中的四件事分别记下：连续能量、不可逆处理、空间离散、非线性求解器。
> 很多看似冲突的结论，实际上是这四项设置不同。



1. O. C. Zienkiewicz, R. L. Taylor, J. Z. Zhu. *The Finite Element Method:
   Its Basis and Fundamentals*. 6th ed., Elsevier, 2005.
2. M. Raissi, P. Perdikaris, G. E. Karniadakis. “Physics-informed neural
   networks: A deep learning framework for solving forward and inverse problems
   involving nonlinear partial differential equations.” *Journal of
   Computational Physics* 378 (2019): 686-707.
   https://doi.org/10.1016/j.jcp.2018.10.045
3. B. Yu, W. E. “The Deep Ritz Method: A Deep Learning-Based Numerical
   Algorithm for Solving Variational Problems.” *Communications in Mathematics
   and Statistics* 6 (2018): 1-12. https://doi.org/10.1007/s40304-018-0127-z
4. E. Samaniego et al. “An energy approach to the solution of partial
   differential equations in computational mechanics via machine learning.”
   *Computer Methods in Applied Mechanics and Engineering* 362 (2020): 112790.
   https://doi.org/10.1016/j.cma.2019.112790
5. S. Berrone, C. Canuto, M. Pintore. “Variational Physics Informed Neural
   Networks: The Role of Quadratures and Test Functions.” *Journal of
   Scientific Computing* 92 (2022): 100.
   https://doi.org/10.1007/s10915-022-01939-z
6. G. A. Francfort, J.-J. Marigo. “Revisiting brittle fracture as an energy
   minimization problem.” *Journal of the Mechanics and Physics of Solids* 46
   (1998): 1319-1342. https://doi.org/10.1016/S0022-5096(98)00034-9
7. B. Bourdin, G. A. Francfort, J.-J. Marigo. “Numerical experiments in
   revisited brittle fracture.” *Journal of the Mechanics and Physics of
   Solids* 48 (2000): 797-826.
   https://doi.org/10.1016/S0022-5096(99)00028-9
8. B. Bourdin, G. A. Francfort, J.-J. Marigo. “The Variational Approach to
   Fracture.” *Journal of Elasticity* 91 (2008): 5-148.
   https://doi.org/10.1007/s10659-007-9107-3
9. K. Pham, H. Amor, J.-J. Marigo, C. Maurini. “Gradient Damage Models and
   Their Use to Approximate Brittle Fracture.” *International Journal of
   Damage Mechanics* 20 (2011): 618-652.
   https://doi.org/10.1177/1056789510386852
10. H. Amor, J.-J. Marigo, C. Maurini. “Regularized formulation of the
    variational brittle fracture with unilateral contact: Numerical
    experiments.” *Journal of the Mechanics and Physics of Solids* 57 (2009):
    1209-1229. https://doi.org/10.1016/j.jmps.2009.04.011
11. C. Miehe, F. Welschinger, M. Hofacker. “Thermodynamically consistent
    phase-field models of fracture: Variational principles and multi-field FE
    implementations.” *International Journal for Numerical Methods in
    Engineering* 83 (2010): 1273-1311. https://doi.org/10.1002/nme.2861
12. C. Miehe, M. Hofacker, F. Welschinger. “A phase field model for
    rate-independent crack propagation: Robust algorithmic implementation
    based on operator splits.” *Computer Methods in Applied Mechanics and
    Engineering* 199 (2010): 2765-2778.
    https://doi.org/10.1016/j.cma.2010.04.011
13. M. Ambati, T. Gerasimov, L. De Lorenzis. “A review on phase-field models
    of brittle fracture and a new fast hybrid formulation.” *Computational
    Mechanics* 55 (2015): 383-405.
    https://doi.org/10.1007/s00466-014-1109-y
14. T. Gerasimov, L. De Lorenzis. “On penalization in variational phase-field
    models of brittle fracture.” *Computer Methods in Applied Mechanics and
    Engineering* 354 (2019): 990-1026.
    https://doi.org/10.1016/j.cma.2019.05.038
15. M. Manav, R. Molinaro, S. Mishra, L. De Lorenzis. “Phase-field modeling
    of fracture with physics-informed deep learning.” *Computer Methods in
    Applied Mechanics and Engineering* 429 (2024): 117104.
    https://doi.org/10.1016/j.cma.2024.117104
16. I. E. Lagaris, A. Likas, D. I. Fotiadis. “Artificial neural networks for
    solving ordinary and partial differential equations.” *IEEE Transactions
    on Neural Networks* 9 (1998): 987-1000.
    https://doi.org/10.1109/72.712178
17. T. L. Anderson. *Fracture Mechanics: Fundamentals and Applications*.
    4th ed., CRC Press, 2017.