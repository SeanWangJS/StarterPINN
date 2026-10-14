# 从强形式 PINN 到深度能量法：相场断裂的推导教程

## 1. 目的与范围

> **配套导读**
>
> 整篇文章围绕一个具体的问题展开：一块矩形板，中间有一条水平的预制裂纹，把板的上下两边慢慢拉开。开始时板只是均匀变形，裂纹张开一点；拉到一定程度，裂纹会从两端向外延伸。我们想用神经网络把这个过程算出来。
>
> 要描述这个过程，需要两个场。一个是位移场 $\boldsymbol u(\boldsymbol x)$，告诉我们板上每一点移动了多少；另一个是相场 $\phi(\boldsymbol x)$，告诉我们每一点损伤到什么程度，$\phi=0$ 是完好，$\phi=1$ 是完全断开。相场法不画一条无限细的裂纹线，而是让 $\phi$ 在裂纹附近从 1 平滑地降到 0，形成一条有一定宽度的损伤带。
>
> 这两个场互相牵制。板被拉伸后储存了弹性能，弹性能推动损伤增长；损伤一增长，那一带的材料就变软，整块板的应力需要重新分布，位移也跟着变。所以哪怕材料本身是线弹性的，整个问题也是非线性的：没法解一个线性方程组就得到答案。
>
> 一下子处理两个耦合的场太难，正文分两步走。第 2 到第 6 节先把损伤分布当作已知，只求位移。这一部分从你熟悉的强形式 PINN 讲起，经过弱形式，落到深度能量法（DEM）。第 7 节才让相场动起来。第 8 节处理单位和数值尺度，第 9 节把这些内容拼成完整的中心裂纹算例。
>
> 还有一个"时间"的概念需要先说清楚。这里研究的是准静态加载：每次把边界多拉开一点，然后求这一时刻的平衡状态，不考虑惯性和振动。这里的"时间"其实就是加载的进度，每一次加载叫一个载荷步。在一个载荷步里，位移和相场往往要来回迭代几次才能互相适应，然后才进入下一步。所以训练一次网络只是在解其中一个子问题，算完一整条裂纹扩展路径需要很多次这样的求解。



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

> **配套导读**
>
> 先从位移说起。网络给出的是每个点的位移，但材料受不受力，看的不是位移本身。设想整块板向右平移 1 毫米，每个点都动了，却没有任何一处被拉长或压短，材料里也就没有应力。真正引起应力的是相邻两点之间位移的差别，也就是位移的空间导数。
>
> 位移梯度 $\nabla\boldsymbol u$ 里混着两种运动：一种是真正的伸缩和剪切，另一种是局部的小转动。转动不会让材料储能，所以小应变只取对称部分：
>
> $$
> \boldsymbol\varepsilon=\tfrac12(\nabla\boldsymbol u+\nabla\boldsymbol u^T).
> $$
>
> 这一步依赖"小变形"假设。转角大了以后，忽略的二次项就不能再忽略，这时需要有限变形理论，本文不涉及。
>
> 剪切分量有一个容易踩的坑。张量剪应变 $\varepsilon_{xy}=\tfrac12(u_{,y}+v_{,x})$，工程剪应变 $\gamma_{xy}=u_{,y}+v_{,x}$，两者差一倍。两种写法本身都对，只是能量公式和本构矩阵里的系数要跟着变。如果写能量时用了一种，组装向量时又换成另一种，剪切能就会差出 2 倍或 4 倍，而且这类错误在拉伸算例里完全看不出来。
>
> 有了应力，平衡方程 $\nabla\cdot\boldsymbol\sigma+\boldsymbol b=\boldsymbol 0$ 说的是：从板里任意切出一小块，作用在它表面上的力和体力加起来为零。边界上的 $\boldsymbol\sigma\boldsymbol n$ 是沿外法向截出来的面力，也就是牵引力。
>
> 接下来是边界条件，这一步决定了我们到底在解哪个问题。以中心裂纹板为例：上下两边规定竖向位移，一边向上拉 $\bar u$，一边向下拉 $\bar u$；上下边的横向位移不加限制，允许板因泊松效应自由收缩；左右两边什么都不规定，是自由边，牵引为零。可以看到，同一条边的两个方向可以分别给不同类型的条件。
>
> 这里还有一个细节。上下边的竖向位移已经把竖直方向的平移和整体转动都锁住了，但整块板仍然可以左右平移而不产生任何应变。如果不处理，网络输出可以整体漂移，优化器会在一个能量完全平坦的方向上来回晃。常见做法是只固定一个点的横向位移，例如板中心，而不去约束整条中线或整条侧边，否则会把本该自由的泊松收缩也锁住。
>
> 最后说一下正文里的 $H^1$。它是这样一类函数：函数本身和一阶导数的平方都能在区域上积分。有限元里常用的分片线性函数，在单元交界处导数会跳，但仍然属于 $H^1$。后面会看到，能量形式只要求这么多，比强形式的要求低。



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

> **配套导读**
>
> 这一节的公式比较多，可以先从一根弹簧想起。刚度为 $c$ 的弹簧伸长 $q$，储存的能量是 $\tfrac12cq^2$，对 $q$ 求导就得到弹簧力 $cq$。材料里也是一样的关系：应变能密度 $\psi$ 对应变求导，得到应力，
>
> $$
> \boldsymbol\sigma=\frac{\partial\psi}{\partial\boldsymbol\varepsilon}.
> $$
>
> 为什么正文要先写能量、再求应力，而不是直接给出应力公式？因为第 6 节的 DEM 要最小化的是能量。如果应力和能量是分别写的，两者不配套，那么"让应力平衡"和"让能量最小"其实在解两个不同的问题，算出来的结果也就没法互相印证。先写能量再求导，可以从源头上保证它们一致。
>
> 对各向同性材料，能量可以拆成两部分：改变体积的部分和改变形状的部分。
>
> $$
> \psi_0=\tfrac12K(\operatorname{tr}\boldsymbol\varepsilon)^2+\mu\,\boldsymbol e:\boldsymbol e,
> \qquad
> \boldsymbol e=\boldsymbol\varepsilon-\tfrac13(\operatorname{tr}\boldsymbol\varepsilon)\boldsymbol I.
> $$
>
> 均匀膨胀只改变体积，由体积模量 $K$ 控制；纯剪切只改变形状，由剪切模量 $\mu$ 控制。
>
> 我们算的是二维问题，但材料仍是三维的。平面应变假设板在厚度方向上不能伸缩，即 $\varepsilon_{zz}=0$，这并不等于厚度方向没有应力。正因为如此，偏应变的定义里要减去三分之一的迹，而且要带上 $e_{zz}=-\tfrac13\operatorname{tr}\boldsymbol\varepsilon$。
>
> 举个数字例子：只有 $\varepsilon_{xx}=\varepsilon$，其他分量为零。按三维定义，$e_{xx}=\tfrac23\varepsilon$，$e_{yy}=e_{zz}=-\tfrac13\varepsilon$，于是 $\boldsymbol e:\boldsymbol e=\tfrac23\varepsilon^2$。如果图省事改成二维的"减二分之一"，算出来是 $\tfrac12\varepsilon^2$，形状改变能少了四分之一。
>
> 现在加入损伤。最自然的想法是让损伤处的材料变软，把弹性能乘上一个随 $\phi$ 减小的系数 $g(\phi)$。但如果把全部弹性能都乘上 $g$，压缩状态下材料也能靠"开裂"来释放能量，这与受压时裂纹应当闭合、继续承载的直觉不符。
>
> Amor 分解的办法是：体积压缩的那部分能量 $\psi^-$ 保持不变，只让其余部分 $\psi^+$ 随损伤降低，
>
> $$
> \psi=g(\phi)\psi^++\psi^-,\qquad
> \psi^+=\tfrac12K\langle\operatorname{tr}\boldsymbol\varepsilon\rangle_+^2+\mu\,\boldsymbol e:\boldsymbol e,\qquad
> \psi^-=\tfrac12K\langle\operatorname{tr}\boldsymbol\varepsilon\rangle_-^2.
> $$
>
> $\psi^+$ 常被叫作"拉伸能"，这个名字容易误导，因为它包含了全部形状改变能。比如单轴压缩 $\varepsilon_{xx}=-\varepsilon$：体积是缩小的，体积能全部归入 $\psi^-$；但形状改变能 $\tfrac23\mu\varepsilon^2$ 仍然在 $\psi^+$ 里，仍然可以驱动损伤。所以 Amor 分解保护的只是静水压缩，它也不是真正的裂纹面接触模型。以后在受压区看到一点损伤，可能就是模型本身的行为，不一定是程序写错了。
>
> 退化函数取 $g(\phi)=(1-\phi)^2+k$，其中 $k$ 是一个很小的残余刚度，比如 $10^{-3}$。没有它，完全损伤的区域刚度为零，位移在那里可以随便取，求解会出问题。有了它，问题稳定了，代价是"断开"的地方仍然保留千分之一的刚度，能传一点点力。另外，按这个写法 $g(0)=1+k$，完好材料的刚度也被放大了千分之一。这些小量都是模型的一部分，比较不同方法的结果时，$k$ 和 $g$ 的写法必须一致。



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

> **配套导读**
>
> 到这里，损伤分布暂时固定，问题就是一个求位移的弹性问题。最直接的神经网络做法就是强形式 PINN：在板内撒一批配点，网络给出每点的位移，自动微分算出应变，代入本构得到应力，再对应力求一次散度，得到每点的不平衡力 $\boldsymbol r$。把 $\|\boldsymbol r\|^2$ 的平均值当作 loss，训练网络让它变小。整个过程不需要任何位移标签，平衡方程自己就提供了"目标值为零"的监督信号。
>
> 计算上的负担出在最后一步。位移到应变是一次空间导数，应力到散度又是一次，所以残差里包含位移的二阶导数。自动微分可以算，但计算图更深，显存更大，对激活函数也有要求。ReLU 网络在每个线性片内的二阶导数恒为零，拿它算二阶残差，大部分地方得到的都是零，所以强形式 PINN 一般用 $\tanh$ 之类的光滑激活。可光滑激活又不擅长表示裂尖附近陡峭的变化，这两个要求之间存在张力。
>
> 边界条件有两种处理方式。软约束把"边界位移偏离多少"作为一个罚项加进 loss，再配一个权重。这样实现简单，但结果会随权重变化。硬约束把边界条件直接写进位移表达式，例如
>
> $$
> \boldsymbol u_\theta=\boldsymbol u_D+B(\boldsymbol x)\,\boldsymbol N_\theta(\boldsymbol x),
> $$
>
> 其中 $B$ 在规定位移的边界上恒为零。无论网络参数怎么变，边界条件都精确成立。难点在于构造 $B$：它要在该为零的地方为零，又不能在不该约束的地方把位移压住。
>
> 强形式 PINN 在裂纹问题上有一个特别的隐患。裂尖附近面积很小，却是变化最剧烈、最难拟合的地方。全域平均残差可能已经很小，裂尖处却仍然算得很差，因为它在平均值里只占很小的比重。所以除了看 loss，还要在独立的检查点上画出残差的空间分布，专门盯着裂尖和损伤带看。



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

> **配套导读**
>
> 强形式要求每一点都平衡，代价是二阶导数。能不能换一种等价的表达，把导数阶数降下来？弱形式就是答案。
>
> "弱形式"这个名字听上去像是降低了要求，比如只要求整体平均平衡。其实不然。只检查合力为零显然不够，局部不平衡完全可以互相抵消。弱形式要求的是：对每一个允许的测试函数，某个积分关系都成立。测试函数可以选成集中在任意一个小区域上的形状，所以局部的不平衡逃不过去。
>
> 在力学里，测试函数有一个直观的名字：虚位移 $\delta\boldsymbol u$。可以把它想成一次假想的微小扰动：让各点稍微挪一下，看看内力和外力在这次挪动中各做了多少功。它不是真实发生的运动，也不是另一个要求的未知量。规定了位移的边界不能挪，所以虚位移在那里为零。
>
> 具体做法是把平衡方程乘上 $\delta\boldsymbol u$，在全域上积分，然后对含 $\nabla\cdot\boldsymbol\sigma$ 的那一项分部积分。分部积分做了两件事。第一，原本作用在应力上的导数，有一部分转移到了虚位移上，于是只需要应力本身，也就只需要位移的一阶导数。第二，分部积分会产生一个边界积分 $\int\delta\boldsymbol u\cdot\boldsymbol\sigma\boldsymbol n\,d\Gamma$，规定的牵引力 $\bar{\boldsymbol t}$ 正是从这里进入方程的。牵引条件被称为"自然边界条件"，就是因为它在推导中自然出现，不用另外施加。
>
> 推导的结果用力学语言说就是虚功原理：对任意允许的虚位移，内力虚功等于外力虚功。
>
> 还有一点要和 PINN 的做法区分开。把残差平方后积分，$\int r^2\,d\Omega$，是最小二乘；弱形式是 $\int w\,r\,d\Omega=0$ 对所有 $w$ 成立。这是两类不同的方法。VPINN 之类的方法会显式选一组测试函数；下一节的 DEM 不需要显式挑测试函数，原因在于能量。



强形式要求式 (1)–(3) 逐点成立，其中式 (1) 含应力的空间导数，进而含位移的二阶
导数。本节把它改写为一个积分恒等式，使位移只需一阶导数，并说明两种表述在何种
意义下等价。推导使用指标记法，重复指标求和，$(\cdot)_{,j}=\partial(\cdot)/\partial x_j$。

### 5.1 试探空间与测试空间

先区分两类函数。待求位移属于**试探空间**

$$
\mathcal U_{\bar u}
=\{\boldsymbol u\in(H^1(\Omega))^2:\ \boldsymbol u=\bar{\boldsymbol u}
\ \text{on }\Gamma_u\},
$$

即事先满足位移边界条件 (2)。用来检验平衡的虚位移属于**测试空间**

$$
\mathcal V_0
=\{\delta\boldsymbol u\in(H^1(\Omega))^2:\ \delta\boldsymbol u=\boldsymbol 0
\ \text{on }\Gamma_u\}.
$$

$\mathcal V_0$ 中的函数是两个可行位移之差：若
$\boldsymbol u_1,\boldsymbol u_2\in\mathcal U_{\bar u}$，则
$\boldsymbol u_1-\boldsymbol u_2$ 在 $\Gamma_u$ 上为零。因此虚位移描述的是"在不破坏
位移约束的前提下，位移场还能怎样变化"。

### 5.2 加权积分

设 $\boldsymbol u$ 足够光滑并满足式 (1)。对任意 $\delta\boldsymbol u\in\mathcal V_0$，
将式 (1) 与它做点积并在区域上积分：

$$
\int_\Omega\delta u_i\,\sigma_{ij,j}\,d\Omega
+\int_\Omega\delta u_i\,b_i\,d\Omega=0. \tag{5-1}
$$

这一步只是把逐点为零的量加权求和，仍然等价于原方程，导数阶数尚未改变。

### 5.3 分部积分：乘积求导与散度定理

对第一项使用乘积求导法则：

$$
(\delta u_i\,\sigma_{ij})_{,j}
=\delta u_{i,j}\,\sigma_{ij}+\delta u_i\,\sigma_{ij,j}. \tag{5-2}
$$

移项后积分，并对左端的散度项使用散度定理
$\int_\Omega F_{j,j}\,d\Omega=\int_{\partial\Omega}F_jn_j\,d\Gamma$，取
$F_j=\delta u_i\sigma_{ij}$：

$$
\int_\Omega\delta u_i\,\sigma_{ij,j}\,d\Omega
=\int_{\partial\Omega}\delta u_i\,\sigma_{ij}n_j\,d\Gamma
-\int_\Omega\delta u_{i,j}\,\sigma_{ij}\,d\Omega. \tag{5-3}
$$

这就是多维的分部积分：应力的导数被移到虚位移上，同时留下一个边界积分。
代入式 (5-1)，得到

$$
\int_\Omega\delta u_{i,j}\,\sigma_{ij}\,d\Omega
=\int_\Omega\delta u_i\,b_i\,d\Omega
+\int_{\partial\Omega}\delta u_i\,\sigma_{ij}n_j\,d\Gamma.
$$

### 5.4 利用应力对称性化为应变

把虚位移梯度拆成对称与反对称部分：

$$
\delta u_{i,j}
=\underbrace{\tfrac12(\delta u_{i,j}+\delta u_{j,i})}_{\varepsilon_{ij}(\delta\boldsymbol u)}
+\underbrace{\tfrac12(\delta u_{i,j}-\delta u_{j,i})}_{\omega_{ij}(\delta\boldsymbol u)}.
$$

由于 $\sigma_{ij}=\sigma_{ji}$ 而 $\omega_{ij}=-\omega_{ji}$，交换哑指标可得

$$
\omega_{ij}\sigma_{ij}=\omega_{ji}\sigma_{ji}=-\omega_{ij}\sigma_{ij}
\quad\Longrightarrow\quad
\omega_{ij}\sigma_{ij}=0.
$$

因此 $\delta u_{i,j}\sigma_{ij}=\varepsilon_{ij}(\delta\boldsymbol u)\,\sigma_{ij}$。
力学上，这表示虚位移中的局部刚体转动不做内功，与第 2 节"转动不产生应变"一致。

### 5.5 拆分边界项

按 $\partial\Omega=\Gamma_u\cup\Gamma_t$ 拆分边界积分：

- 在 $\Gamma_u$ 上，$\delta\boldsymbol u=\boldsymbol 0$，该部分为零。这里的面力是
  未知反力，测试空间的选取恰好使它不出现在方程中；
- 在 $\Gamma_t$ 上，由式 (3)，$\sigma_{ij}n_j=\bar t_i$，用已知面力替换。

于是得到位移问题的弱形式：求 $\boldsymbol u\in\mathcal U_{\bar u}$，使对所有
$\delta\boldsymbol u\in\mathcal V_0$，

$$
\int_\Omega
\boldsymbol{\varepsilon}(\delta\boldsymbol{u}):\boldsymbol{\sigma}(\boldsymbol u)\,d\Omega
=\int_\Omega\delta\boldsymbol{u}\cdot\boldsymbol{b}\,d\Omega
+\int_{\Gamma_t}\delta\boldsymbol{u}\cdot\bar{\boldsymbol{t}}\,d\Gamma. \tag{5-4}
$$

左端是内力虚功，右端是外力虚功，这就是虚功原理 [1]。三类方程在式 (5-4) 中的处理
方式各不相同：式 (1) 以积分形式保留；式 (3) 被代入边界项；式 (2) 不出现在方程里，
由试探空间 $\mathcal U_{\bar u}$ 保证。

### 5.6 二维展开

按第 2 节的张量剪应变，内虚功的被积函数为

$$
\boldsymbol\varepsilon(\delta\boldsymbol u):\boldsymbol\sigma
=\delta u_{,x}\,\sigma_{xx}
+\delta v_{,y}\,\sigma_{yy}
+(\delta u_{,y}+\delta v_{,x})\,\tau_{xy}.
$$

剪切项来自 $\varepsilon_{xy}\sigma_{xy}+\varepsilon_{yx}\sigma_{yx}
=2\varepsilon_{xy}\tau_{xy}$，系数 2 被工程剪应变
$\delta\gamma_{xy}=\delta u_{,y}+\delta v_{,x}$ 吸收。若改用 Voigt 记号
$\{\varepsilon_{xx},\varepsilon_{yy},\gamma_{xy}\}$，该式就是
$\delta\boldsymbol\varepsilon^T\boldsymbol\sigma$，有限元刚度矩阵
$\int\boldsymbol B^T\boldsymbol D\boldsymbol B\,d\Omega$ 即由此离散得到。

### 5.7 从弱形式回到强形式

上面的推导说明强解满足式 (5-4)。反过来，若式 (5-4) 的解足够光滑，把 5.3–5.5 节逆向
进行，可得对所有 $\delta\boldsymbol u\in\mathcal V_0$：

$$
\int_\Omega\delta\boldsymbol u\cdot(\nabla\cdot\boldsymbol\sigma+\boldsymbol b)\,d\Omega
-\int_{\Gamma_t}\delta\boldsymbol u\cdot(\boldsymbol\sigma\boldsymbol n-\bar{\boldsymbol t})\,d\Gamma
=0. \tag{5-5}
$$

分两步选取测试函数：

1. 先取在整个边界上为零的 $\delta\boldsymbol u$，边界项消失。由变分法基本引理
   （若连续函数 $\boldsymbol f$ 满足 $\int_\Omega\delta\boldsymbol u\cdot\boldsymbol f\,d\Omega=0$
   对所有这样的 $\delta\boldsymbol u$ 成立，则 $\boldsymbol f\equiv\boldsymbol 0$），
   得到式 (1) 在 $\Omega$ 内成立；
2. 于是式 (5-5) 的区域积分恒为零，只剩边界项。由于 $\delta\boldsymbol u$ 在 $\Gamma_t$
   上可以任意取值，同理得到式 (3) 在 $\Gamma_t$ 上成立。

这说明牵引条件不必事先施加，它是弱形式对任意虚位移成立的**结果**，故称自然边界
条件；位移条件则必须事先写进试探空间，故称本质边界条件。第 1 步也说明，弱形式
并非只要求"整体平均平衡"：测试函数可以只在任意小的区域内非零，因而能检出任何
局部不平衡。

### 5.8 正则性要求的变化

式 (5-4) 中，$\boldsymbol u$ 和 $\delta\boldsymbol u$ 都只出现一阶导数，因此只需
$\boldsymbol u\in(H^1(\Omega))^2$，即位移及其一阶弱导数平方可积；强形式则要求应力
可微，位移通常需二阶可导。分片线性有限元在单元边界处导数间断，不能直接代入式 (1)，
却是式 (5-4) 的合法试探函数。对神经网络而言，式 (5-4) 避免了显式构造应力的空间散度。

弱形式并不等于"把残差平方后积分"：$\int_\Omega|\boldsymbol r|^2\,d\Omega$ 是最小
二乘目标，而式 (5-4) 要求对所有允许的虚位移成立。Galerkin 有限元取有限个形函数作为
测试函数，VPINN 类方法显式选取一组测试函数。在线弹性或存在势能的本构中，式 (5-4)
还等价于总势能的驻值条件，这给出了 DEM 可以直接最小化的标量目标 [3,4]，见第 6 节。

## 6. 从弱形式到 DEM：总势能驻值

> **配套导读**
>
> 上一节得到了虚功原理，但它还是一个"对所有测试函数成立"的方程，不能直接当成训练的 loss。如果能找到一个标量，它的一阶变分恰好就是虚功原理，那么最小化这个标量就等价于求解平衡。第 3 节先写能量的好处在这里显现：这个标量就是总势能。
>
> 还是用弹簧来理解。弹簧受恒定拉力 $P$，总势能是
>
> $$
> \Pi(q)=\tfrac12cq^2-Pq.
> $$
>
> 对 $q$ 求导并令其为零，得到 $cq=P$，正是平衡条件。减号那一项是外力的势能，如果漏掉它，最小化 $\tfrac12cq^2$ 只会得到 $q=0$，弹簧根本没有被拉开。顺便算一下平衡时的能量：$q=P/c$，$\Pi=-P^2/(2c)$，是个负数。所以 energy loss 不会像残差那样降到零，在力加载问题里它甚至收敛到一个负值。判断收敛要看梯度、平衡残差和反力是否稳定，不能看能量是否接近零。
>
> 换成位移控制，情况又不一样。如果弹簧的伸长被规定为 $\bar q$，那就没有什么可以优化的了，$\bar q$ 不是一个候选值，而是约束。对应到中心裂纹板，上下边规定了位移，这些规定限制了允许搜索的位移场；没有体力、其余边又是自由边，所以总势能就是弹性能本身。加载端的反力是求解之后算出来的结果，不能再当作已知外力加进势能里，否则同一个载荷被施加了两次。
>
> DEM 的做法就是用网络表示这些候选位移场，然后调整参数让离散的总势能下降。和强形式比，计算链在算出应变能后就结束了，不用再对应力求散度，所以只需要一阶空间导数。
>
> 这里有一个从机器学习角度很好理解的限制。连续问题的平衡要求势能对所有允许方向的变化都不再下降，而优化器只能沿着网络参数能改变的方向去调整位移，也就是 $\partial\boldsymbol u_\theta/\partial\theta$ 张成的那些方向。参数梯度为零，只说明在网络能表示的方向上已经到底了，连续问题未必解准了。加宽网络、加入特征映射、在局部加密，本质上都是在增加可以检验的方向。
>
> 积分是 DEM 里分量最重的一个环节，因为 loss 本身就是一个积分。举个例子：板的面积是 8，裂尖附近的加密区面积是 0.8，只占十分之一，但你在那里放了一半的采样点。如果直接对所有点的能量密度求普通平均，这块区域在总能量里的分量就从 10% 变成了 50%，等于把裂尖附近的能量放大了 5 倍。优化器最小化的已经是另一个能量了。所以非均匀采样必须配上相应的权重：随机采样用重要性权重，确定性求积用面积权重。加密是为了把原来那个积分算得更准，不能改变各部分的比重。
>
> 自然边界条件也可以从这个角度理解。自由边上牵引为零，对应的外力项就是零，所以 DEM 不需要为自由边额外加罚项。但有限的网络训练完以后，自由边上的应力只是近似为零，需要单独检查。另外，相场的损伤带不是一条真正的几何边界，那里的应力释放靠的是 $g(\phi)$ 把材料变软，而且因为 $k>0$，应力并不严格为零。
>
> 正文最后提到的 log-energy，是把 loss 换成 $\log\mathcal E_\theta$。在能量为正时，$\nabla\log\mathcal E=\nabla\mathcal E/\mathcal E$，驻点不变，只是梯度被除以当前能量，相当于做了一个自适应的缩放。这是一个优化技巧，不涉及新的物理。如果 loss 里还有边界罚项或正则项，只给能量取对数会改变各项之间的相对权重；如果总势能可能为负，对数就不能直接用了。



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

> **配套导读**
>
> 前面一直把损伤分布当成给定的。现在要回答：裂纹为什么会长，又往哪里长？
>
> Griffith 的思路是能量平衡。裂纹每扩展一个单位面积，要付出 $G_c$ 的能量来形成新表面；同时，裂纹扩展会让结构变软，释放一部分弹性能。释放的能量够付这笔账，裂纹就扩展，不够就不扩展。这个思路很清楚，但直接计算却很麻烦，因为未知量里包含裂纹本身的几何形状，而裂纹可能转弯、分叉、合并。
>
> 相场法用连续的 $\phi$ 代替裂纹几何，把"找一条线"换成"求一个场"。AT2 模型用下面这个密度来近似单位面积的裂纹表面：
>
> $$
> \gamma(\phi,\nabla\phi)=\frac{\phi^2}{2l_0}+\frac{l_0}{2}|\nabla\phi|^2.
> $$
>
> 两项的作用正好相反。第一项惩罚损伤的分布范围，损伤铺得越宽越贵；第二项惩罚损伤的变化速度，从 1 降到 0 降得越陡越贵。一条裂纹在这两种代价之间找到平衡，形成一条宽度为 $l_0$ 量级的过渡带。
>
> 对于一条无限长的直裂纹，可以把这个平衡解出来。在垂直于裂纹的方向上，最优剖面是 $\phi(s)=e^{-|s|/l_0}$。把它代回去积分，两项各贡献 $\tfrac12$，合起来恰好是 1。也就是说，这条弥散的损伤带乘上 $G_c$，正好等于单位长度裂纹应有的表面能。这是 AT2 模型里 $\tfrac{1}{2l_0}$ 和 $\tfrac{l_0}{2}$ 这两个系数的来历，也是很多代码用指数剖面来初始化预裂纹的原因。这个积分自己推一遍只要几行，值得动手算一算。
>
> $l_0$ 是模型参数，不是数值上的平滑参数。$l_0$ 越小，弥散裂纹越接近真正的尖锐裂纹，但损伤带也越窄，需要更密的积分点和表示能力更强的网络。指数剖面在离裂纹 $3l_0$ 处还有约 5%，所以积分至少要把几个 $l_0$ 宽的范围分辨清楚。如果固定积分分辨率不断减小 $l_0$，结果变差是因为没有分辨率，不能归咎于相场理论或优化器。
>
> 把裂纹能和退化弹性能放在一起，就得到耦合的总能量。$\phi$ 增大，$g(\phi)\psi^+$ 减小，弹性能被释放；同时 $G_c\gamma$ 增大，要付出断裂能。相场的分布就是这两者在全场竞争的结果，不需要人为指定裂纹往哪个方向走。
>
> AT2 有一个特点会影响你解读结果：它没有严格的起裂门槛。冻结历史场后，在没有空间变化的均匀状态下，相场方程化为 $\tfrac{G_c}{l_0}\phi=2(1-\phi)\mathcal H$，解出来是
>
> $$
> \phi=\frac{2\mathcal H}{G_c/l_0+2\mathcal H},
> $$
>
> 只要驱动力 $\mathcal H$ 大于零，$\phi$ 就大于零。所以加载以后全场出现一层很淡的损伤是正常的，不代表裂纹已经扩展。AT1 模型在这一点上不同，它有明确的弹性阶段。
>
> 接下来是卸载的问题。把拉力减小，弹性能下降，损伤的驱动力也下降。如果每一步都只是最小化当前能量，模型会发现把损伤减小更划算，裂纹就"愈合"了。脆性断裂不允许这样，所以要加约束 $\phi_n\ge\phi_{n-1}$：每一点的损伤不能比上一步小。
>
> 从优化的角度看，这就是一个带不等式约束的最小化问题。约束不起作用的地方，能量对 $\phi$ 的导数为零，和无约束时一样；约束起作用的地方，能量其实想让 $\phi$ 变小，但被下界挡住了，导数不为零，只能是正的。两种情况合起来，就是正文里的变分不等式和 KKT 互补条件。
>
> 处理这个约束有三种常见办法。第一种是直接约束，比如投影或主动集，最接近原始问题，但在神经网络里，参数和逐点约束之间的关系很间接，不好做。第二种是罚函数，对违反约束的部分加一个 $\tfrac{\gamma_{\rm ir}}2\langle\phi-\phi_{n-1}\rangle_-^2$ 的惩罚，loss 仍然是一个可微的标量，适合 DRM；罚参数小了挡不住愈合，大了问题会变得病态。第三种是历史场：每一点记住过去出现过的最大活性能
>
> $$
> \mathcal H_n=\max(\mathcal H_{n-1},\psi_n^+),
> $$
>
> 求相场时用 $\mathcal H$ 代替当前的 $\psi^+$。卸载时 $\psi^+$ 下降了，$\mathcal H$ 不降，相场的驱动力也就不降。
>
> 历史场简单好用，但它记住的是"最大驱动力"，不是"损伤的下界"，这两者并不等价。最典型的后果出在预裂纹上：零载荷时 $\mathcal H=0$，按上面的均匀解，相场的最优值就是 $\phi=0$，你精心初始化的预裂纹会被直接抹掉。想保住它，要么显式加上 $\phi\ge\phi_0$ 的约束或罚项，要么给一个与预裂纹相容的初始历史场 $\mathcal H_0$。
>
> 冻结历史场后的相场方程是
>
> $$
> \frac{G_c}{l_0}\phi-G_cl_0\Delta\phi-2(1-\phi)\mathcal H=0.
> $$
>
> 三项可以分别理解：第一项把 $\phi$ 往完好状态拉，第二项让 $\phi$ 在空间上平滑过渡，第三项是历史场推动损伤增长。$\mathcal H$ 固定时，这是关于 $\phi$ 的线性椭圆方程，很好解。但 $\mathcal H$ 来自位移，位移又依赖 $\phi$，所以整体上仍是非线性的，需要在一个载荷步内交替迭代：固定 $\phi$ 求位移，更新 $\mathcal H$，固定 $\mathcal H$ 求 $\phi$，如此反复，直到两个场都不再明显变化，再进入下一个载荷步。
>
> 神经网络求解这个耦合问题有几条路线。两个场都用强形式 PINN，都需要二阶导数；分块 DEM 交替最小化位移能量和相场能量，各自只需要一阶导数，而且每个子问题出了问题容易定位；联合 DRM 用一个总能量同时训练两个场，形式紧凑，但面对的是一个非凸优化问题，可能落到不同的局部极小值。做方法研究时，先把分块的版本做扎实，再去尝试联合训练，会省很多排查的工夫。



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

> **配套导读**
>
> 同一块板，长度写成 1 毫米还是 0.001 米，物理结果不应有任何区别。但对网络来说，输入是 $10^{-3}$ 还是 $1$，应力是 $10^{11}$ 还是 $1$，差别很大：激活函数可能饱和，各项梯度的量级可能相差十几个数量级，优化器的步长没法兼顾。无量纲化就是先选一组特征尺度，把问题换成量级都在 1 附近的数字，物理关系保持不变。
>
> 正文用同一个长度 $L$ 去除坐标和位移，用 $E$ 去除应力。坐标和位移用同一个尺度，应变就不变：原长和伸长量同时换了单位，伸长的比例不会变。如果位移另选了一个尺度 $U$，应变就要多乘一个 $U/L$。自动微分只会忠实地对你写下的表达式求导，表达式里漏掉的换算系数，它不会替你补上。
>
> 无量纲化后，能量里会出现一个组合 $\kappa=G_c/(EL)$。它的来历很直接：弹性能密度的量级是 $E$，裂纹能密度的量级是 $G_c/L$（$G_c$ 是单位面积的能量，除以一个长度变成单位体积），两者之比就是 $\kappa$。它衡量的是在这个结构尺度上，开裂的代价和储存弹性能相比有多大。这个数是物理决定的，不能当成 loss 权重随意调大调小。
>
> 历史场的尺度最容易出错。$\mathcal H$ 可以除以 $E$，也可以除以 $G_c/l_0$，还可以除以 $G_c/L$，每一种都成立，但相场方程里的系数会跟着变。例如用 $\mathcal H l_0/G_c$ 作为无量纲历史场时，拉普拉斯项前面是 $(l_0^*)^2$，而用 $\mathcal HL/G_c$ 时是 $l_0^*$。实际代码里真正危险的情况是：历史场按一种尺度归一化，方程系数却照抄了另一种写法。看到一条无量纲方程时，先找到它的历史场是怎么定义的，再核对系数。
>
> 最后，loss 的权重也和尺度有关。能量是体积分，边界罚项通常是边界点上位移偏差的平均值，两者单位不同。即使物理量都已经无量纲化了，代码里用 `sum` 还是 `mean`，也会让权重随采样点数变化。比较稳妥的做法是先写清楚每一项 loss 对应哪个连续量，再推出离散的系数。



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

> **配套导读**
>
> 现在把前面的内容都放回中心裂纹板上。正文用的是半尺寸：板宽 $2W$，高 $2H$，裂纹长 $2a$；上下边分别拉 $+\bar u$ 和 $-\bar u$，所以两边的相对位移是 $2\bar u$。这些"半"字看起来只是符号习惯，但和解析公式、文献结果比较时，差一个 2 就是完全不同的载荷。
>
> 首先要确保预裂纹真的存在于求解中。在相场模型里，画一条线段 $\Gamma_0$ 并不会让材料在那里失去承载能力，需要把它转成一个初始相场，常用的就是上一节的指数剖面 $\phi_0=e^{-d/l_0}$，其中 $d$ 是到线段的距离。然后还要保证后续求解不会把它抹掉，第 7 节已经说过，单靠历史场在零载荷时做不到。另外，指数剖面是无限长直裂纹的精确解，用在有限长裂纹的两端和有限大的板上只是近似。
>
> 也可以在几何上真的挖掉一条狭缝，这时狭缝的上下表面是真正的自由边。只是要注意别在同一个位置再放一条完整的相场裂纹，否则同一条裂纹的表面能会被计算两次。
>
> Griffith 的经典结果给了一个参照。对无限大板中长 $2a$ 的裂纹，在远处施加拉应力 $\sigma_\infty$：
>
> $$
> K_I=\sigma_\infty\sqrt{\pi a},\qquad
> G=\frac{K_I^2}{E'},\qquad
> \sigma_c=\sqrt{\frac{E'G_c}{\pi a}}.
> $$
>
> $K_I$ 描述尖锐裂纹尖端应力集中的强度，$G$ 是裂纹每扩展单位面积释放的能量，令 $G=G_c$ 就得到临界应力。平面应变时 $E'=E/(1-\nu^2)$。
>
> 这个公式很有用，但它对应的是一个理想化的问题：无限大的板，无限尖的裂纹，力加载。你的算例和它至少有几处不同。板的边界离裂纹只有几个裂纹长度，会改变裂尖附近的应力场，需要乘一个依赖 $a/W$、$H/W$ 和加载方式的修正因子。相场裂纹有 $l_0$ 量级的宽度，裂尖是圆钝的，不会出现理想解里无穷大的应力。残余刚度 $k$ 和预裂纹的构造方式也会带来偏差。所以算出来的临界载荷和 $\sigma_c$ 对不上，是预料之中的事，这个公式适合用来检查量级和趋势，比如 $\sigma_c$ 随 $a$ 的变化规律，而不是用来逐位比对。
>
> 想要定量地判断算得准不准，需要一个解的是同一个问题的独立参考，比如有限元：同样的几何、边界、$l_0$、$k$ 和不可逆处理。用有限元做参考，并不意味着要把你的神经网络求解器改成有限元。
>
> 和参考比较时，反力—位移曲线是最有信息量的一张图：初始斜率反映刚度，峰值反映承载力，峰后的下降反映裂纹扩展。但只看这条曲线还不够，要结合相场剖面、弹性能和裂纹能随载荷的变化一起看，才能分清是合理的裂纹扩展，还是损伤带被算宽了、积分漏掉了局部能量之类的数值问题。
>
> 收敛也要分层做。先固定 $l_0$，加密积分点、增加网络容量，直到结果不再明显变化，说明这个 $l_0$ 下的离散误差已经控制住了；然后再减小 $l_0$，看结果怎样趋近尖锐裂纹的极限。两件事混在一起做，结果变了也说不清是哪个因素造成的。



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

> **配套导读**
>
> 回头看"从 PINN 换成 DEM"到底换了什么。在相场固定、本构保守、边界条件一致的前提下，两者解的是同一个平衡问题，应该得到同一个稳定解。不同的是求解方式：一个让每个点的残差变小，一个让整体的势能变小。如果两者算出的位移不一样，先去查积分、边界、网络表示和优化，而不是怀疑力学模型给出了两个答案。
>
> 但如果在换求解方法的同时，还把历史场换成了罚函数，或者换了拉压分解，那就不只是换了求解方法，模型本身也变了，两组结果之间的差异就不能全部归到求解方法上。完整的断裂问题还可能有多个局部极小值，不同的初始化或不同的求解顺序走到不同的状态，loss 更低的那个也不一定更符合真实的加载过程。
>
> 这些都是在划定结论的适用范围。能量下降，说明优化在推进；独立检查点上的残差变小，说明平衡在改善；和同一问题的独立参考吻合，才说明解是准的。三种证据各说明一件事，不能互相替代。



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

> **配套导读**
>
> 假设训练跑完了，裂纹也出来了，但反力明显不对。这时如果直接加宽网络或加长训练，很难判断是在改进求解，还是在用更多的算力掩盖一个公式错误。正文列出的检查是按从便宜到昂贵的顺序排的，每一步只回答一个问题。
>
> 最便宜的是本构检查，根本不用训练网络。给一个应变状态，把某个分量加减一个小量 $h$，用中心差分算能量的变化率，看它是否等于对应的应力分量。剪切分量要用对称的扰动方向，否则会差一个 2。测试的应变状态要覆盖体积拉伸、体积压缩、纯剪切，以及体积应变接近零、Amor 分解正要切换的地方。
>
> 然后是积分检查。用常数 1 去积分，确定性求积的权重和应当正好等于面积；用多项式去积分，可以验证求积的精度阶数。非均匀采样时，检查权重是否让加密区的面积占比恢复正常。
>
> 再往上是单场问题。固定相场只求位移：如果这时边界位移、反力或自由边牵引已经不对，问题就出在位移这一侧，不用把相场牵扯进来。反过来，给定历史场只求相场：检查损伤带的宽度和剖面，可以和第 7 节的指数解或均匀解对照。
>
> 两个单场都可信以后，再加入载荷步和交替迭代。这时最有用的测试是先加载再卸载：弹性位移应当回退，损伤不应回退。用历史场的话，要分别输出当前活性能和历史最大值，确认代码没有把这两个量混成一个。有预裂纹的算例，从第一个载荷步就要检查它是否还在，不要等到扩展阶段才发现初始裂纹早就没了。
>
> 如果怀疑误差来自损伤带的积分不够，就只加密那一块的求积，其他设置全部不动，看相场剖面、反力和能量是否趋于稳定。即使结果没有改善，这次实验也排除了一个可能。同时改动网络、采样和优化器，结果变好或变坏都说明不了是哪个改动起的作用。



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

> **配套导读**
>
> 参考文献不用从头读到尾，可以带着读正文时卡住的问题去找。
>
> 如果不明白"乘上虚位移再积分"为什么还是平衡，看教材 [1] 讲虚功原理和最小势能原理的部分。如果力学部分已经清楚，只是不明白神经网络怎么成为试探函数，看 Deep Ritz [3] 和 DEM [4]。强形式 PINN 的原始框架是 Raissi 等 [2]。
>
> 相场部分，建议先读综述 [13]，搞清楚裂纹能、拉压分解和不可逆处理各自解决什么问题，再回到 [6–9] 看变分断裂和正则化是怎么来的。本文用的 Amor 分解出自 [10]，历史场和分步算法出自 [11,12]，想改用罚函数处理不可逆性时，[14] 给出了罚参数怎么选。应力强度因子、能量释放率和 $J$ 积分可以查 Anderson 的教材 [17]。
>
> Manav 等 [15] 和你的研究方向最接近，值得细读。读的时候要把能量模型和数值实现分开看：按正文的介绍，他们的网络输出的是有限元节点上的场值，单元内的梯度用有限元形函数计算，再做 Gauss 求积，空间梯度并不是对网络做自动微分得到的。借鉴他们的训练设置时，要连同这些前提一起考虑，不能只拿走优化器或 log-energy 这一个技巧。
>
> 读一篇和你方案接近的论文，可以试着用自己的话回答四个问题：它最小化的是什么能量，怎样防止裂纹愈合，空间导数和积分是怎么算的，载荷是怎样一步步推进的。这四点对上了，两篇论文的结果才有可比性；很多看起来互相矛盾的结论，其实就是这四点里有某一点设置不同。



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
