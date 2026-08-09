# 时间快照组合子机 (Time-Snapshot Combinator Machine, TSCM)

**一个关于“运行时自我收缩能力边界”的计算理论假说与实验验证项目。**

TSCM 并非一台已实现的机器，而是一个持续演进的研究框架。它提出一台刻意放弃图灵完备性的抽象机器，通过**禁忌标记**从崩溃中学习拒绝，更关键的是，它以**“连续几何探索 + 离散坍缩决策”的交替循环**取代传统的“取指-译码-执行-写回”——并行性不是事后叠加的优化，而是探索阶段的内建属性。项目分阶段推进，从纯软件仿真到混合信号原型，最终探索光学与量子实现。

---

## 🧠 四个核心思想

| 思想 | 核心主张 |
|------|----------|
| **① 能力边界的动态收缩** | 机器的可达状态空间随运行历史单调递减——每次崩溃后永久屏蔽相关路径，计算能力越用越少，但对已知错误的规避能力逐步增强。 |
| **② 负结果驱动演化** | 将崩溃视为最有价值的学习信号，而非需要掩盖的故障。系统提取崩溃点的拓扑特征并永久标记，完成经验驱动的能力收缩。 |
| **③ 祛魅图灵完备性** | 图灵完备性并非无代价的善——它允许所有可计算行为，也允许所有可计算错误。在不同完备性等级之间可以做出理性取舍。 |
| **④ 连续探索与离散决策的交替** | 计算循环不必是“取指-译码-执行-写回”。它可以是连续几何探索与离散坍缩采样的交替——探索阶段并行考虑所有路径，决策阶段由内部序参量自发触发。 |

---

## 🧪 实验验证（Phase 1）

为了在可控环境中检验禁忌标记的持续学习效应，我们剥离了原理论中的连续动力学，将其抽象为一个带记忆的离散马尔可夫链：

- **状态转移**：\( h_{t+1} = \tanh(W \cdot h_t) \)，\( W \) 取自三类随机矩阵族，模拟不同崩溃模式。
- **禁忌标记**：崩溃后提取指纹，映射为掩码屏蔽转移矩阵的相应行。
- **规模**：1 000 轮 × 500 序列的大规模对照实验，采用 Mann‑Kendall 趋势检验。

**关键结果**：
- ✅ 禁忌标记展现强大的即时过滤能力，崩溃率从 32% 骤降至 1.6%（约 20 倍）。
- ❌ 持续学习效应缺失——崩溃率在最初几轮后完全平坦，学习迅速饱和。
- ⚙️ 通过调节屏蔽强度可避免“能力脑死亡”，但无法启动持续学习。

→ 结论：**静态环境下，学习饱和不可避免；动态自限性需要持续多样化的负结果信号。**

*实验代码纯 NumPy 实现，无 JAX/Diffrax 依赖，可在数分钟内完成全量分析。*

---

## 🗺️ 研究路线图（Phase 1–5）

| 阶段 | 目标 | 物理基底 |
|------|------|----------|
| **Phase 1** ✅ | 软件仿真验证 | CPU / GPU |
| **Phase 2** | 混合信号原型（模拟 MPE + 数字 MCC） | 分立元件 + MCU |
| **Phase 3** | 光学 MPE 探索 | 硅光芯片 |
| **Phase 4** | 全系统集成 | 混合信号 SoC / 光电集成 |
| **Phase 5** | 量子原型（QC‑TSCM） | 参数化量子电路 |

---

## 📁 仓库结构

```
tscm/
├── theory/              # 理论文档与系列博客
├── phase1/              # Phase 1 实验
│   ├── core/            # 马尔可夫链、禁忌管理器、崩溃检测
│   ├── experiments/     # 统计分析与绘图脚本
│   └── data/            # 实验原始数据
├── phase2/              # Phase 2 电路设计与仿真（准备中）
├── papers/              # 相关预印本与投稿
└── README.md
```

---

## 📖 引用与扩展阅读

- 系列博客：https://www.zhihu.com/column/c_2067727111268439020
- 实验报告：*动态自限性假说的实验检验：禁忌标记的即时有效性与持续学习的缺失*

---

## 🤝 参与贡献

欢迎对非传统计算模型、模拟/混合信号电路、光学计算或理论计算机科学感兴趣的研究者参与。您可以：

- 复现 Phase 1 实验并提出改进方案
- 参与 Phase 2 的电路仿真或原型搭建
- 提出新的崩溃模式生成策略以突破学习饱和
- 贡献文献综述或理论分析

请通过 Issue 或 Pull Request 联系我们。




## 快速开始
```bash

conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
conda create -n tscm python=3.10 numpy scipy pandas matplotlib statsmodels -y
conda activate tscm
python -m pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ lifelines

```


# Phase 1：软件仿真验证

通过离散马尔可夫链模型验证禁忌标记的即时有效性及持续学习效应。
```bash

python -m phase1.experiments.run_experiment  # 运行实验并保存数据
python -m phase1.experiments.statistical_analysis  # 统计分析及可视化

```
