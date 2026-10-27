# 论文复现实验室 · Paper Repro Lab

> 把顶刊论文"跑起来"的长期档案库：每个复现项目都提供**源码 + 已训练模型 + 结果图 + 复现报告**，并附「复现了什么 / 没复现什么」的诚实清单。

🌐 **在线下载主页**：https://alice-p197.github.io/paper-repro-lab/

| # | 论文 | 期刊 | 状态 | 资源 |
|---|---|---|---|---|
| 1 | [Optical generative models](https://www.nature.com/articles/s41586-025-09230-z)（光学生成模型） | *Nature* 644, 903–911 (2025) | ✅ v1.0.0 已复现 | [Release 下载](https://github.com/Alice-P197/paper-repro-lab/releases/latest) · [项目目录](reproductions/optical-generative-models/) |

---

## 项目一：光学生成模型（Nature 2025）

复现论文的**快照（snapshot）光学生成模型**：一个相位种子经 10× 上采样进入 SLM，经过角谱传播打到一片可学习的衍射相位解码层，再传播到传感器成像——生成模型的"画笔"从 GPU 换成了一束光。

**本复现成绩（纯 CPU，无 GPU）：**

| 指标 | MNIST | Fashion-MNIST |
|---|---|---|
| 50 个相位种子平均拟合 MSE | **0.0082** | **0.0089** |
| 类别条件生成覆盖 | 10 / 10 | 10 / 10 |
| 免训练生成一批（80 张） | < 1 s | < 1 s |

### 快速上手（免训练，约 1 分钟）

```matlab
% 要求：MATLAB R2021b+ · Deep Learning Toolbox · Image Processing Toolbox
git clone https://github.com/Alice-P197/paper-repro-lab.git
cd paper-repro-lab/reproductions/optical-generative-models/matlab
matlab -batch "download_data('mnist')"   % 下载数据集（约 12 MB）
matlab -batch "quickstart"               % 读入已训练模型，输出 10 类 × 8 样本
```

> 不想装 MATLAB？直接到 [Releases](https://github.com/Alice-P197/paper-repro-lab/releases/latest) 下载 6.5 MB 资源包，里面有全部结果图和报告 PDF。

### 完整重训（可选，纯 CPU）

```matlab
matlab -batch "setenv('OGEN_DATASET','mnist');   train_prototypes"     % ~18 min
matlab -batch "setenv('OGEN_DATASET','fashion'); train_prototypes"     % ~18 min
matlab -batch "setenv('OGEN_DATASET','mnist');   generate_mixture"     % ~1 min
matlab -batch "setenv('OGEN_DATASET','mnist');   train_encoder_distill" % ~1.8 h（可选）
```

### 与论文的诚实差异

1. **训练程序**：论文为 GPU 端到端训练（含 DDPM 教师蒸馏）；本复现为两阶段分解训练（CPU 可行），最终物理结构一致。
2. **生成多样性**：论文由 DDPM 教师提供；本复现由类内种子的场空间凸混合提供。
3. **未复现**：迭代式生成模型（论文 Fig. 4 梵高，需 5 层解码器 + 1000 步 DDPM + RGB + 数据中心算力）。
4. 编码器蒸馏管线仍在收敛中（MNIST 15000 步，种子 MSE ≈ 0.53），主结果以「原型拟合 + 场混合」为准。

完整分析见 [项目报告 PDF](reproductions/optical-generative-models/report/reproduction_report_cn.pdf)。

---

## 目录结构

```
paper-repro-lab/
├── docs/                                 # GitHub Pages 下载主页
│   └── index.html
└── reproductions/
    └── optical-generative-models/
        ├── matlab/          # 全部 MATLAB 源码
        ├── models/          # 已训练模型（prototypes_*.mat）
        ├── figures/         # 结果图 + 论文原图对照
        ├── report/          # 复现报告 PDF
        ├── logs/            # 训练日志
        └── README.md        # 项目详细说明（原理/参数/踩坑记录）
```

## 新增一个复现项目的规范

1. 在 `reproductions/` 下建 `<论文名>/`，按上面的子目录结构放代码 / 模型 / 图 / 报告 / 日志；
2. 写一个项目 `README.md`，必须包含「复现了什么 / 与论文差异 / 未复现部分」三节；
3. 打一个 `vX.Y.Z` Release，把整理好的资源 zip 挂为附件；
4. 在本页顶部的项目表格和主页 `docs/index.html` 的数据区各加一行。

## 引用

```bibtex
@article{chen2025optical,
  title   = {Optical generative models},
  author  = {Chen, Shaowei and Li, Yuhang and Wang, Yide and Chen, Hong and Ozcan, Aydogan},
  journal = {Nature},
  volume  = {644},
  pages   = {903--911},
  year    = {2025}
}
```

## 许可

- 本仓库代码：[MIT](LICENSE)
- 论文原图：版权归原出版方所有，仅用于学术交流与结果对照
