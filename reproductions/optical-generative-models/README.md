# 光学生成模型 MATLAB 复现

复现论文：**S. Chen, Y. Li, Y. Wang, H. Chen, A. Ozcan, "Optical generative models," *Nature* 644, 903–911 (2025)** 中的**快照（snapshot）光学生成模型**——MNIST / Fashion-MNIST 的类别条件生成（对应论文 Fig. 1 与 Fig. 3c 实验结果）。

## 最终结果

| 结果图 | 内容 |
|---|---|
| `fig_generation_mnist.png` | **10 类手写数字条件生成**（每类 8 个样本，全部成形、样本间有笔画变化）|
| `fig_generation_fashion.png` | **10 类服饰条件生成**（T恤/裤子/套衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/靴）|
| `fig_prototypes_mnist.png` | 目标图像 vs 光学拟合相位种子的对比 |
| `fig_distilled_mnist.png` | 数字编码器 + 光学解码器全管线生成 |

拟合精度：50 个原型种子平均 MSE **0.0082**（MNIST）/ **0.0089**（Fashion）。

## 原理（与论文 Methods 的对应关系）

```
相位种子 φ (32×32) ──► 相位编码（式(1)）：上采样 10× → 320×320 @ 8 μm
        ▼
SLM（相位型） U₀ = exp(iφ)，SLM 区域外无光（暗孔径）
        ▼
角谱传播 d₀,₁ = 120.1 mm（式(2)(3)）
        ▼
衍射解码层：400×400 可学习相位（式(4)）
        ▼
角谱传播 d₁,₂ = 96.4 mm → 传感器强度 O = |U|²（式(5)）
        ▼
中心裁剪 320×320 → 10× 下采样 → 32×32
```

- λ = 520 nm，像素 8 μm，全部与论文实验装置一致
- 数字编码器（3 层 FC + 类别嵌入，式(8)）→ 相位种子，构成完整生成管线

## 复现策略（CPU 关键创新）

论文用数据中心 GPU 端到端训练（batch 100 × 数万迭代 + DDPM 教师蒸馏）。
纯 CPU 下端到端训练会塌缩到"条件均值解"（所有类生成同一团块，详见下
"教训"节）。本复现采用**两阶段分解**：

1. **阶段一 `train_prototypes.m`**——计算全息式直接优化：
   每类 K=5 个相位种子（共 50 个）+ 共享衍射解码层作为自由参数，
   以"0.5 类均值 + 0.5 真实样本"混合为目标。
   - 阶段 A（任务流）：B=1 逐任务轮流拟合 (种子ₖ + 解码层)，解码层低学习率
   - 阶段 B（抛光）：冻结解码层，热启动重拟合每个种子
   - **平均 MSE 0.008**，每个类别渲染出清晰的对应数字/服饰形状
2. **阶段二 `generate_mixture.m`**——类别条件生成：
   对类内种子做**场空间随机凸混合**（复数叠加取相位，物理上=SLM 相位
   图样的叠加）→ 光学前向 → 每类任意数量新样本，样本间自然变化。
3. **阶段三 `train_encoder_distill.m`**——数字编码器蒸馏（可选）：
   编码器 (噪声 z + 类别嵌入) → 相位种子，监督目标为阶段一种子。
   纯数字回归（无光学损失），完成后即得论文的完整架构。

## 文件

| 文件 | 说明 |
|---|---|
| `train_prototypes.m` | 阶段一：两阶段原型优化（任务流 + 抛光） |
| `forward_from_phase.m` | 相位种子 → 光学前向（式(2)-(5)，被阶段一/二共用） |
| `optical_forward.m` | 完整前向（编码器相位 → 光学 → 强度） |
| `encoder_phase.m` | 数字相位编码器（式(1)(8)，FiLM-lite 类别通路） |
| `as_propagate.m` | 角谱传播（式(2)(3)，含传递函数缓存） |
| `load_idx.m` | MNIST/Fashion IDX 数据解析（大端字节序） |
| `generate_mixture.m` | 阶段二：场混合采样生成 Fig-3c 风格网格 |
| `train_encoder_distill.m` | 阶段三：编码器蒸馏 |
| `train_optical_gen.m` | （保留）端到端训练脚本——CPU 上会塌缩，见教训 |
| `train_ddpm_teacher.m` | （可选）完整 DDPM 教师 |
| `prototypes_{mnist,fashion}.mat` | 阶段一产物：50 种子 + 解码层 |
| `fig_generation_{mnist,fashion}.png` | **最终生成结果** |

## 运行

需要 MATLAB R2021+ 与 Deep Learning Toolbox。全量数据已下载至 `../data/`。

```matlab
% 阶段一（每数据集约 18 min，可过夜跑两个数据集）
setenv('OGEN_DATASET','mnist');   train_prototypes
setenv('OGEN_DATASET','fashion'); train_prototypes

% 阶段二：生成（~1 min）
setenv('OGEN_DATASET','mnist');   generate_mixture
setenv('OGEN_DATASET','fashion'); generate_mixture

% 阶段三：编码器蒸馏（~1.5 h / 数据集）
setenv('OGEN_DATASET','mnist');   train_encoder_distill
```

## 为什么端到端训练会塌缩（重要教训）

1. **物理建模错误（根本原因）**：SLM 相位用 0 填充到 400×400 时，
   `exp(1i·0)=1` 会注入一圈幅度为 1 的平面波环带——其衍射图样与种子
   无关、恒定不变，主导传感器读数，使一切训练塌缩到同一个"平均图像"
   （看起来像个 8）。**修复：SLM 区域外复数场置零（暗孔径）**。
2. **共享解码层的条件均值吸引子**：解码层从随机初始化起、对随机散斑
   种子做批量训练时，梯度平均使它先拟合"所有目标的均值"，随后把种子
   锁死在该盆地（并行、交替、GS 算法均逃不出）。逐任务流 + 低解码层
   学习率 + 冻结抛光的分解训练可以避开。
3. **float32 梯度噪声地板**：角谱传播的 1/N² 归一化使原始损失梯度仅
   ~1e-7（单精度舍入边缘），损失需乘 1e6 抬高信噪比。
4. **编码器间接学条件**：类别嵌入（32 维）混入 1024 维噪声后，梯度
   信号被淹没；Adam 下放大梯度无效，须独立学习率组 + FiLM 直通通路
   ——但即便如此 CPU 规模下仍不收敛，故采用分解方案。

## 与论文的差异（诚实声明）

1. **训练程序**：论文端到端（GPU）；本复现两阶段分解（CPU 可行）。
   最终系统的物理结构与论文完全一致（相位种子 → 衍射解码层 → 强度）。
2. **生成多样性来源**：论文由 DDPM 教师提供样本多样性；本复现由类内
   种子的场空间混合提供（5 个原型/类的凸组合流形）。
3. **强度归一化**：逐目标亮度匹配（替代论文的逐图最大值归一化）。
4. 未复现：多波长（RGB）、迭代式模型、Van Gogh 高分辨率生成
   （需 5 层解码器 + 1000 步 DDPM + 数据中心算力）。

## 已知 MATLAB 兼容性坑（供后人参考）

- `fft2` 对复数 dlarray 要求 batch 维 >1（`as_propagate.m` 已对 B=1 自动复制）
- **本机 MATLAB 2025 隐式扩展 `(1:5)' + (128×1)` 报"大小不兼容"**——
  需 `repmat` 显式展开（极反常，注意）
- `mse()` 对非格式化 dlarray 需 `DataFormat` → 直接用 `mean((a-b).^2,'all')`
- `[a,b]` 对 'SSCB' dlarray 沿 S2 拼接；按通道拼接必须 `cat(3,...)`
- `dlconv` 权重须 'SSCU'、bias 须 'C'，且需显式 `'Padding','same'`
- IDX 文件是大端：`fread(...,'uint8=>single',0,'b')`，维度 permute [2 1 4 3]
- dlarray 复数 fft2 的批量梯度是正确的（曾误判为 bug——方向导数换算错误）
