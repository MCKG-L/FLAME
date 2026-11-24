# FLAME 复现与代码学习指南

> 本文对应 *Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization*（ICML 2026）。以本地论文和源码提交 `5d486d5b0be7f72701294e69dba4e4a95cf94420` 为依据，结合发布版 v0.1.0 的模型配置。编写日期为 2026-11-21。
>
> 学习目标是从局部能量公式推到张量计算，再读懂粗定位、视觉特征融合、提示解码和训练监督。第 17 节给出当前工作区的复现路线，第 20 节说明核验范围。

## 阅读导航

1. [学习顺序与复现范围](#reading)
2. [任务定义与整体设计](#overview)
3. [代码目录、路径与版本](#source)
4. [局部能量假设与 LAD 公式](#theory)
5. [LAD、LADMulti 与输入域](#lad)
6. [LAD-Net 的卷积主干与双任务头](#ferret)
7. [SAM 语义分支、尺寸与 ROI](#sam)
8. [适配器与可学习残差](#adapter)
9. [密集提示、掩码解码与梯度](#prompt)
10. [从输入到输出的形状总表](#shapes)
11. [数据读取、采样与 EditStream](#data)
12. [损失函数与有效样本](#loss)
13. [训练循环、调度、EMA 与恢复](#train)
14. [权重加载、推理与分块](#inference)
15. [评价协议与论文结果](#evaluation)
16. [论文、源码与发布权重的差异](#differences)
17. [环境与可执行复现路线](#reproduce)
18. [入口排查与训练准备](#troubleshooting)
19. [精读、手算练习与消融](#study)
20. [核验范围与资料索引](#verification)

<a id="reading"></a>

## 1. 学习顺序与复现范围

先看第 2、4～6 节，理解一张 RGB 图怎样变成一张或多张 LAD 图，以及为什么能量图不能直接当掩码。再看第 7～12 节，把语义分支尺寸、提示 logits 和三种监督目标写出来。最后对照第 13～18 节执行推理与训练准备。

| 目标 | 所需条件 | 可以得到的结果 |
|---|---|---|
| 算子与主干核验 | PyTorch 和仓库代码 | 检查 LAD 数值、主干形状、输出和梯度 |
| 发布模型推理 | SAM 2.1 基座、FLAME 权重、配套 JSON | 对给定图像得到定位与检测结果 |
| 数据集复测 | 完整切分、GT、真实样本和明确评价口径 | 测出指定条件下的可比较指标 |
| 重训或微调 | 完整数据、环境、可运行训练入口 | 验证训练、EMA、保存和评价链路 |
| 重建论文 EditStream 实验 | 生成流程、对应已编辑数据及划分 | 评估持续更新数据对泛化的影响 |

文中用**论文**指学术描述，用**源码**指当前提交的实际执行路径，用**发布配置**指已核对 SHA256 的 v0.1.0 JSON。三者在 SAM 后端、LAD 温度、损失和评价上都有需要区分的细节。

<a id="overview"></a>

## 2. 任务定义与整体设计

FLAME 的主要任务是 **AI 编辑区域定位**。给定一张可能经过指令编辑的图，输出每个像素属于编辑区域的分数，同时支持图像级真假检测。原图、GT 和编辑指令可以出现在数据样本里，用于标注、配对或展示。模型前向从待测 RGB 获取取证与语义特征。

相比 TruFor 的 Noiseprint++，FLAME 使用显式的邻域能量统计。相比 PSCC-Net 的逐级相关性掩码，FLAME 把取证粗图作为 SAM 的密集提示。相较于 HiFi-Net 的生成方法归因，FLAME 的重点是同一张编辑图内的局部定位。论文的跨生成器实验支持在其测试分布上的比较，实际输入上的误报还取决于图像内容、分辨率、压缩和所用权重。

~~~mermaid
flowchart TD
    X["待测 RGB"] --> LAD["LAD 或多 τ LAD"]
    LAD --> F["LAD-Net 取证特征"]
    F --> D["整图检测 logit"]
    F --> C["粗定位 logits"]
    X --> S["冻结的 SAM 图像编码器"]
    S --> A["三尺度特征适配器"]
    F --> A
    C --> P["SAM 密集提示编码"]
    P --> M["SAM 掩码解码器"]
    A --> M
    M --> L["原始分割 logits 与 IoU 估计"]
~~~

这里有四个容易混淆的量。LAD 是像素邻域能量，粗图是取证头的定位 logits，最终图是 SAM 解码后的定位 logits，IoU 预测是解码器对其掩码质量的标量估计。检测 logit 来自取证主干上的分类头。

<a id="source"></a>

## 3. 代码目录、路径与版本

~~~text
FLAME/                             仓库根目录
├── README.md
├── docs/                          本地 ICML 论文
├── doc/                           本指南
└── FLAME/                         Python 模块根目录
    ├── train.py                   CLI、训练循环、损失、优化器、保存
    ├── test.py                    单图可视化入口，构造配置有版本差异
    ├── test_dataset.py            数据集评价、配置驱动建模、扰动评估
    ├── model/
    │   ├── ferret_backbone.py     LAD/LADMulti、取证 CNN、粗图和检测头
    │   ├── forgerylocalizer.py     SAM 后端、适配器接入、提示和解码
    │   ├── adapters.py           SharedAdapter、NormGatedAdapter
    │   ├── glp_module.py          独立热图模块，发布路径不调用
    │   └── ema.py
    ├── utils/
    │   ├── localforgerydataset.py 图片、掩码、真图与局部/全局输入
    │   ├── dataset_config.py      数据集 JSON、ConcatDataset 与 loader
    │   ├── validate.py           逐图定位与整图检测指标
    │   ├── tiling.py             分块推理与全局特征缓存
    │   └── checkpoint_state.py    权重快照、兼容加载
    ├── sam2configs/               Hiera B+ 配置及基座权重预期位置
    ├── checkpoints/               FLAME 发布权重预期位置
    ├── tests/                     取证主干自检
    └── demo_test_set/             MagicBrush 示例
~~~

本文的 CLI 命令从仓库根目录 `/home/mckg/FLAME` 发起。直接导入 `model.*` 时，从内层 `/home/mckg/FLAME/FLAME` 工作，或将其加入 Python 模块搜索路径。训练配置 JSON 用于组织数据，发布文件 `model_params.json` 用于重建模型，两份 JSON 的用途不同。

当前仓库有 SAM2 主路径和可选的 `sam3_interactive` 分支。论文图文和附录涉及 SAM 3，公开发布 JSON 指定 SAM 2.1 Hiera B+。以下张量表默认按发布版配置阅读，SAM3 的 native prompt、双路解码等选项单独讨论。

<a id="theory"></a>

## 4. 局部能量假设与 LAD 公式

### 4.1 论文为什么比较邻接像素

论文 §3 将像素组成的图像视作格点，写出 Gibbs 形式 $p(x)\propto\exp[-E(x)]$，使用相邻像素 RGB 差异构造局部二元势函数。差异大意味着存在较强局部变化，可能对应纹理、边缘、成像噪声或编辑边界。

在论文的局部潜空间编辑模型中，编辑区域使用生成轨迹，保留区域使用参考轨迹。作者在谱能量不等式假设和潜空间合成模型下，分析两类统计现象：

- **内部能量差**。生成区域的高频局部变化可能较弱，出现低能量区域。
- **边界能量峰**。生成与保留区域相接时，邻域协方差变化可能造成较高的跨边界能量。

这些是特定假设下的统计线索。真实平坦墙面也会产生低局部能量，真实物体轮廓也会出现高能量。模型用 CNN 判断区域模式，再用 SAM 为定位提供结构信息。论文中的解释和简单的“低能量即伪造”阈值规则会产生不同结论。

### 4.2 从二元势函数到 LAD

设 $I(p)\in[0,1]^3$，$\mathcal N_p$ 是中心挖空的奇数邻域。论文式 (5) 为

$$
\operatorname{LAD}(p)=\frac{1}{|\mathcal N_p|}
\sum_{q\in\mathcal N_p}
\tanh\left(\frac{\|I(p)-I(q)\|_2^2}{\tau^2}\right).
$$

注意顺序。先对三个 RGB 通道平方求和，再对每条邻接边施加 $\tanh$，最后平均邻居。对单边 RGB 差异，如果只有一个通道变化 $1/255$，$\tau=0.004$ 时势值约为 0.744。若三个通道各变化 $1/255$，势值接近 0.994。若只有一个邻居变化，这两个值还要各除以 8。

小差异区间内，$\tanh(u)\approx u$。差异远大于 $\tau$ 时势函数趋近 1，避免强边缘的幅值无限主导结果。$\tau$ 是在当前 RGB 数值域中使用的温度，跨输入尺度复制其数值会改变响应。论文将这一温度下的敏感性与 8-bit 量化步长联系起来，详见论文附录 E、G。

<a id="lad"></a>

## 5. LAD、LADMulti 与输入域

### 5.1 源码的每个张量轴

以 `LADOperator(kernel_size=3)` 为例，`F.unfold` 产出 `[B,3×9,HW]`。重排成 `[B,3,9,HW]` 后，索引 4 是中心，剩余 8 个索引是邻居。RGB 差值形状为 `[B,3,8,HW]`，沿 RGB 求平方和得到 `[B,8,HW]`。最后对邻居平均并恢复到 `[B,1,H,W]`。主要实现位于 [ferret_backbone.py](../FLAME/model/ferret_backbone.py#L173)。

源码以零 padding 展开邻域。即使输入是均匀非零图，内区 LAD 为 0，边界仍可能响应，因为窗口与零 padding 比较。对于 $3×3$ 和足够大的常量，同一条边的非角像素约有 3 个外侧邻居，角点有 5 个。分析编辑边缘时应区分图像外缘效应。

### 5.2 发布版为什么有四个通道

| 取证算子 | 输入主干的通道 | 计算 |
|---|---:|---|
| `lad` | 1 | 单个 $\tau$，通常为 0.004 |
| `lad_multi` | 4（发布版） | 依次计算 0.016、0.032、0.064、0.128，沿通道拼接 |
| `mldc` | 3 | RGB 每通道八方向差分，24 通道经可学习投影 |
| `rgb`、`image`、`identity` | 3 | 直接使用 RGB |
| `lad_mldc_hybrid` | LAD 主输入与并行 MLDC | 在高分辨率特征层通过投影及 sigmoid 门控注入 |

四个温度共享同一套邻域定义，数值越大，同样的局部差异通常响应越小。多温度使主干能够同时读取早饱和、晚饱和的统计，而不是求四个温度的平均。`AdaptiveTauFusion` 是部分提示头启用的可选分支，学习乘数 $1+\delta\tanh(z)$，最后一层零初始化使初始乘数为 1。发布版采用普通 `mask_compressor`，因此没有启用该融合头。

### 5.3 为什么要保存 raw RGB

训练入口 `select_local_input()` 优先读取 `local_patch_raw`，其值为 `[0,1]`。传统字段 `local_patch` 和训练态 `orig` 包含 SAM/ImageNet 标准化。把标准化数据交给 LAD 会按通道放大差值，改变 $\|I(p)-I(q)\|^2/\tau^2$ 的分布。

学习时可以打印 raw 和 normalized 张量的范围。像素颜色通道相减时均值抵消，除以标准差的缩放仍保留。第 14 节说明分块入口需要额外核对这一点。

<a id="ferret"></a>

## 6. LAD-Net 的卷积主干与双任务头

### 6.1 论文名字与代码名字

论文 LAD-Net 对应 `FerretBackbone`。当前发布版使用 96 为基础宽度，取证图经两个 stem 卷积、两个下采样 stage 和最终投影。构造函数内把 `depths` 重新设为 `[2,2]`，传入自定义 depths 时需检查实际生效结构。

~~~text
LADMulti [B,4,512,512]
  → cbr1: Conv 3×3, stride 1 + IN + ReLU → [B,48,512,512]
  → cbr2: Conv 3×3, stride 2 + IN + ReLU → [B,96,256,256]
  → Stage 1: 2 个 DSBlock，首块 stride 2 → [B,192,128,128]
  → Stage 2: 2 个 DSBlock，首块 stride 2 → [B,384,64,64]
  → final_conv: 1×1 → [B,384,64,64]
~~~

`DSBlock` 先在两个并行的 3×3 分组卷积上使用 dilation 1 和 2，对应 3×3 和 5×5 的感受野。这两路形状相同，相加后经 1×1 投影、IN 和 ReLU。随后用 depthwise 3×3 加 pointwise 1×1 的可分离卷积，再接 IN 和 SE。SE 将每通道池化为标量，经压缩比 8 的小网络和 sigmoid 后乘回特征。

残差分支在尺寸或通道变化时用 1×1 卷积和 IN 对齐，其他情况为恒等。主路与残差相加后执行 ReLU。IN 以单样本的空间统计归一化各通道，减弱对 batch 运行统计的依赖。建议用 hook 核对 stride=2 首块与 stride=1 后续块的差别。

### 6.2 粗定位头

发布版 `mask_compressor` 对深层特征执行 `3×3 Conv(384,96) → IN → ReLU → 1×1 Conv(96,1)`，输出 `[B,1,64,64]`，再双线性插值为 `[B,1,256,256]`。其输出是正负都可取的 logit。附录 F 描述 `1×1` 输入投影，发布 JSON 明确使用 `mask_compressor_kernel_size=3`。

`mask_compressor` 路径把同一组原始粗图 logits 同时作为粗监督和提示来源。采用实验性 split head 时，`coarse_mask` 可以维持旧头输出，而 `dense_prompt_mask` 增加另一条残差，二者的监督和用途随之分离。

### 6.3 图像级检测头

深层 `[B,384,64,64]` 特征经 GAP 成 `[B,384]`。发布结构的分类头为 `LayerNorm → Linear(384,192) → GELU → Dropout(0.2) → Linear(192,1)`。早期 `legacy_logit_head` 用 Dropout 和一层 Linear，加载时键名与参数形状会有区别。

分类头每次前向计算 logit。是否有图像级 BCE 梯度由训练调用中的真实样本比例等条件决定。推理时 sigmoid 后数值越高越倾向于编辑，和掩码最大值是两种可独立抽取的分数。

<a id="sam"></a>

## 7. SAM 语义分支、尺寸与 ROI

### 7.1 语义表示从哪里来

`ForgeryLocalizer` 构建 SAM 编码器、prompt encoder 与 mask decoder。对 SAM2，图像编码在 `torch.no_grad()` 下提取 `backbone_fpn`。将低分辨率语义嵌入与两个高分辨率特征送给 adapter。代码先用解码器的 `conv_s0`、`conv_s1` 把高分辨率通道投影到 32、64；主嵌入保持 256 通道，并可加静态的 no-memory embedding。

仓库 YAML 提供 Hiera B+ 的构造结构。YAML 内 `image_size: 1024` 属于 SAM 基座配置，FLAME 将 `sam.image_size` 和提示嵌入尺寸按自己的全局输入设置。`output_resolution` 还会单独决定送给 SAM2 decoder 的三尺度空间尺寸。发布版训练和数据集加载通常使用 512，不能简单从 YAML 中的 1024 推导所有张量。

### 7.2 全局图与局部图

训练数据可同时提供：

- `local_patch_raw`，LAD 分支计算的局部 RGB。
- `global_image`，待测图的全局缩放及标准化结果。
- `norm_coords=[x_1,y_1,x_2,y_2]`，局部 patch 在原图中的归一化坐标。

模型先编码全局图，再通过 ROI Align 从语义金字塔截取局部对应范围。整图 resize 的坐标为 `[0,0,1,1]`，三个尺度形状也已对齐时会直接复用特征。裁剪时，将归一化坐标乘全局图宽高，以 `[batch_index, x1, y1, x2, y2]` 送给 ROI Align，并用特征图相对全局图的分辨率换算坐标。靠边 patch 若包含反射 padding，ROI 仍限制在原图范围内。

若提供 `global_context`，直接复用此前已编码的语义特征，并要求同时提供 `norm_coords`。默认非训练数据集没有这两个全局字段时，forward 回退为 `encode_global(orig)`。SAM2 的这个回退函数直接将输入交给 encoder，而训练全局视图已在数据侧用 SAM2 transform 处理。自行对比训练与评价时，应把这两个路径的归一化差异当成变量单独检查。

<a id="adapter"></a>

## 8. 适配器与可学习残差

### 8.1 发布版使用 SharedAdapter

每个尺度把最终 384 通道取证图插值到对应语义特征大小，拼接后经 `1×1 Conv → GELU → Dropout2d → 1×1 Conv → GELU → 1×1 Conv`，再加回原语义分支。

$$S_i^{\mathrm{adapt}}=S_i + \eta A_i\big([S_i;\operatorname{Resize}(F_{\mathrm{tex}})]\big).$$

发布配置 $\eta=1$、隐藏宽度 64、所有尺度开启。源码 `SharedAdapter` 是一个多尺度容器，每个尺度在三个 `ModuleList` 中持有独立投影。因此不同尺度的卷积参数彼此独立。

| 尺度索引 | 语义通道 $C_i$ | 拼接通道 | 残差投影 |
|---|---:|---:|---|
| 0 | 256 | 640 | 64 → 256 |
| 1 | 32 | 416 | 64 → 32 |
| 2 | 64 | 448 | 64 → 64 |

### 8.2 恒等初始化对反向有什么影响

前部卷积用 Xavier 初始化，最后的 `mlp_up` 权重和 bias 为零。因此第 0 步输出等于原语义特征，对冻结的 SAM 解码器而言初始特征保持原值。第一次反向，最后的投影可以得到梯度，而仅经零投影回传到 adapter 前部的梯度初始为零。随着残差投影更新，前部才逐步获得这一条路径的梯度。

取证主干可同时通过粗图损失、检测损失和提示路径接收梯度，因此 adapter 的初始状态不意味着整个 LAD-Net 锁住。可选 `norm_gated` 还增加 GroupNorm、尺度 $\gamma$ 和样本门控。发布路径采用 SharedAdapter，相关门控值无须套到已发布模型上。

<a id="prompt"></a>

## 9. 密集提示、掩码解码与梯度

### 9.1 提示进入解码器之前

对发布的 SAM2 路径，适配后的 `image_embeddings` 在 512 输出分辨率下是 `[B,256,32,32]`。prompt encoder 要求掩码提示空间大小为 embedding 的 4 倍，因此取证头固定上采样的 256×256 粗 logits 在进入 prompt encoder 前再缩到 128×128。

发布配置 `coarse_prompt_transform=none`，scale=1，bias=0。代码也提供按样本减均值、z-score 及可学习偏置或残差细化。这些变换作用于提示分支。用于辅助 BCE 的 `coarse_mask` 仍可保持原值。

常规 SAM2 解码调用只给 `masks=coarse_prompt`，点和框为 `None`。prompt encoder 产生 dense embedding，解码器同时读取图像嵌入、位置编码和两路高分辨率特征。`multimask_output=False` 表示只要单掩码，`repeat_image=False` 保持批内每图一组输入。解码后通过 `postprocess_masks` 恢复模型输出分辨率，推理端才 sigmoid 和阈值化。

### 9.2 参数冻结和梯度通路

| 模块 | 主路径的训练状态 | 来自何处的梯度 |
|---|---|---|
| SAM 图像编码器 | 冻结并保持 eval，编码前向包在 no_grad 内 | 不更新 |
| SAM 提示编码器 | 参数冻结且 eval | 掩码损失仍可经它传到输入提示 |
| SAM 掩码解码器 | 参数冻结，IoU head 视开关解冻 | 掩码损失经解码运算传到 adapter / 提示 |
| IoU prediction head | 发布配置启用 | 预测 IoU 与真实 IoU 的 L1 |
| SharedAdapter | 默认训练 | 最终图相关目标 |
| LAD-Net 和粗定位头 | 默认训练 | 粗 BCE、分类和来自提示/adapter 的梯度 |

冻结权重与停止输入梯度是不同操作。`set_frozen_sam_eval()` 固定 SAM 模块的 eval 状态，仍保留适配器需要的梯度计算。`glp_module.py` 有独立双分支热图模块，当前 wrapper 的导入已经注释。发布模型的提示直接来自 coarse head。

### 9.3 建议怎么读实验性提示头

先理解 `mask_compressor` 路径，再阅读 `_make_coarse_and_dense_prompt_logits()` 中某个实验分支。`split_multiscale` 将可学习提示残差和粗监督分开，带门控的版本可以按位置扩张或抑制。许多模块为了兼容旧权重，将残差出口或某个门初始设到接近恒等。训练里对应的 gate、signed residual、oracle 和面积项各自有权重，只有运行配置明确启用才纳入实际训练目标。

需关注 `extras["coarse_mask"]`、`extras["dense_prompt_mask"]` 和 `extras["coarse_prompt"]` 的区别。第一项是粗定位监督对象，第二项是提示候选原图，第三项是进入 prompt encoder 前已经调整尺寸和数值的张量。改变其中一个并不自动改变另外两个。

<a id="shapes"></a>

## 10. 从输入到输出的形状总表

下表是多 τ 发布结构、512×512 输入和 512×512 输出时的设计形状。LAD-Net 部分可直接用现有测试和结构探针核验，SAM 细化部分按配置与调用尺寸推导。

| 节点 | 形状 | 说明 |
|---|---|---|
| 待测图 | `[B,3,512,512]` | LAD 输入取 raw RGB |
| LADMulti | `[B,4,512,512]` | 四个温度 |
| 取证 CNN 最终特征 | `[B,384,64,64]` | 被送往 adapter、粗头和分类头 |
| 粗头低分辨率图 | `[B,1,64,64]` | 原始 logits |
| `coarse_mask`、`dense_prompt_mask` | `[B,1,256,256]` | Ferret 输出尺寸 |
| 分类 logit | `[B,1]` | GAP 后分类 |
| 三尺度解码特征 | `[B,256,32,32]`、`[B,32,128,128]`、`[B,64,64,64]` | 按源码顺序，最高分辨率不排在首位 |
| SAM2 掩码提示 | `[B,1,128,128]` | 四倍 embedding 空间尺寸 |
| SAM2 dense embedding | `[B,256,32,32]` | prompt encoder 的输出 |
| 单掩码低分辨率输出 | `[B,1,128,128]` | decoder 上采样 4 倍 |
| 最终掩码 logits | `[B,1,512,512]` | postprocess 后与 GT 对齐 |
| SAM IoU 估计 | `[B,1]` | 掩码质量监督 |

阅读返回值要注意代码注释可能沿用旧尺寸。例如 Ferret 的粗图注释写 `H/4`，真实代码插值使用固定 256。换成 512 输入时，$H/4=128$，这两个数当然不同。

<a id="data"></a>

## 11. 数据读取、采样与 EditStream

### 11.1 论文数据与仓库数据目录

论文在 MagicBrush、SID 上训练及测试，用 CoCoGLIDE、AutoSplice 和三组 EditStream 数据测试分布外表现。MagicBrush 多轮编辑的论文 GT 取历轮区域的并集。附录 D.1 给出 MagicBrush 验证 528、测试 1,053，SID 训练 10,000、验证 528 的实验设置。当前 JSON 只描述数据目录和采样方式，数据准备者需要把相应划分和累计掩码整理成预期格式。随仓库提供的 MagicBrush 三联图用来检查读图和评价入口。

~~~text
dataset_root/
├── target/
│   ├── 30.png
│   └── 149.png
├── mask/
│   ├── 30.png
│   └── 149.png
└── source/
    ├── 30.png
    ├── 149.png
    └── 149.txt       可选指令
~~~

平铺结构按文件 stem 配对 `mask/<stem>.png`，source 匹配常见 PNG/JPG。缺 source 时，读取路径使用 target 作展示回退，同时无法从该目录补充真实样本。`allow_multiple_targets=true` 以最后一个下划线前的名称分组，多个 target 可能共用同一 source 和 mask。每轮标注不同时，应事先创建每个目标对应的独立标签。另一种嵌套结构递归配对 `Forged/` 与 `Mask/`，按 stem 取第一个匹配 mask，目录内存在重名时要人工核对。

### 11.2 掩码解析需要预检

平铺模式会打开 mask PNG，再根据阵列形态转换：

| 输入格式 | 当前二值化逻辑 |
|---|---|
| 灰度 | 整除 255 |
| 全不透明 RGBA | 红通道大于 0 |
| 含透明区域 RGBA | alpha 为零的区域取前景 |
| 普通 RGB | 第一通道大于 0 |

建议自建数据保存为灰度 **0/255 PNG**。若灰度文件直接存 0/1，源码的整除会将前景抹为 0。软灰度边缘也只有达到 255 才保留。`create_combined_mask()` 支持可选源图差分与标注求交，`use_diff` 默认为 false。发布推理只需要一张待测图。

### 11.3 缩放、裁剪与批处理

| 情况 | RGB | 掩码 | 全局字段 |
|---|---|---|---|
| 训练且 force resize | 缩到固定正方形 | 最近邻同步缩放 | 标准化全局图和整图坐标 |
| 训练且 crop | 小图反射填充，按前景中心或随机裁剪 | 同一几何变换 | 标准化全局图和裁剪坐标 |
| 默认非训练 loader | resize 到 `img_size` | 最近邻对齐 | 不提供 global_image / norm_coords |

训练的 `orig` 和 `local_patch` 是兼容用的标准化字段，`local_patch_raw` 才用于 LAD。评价态 `orig` 保持 RGB 原始数值范围，但已经 resize。可视化 `source` 与 `instruction` 单独存放。`streams` 是旧接口保留字段，当前 Ferret-SAM 数据返回空列表。显式鲁棒性扰动在 `test_dataset.py` 的 `InputPerturbationDataset` 进行。

`custom_collate_fn` 跳过缺图或缺 mask 的样本，全部无效时返回空字典。混合尺寸无法 stack 时回退到列表，使用普通验证路径前应保证尺寸一致。日志表里的加载样本数、有效样本数和参与指标的样本数可能不同。

### 11.4 真实样本比例和两套标签

若 $0<r<1$，数据集按 $n_{\rm real}=\lfloor n_{\rm fake}r/(1-r)\rfloor$ 从根目录下的 `source/` 抽样。$r=0.2$ 表示理想情况下真实图片占总数 20%，相当于每 4 假配 1 真。$r=1.0$ 特判为读取全部可用 source，并不是删除所有假图。真实图片数量还受 source 实际文件数约束。

`authentic_source_dir` 在当前实现中不会改变实际目录，`DatasetManager` 传给数据集的值也是 `None`。数据集配置的比率为零时可被全局 `--authentic_ratio` 覆盖，CLI 默认 0.2。训练 loader 做普通 shuffle，无逐批配比保证。

存在两套不同的样本属性：

- `sample_type` 表示样本来自 target 还是额外采样的 source。
- `is_authentic` 对 target 按当前 mask 是否为空设定。随机裁掉编辑区域，或 mask 格式导致全空时，它会变成 true，而 `sample_type` 仍是 forgery。

损失与图像级评价优先使用 `is_authentic`，定位平均值则按 `sample_type` 排除额外加入的真图。复现时把这两个字段一起打印，避免把“原图有编辑”和“当前 patch 有编辑”混成一个含义。

小数据还需处理边界情况。验证集只有三张假图且 $r=0.2$ 时，公式会得到 0 张真图。当前非训练抽样分支随后用这个 0 作步长计算的除数。示例评价命令用 `--authentic_ratio 1.0`。自定义重训小验证集应检查计算出的真图数量。

### 11.5 数据配置和验证清单

[datasets_default.json](../FLAME/configs/datasets_default.json) 默认将 MagicBrush 和 SID 分成训练、验证条目，经 `ConcatDataset` 合并。相对数据路径在 `data_root` 下解析，`data_root` 为相对值时受启动目录影响。目录缺失条目会被跳过，建议正式重训使用绝对数据根路径并核对每个条目的装载数。

配置级 `force_resize` 非空时会覆盖全局设置。训练 batch size 由 CLI 覆写 train loader，JSON 中的验证 batch size 单独生效。`sample_ratio` 为 0～1 之间的子集比例。可用 `--val_manifest` 绑定样本身份来复测同一批验证图，这比只保留随机种子更直接。[验证清单实现](../FLAME/utils/validation_manifest.py)

### 11.6 EditStream 在复现中的位置

论文把 EditStream 描述为数据生产闭环。选择原图、计划编辑及区域、调用编辑模型、筛除不合格样本，再用新样本和历史数据做适配。论文给出 Qwen-Image-Edit 500 张用于微调、2,500 张用于测试，并对 Flux Kontext 和 Nano Banana 等新域进行评价。微调策略在 batch 内混入 20% 旧域数据，目的是比较新域适配时是否遗忘 MagicBrush 和 SID。

当前仓库提供现成数据的读取和训练，未包含上述代理式生成闭环的完整运行入口。重新训练 FLAME 和重建 EditStream 是两个实验。使用 Qwen 数据微调后，Qwen 对 FLAME-F 成为 ID，解读跨模型泛化时需要把这一列单独识别。

<a id="loss"></a>

## 12. 损失函数与有效样本

### 12.1 从论文目标到代码组合

论文 §4.2 写出

$$
L_{\rm paper}=L_{\rm Dice}+20L_{\rm Focal}+L_{\rm IoU}+L_{\rm Det}.
$$

源码 [train.py 的 compute_loss](../FLAME/train.py#L2603) 增加了对粗图的直接监督，并有不同默认权重。以发布 JSON 的主要选项表示，忽略防溢出裁剪后可写成

$$
L_{\rm release\ config}
=L_{\rm Dice}+10L_{\rm Focal}
+0.2L_{\rm coarse\ BCE}
+0.25L_{\rm IoU}
+\lambda_{\rm det}L_{\rm Det}.
$$

发布 JSON 记录的 coarse Dice 为 0，CLI 的 $\lambda_{\rm det}$ 默认为 1。最后三项是否进入总损失还由样本和开关决定。发布 JSON 所记录的配置可用于对照，论文公式和值应分别注明来源。

### 12.2 哪些图片训练定位

首先选取 `forgery_mask`。优先用 `~is_authentic`，其次由 `sample_type != "authentic"` 判定，最后退回检查 GT 前景像素。

Focal、Dice、粗 BCE 和 IoU 回归均只在这个子批上算。真实样本的全零标签由图像级 BCE 使用，不直接传给上述定位损失。若整个 batch 均判为真实，定位损失为 0。检测也关闭时，检查训练循环对无梯度 loss 的处理，再决定是否使用纯真实批。

### 12.3 Focal 与 Dice 的实现

令最终分割 logit 为 $z$、$p=\sigma(z)$、$y\in\{0,1\}$。

$$
p_t=yp+(1-y)(1-p),\quad
\alpha_t=y\alpha+(1-y)(1-\alpha),
$$

$$
L_{\rm Focal}=\operatorname{mean}\left[
\alpha_t(1-p_t)^\gamma \operatorname{BCEWithLogits}(z,y)
\right].
$$

helper 先将输入 logits 局部截断到 `[-10,10]`，再限制 $p_t$，最后 `nanmean`。发布 JSON 的 $\alpha=0.55,\gamma=1$。$\gamma$ 减弱已正确分类的像素对目标的贡献。

Dice 在每个样本上先计算概率形式的交叠：

$$
L_{{\rm Dice},i}
=1-\frac{2\sum_j p_{ij}y_{ij}+\epsilon}
{\sum_jp_{ij}+\sum_jy_{ij}+\epsilon}.
$$

再对有效图片求均值。helper 遇到 GT 和阈值后二值预测都为空时跳过该图，GT 空但预测不空时仍会惩罚。最终分割 logits 直接进入 Dice，未共享 Focal 的 logit 截断。不要把该实现误写成 TruFor 的平方分母 Dice。

### 12.4 为什么粗图需要辅助 BCE

粗 logits 先以 nearest 插值到 GT 大小，再 clamp 到 `[-10,10]`。BCE 直接约束其前景与背景语义，使取证 CNN 有局部目标，也为送入 SAM 的提示建立位置先验。没有辅助监督时，提示也可能通过最终分割反向学习，但数值可偏离可解释的粗定位图。

粗 Dice 的代码仍会计算，权重为 0 时不贡献梯度。对于 split head，应分别检查 coarse/dense prompt 的监督。当前发布结构下两者共享同一粗头。

### 12.5 IoU 质量与整图检测

**IoU 回归。** 当 `train_sam_iou` 开启且有假图时，在 no_grad 下将最终分割按 0.5 二值化，逐图计算 IoU。预测和 GT 都空时目标为 1。SAM 输出的质量分数 clamp 到 0～1 后与该目标做 L1。该目标是基于当前模型预测而计算，随训练变化，且不会经阈值计算反传。

**图像级 BCE。** 当 `authentic_ratio > 0` 且 extras 提供检测 logits 时，按假为 1、真为 0 做二元 BCE。它训练 CNN 的 GAP/分类头。评估中应使用 sigmoid 后的同一分数，或明确切到论文附录的 mask max 协议。

**其他约束。** 主训练循环还能给 adapter 的特征增量、提示面积、门控及最终 logits 校准增加目标。发布 JSON 中这些额外权重为 0。源码对一些 loss 和总目标设上界，超过上界的分支会进入常数区，诊断异常梯度时应同时记录裁剪前后的数值。

<a id="train"></a>

## 13. 训练循环、调度、EMA 与恢复

### 13.1 三种配置不能混用

| 项目 | 论文公式或附录 H | CLI 默认 | 发布 JSON |
|---|---|---|---|
| LAD | 单 τ 0.004 | `lad` / 0.004 | `lad_multi` / 0.016, 0.032, 0.064, 0.128 |
| SAM | 附录 F 涉及 SAM 3 | `sam2` | `sam2` |
| 基础 LR | 0.001 | 0.001 | 0.00012 |
| Focal 权重 | 20 | 10 | 10 |
| Focal α / γ | 0.6 / 1.0 | 0.55 / 1.0 | 0.55 / 1.0 |
| IoU 权重 | 1 | 0.25 | 0.25 |
| 粗图 BCE 权重 | 主公式未单列 | 0.2 | 0.2 |
| 优化器/调度器 | 表 H 写 Adam 权重衰减 | AdamW / cosine | JSON 记载 scheduler 为 none |
| batch / epoch | 表 H batch 8 | 8 / 3 | JSON 未记录这两个字段 |
| 训练尺寸 | 对照论文实验 | 512 | 512 |
| EMA decay | 参照源码 | 0.975 | 0.975 |
| 保存模式 | 参照源码 | full | trainable_only |

论文的超参搜索最优值和下载权重的训练记录用途不同。以 CLI 默认运行，可作为本地基线，不能据此推断重现了发布权重的训练过程。

### 13.2 一次训练迭代

1. `train_epoch()` 将模型设为 train，wrapper 保持冻结 SAM 模块为 eval。
2. 从 batch 选 raw patch、GT、全局视图和 ROI 坐标，进入 autocast。
3. 模型返回 logits 与 extras，计算主损失和已启用的附加目标。
4. 用 GradScaler 缩放 loss 并反向。`batch_idx > 2` 时 unscale 后以范数 1.0 裁剪。
5. 调用 scaler/optimizer step，更新 scaler，清空梯度，有效条件下推进 scheduler。
6. 更新可训练权重的 EMA，在验证步点替换参数并评价。

优化器为 AdamW，默认 weight decay `1e-4`。adapter、Ferret 可以按选项冻结，IoU head 单独控制。提示校准器与提示头还有学习率倍率。源码设置 torch、NumPy 和 Python seeds 为 0，同时开启 cuDNN benchmark，重复实验应记录依赖版本、数据清单和实际 LR 曲线。

### 13.3 调度器的时间单位

`train_epoch` 在训练 batch 内调用 `scheduler.step()`。OneCycle 的初始化按 epoch 数和每轮 batch 数计算步数。cosine 的 `T_max` 却设置为 `args.epochs`，与按 batch step 的频率不一致，实际 LR 会比预期的逐 epoch 衰减更快到达极小值，并可能回升。实验建议使用发布 JSON 的 `none` 先对齐，再用正确单位设计比较。用日志中的 LR 曲线核实行为。

### 13.4 EMA、选点与保存

EMA 对可训练参数保存影子副本，并更新 $w_s\leftarrow \rho w_s+(1-\rho)w$。内部验证前应用 EMA，保存最佳结果时复制当时的参数，随后恢复训练参数。`clone_state_dict` 取 CPU clone，避免 `state_dict()` 的张量引用在 restore 后指向更新过的数据。

训练按每轮内的 `(batch_idx+1) % val_steps == 0` 触发验证，CLI 默认 2000。最佳值使用 `F1+IoU`，与图像级 AP 无直接关系。小数据实验必须降低 `val_steps`，否则可能没有触发内部验证与最佳权重保存。`checkpoint_save_mode=none` 时只更新最佳分数。

| 保存方式 | 内容和用途 |
|---|---|
| `full` | 完整模型参数/缓冲及可选 EMA 字典 |
| `trainable_only` | 只存保存时可训练的参数，构建时依赖匹配的 SAM 基座 |
| `none` | 用于短诊断，不生成相同入口的模型 checkpoint |

主配置会保存在运行目录，带分数的 checkpoint 配有对应参数 JSON。仓库构造 payload 的字段为 `model,epoch,batch,score,checkpoint_save_mode`，full 条件下可有 `ema`。它没有 optimizer/scaler/scheduler/RNG 的完整状态。`--resume_checkpoint` 恢复权重并从记录 epoch 加一开始，更接近带元数据的微调起点。即便 checkpoint 在 epoch 中间保存，也没有从该 batch 精确续跑的状态。

<a id="inference"></a>

## 14. 权重加载、推理与分块

### 14.1 先选入口，再对齐架构

`test_dataset.py` 从发布 JSON 读取 SAM 后端、取证算子、多 τ 列表、分辨率、适配器与提示配置。先构造基座，再加载 FLAME 权重叠加。`load_matching_state_dict()` 检查键名与 shape，打印跳过的变量。若从单 τ 扩展为多 τ，这个 helper 有一条将 stem 权重复制再除以新通道数的明确适配规则。正常复现发布模型时，多 τ 的 stem 应当直接形状匹配。

trainable-only checkpoint 下冻结 SAM 参数的 missing keys 可由基座补齐。若 `ferret_backbone.cbr1.0.weight`、粗头、adapter 或 IoU head 的应加载权重缺失，结果应先视作配置不匹配。若使用 `--use_ema`，加载器要求 checkpoint 含独立 `ema` 字典。训练保存时模型本身可能已是验证用 EMA 参数，需按文件结构决定是否再覆盖。

`test.py` 的构造没有转发 `forensic_operator`、`lad_multi_taus`、`output_resolution` 等设置，按构造默认得到单 τ LAD 和 256 模型输出。v0.1.0 配套结构使用四通道 stem。相同键的 4 对 1 通道 shape 冲突即使 `strict=False` 仍会报错。因此复现发布版优先走配置完整的 `test_dataset.py`。要做单图展示，可构造只含一组 target/mask 的小目录，或统一使用数据集入口的建模函数。

### 14.2 固定尺寸推理

1. target resize 为 `--img_size`（CLI 默认 512），转为 raw RGB tensor。
2. 加载模型配置中的训练输出尺寸（发布版为 512）。
3. 得到最终 logits、粗 logits 和分数，按需要对概率和 GT 对齐。
4. 样本按阈值转为二值掩码，评价输出 JSON 和可选可视化。

CLI 的数据 resize 与 JSON 的模型输出分辨率是不同参数。若二者不一致，常规验证代码会插值概率图供 F1 使用，但 logits 路径计算 IoU 时直接用原 logits 送 `compute_iou()`。保持两者一致，避免同一次评价中出现不同空间张量。

### 14.3 全局引导的分块推理

`--use_tiling` 调用 `overlap_tile_predict_logits_global_guided()`。默认窗口 512、有效中心 384、stride 384。每边保留 64 像素上下文，反射 padding 后扫描 tile，保留中心 logits。重叠部分按覆盖次数平均，再裁回输入网格。全局语义只编码一次，局部 tile 用其坐标取得语义 ROI。图像级分数汇总为 tile detection logit 的最大值。

当前公开路径有两点值得独立检查。第一，`test_dataset.py` 的 loader 先 resize 输入，因此开启 tiling 后默认操作的是 loader 的输出图，保留的分辨率不等于文件原图。第二，tiling 函数在局部 tile 上调用 `model.transforms.transforms`，再把它作为 forward 的 `orig` 送入 LAD，该输入域和训练的 raw patch 有差别。对发布权重先以 512 resize 作为主复现口径。若研究原图分块，将 raw local 与 normalized global 输入分别控制，并与同一 checkpoint 的 resize 结果比较。

<a id="evaluation"></a>

## 15. 评价协议与论文结果

### 15.1 像素级定位

像素分数经 sigmoid，以严格大于 0.5 判前景。每张图计算

$$
IoU=\frac{TP}{TP+FP+FN},\qquad
F1=\frac{2TP}{2TP+FP+FN}.
$$

源码对纳入的图逐图平均，区别于将全体像素混淆计数累加后再计算。默认排除额外加入的 source 真图，保留 `sample_type=forgery` 的目标，哪怕某个 GT 因裁剪或编码为空。GT 与预测均空时为 1，只有一方空时为 0。

还按 GT 前景面积比例分三档，`<5%`、`[5%,20%)`、`≥20%`。应同时报告各档样本数、无效样本或批次数。总体结果可能因为小样本、空标注或数据划分变化而偏移。

### 15.2 图像级检测有两种协议

| 来源 | 连续分数 | 报告方式 |
|---|---|---|
| 论文附录 D.2 | 最终 mask 概率最大值 | ACC@0.5、AP |
| `validate.py` | CNN 的 `sigmoid(detection_logit)` | ACC、AP、Precision/Recall/F1 与真假分类准确率 |

两者可能对同一张图给出相反决策。论文附录的 mask max 对单个热点敏感，代码的 CNN score 由全局池化特征与显式图像级目标决定。复测时可以同时记录两个分数，用同一真/假清单计算两套结果。AP 需要有正负两类，源码只有一类时报告 0 作为占位。它与 ROC-AUC 是不同指标。

### 15.3 论文表格作为参考坐标

以下是本地论文 Table 1 部分结果。数字是作者报告值。FLAME-F 使用 Qwen-Image-Edit 的适配集，表的最后五列平均包含已变为 ID 的 Qwen 测试列，不能把 FLAME-F 的该均值直接称为五个严格 OOD 域的结果。

| 方法 | MagicBrush IoU/F1 | SID IoU/F1 | CoCoGLIDE IoU/F1 | AutoSplice IoU/F1 | Table 1 最后五列均值 |
|---|---|---|---|---|---|
| TruFor | 0.281 / 0.391 | 0.188 / 0.243 | 0.371 / 0.457 | 0.364 / 0.483 | 0.247 / 0.324 |
| FLAME | 0.538 / 0.650 | 0.580 / 0.677 | 0.469 / 0.576 | 0.501 / 0.624 | 0.358 / 0.459 |
| FLAME-F | 0.507 / 0.632 | 0.569 / 0.650 | 0.481 / 0.602 | 0.498 / 0.618 | 0.460 / 0.565 |

Table 2 中 Base FLAME 的 MagicBrush ACC/AP 为 0.901/0.917，SID 为 0.916/0.922。该表印出的 Base FLAME Average 为 0.639/0.677，但用表内最后五列的公开数值直接做算术平均得到约 0.739/0.777。引用时注明是原表值还是重算值，并保留所用集合。不要以此表的 average 列作为本地 checkpoint 加载正确性的判断依据。

### 15.4 鲁棒性实验

论文 Table 3 报告 JPEG、Gaussian Blur 和 Gaussian Noise 条件下的表现。仓库评价入口提供 `--input_perturbation_type` 与 intensity。令 intensity 为 $a$，代码使用 $\max(10,\operatorname{int}(95-56.67a))$ 作为 JPEG quality，blur sigma 为 $a$，noise std 为 $0.2a$。包装类在原数据集取出样本后扰动张量，因此默认测试路径会先 resize。复测论文时应记录实际质量、核大小、sigma 和噪声标准差。

<a id="differences"></a>

## 16. 论文、源码与发布权重的差异

| 项目 | 当前证据 | 复现时如何处理 |
|---|---|---|
| SAM 代际 | 论文附录涉及 SAM 3，发布 JSON 指定 SAM2 | 下载 Hiera B+ 基座，按 release 配置建模 |
| 取证输入 | 论文式 (5) 单 τ，发布主干首层需要 4 通道 | 保留四个 τ 的顺序 |
| 粗头结构 | 附录提到 1×1，发布首个粗头卷积为 3×3 | 使用发布的 kernel size |
| 适配器初始化 | 前部 Xavier，源码末层零初始化 | 读出第 0 步恒等和梯度路径 |
| 优化目标 | 论文 Focal/IoU 为 20/1，发布为 10/0.25，另有粗 BCE | 分开报告论文与 release 设置 |
| 检测计分 | 论文附录用 mask max，代码默认读分类头 | 报告分数来源和阈值 |
| 单图权重加载 | `test.py` 没有完整传递 release 参数 | 先用数据集入口 |
| 训练入口 | 缺少被顶层 import 的诊断模块 | 确认可运行版本后再做训练 |
| 评估输入 | 普通 resize 回退和训练全局输入归一化不同 | 做对照时使用固定入口与预处理 |
| 分块 | 当前 loader 先 resize，tile 后的 raw 域也不同 | 先建立普通 resize 基线 |
| 断点恢复 | 只有模型权重和有限 epoch/batch 元数据 | 将 resume 当成初始化加部分进度恢复 |
| 数据合成 | 论文描述 EditStream，仓库面向现成图/掩码 | 把数据生成和模型训练作为独立复现实验 |

这些差异分别影响**能否启动、运行的模型是否正确、指标是否可比较**。排查顺序从文件与 CLI 开始，接着看权重形状、数据输入和指标定义。只看可视化结果很难定位问题属于哪一层。

<a id="reproduce"></a>

## 17. 环境与可执行复现路线

### 17.1 准备环境

以下命令从仓库根目录开始。使用符合本机 GPU 的 PyTorch/torchvision 组合，再安装依赖。当前 Meta SAM2 文档要求 Python ≥3.10、PyTorch ≥2.5.1、torchvision ≥0.20.1，严于 FLAME requirements 中给出的旧下界。若 PyPI 环境无法导入项目使用的 SAM2 接口，按 Meta 的安装说明固定其官方实现版本。[SAM2 安装说明](https://github.com/facebookresearch/sam2#installation) · [FLAME 依赖文件](../FLAME/requirements.txt)

~~~bash
cd /home/mckg/FLAME
conda create -n flame python=3.10 -y
conda activate flame
python -m pip install -r FLAME/requirements.txt
python -c 'import torch, torchvision, hydra; from sam2.build_sam import build_sam2; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
~~~

推理 CLI 可选 device，公开构造调用 `build_sam2(...)` 时没有传入该 device。在 CPU-only 环境需先核对安装版本中 `build_sam2` 的默认设备，不宜只依赖外层 `--device cpu`。以下以 CUDA 为例。

### 17.2 下载并检查两套权重

需要三个文件，分别是 SAM2.1 Hiera B+ 基座、FLAME 权重、配套 JSON。基座默认位置在内层仓库的 `sam2configs`。发布文件的 SHA256 可用来排除下载损坏、重定向网页或错误版本。[FLAME 发布页](https://github.com/phoenixnir/FLAME/releases/tag/v0.1.0) · [SAM2.1 模型列表](https://github.com/facebookresearch/sam2#model-description)

~~~bash
curl -fL https://github.com/phoenixnir/FLAME/releases/download/v0.1.0/flame_g2_ladmulti_sam2.pth \
  -o FLAME/checkpoints/flame_g2_ladmulti_sam2.pth
curl -fL https://github.com/phoenixnir/FLAME/releases/download/v0.1.0/model_params.json \
  -o FLAME/checkpoints/model_params.json
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
  -o FLAME/sam2configs/sam2.1_hiera_base_plus.pt
sha256sum FLAME/checkpoints/flame_g2_ladmulti_sam2.pth FLAME/checkpoints/model_params.json
~~~

预期 FLAME 权重 SHA256 为 `9b6c67e3c35f647a3c9207275b1f1406c36780a2a3e0ee6b0f696d82161991bb`，JSON 为 `e46391983ca732313d1dc306e14ef3799c67a5bdf899a54a5fac8fef7e87ed42`。

### 17.3 框架外先核验 LAD-Net

以下片段只需要 PyTorch，从仓库根目录执行。用发布结构的参数构造随机权重主干，检查张量接口与输入域。输出只用于形状检查。

~~~bash
PYTHONPATH=FLAME python - <<'PY'
import torch
from model.ferret_backbone import FerretBackbone

torch.manual_seed(0)
torch.set_num_threads(2)
net = FerretBackbone(
    forensic_operator="lad_multi",
    lad_multi_taus=[0.016, 0.032, 0.064, 0.128],
    mask_compressor_kernel_size=3,
).eval()
with torch.inference_mode():
    x = torch.rand(1, 3, 512, 512)
    features, coarse, det, prompt = net(x, return_dense_prompt=True)
    assert features.shape == (1, 384, 64, 64)
    assert coarse.shape == prompt.shape == (1, 1, 256, 256)
    assert det.shape == (1, 1)
    assert all(torch.isfinite(t).all() for t in (features, coarse, det, prompt))
    assert torch.equal(coarse, prompt)
print("Ferret shapes:", features.shape, coarse.shape, det.shape)
PY
~~~

完整仓库主干测试可在内层目录用 `python -m pytest -q tests/test_ferret_backbone_lad.py`。需安装 pytest。该测试文件还覆盖部分非发布提示头。

### 17.4 跑仓库 MagicBrush 小样本

建议以普通 resize 启动数据集入口，本地 demo 的三个 target 与三个 source 可用于同时观察定位和检测。

~~~bash
python FLAME/test_dataset.py \
  --dataset FLAME/demo_test_set/magicbrush \
  --model FLAME/checkpoints/flame_g2_ladmulti_sam2.pth \
  --config FLAME/checkpoints/model_params.json \
  --output FLAME/outputs/demo_release_512 \
  --device cuda --img_size 512 --batch_size 1 --num_workers 0 \
  --authentic_ratio 1.0 --save_all_vis
~~~

可检查 `evaluation_results.json`、可视化文件以及终端加载日志。预计定位池为三张 target，检测池还会包含三张 source。若有效数量不同，先看读取报错和缺失字段。该目录的指标只能作为样例运行记录。

### 17.5 复测一个完整测试集

将上例中的 `--dataset` 换为整理好的测试目录，把输出放到独立 run 目录。确认输入和输出尺寸、GT 版本、真图来源、分数类型、阈值、权重 hash 一致。检测和定位各留一套样本列表及成功计数，避免跨数据集比较时遗漏错误跳过的样本。

建议首个完整实验以 MagicBrush 或 SID 的独立测试集为对象。完成同域基线后再对 CoCoGLIDE 或 AutoSplice 做跨域测试，确认每个域上的编辑模型与训练来源。

<a id="troubleshooting"></a>

## 18. 入口排查与训练准备

### 18.1 先处理训练入口缺失依赖

当前提交的 `train.py` 在顶层执行

~~~python
from utils.live_reload_diagnostics import compare_models_on_batches
~~~

工作区和当前提交缺该文件。即使关闭诊断，解释器仍会先执行顶层 import。装齐依赖后，`python FLAME/train.py --help` 是第一项启动检查。确认上游有配套模块时使用同一版本。自行补兼容时，可将 import 移到实际使用该功能的 `run_live_reload_diagnostic()` 路径，保留启用时的明确错误，然后以关闭诊断的配置验证。本文记录这个阻塞和排查建议。

### 18.2 用独立数据做小规模训练检查

复制 `datasets_default.json` 为自己的配置，设置绝对 `data_root`。每个启用条目准备相应 `target/mask/source`，训练与验证按原始图 ID 独立切分，避免同图多次编辑跨集合。训练和验证的 `is_training` 分别为 true/false。首次 smoke 可用 `num_workers=0`，训练 `drop_last` 与样本数相容。

提前统计前景像素数、读图有效率、真实数量和每轮 batch 数。使用全局 0.2 时，确保验证集的真实样本抽样计算不会遇到第 11.4 节的 0 分母。过小的数据可为验证条目显式指定 `authentic_ratio: 1.0` 来使用其全部 source，或者选择只测定位的配置并核实全局覆写逻辑。

### 18.3 与发布结构对齐的训练启动示例

在缺失模块问题处理完、`--help` 可用并准备好配置后，下面的命令做一轮训练检查。`datasets_local.json` 是自己整理的文件。验证间隔设为 1 是为了确认小集能触发验证与保存，正式训练应按本地验证开销调整。

~~~bash
python FLAME/train.py \
  --dataset_config FLAME/configs/datasets_local.json \
  --sam_config sam2.1_hiera_b+.yaml \
  --sam_ckpt FLAME/sam2configs/sam2.1_hiera_base_plus.pt \
  --sam_backend sam2 --img_size 512 --train_force_resize --val_force_resize \
  --forensic_operator lad_multi --lad_multi_taus 0.016,0.032,0.064,0.128 \
  --coarse_prompt_head mask_compressor --train_sam_iou \
  --lr 0.00012 --scheduler_type none --ema_decay 0.975 \
  --lambda_focal 10 --focal_alpha 0.55 --focal_gamma 1 --lambda_iou 0.25 \
  --coarse_loss_weight 0.2 --authentic_ratio 0.2 \
  --batch_size 1 --epochs 1 --val_steps 1 --checkpoint_save_mode trainable_only
~~~

Hydra 由 utils 初始化配置搜索目录，`--sam_config` 传文件名让它在 `sam2configs` 找 YAML，`--sam_ckpt` 则传从当前目录可访问的实际权重路径。batch/epoch/val_steps 是检查用设置。只用小集训练一轮后，先检查总 loss 有限、粗图和最终图分支的梯度、真实检测标签、验证是否进入以及保存的 checkpoint 是否可重载。

微调时可把诊断通过的权重作为 `--resume_checkpoint`，同时核对起始 epoch、目标 epoch 和 best score。若恢复权重的 epoch 已经达到 `--epochs`，主循环没有可执行轮次。根据是否更换了验证数据，决定是否重新选择 best score。

### 18.4 常见故障定位表

| 症状 | 先核对 |
|---|---|
| `No module named 'model'` | 测试从内层目录启动或设置 PYTHONPATH |
| `No module named 'utils.live_reload_diagnostics'` | 当前提交缺该训练诊断依赖 |
| Hydra 找不到配置 | SAM backend、配置搜索路径、YAML 名称 |
| `cbr1.0.weight` 4 对 1 通道 | checkpoint 与发布配置是否一起加载 |
| 推理可运行但适配器权重被跳过 | adapter_type、活动层、stem 宽度与加载报告 |
| 假图标签大量变真 | mask 0/1 存盘、RGBA 方向、crop 后前景 |
| 定位指标近 0 且真图误报高 | 输入域、尺寸、logits 是否重复 sigmoid、两个阈值 |
| 整图 AP 为 0 | 是否只载入了一个类别，source 是否存在 |
| 小数据没有 best_model.pth | 每轮步数、val_steps、是否实际进入验证及是否改善 |
| 分块没有原图细节收益 | 进入 tiling 前 loader 是否已 resize |

<a id="study"></a>

## 19. 精读、手算练习与消融

### 19.1 建议精读路线

| 顺序 | 位置 | 读完应能回答 |
|---|---|---|
| 1 | 论文 §3 与 `LADOperator` | RGB 差异在哪个轴求和，为什么先 tanh 再平均 |
| 2 | `FerretBackbone.forward` | 四通道如何经过 stride 走到 384 通道 |
| 3 | `ForgeryLocalizer.encode_global` | 模型何时使用 ROI 与语义特征缓存 |
| 4 | `SharedAdapter.forward` | 高层和取证通道如何拼接，零初始化会改变哪条梯度 |
| 5 | `_decode_with_dense_prompt` | 粗图为什么从 256 又变成 128 |
| 6 | `compute_loss` | 什么样的图参与分割监督，什么样的图只参与分类 |
| 7 | `validate` | IoU/F1 和 ACC/AP 分别用什么图、什么分数 |
| 8 | `test_dataset.load_and_initialize_model` | 哪些 JSON 字段必须传进构造器 |

### 19.2 五个理解练习

1. **边界手算。** 对常量非零 RGB 图，求 3×3 LAD 的内部、行边界和角点响应。预测分别接近 0、3/8 和 5/8，再用随机小图验证 unfold 的中心位置。
2. **温度曲线。** 固定 RGB 差值，画出或计算四个 τ 的响应。解释在同一位置保留四个通道和直接平均四个通道有何区别。
3. **输出坐标。** 对 512 输入，逐层写出卷积 stride、粗 logits、SAM 主嵌入和 mask prompt 的尺寸。设模型输出尺寸变为 256，推导哪几项会改变。
4. **梯度切割。** 构造一真一假的 batch。只对定位损失反向时，说明哪些样本的最终 logits、粗 logits 和 SAM IoU 分数接收梯度。
5. **分数来源。** 找一张存在孤立 mask 热点但分类分数低的图，对比 mask max 与 detection head 的结果，同时报告其中一个误报的原因。

### 19.3 消融如何做到可解释

| 研究问题 | 对照 | 需要固定的条件 |
|---|---|---|
| 低层统计是否有贡献 | RGB-only、单 τ、多 τ | 相同数据、分辨率、训练预算和重训规则 |
| SAM 是否改善边界 | 粗定位 vs 最终定位 | 对齐到同一 GT 网格，使用同一阈值 |
| 适配器是否改善泛化 | 完整 adapter vs 改为恒等 | 固定 SAM 基座、提示与数据 |
| 哪个检测分数更稳健 | CNN head vs mask max | 相同真/假清单，计算 ACC 和 AP |
| 新生成器适应是否会遗忘 | Base vs 微调，with/without replay | 固定新旧评估域和微调集 |
| 局部能量是否经得起压缩 | clean vs JPEG 等强度梯度 | 同一图像和 GT，记录实际扰动参数 |

论文 Table 4 在其消融设置中报告完整 FLAME 0.567/0.662 IoU/F1，无 LAD、无 adapter、无 SAM 的结果均较低。自训消融可先从同一 checkpoint 上对比 coarse 与 final，验证流程后再控制训练条件比较模块贡献。

<a id="verification"></a>

## 20. 核验范围与资料索引

### 20.1 本次实际核验

| 核验项 | 结果及范围 |
|---|---|
| 论文 | 本地 27 页 PDF 成功抽取文本，并核对方法、附录与表格。MinerU 对公开稿的抽取服务返回暂时不可用，改用 pypdf |
| 发布 JSON | 从 v0.1.0 下载，SHA256 与发布页一致 |
| 现有测试 | 从内层目录执行 `tests/test_ferret_backbone_lad.py`，44 项通过 |
| 512 形状 | 用发布结构的随机权重取证主干，输出 `[1,384,64,64]`、`[1,1,256,256]`、`[1,1]`，粗图与提示相等 |
| 参数量 | 实际构造的发布结构 Ferret 为 1,557,026，三尺度 SharedAdapter 为 131,808。数字不包含 SAM 及其 IoU head |
| LAD 边界 | 5×5 常量图的内区为 0，角点为 5/8 |
| 首步 adapter | 零出口使前向严格恒等。第一次反向出口有梯度，上游 fusion 卷积梯度为零 |
| 权重冲突 | 用对应结构的 state dict 验证四通道 stem 加载到单通道主干时，`strict=False` 仍产生 shape 错误 |
| 损失梯度 | 提取源码函数，使用一真一假合成 batch。只在假样本位置得到分割、粗图和 IoU 梯度，两个样本都得到检测梯度 |
| 训练依赖 | 检查源码 import 和文件列表，确认诊断模块在当前提交缺失 |

本次在已有 PyTorch 2.11.0+cu128 环境中运行了算子、主干与损失的有限检查。完整 SAM 权重推理、真实数据集评价、训练和论文指标复现仍需按第 17～18 节执行。

### 20.2 参考资料

- 本地论文，含主文、超参数搜索和实验附录。[本地 PDF](../docs/FICML_2026_Order%20within%20Chaos.pdf)
- 论文正式出版页与公开预印本。[PMLR 306](https://proceedings.mlr.press/v306/wang26iy.html) · [arXiv:2606.02178](https://arxiv.org/abs/2606.02178)
- 发布权重及配置校验值。[FLAME v0.1.0](https://github.com/phoenixnir/FLAME/releases/tag/v0.1.0)
- SAM2 的安装与 Hiera 系列权重。[Meta SAM2](https://github.com/facebookresearch/sam2)
- 本地项目入口说明与公示文件。[README](../README.md) · [NOTICE](../NOTICE)

### 20.3 源码索引

| 内容 | 源码 |
|---|---|
| LAD、取证 CNN、粗提示头 | [ferret_backbone.py](../FLAME/model/ferret_backbone.py) |
| SAM 接入、ROI、密集提示、解码 | [forgerylocalizer.py](../FLAME/model/forgerylocalizer.py) |
| 尺度适配、残差诊断与正则 | [adapters.py](../FLAME/model/adapters.py) |
| 数据、配置、collate | [localforgerydataset.py](../FLAME/utils/localforgerydataset.py) · [dataset_config.py](../FLAME/utils/dataset_config.py) · [train_utils.py](../FLAME/utils/train_utils.py) |
| 训练和主损失 | [train.py](../FLAME/train.py) · [metrics.py](../FLAME/utils/metrics.py) |
| 评价和 tile | [validate.py](../FLAME/utils/validate.py) · [tiling.py](../FLAME/utils/tiling.py) |
| 权重与 EMA | [checkpoint_state.py](../FLAME/utils/checkpoint_state.py) · [ema.py](../FLAME/model/ema.py) |
| 单图与配置驱动评价入口 | [test.py](../FLAME/test.py) · [test_dataset.py](../FLAME/test_dataset.py) |
| 主干测试 | [test_ferret_backbone_lad.py](../FLAME/tests/test_ferret_backbone_lad.py) |
