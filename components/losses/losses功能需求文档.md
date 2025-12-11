# Losses 模块功能需求文档

## 模块概述

Losses 模块是系统的损失函数组件库，提供分类、检测、分割、蒸馏等任务的损失函数实现。支持损失权重配置、样本加权、难样本挖掘等功能，通过注册器管理实现配置驱动的损失函数构建与组合。

---

## 目录结构

```
losses/
├── __init__.py              # 模块初始化
├── builder.py               # 损失函数构建器
├── base_loss.py             # 损失函数基类
│
├── classification/          # 分类损失
│   ├── __init__.py
│   ├── cross_entropy.py     # 交叉熵损失
│   ├── focal_loss.py        # Focal Loss
│   ├── label_smooth.py      # 标签平滑
│   └── asymmetric_loss.py   # 非对称损失
│
├── detection/               # 检测损失
│   ├── __init__.py
│   ├── iou_loss.py          # IoU 系列损失
│   ├── smooth_l1.py         # Smooth L1
│   ├── gfocal_loss.py       # Generalized Focal Loss
│   └── varifocal_loss.py    # VariFocal Loss
│
├── segmentation/            # 分割损失
│   ├── __init__.py
│   ├── dice_loss.py         # Dice 损失
│   ├── lovasz_loss.py       # Lovasz 损失
│   ├── boundary_loss.py     # 边界损失
│   └── ohem_loss.py         # OHEM 损失
│
├── distillation/            # 蒸馏损失
│   ├── __init__.py
│   ├── kd_loss.py           # KD 损失
│   ├── feature_loss.py      # 特征损失
│   └── relation_loss.py     # 关系损失
│
└── utils/                   # 工具函数
    ├── __init__.py
    ├── loss_utils.py        # 损失工具
    └── weight_reduce.py     # 加权归约
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_loss(config)` - 损失函数构建
* `LOSSES` - 损失函数注册器
* 所有损失函数类

---

### 2. `builder.py`

**功能**：损失函数构建器

**核心职责**：

* 根据配置构建单个损失函数
* 构建组合损失函数
* 处理损失权重

**构建函数**：

| 函数                          | 功能      |
| --------------------------- | ------- |
| `build_loss(config)`        | 构建单个损失  |
| `build_multi_loss(configs)` | 构建多损失组合 |

**组合损失配置**：

```yaml
loss:
  - type: CrossEntropyLoss
    weight: 1.0
  - type: DiceLoss
    weight: 0.5
```

---

### 3. `base_loss.py`

**功能**：所有损失函数的抽象基类

**核心职责**：

* 定义损失函数统一接口
* 损失权重管理
* 样本加权处理
* 归约方式控制

**基类接口**：

| 方法                                | 功能           |
| --------------------------------- | ------------ |
| `forward(pred, target, **kwargs)` | 计算损失         |
| `_forward(pred, target)`          | 实际损失计算（子类实现） |

**通用配置参数**：

| 参数            | 说明                     | 默认值    |
| ------------- | ---------------------- | ------ |
| `loss_weight` | 损失权重                   | 1.0    |
| `reduction`   | 归约方式：none / mean / sum | 'mean' |
| `avg_factor`  | 平均因子                   | None   |
| `loss_name`   | 损失名称（用于日志）             | None   |

**样本加权**：

| 参数             | 说明      |
| -------------- | ------- |
| `weight`       | 每个样本的权重 |
| `class_weight` | 每个类别的权重 |
| `ignore_index` | 忽略的标签值  |

---

## 二、classification/ 分类损失

### 1. `classification/__init__.py`

**功能**：导出分类损失函数

---

### 2. `classification/cross_entropy.py`

**功能**：交叉熵损失及变体

**损失类型**：

| 类                        | 说明     |
| ------------------------ | ------ |
| `CrossEntropyLoss`       | 标准交叉熵  |
| `BinaryCrossEntropyLoss` | 二分类交叉熵 |
| `SoftCrossEntropyLoss`   | 软标签交叉熵 |

**CrossEntropyLoss 配置参数**：

| 参数                | 说明     | 默认值  |
| ----------------- | ------ | ---- |
| `class_weight`    | 类别权重   | None |
| `ignore_index`    | 忽略标签   | -100 |
| `label_smoothing` | 标签平滑系数 | 0.0  |

**BinaryCrossEntropyLoss 配置参数**：

| 参数            | 说明           | 默认值  |
| ------------- | ------------ | ---- |
| `pos_weight`  | 正样本权重        | None |
| `with_logits` | 输入是否为 logits | True |

**SoftCrossEntropyLoss**：

* 支持软标签（概率分布）
* 用于知识蒸馏、MixUp 等

---

### 3. `classification/focal_loss.py`

**功能**：Focal Loss 系列

**损失类型**：

| 类                     | 说明             |
| --------------------- | -------------- |
| `FocalLoss`           | 标准 Focal Loss  |
| `BinaryFocalLoss`     | 二分类 Focal Loss |
| `MultiLabelFocalLoss` | 多标签 Focal Loss |

**FocalLoss 配置参数**：

| 参数             | 说明     | 默认值  |
| -------------- | ------ | ---- |
| `alpha`        | 类别平衡因子 | 0.25 |
| `gamma`        | 聚焦因子   | 2.0  |
| `class_weight` | 类别权重   | None |

**损失公式**：

```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

**适用场景**：

* 类别不平衡
* 难样本挖掘

---

### 4. `classification/label_smooth.py`

**功能**：标签平滑损失

**损失类型**：

| 类                 | 说明     |
| ----------------- | ------ |
| `LabelSmoothLoss` | 标签平滑损失 |

**配置参数**：

| 参数            | 说明   | 默认值        |
| ------------- | ---- | ---------- |
| `smoothing`   | 平滑系数 | 0.1        |
| `num_classes` | 类别数  | None（自动推断） |

**标签平滑公式**：

```
y_smooth = (1 - smoothing) * y_onehot + smoothing / num_classes
```

**适用场景**：

* 防止过拟合
* 提高泛化能力

---

### 5. `classification/asymmetric_loss.py`

**功能**：非对称损失（多标签分类）

**损失类型**：

| 类                         | 说明    |
| ------------------------- | ----- |
| `AsymmetricLoss`          | 非对称损失 |
| `AsymmetricLossOptimized` | 优化版本  |

**配置参数**：

| 参数                         | 说明      | 默认值  |
| -------------------------- | ------- | ---- |
| `gamma_neg`                | 负样本聚焦因子 | 4    |
| `gamma_pos`                | 正样本聚焦因子 | 1    |
| `clip`                     | 概率裁剪阈值  | 0.05 |
| `disable_torch_grad_focal` | 禁用梯度优化  | True |

**适用场景**：

* 多标签分类
* 类别严重不平衡

---

## 三、detection/ 检测损失

### 1. `detection/__init__.py`

**功能**：导出检测损失函数

---

### 2. `detection/iou_loss.py`

**功能**：IoU 系列回归损失

**损失类型**：

| 类          | 说明              |
| ---------- | --------------- |
| `IoULoss`  | 标准 IoU 损失       |
| `GIoULoss` | Generalized IoU |
| `DIoULoss` | Distance IoU    |
| `CIoULoss` | Complete IoU    |
| `EIoULoss` | Efficient IoU   |
| `SIoULoss` | Scylla IoU      |

**通用配置参数**：

| 参数            | 说明                      | 默认值    |
| ------------- | ----------------------- | ------ |
| `eps`         | 数值稳定性                   | 1e-6   |
| `mode`        | 计算模式：iou / linear / log | 'log'  |
| `bbox_format` | 边框格式：xyxy / xywh        | 'xyxy' |

**损失公式**：

| 类型   | 公式                          |
| ---- | --------------------------- |
| IoU  | `1 - IoU`                   |
| GIoU | `1 - IoU + (C - Union) / C` |
| DIoU | `1 - IoU + d² / c²`         |
| CIoU | `1 - IoU + d² / c² + αv`    |

**适用场景**：

* 目标检测边框回归
* 推荐使用 CIoU 或 SIoU

---

### 3. `detection/smooth_l1.py`

**功能**：Smooth L1 损失及变体

**损失类型**：

| 类                | 说明           |
| ---------------- | ------------ |
| `SmoothL1Loss`   | 标准 Smooth L1 |
| `L1Loss`         | L1 损失        |
| `BalancedL1Loss` | 平衡 L1 损失     |

**SmoothL1Loss 配置参数**：

| 参数     | 说明   | 默认值 |
| ------ | ---- | --- |
| `beta` | 平滑阈值 | 1.0 |

**损失公式**：

```
L = 0.5 * x² / beta    if |x| < beta
L = |x| - 0.5 * beta   otherwise
```

**BalancedL1Loss 配置参数**：

| 参数      | 说明   | 默认值 |
| ------- | ---- | --- |
| `alpha` | 平衡因子 | 0.5 |
| `gamma` | 调制因子 | 1.5 |
| `beta`  | 平滑阈值 | 1.0 |

---

### 4. `detection/gfocal_loss.py`

**功能**：Generalized Focal Loss

**损失类型**：

| 类                       | 说明            |
| ----------------------- | ------------- |
| `QualityFocalLoss`      | 质量 Focal Loss |
| `DistributionFocalLoss` | 分布 Focal Loss |

**QualityFocalLoss 配置参数**：

| 参数            | 说明         | 默认值  |
| ------------- | ---------- | ---- |
| `beta`        | 聚焦因子       | 2.0  |
| `use_sigmoid` | 使用 sigmoid | True |

**特点**：

* 联合学习分类和定位质量
* 软标签形式

**DistributionFocalLoss 配置参数**：

| 参数        | 说明     | 默认值 |
| --------- | ------ | --- |
| `reg_max` | 最大回归范围 | 16  |

**特点**：

* 边框预测为离散分布
* 用于 GFocal / FCOS 等

---

### 5. `detection/varifocal_loss.py`

**功能**：VariFocal Loss

**损失类型**：

| 类               | 说明           |
| --------------- | ------------ |
| `VariFocalLoss` | VariFocal 损失 |

**配置参数**：

| 参数            | 说明         | 默认值  |
| ------------- | ---------- | ---- |
| `alpha`       | 正样本权重      | 0.75 |
| `gamma`       | 聚焦因子       | 2.0  |
| `use_sigmoid` | 使用 sigmoid | True |

**特点**：

* IACS（IoU-Aware Classification Score）
* 非对称处理正负样本

---

## 四、segmentation/ 分割损失

### 1. `segmentation/__init__.py`

**功能**：导出分割损失函数

---

### 2. `segmentation/dice_loss.py`

**功能**：Dice 系列损失

**损失类型**：

| 类                     | 说明         |
| --------------------- | ---------- |
| `DiceLoss`            | 标准 Dice 损失 |
| `BinaryDiceLoss`      | 二分类 Dice   |
| `GeneralizedDiceLoss` | 广义 Dice    |

**DiceLoss 配置参数**：

| 参数             | 说明       | 默认值   |
| -------------- | -------- | ----- |
| `smooth`       | 平滑因子     | 1.0   |
| `exponent`     | 指数       | 2     |
| `ignore_index` | 忽略标签     | 255   |
| `naive_dice`   | 是否使用朴素计算 | False |

**损失公式**：

```
Dice = 2 * |P ∩ G| / (|P| + |G|)
Loss = 1 - Dice
```

**GeneralizedDiceLoss**：

* 类别权重与面积成反比
* 更好处理小目标

---

### 3. `segmentation/lovasz_loss.py`

**功能**：Lovasz 损失

**损失类型**：

| 类                 | 说明                   |
| ----------------- | -------------------- |
| `LovaszLoss`      | Lovasz-Softmax 损失    |
| `LovaszHingeLoss` | Lovasz-Hinge 损失（二分类） |

**LovaszLoss 配置参数**：

| 参数             | 说明                 | 默认值       |
| -------------- | ------------------ | --------- |
| `per_image`    | 是否按图像计算            | False     |
| `ignore_index` | 忽略标签               | 255       |
| `classes`      | 类别模式：all / present | 'present' |

**特点**：

* 直接优化 IoU
* 可微分的 IoU 代理

---

### 4. `segmentation/boundary_loss.py`

**功能**：边界损失

**损失类型**：

| 类              | 说明             |
| -------------- | -------------- |
| `BoundaryLoss` | 边界损失           |
| `HDLoss`       | Hausdorff 距离损失 |

**BoundaryLoss 配置参数**：

| 参数       | 说明   | 默认值 |
| -------- | ---- | --- |
| `theta0` | 边界宽度 | 3   |
| `theta`  | 衰减宽度 | 5   |

**HDLoss 配置参数**：

| 参数      | 说明   | 默认值 |
| ------- | ---- | --- |
| `alpha` | 距离权重 | 2.0 |

**适用场景**：

* 边界精细化
* 医学图像分割

---

### 5. `segmentation/ohem_loss.py`

**功能**：在线难样本挖掘损失

**损失类型**：

| 类                      | 说明       |
| ---------------------- | -------- |
| `OHEMCrossEntropyLoss` | OHEM 交叉熵 |
| `OHEMBinaryLoss`       | OHEM 二分类 |

**OHEMCrossEntropyLoss 配置参数**：

| 参数             | 说明      | 默认值    |
| -------------- | ------- | ------ |
| `thresh`       | 难样本阈值   | 0.7    |
| `min_kept`     | 最小保留像素数 | 100000 |
| `ignore_index` | 忽略标签    | 255    |

**挖掘策略**：

1. 计算所有像素损失
2. 按损失排序
3. 保留损失高于阈值或前 K 个

---

## 五、distillation/ 蒸馏损失

### 1. `distillation/__init__.py`

**功能**：导出蒸馏损失函数

---

### 2. `distillation/kd_loss.py`

**功能**：知识蒸馏 Logit 损失

**损失类型**：

| 类                           | 说明       |
| --------------------------- | -------- |
| `KnowledgeDistillationLoss` | 标准 KD 损失 |
| `DKDLoss`                   | 解耦 KD 损失 |
| `DISTLoss`                  | DIST 损失  |

**KnowledgeDistillationLoss 配置参数**：

| 参数            | 说明   | 默认值         |
| ------------- | ---- | ----------- |
| `temperature` | 蒸馏温度 | 4.0         |
| `reduction`   | 归约方式 | 'batchmean' |

**损失公式**：

```
L_KD = T² * KL(softmax(z_s/T) || softmax(z_t/T))
```

**DKDLoss 配置参数**：

| 参数            | 说明     | 默认值 |
| ------------- | ------ | --- |
| `temperature` | 蒸馏温度   | 4.0 |
| `alpha`       | 目标类权重  | 1.0 |
| `beta`        | 非目标类权重 | 8.0 |

---

### 3. `distillation/feature_loss.py`

**功能**：特征蒸馏损失

**损失类型**：

| 类                   | 说明                             |
| ------------------- | ------------------------------ |
| `FeatureMSELoss`    | MSE 特征损失                       |
| `FeatureL1Loss`     | L1 特征损失                        |
| `FeatureCosineLoss` | 余弦特征损失                         |
| `AttentionLoss`     | 注意力损失                          |
| `MGDLoss`           | Masked Generative Distillation |

**FeatureMSELoss 配置参数**：

| 参数                 | 说明    | 默认值   |
| ------------------ | ----- | ----- |
| `normalize`        | 是否归一化 | False |
| `student_channels` | 学生通道数 | None  |
| `teacher_channels` | 教师通道数 | None  |

**AttentionLoss 配置参数**：

| 参数          | 说明    | 默认值  |
| ----------- | ----- | ---- |
| `p`         | 范数类型  | 2    |
| `normalize` | 是否归一化 | True |

**MGDLoss 配置参数**：

| 参数           | 说明   | 默认值     |
| ------------ | ---- | ------- |
| `alpha_mgd`  | 损失权重 | 0.00007 |
| `lambda_mgd` | 掩码比例 | 0.65    |

---

### 4. `distillation/relation_loss.py`

**功能**：关系蒸馏损失

**损失类型**：

| 类                 | 说明       |
| ----------------- | -------- |
| `RKDDistanceLoss` | RKD 距离损失 |
| `RKDAngleLoss`    | RKD 角度损失 |
| `CRDLoss`         | 对比表示蒸馏损失 |
| `PKTLoss`         | 概率知识迁移损失 |

**RKDDistanceLoss 配置参数**：

| 参数        | 说明     | 默认值   |
| --------- | ------ | ----- |
| `eps`     | 数值稳定性  | 1e-12 |
| `squared` | 是否平方距离 | False |

**RKDAngleLoss 配置参数**：

| 参数    | 说明    | 默认值   |
| ----- | ----- | ----- |
| `eps` | 数值稳定性 | 1e-12 |

**CRDLoss 配置参数**：

| 参数            | 说明   | 默认值   |
| ------------- | ---- | ----- |
| `temperature` | 温度   | 0.07  |
| `neg_num`     | 负样本数 | 16384 |

---

## 六、utils/ 工具函数

### 1. `utils/__init__.py`

**功能**：导出工具函数

---

### 2. `utils/loss_utils.py`

**功能**：损失计算工具函数

**工具函数**：

| 函数                                               | 功能                 |
| ------------------------------------------------ | ------------------ |
| `convert_to_one_hot(labels, num_classes)`        | 转换为 one-hot        |
| `get_class_weight(labels, method)`               | 计算类别权重             |
| `focal_weight(p, gamma)`                         | 计算 focal 权重        |
| `sigmoid_focal_loss(pred, target, alpha, gamma)` | sigmoid focal loss |
| `compute_iou(box1, box2)`                        | 计算 IoU             |
| `compute_giou(box1, box2)`                       | 计算 GIoU            |

**类别权重计算方法**：

| 方法              | 说明      |
| --------------- | ------- |
| `inverse`       | 频率倒数    |
| `sqrt_inverse`  | 频率倒数平方根 |
| `effective_num` | 有效样本数   |
| `median_freq`   | 中值频率平衡  |

---

### 3. `utils/weight_reduce.py`

**功能**：加权归约函数

**归约函数**：

| 函数                                                        | 功能         |
| --------------------------------------------------------- | ---------- |
| `weight_reduce_loss(loss, weight, reduction, avg_factor)` | 通用加权归约     |
| `reduce_loss(loss, reduction)`                            | 简单归约       |
| `weighted_loss(loss_func)`                                | 装饰器：添加加权功能 |

**归约方式**：

| 方式     | 说明          |
| ------ | ----------- |
| `none` | 不归约，返回每元素损失 |
| `mean` | 平均          |
| `sum`  | 求和          |

**avg_factor 说明**：

* 自定义平均分母
* 用于正样本平均等场景

---

## 配置示例

### 分类损失

```yaml
loss:
  type: CrossEntropyLoss
  loss_weight: 1.0
  label_smoothing: 0.1
  class_weight: [1.0, 2.0, 1.5]  # 可选
```

### Focal Loss

```yaml
loss:
  type: FocalLoss
  alpha: 0.25
  gamma: 2.0
  loss_weight: 1.0
```

### 检测损失组合

```yaml
loss:
  cls_loss:
    type: QualityFocalLoss
    beta: 2.0
    loss_weight: 1.0
  bbox_loss:
    type: CIoULoss
    loss_weight: 2.0
```

### 分割损失组合

```yaml
loss:
  - type: CrossEntropyLoss
    loss_weight: 1.0
    ignore_index: 255
  - type: DiceLoss
    loss_weight: 1.0
    ignore_index: 255
```

### 蒸馏损失

```yaml
loss:
  task_loss:
    type: CrossEntropyLoss
    loss_weight: 1.0
  kd_loss:
    type: KnowledgeDistillationLoss
    temperature: 4.0
    loss_weight: 0.5
  feature_loss:
    type: FeatureMSELoss
    loss_weight: 0.1
    normalize: true
```

### OHEM 损失

```yaml
loss:
  type: OHEMCrossEntropyLoss
  thresh: 0.7
  min_kept: 100000
  ignore_index: 255
  loss_weight: 1.0
```

---

## 损失函数选择指南

| 任务       | 推荐损失                         | 说明     |
| -------- | ---------------------------- | ------ |
| 图像分类     | CrossEntropy + LabelSmooth   | 标准配置   |
| 多标签分类    | AsymmetricLoss               | 处理不平衡  |
| 目标检测（分类） | FocalLoss / QualityFocalLoss | 难样本挖掘  |
| 目标检测（回归） | CIoULoss / SIoULoss          | IoU 优化 |
| 语义分割     | CE + Dice                    | 组合使用   |
| 实例分割     | CE + Dice + Boundary         | 边界优化   |
| 知识蒸馏     | KDLoss + FeatureLoss         | 多层级蒸馏  |

---

## 依赖关系

**被依赖方**：

* Models（Head）- 模型损失计算
* Training Engine - 训练损失汇总
* Distillation - 蒸馏训练

**依赖项**：

* `Registry` - 组件注册与构建
* `torch.nn` - PyTorch 基础损失
* `torch.nn.functional` - 函数式接口
