# Models 模块功能需求文档

## 模块概述

Models 模块是系统的模型组件库，提供模块化、可组合的深度学习模型构建能力。采用 Backbone-Neck-Head 架构范式，支持分类、检测、分割三大任务，所有组件通过注册器管理，支持配置驱动构建。

---

## 目录结构

```
models/
├── __init__.py
├── builder.py               # 模型构建器
├── base_model.py            # 模型基类
│
├── backbones/               # 骨干网络
├── necks/                   # 特征融合层
├── heads/                   # 任务头
├── detectors/               # 检测器（完整模型）
├── segmentors/              # 分割器（完整模型）
├── classifiers/             # 分类器（完整模型）
├── layers/                  # 基础层组件
└── utils/                   # 模型工具
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_model(config)` - 模型构建入口
* `build_backbone(config)` - 骨干网络构建
* `build_neck(config)` - Neck 构建
* `build_head(config)` - Head 构建
* `BaseModel` - 模型基类
* 所有具体模型类（按需导入）

---

### 2. `builder.py`

**功能**：模型构建器，配置驱动的模型实例化

**核心职责**：

* 根据配置构建完整模型
* 根据配置构建模型子组件
* 处理预训练权重加载
* 模型组件自动组装

**构建函数**：

| 函数                       | 功能                    |
| ------------------------ | --------------------- |
| `build_model(config)`    | 构建完整模型（检测器/分割器/分类器）   |
| `build_backbone(config)` | 构建骨干网络                |
| `build_neck(config)`     | 构建特征融合层               |
| `build_head(config)`     | 构建任务头                 |
| `build_loss(config)`     | 构建损失函数（委托给 losses 模块） |

**预训练加载**：

* 支持 torchvision 预训练权重
* 支持 timm 预训练权重
* 支持本地权重文件
* 支持 URL 下载
* 支持部分加载（strict=False）
* 自动处理权重键名映射

---

### 3. `base_model.py`

**功能**：所有模型的抽象基类

**核心职责**：

* 定义模型通用接口
* 提供前向传播框架
* 统一训练/推理模式切换
* 提供模型信息查询

**基类接口**：

| 方法                                           | 功能              |
| -------------------------------------------- | --------------- |
| `forward(inputs, targets=None, mode='loss')` | 统一前向接口          |
| `forward_train(inputs, targets)`             | 训练前向（返回 loss）   |
| `forward_test(inputs)`                       | 推理前向（返回预测）      |
| `forward_features(inputs)`                   | 特征提取（返回中间特征）    |
| `init_weights()`                             | 权重初始化           |
| `load_pretrained(path)`                      | 加载预训练权重         |
| `get_model_info()`                           | 获取模型信息（参数量、结构等） |

**模式控制**：

* `mode='loss'`：训练模式，返回损失字典
* `mode='predict'`：推理模式，返回预测结果
* `mode='features'`：特征模式，返回特征图

---

## 二、backbones/ 骨干网络

### 1. `backbones/__init__.py`

**功能**：导出所有骨干网络

**导出内容**：

* 所有 Backbone 类
* `BACKBONES` 注册器引用

---

### 2. `backbones/resnet.py`

**功能**：ResNet 系列骨干网络

**支持变体**：

* ResNet-18/34/50/101/152
* ResNetV1c（深层 stem）
* ResNetV1d（平均池化下采样）
* ResNeXt-50/101
* Wide ResNet
* Res2Net

**配置参数**：

| 参数                | 说明                    |
| ----------------- | --------------------- |
| `depth`           | 网络深度：18/34/50/101/152 |
| `num_stages`      | 使用的 stage 数量（1-4）     |
| `out_indices`     | 输出特征层索引               |
| `frozen_stages`   | 冻结的 stage 数量          |
| `norm_cfg`        | 归一化层配置                |
| `style`           | 风格：pytorch / caffe    |
| `deep_stem`       | 是否使用深层 stem           |
| `avg_down`        | 是否使用平均池化下采样           |
| `base_channels`   | 基础通道数                 |
| `groups`          | 分组卷积组数（ResNeXt）       |
| `width_per_group` | 每组宽度                  |

**输出特征**：

* 多尺度特征图列表
* 各层输出通道数可查询

---

### 3. `backbones/efficientnet.py`

**功能**：EfficientNet 系列骨干网络

**支持变体**：

* EfficientNet-B0 ~ B7
* EfficientNetV2-S/M/L

**配置参数**：

| 参数               | 说明                    |
| ---------------- | --------------------- |
| `arch`           | 架构变体：b0-b7 / v2_s/m/l |
| `out_indices`    | 输出特征层索引               |
| `frozen_stages`  | 冻结的 stage 数量          |
| `drop_path_rate` | DropPath 概率           |
| `norm_cfg`       | 归一化层配置                |

**特性**：

* 复合缩放（深度/宽度/分辨率）
* SE 注意力模块
* MBConv / Fused-MBConv 块

---

### 4. `backbones/vit.py`

**功能**：Vision Transformer 骨干网络

**支持变体**：

* ViT-Tiny/Small/Base/Large/Huge
* DeiT 变体

**配置参数**：

| 参数                 | 说明                            |
| ------------------ | ----------------------------- |
| `arch`             | 架构：tiny/small/base/large/huge |
| `img_size`         | 输入图像尺寸                        |
| `patch_size`       | Patch 尺寸                      |
| `embed_dim`        | 嵌入维度                          |
| `depth`            | Transformer 层数                |
| `num_heads`        | 注意力头数                         |
| `mlp_ratio`        | MLP 扩展比例                      |
| `out_indices`      | 输出层索引                         |
| `drop_rate`        | Dropout 概率                    |
| `drop_path_rate`   | DropPath 概率                   |
| `with_cls_token`   | 是否使用 CLS token                |
| `output_cls_token` | 是否输出 CLS token                |
| `interpolate_mode` | 位置编码插值模式                      |

**特性**：

* 可变输入分辨率（位置编码插值）
* 多尺度特征输出支持
* 支持特征金字塔提取

---

### 5. `backbones/swin_transformer.py`

**功能**：Swin Transformer 骨干网络

**支持变体**：

* Swin-Tiny/Small/Base/Large
* SwinV2 变体

**配置参数**：

| 参数               | 说明                       |
| ---------------- | ------------------------ |
| `arch`           | 架构：tiny/small/base/large |
| `img_size`       | 输入图像尺寸                   |
| `patch_size`     | Patch 尺寸                 |
| `embed_dim`      | 嵌入维度                     |
| `depths`         | 各 stage 层数               |
| `num_heads`      | 各 stage 注意力头数            |
| `window_size`    | 窗口大小                     |
| `mlp_ratio`      | MLP 扩展比例                 |
| `out_indices`    | 输出 stage 索引              |
| `drop_path_rate` | DropPath 概率              |
| `use_checkpoint` | 是否使用梯度检查点                |

**特性**：

* 分层特征图（类 CNN）
* 滑动窗口注意力
* 原生多尺度输出

---

### 6. `backbones/convnext.py`

**功能**：ConvNeXt 骨干网络

**支持变体**：

* ConvNeXt-Tiny/Small/Base/Large/XLarge
* ConvNeXtV2 变体

**配置参数**：

| 参数                       | 说明                |
| ------------------------ | ----------------- |
| `arch`                   | 架构变体              |
| `depths`                 | 各 stage 层数        |
| `dims`                   | 各 stage 通道数       |
| `out_indices`            | 输出 stage 索引       |
| `drop_path_rate`         | DropPath 概率       |
| `layer_scale_init_value` | LayerScale 初始值    |
| `gap_before_final_norm`  | 是否在最终 norm 前做 GAP |

**特性**：

* 现代化 CNN 设计
* 7x7 深度卷积
* LayerNorm 替代 BatchNorm

---

### 7. `backbones/mobilenet.py`

**功能**：MobileNet 系列轻量级骨干

**支持变体**：

* MobileNetV2
* MobileNetV3-Small/Large

**配置参数**：

| 参数              | 说明                          |
| --------------- | --------------------------- |
| `arch`          | 架构：v2 / v3_small / v3_large |
| `out_indices`   | 输出特征层索引                     |
| `frozen_stages` | 冻结的 stage 数量                |
| `width_mult`    | 宽度乘数                        |
| `norm_cfg`      | 归一化层配置                      |

**特性**：

* 深度可分离卷积
* 倒残差结构
* SE 注意力（V3）
* 轻量级，适合移动端

---

### 8. `backbones/darknet.py`

**功能**：Darknet 系列骨干（YOLO 系列）

**支持变体**：

* Darknet-53（YOLOv3）
* CSPDarknet（YOLOv4/v5）
* ELAN（YOLOv7）
* C2f Backbone（YOLOv8）

**配置参数**：

| 参数              | 说明           |
| --------------- | ------------ |
| `arch`          | 架构变体         |
| `deepen_factor` | 深度因子         |
| `widen_factor`  | 宽度因子         |
| `out_indices`   | 输出特征层索引      |
| `frozen_stages` | 冻结的 stage 数量 |
| `norm_cfg`      | 归一化层配置       |
| `act_cfg`       | 激活函数配置       |

**特性**：

* CSP 结构
* 残差连接
* 适配 YOLO 检测器

---

## 三、necks/ 特征融合层

### 1. `necks/__init__.py`

**功能**：导出所有 Neck 组件

---

### 2. `necks/fpn.py`

**功能**：Feature Pyramid Network

**核心职责**：

* 多尺度特征融合
* 自顶向下路径 + 横向连接
* 输出统一通道数的特征金字塔

**配置参数**：

| 参数                        | 说明            |
| ------------------------- | ------------- |
| `in_channels`             | 输入各层通道数列表     |
| `out_channels`            | 输出统一通道数       |
| `num_outs`                | 输出特征层数量       |
| `start_level`             | 起始输入层级        |
| `end_level`               | 结束输入层级        |
| `add_extra_convs`         | 是否添加额外卷积层     |
| `relu_before_extra_convs` | 额外卷积前是否加 ReLU |
| `norm_cfg`                | 归一化层配置        |

---

### 3. `necks/pan.py`

**功能**：Path Aggregation Network

**核心职责**：

* FPN 基础上增加自底向上路径
* 增强底层特征的语义信息

**配置参数**：

* 继承 FPN 参数
* `extra_convs_on_inputs`：额外卷积应用位置

**特性**：

* 双向特征融合
* 更强的多尺度表达

---

### 4. `necks/bifpn.py`

**功能**：Bi-directional Feature Pyramid Network

**核心职责**：

* 双向跨尺度连接
* 可学习的特征融合权重
* 多次重复融合

**配置参数**：

| 参数                | 说明         |
| ----------------- | ---------- |
| `in_channels`     | 输入各层通道数    |
| `out_channels`    | 输出通道数      |
| `num_outs`        | 输出层数       |
| `num_repeats`     | BiFPN 重复次数 |
| `weighted_fusion` | 是否使用加权融合   |
| `norm_cfg`        | 归一化层配置     |

**特性**：

* EfficientDet 默认 Neck
* 快速归一化融合

---

### 5. `necks/yolo_neck.py`

**功能**：YOLO 系列专用 Neck

**支持变体**：

* YOLOv5 Neck（PANet）
* YOLOv7 Neck（E-ELAN）
* YOLOv8 Neck（C2f-based）

**配置参数**：

| 参数               | 说明      |
| ---------------- | ------- |
| `in_channels`    | 输入各层通道数 |
| `out_channels`   | 输出各层通道数 |
| `deepen_factor`  | 深度因子    |
| `widen_factor`   | 宽度因子    |
| `num_csp_blocks` | CSP 块数量 |
| `norm_cfg`       | 归一化层配置  |
| `act_cfg`        | 激活函数配置  |

---

## 四、heads/ 任务头

### 1. `heads/__init__.py`

**功能**：导出所有任务头

---

### 2. `heads/cls_head.py`

**功能**：分类任务头

**支持变体**：

* `LinearClsHead`：线性分类头
* `StackedLinearClsHead`：多层 MLP 分类头
* `VisionTransformerClsHead`：ViT 专用分类头

**配置参数**：

| 参数            | 说明           |
| ------------- | ------------ |
| `num_classes` | 类别数          |
| `in_channels` | 输入通道数        |
| `hidden_dims` | 隐藏层维度（MLP 头） |
| `dropout`     | Dropout 概率   |
| `loss`        | 损失函数配置       |
| `topk`        | Top-K 精度计算   |

**输出**：

* 训练：损失字典
* 推理：类别概率 / logits

---

### 3. `heads/det_head.py`

**功能**：检测任务头

**支持变体**：

* `AnchorHead`：基于 Anchor 的检测头
* `AnchorFreeHead`：无 Anchor 检测头
* `RPNHead`：区域提议网络头
* `YOLOHead`：YOLO 系列检测头
* `FCOSHead`：FCOS 检测头
* `ATSSHead`：ATSS 检测头

**配置参数**：

| 参数                 | 说明            |
| ------------------ | ------------- |
| `num_classes`      | 类别数           |
| `in_channels`      | 输入通道数         |
| `feat_channels`    | 特征通道数         |
| `stacked_convs`    | 堆叠卷积层数        |
| `anchor_generator` | Anchor 生成器配置  |
| `bbox_coder`       | BBox 编解码器配置   |
| `loss_cls`         | 分类损失配置        |
| `loss_bbox`        | 回归损失配置        |
| `loss_obj`         | 置信度损失配置（YOLO） |

**输出**：

* 训练：损失字典
* 推理：检测框列表 `[x1, y1, x2, y2, score, class]`

---

### 4. `heads/seg_head.py`

**功能**：语义分割任务头

**支持变体**：

* `FCNHead`：全卷积分割头
* `ASPPHead`：空洞空间金字塔池化头
* `PSPHead`：金字塔池化头
* `UPerHead`：统一感知解析头
* `SegFormerHead`：SegFormer 轻量头

**配置参数**：

| 参数              | 说明         |
| --------------- | ---------- |
| `num_classes`   | 类别数        |
| `in_channels`   | 输入通道数（列表）  |
| `channels`      | 中间特征通道数    |
| `in_index`      | 使用的输入特征索引  |
| `dropout_ratio` | Dropout 比例 |
| `loss_decode`   | 分割损失配置     |
| `align_corners` | 上采样对齐角点    |

**输出**：

* 训练：损失字典
* 推理：分割掩码 `[H, W]` 或概率图 `[C, H, W]`

---

### 5. `heads/inst_seg_head.py`

**功能**：实例分割任务头

**支持变体**：

* `MaskRCNNHead`：Mask R-CNN 掩码头
* `SOLOHead`：SOLO 实例分割头
* `CondInstHead`：条件实例分割头

**配置参数**：

| 参数              | 说明       |
| --------------- | -------- |
| `num_classes`   | 类别数      |
| `in_channels`   | 输入通道数    |
| `num_convs`     | 卷积层数量    |
| `roi_feat_size` | RoI 特征尺寸 |
| `loss_mask`     | 掩码损失配置   |

**输出**：

* 训练：损失字典（含掩码损失）
* 推理：实例掩码列表

---

## 五、detectors/ 检测器

### 1. `detectors/__init__.py`

**功能**：导出所有检测器

---

### 2. `detectors/base_detector.py`

**功能**：检测器基类

**核心职责**：

* 定义检测器统一接口
* 组合 Backbone + Neck + Head
* 提供训练/推理流程框架

**基类接口**：

| 方法                            | 功能                    |
| ----------------------------- | --------------------- |
| `extract_feat(img)`           | 特征提取（Backbone + Neck） |
| `forward_train(img, targets)` | 训练前向                  |
| `forward_test(img)`           | 推理前向                  |
| `simple_test(img)`            | 单图推理                  |
| `aug_test(imgs)`              | 增强测试（TTA）             |

**检测器类型**：

* 单阶段检测器：直接预测
* 两阶段检测器：RPN + RoI Head

---

### 3. `detectors/faster_rcnn.py`

**功能**：Faster R-CNN 检测器

**组件构成**：

* Backbone：特征提取
* Neck：FPN（可选）
* RPN Head：区域提议
* RoI Head：分类 + 回归

**配置参数**：

| 参数          | 说明            |
| ----------- | ------------- |
| `backbone`  | 骨干网络配置        |
| `neck`      | Neck 配置       |
| `rpn_head`  | RPN 头配置       |
| `roi_head`  | RoI 头配置       |
| `train_cfg` | 训练配置（正负样本策略等） |
| `test_cfg`  | 测试配置（NMS 参数等） |

---

### 4. `detectors/yolo.py`

**功能**：YOLO 系列检测器

**支持变体**：

* YOLOv5
* YOLOv7
* YOLOv8

**配置参数**：

| 参数          | 说明                    |
| ----------- | --------------------- |
| `backbone`  | Darknet/CSPDarknet 配置 |
| `neck`      | YOLO Neck 配置          |
| `head`      | YOLO Head 配置          |
| `size`      | 模型尺寸：n/s/m/l/x        |
| `train_cfg` | 训练配置                  |
| `test_cfg`  | 测试配置                  |

**特性**：

* 端到端单阶段检测
* 多尺度预测
* Anchor-based / Anchor-free

---

### 5. `detectors/fcos.py`

**功能**：FCOS 无锚框检测器

**组件构成**：

* Backbone + FPN
* FCOSHead（逐像素预测）

**特性**：

* 完全无 Anchor
* 中心度预测
* 多尺度分配

---

### 6. `detectors/detr.py`

**功能**：DETR 检测器

**组件构成**：

* Backbone：CNN 特征提取
* Transformer Encoder-Decoder
* 匈牙利匹配

**支持变体**：

* DETR
* Deformable DETR
* DAB-DETR
* DINO

**特性**：

* 端到端无 NMS
* 可学习的 Object Queries
* 二分图匹配训练

---

### 7. `detectors/retinanet.py`

**功能**：RetinaNet 检测器

**特性**：

* 单阶段检测器
* Focal Loss
* FPN 多尺度

---

## 六、segmentors/ 分割器

### 1. `segmentors/__init__.py`

**功能**：导出所有分割器

---

### 2. `segmentors/base_segmentor.py`

**功能**：分割器基类

**核心职责**：

* 定义分割器统一接口
* 组合 Backbone + Decode Head
* 辅助头支持

**基类接口**：

| 方法                              | 功能           |
| ------------------------------- | ------------ |
| `extract_feat(img)`             | 特征提取         |
| `encode_decode(img, img_metas)` | 编码解码流程       |
| `forward_train(img, seg_map)`   | 训练前向         |
| `inference(img)`                | 推理（含滑窗/翻转增强） |
| `slide_inference(img)`          | 滑窗推理（大图）     |
| `whole_inference(img)`          | 整图推理         |

---

### 3. `segmentors/unet.py`

**功能**：U-Net 分割器

**特性**：

* 对称编码器-解码器结构
* 跳跃连接
* 适合医学图像

**配置参数**：

| 参数              | 说明         |
| --------------- | ---------- |
| `in_channels`   | 输入通道数      |
| `num_classes`   | 类别数        |
| `base_channels` | 基础通道数      |
| `num_stages`    | 编码器/解码器阶段数 |
| `strides`       | 各阶段步长      |
| `enc_num_convs` | 编码器每阶段卷积数  |
| `dec_num_convs` | 解码器每阶段卷积数  |
| `norm_cfg`      | 归一化配置      |

---

### 4. `segmentors/deeplabv3.py`

**功能**：DeepLabV3/V3+ 分割器

**特性**：

* 空洞卷积
* ASPP 模块
* 编码器-解码器结构（V3+）

**配置参数**：

| 参数               | 说明         |
| ---------------- | ---------- |
| `backbone`       | 骨干网络配置     |
| `decode_head`    | ASPP 解码头配置 |
| `auxiliary_head` | 辅助头配置（可选）  |
| `dilations`      | ASPP 空洞率   |
| `output_stride`  | 输出步长：8/16  |

---

### 5. `segmentors/segformer.py`

**功能**：SegFormer 分割器

**特性**：

* 层次 Transformer 编码器
* 轻量 MLP 解码器
* 无位置编码

**配置参数**：

| 参数            | 说明                 |
| ------------- | ------------------ |
| `backbone`    | Mix Transformer 配置 |
| `decode_head` | SegFormer Head 配置  |
| `embed_dims`  | 嵌入维度               |
| `num_heads`   | 各层注意力头数            |
| `num_layers`  | 各阶段层数              |

---

### 6. `segmentors/pspnet.py`

**功能**：PSPNet 分割器

**特性**：

* 金字塔池化模块（PPM）
* 多尺度上下文聚合

---

### 7. `segmentors/mask_rcnn.py`

**功能**：Mask R-CNN 实例分割器

**组件构成**：

* Faster R-CNN 检测器
* Mask Head

**特性**：

* 检测 + 实例分割
* RoI Align 特征提取
* 逐实例掩码预测

**配置参数**：

| 参数          | 说明                   |
| ----------- | -------------------- |
| `backbone`  | 骨干网络配置               |
| `neck`      | FPN 配置               |
| `rpn_head`  | RPN 头配置              |
| `roi_head`  | RoI 头配置（含 mask head） |
| `train_cfg` | 训练配置                 |
| `test_cfg`  | 测试配置                 |

---

## 七、classifiers/ 分类器

### 1. `classifiers/__init__.py`

**功能**：导出所有分类器

---

### 2. `classifiers/base_classifier.py`

**功能**：分类器基类

**核心职责**：

* 定义分类器统一接口
* 组合 Backbone + Head
* 提供训练/推理流程

**基类接口**：

| 方法                           | 功能   |
| ---------------------------- | ---- |
| `extract_feat(img)`          | 特征提取 |
| `forward_train(img, labels)` | 训练前向 |
| `forward_test(img)`          | 推理前向 |
| `simple_test(img)`           | 单图推理 |

---

### 3. `classifiers/image_classifier.py`

**功能**：通用图像分类器

**组件构成**：

* Backbone：任意骨干网络
* Neck：可选（GlobalAveragePooling 等）
* Head：分类头

**配置参数**：

| 参数           | 说明          |
| ------------ | ----------- |
| `backbone`   | 骨干网络配置      |
| `neck`       | Neck 配置（可选） |
| `head`       | 分类头配置       |
| `pretrained` | 预训练权重路径     |
| `train_cfg`  | 训练配置        |
| `test_cfg`   | 测试配置        |

**输出**：

* 训练：损失字典 `{'loss': tensor}`
* 推理：预测结果 `{'pred_label': tensor, 'pred_score': tensor}`

---

## 八、layers/ 基础层组件

### 1. `layers/__init__.py`

**功能**：导出所有基础层

**导出内容**：

* 所有自定义层
* 层构建工具函数

---

### 2. `layers/conv_module.py`

**功能**：卷积模块封装

**核心组件**：

| 组件                       | 功能                          |
| ------------------------ | --------------------------- |
| `ConvModule`             | Conv + Norm + Activation 组合 |
| `DepthwiseSeparableConv` | 深度可分离卷积                     |
| `DeformConv2d`           | 可变形卷积                       |
| `ModulatedDeformConv2d`  | 调制可变形卷积                     |

**ConvModule 配置**：

| 参数             | 说明                             |
| -------------- | ------------------------------ |
| `in_channels`  | 输入通道                           |
| `out_channels` | 输出通道                           |
| `kernel_size`  | 卷积核大小                          |
| `stride`       | 步长                             |
| `padding`      | 填充                             |
| `dilation`     | 空洞率                            |
| `groups`       | 分组数                            |
| `bias`         | 是否使用偏置                         |
| `norm_cfg`     | 归一化配置                          |
| `act_cfg`      | 激活函数配置                         |
| `order`        | 组件顺序：`('conv', 'norm', 'act')` |

---

### 3. `layers/norm_layers.py`

**功能**：归一化层

**支持类型**：

| 层                       | 说明              |
| ----------------------- | --------------- |
| `build_norm_layer(cfg)` | 归一化层构建函数        |
| `BatchNorm2d`           | 批归一化（包装）        |
| `SyncBatchNorm`         | 同步批归一化          |
| `GroupNorm`             | 组归一化            |
| `LayerNorm`             | 层归一化            |
| `InstanceNorm`          | 实例归一化           |
| `LayerNorm2d`           | 2D 层归一化（用于 CNN） |

**配置格式**：

```
norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=32)
norm_cfg = dict(type='LN')
```

---

### 4. `layers/activation.py`

**功能**：激活函数

**支持类型**：

| 激活函数                    | 说明           |
| ----------------------- | ------------ |
| `build_activation(cfg)` | 激活函数构建函数     |
| `ReLU`                  | 标准 ReLU      |
| `LeakyReLU`             | 带泄漏 ReLU     |
| `ReLU6`                 | ReLU6（移动端）   |
| `PReLU`                 | 参数化 ReLU     |
| `SiLU` / `Swish`        | SiLU 激活      |
| `GELU`                  | 高斯误差线性单元     |
| `Mish`                  | Mish 激活      |
| `HardSwish`             | Hard Swish   |
| `HardSigmoid`           | Hard Sigmoid |

**配置格式**：

```
act_cfg = dict(type='ReLU', inplace=True)
act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
act_cfg = dict(type='SiLU')
```

---

### 5. `layers/attention.py`

**功能**：注意力机制模块

**支持类型**：

| 模块                    | 说明                     |
| --------------------- | ---------------------- |
| `SELayer`             | Squeeze-and-Excitation |
| `CBAM`                | 通道+空间注意力               |
| `ECALayer`            | 高效通道注意力                |
| `MultiHeadAttention`  | 多头自注意力                 |
| `WindowAttention`     | 窗口注意力（Swin）            |
| `DeformableAttention` | 可变形注意力（DETR）           |
| `CrossAttention`      | 交叉注意力                  |
| `SpatialAttention`    | 空间注意力                  |
| `ChannelAttention`    | 通道注意力                  |

**MultiHeadAttention 参数**：

| 参数          | 说明          |
| ----------- | ----------- |
| `embed_dim` | 嵌入维度        |
| `num_heads` | 注意力头数       |
| `dropout`   | Dropout 概率  |
| `qkv_bias`  | QKV 是否使用偏置  |
| `attn_drop` | 注意力 Dropout |
| `proj_drop` | 投影 Dropout  |

---

### 6. `layers/drop.py`

**功能**：正则化 Drop 层

**支持类型**：

| 层           | 说明                     |
| ----------- | ---------------------- |
| `Dropout`   | 标准 Dropout             |
| `DropPath`  | 随机深度（Stochastic Depth） |
| `DropBlock` | DropBlock 正则化          |

**DropPath 参数**：

| 参数              | 说明       |
| --------------- | -------- |
| `drop_prob`     | Drop 概率  |
| `scale_by_keep` | 是否按保留率缩放 |

---

### 7. `layers/position_encoding.py`

**功能**：位置编码

**支持类型**：

| 编码                            | 说明           |
| ----------------------------- | ------------ |
| `SinusoidalPositionEncoding`  | 正弦位置编码（固定）   |
| `LearnedPositionEncoding`     | 可学习位置编码      |
| `RotaryPositionEncoding`      | 旋转位置编码（RoPE） |
| `ConditionalPositionEncoding` | 条件位置编码（CPE）  |
| `RelativePositionBias`        | 相对位置偏置（Swin） |

**SinusoidalPositionEncoding 参数**：

| 参数            | 说明    |
| ------------- | ----- |
| `num_feats`   | 特征维度  |
| `temperature` | 温度参数  |
| `normalize`   | 是否归一化 |
| `offset`      | 偏移量   |

---

### 8. `layers/blocks.py`

**功能**：复合模块块

**支持类型**：

| 块                         | 说明              |
| ------------------------- | --------------- |
| `BasicBlock`              | ResNet 基础块（2 层） |
| `Bottleneck`              | ResNet 瓶颈块（3 层） |
| `InvertedResidual`        | MobileNet 倒残差块  |
| `CSPLayer`                | CSP 层（YOLO）     |
| `C2f`                     | C2f 模块（YOLOv8）  |
| `TransformerEncoderLayer` | Transformer 编码层 |
| `TransformerDecoderLayer` | Transformer 解码层 |
| `FFN`                     | 前馈网络            |
| `MLP`                     | 多层感知机           |
| `SPPF`                    | 空间金字塔池化快速版      |

**Bottleneck 参数**：

| 参数             | 说明             |
| -------------- | -------------- |
| `in_channels`  | 输入通道           |
| `out_channels` | 输出通道           |
| `stride`       | 步长             |
| `dilation`     | 空洞率            |
| `downsample`   | 下采样层（shortcut） |
| `groups`       | 分组数（ResNeXt）   |
| `base_width`   | 基础宽度           |
| `norm_cfg`     | 归一化配置          |
| `act_cfg`      | 激活配置           |

---

## 九、utils/ 模型工具

### 1. `utils/__init__.py`

**功能**：导出所有模型工具

**导出内容**：

* 权重初始化函数
* 模型工具函数
* 检查点转换工具

---

### 2. `utils/weight_init.py`

**功能**：权重初始化

**初始化方法**：

| 方法                                         | 说明          |
| ------------------------------------------ | ----------- |
| `constant_init(module, val)`               | 常数初始化       |
| `xavier_init(module, gain)`                | Xavier 初始化  |
| `kaiming_init(module, mode, nonlinearity)` | Kaiming 初始化 |
| `normal_init(module, mean, std)`           | 正态分布初始化     |
| `uniform_init(module, a, b)`               | 均匀分布初始化     |
| `trunc_normal_init(module, mean, std)`     | 截断正态初始化     |
| `bias_init_with_prob(prior_prob)`          | 基于先验概率初始化偏置 |

**层类型适配**：

* Conv2d：Kaiming 初始化
* Linear：Xavier 或 Kaiming
* BatchNorm：常数初始化（weight=1, bias=0）
* LayerNorm：常数初始化

**初始化配置格式**：

```
init_cfg = dict(type='Kaiming', layer='Conv2d')
init_cfg = dict(type='Pretrained', checkpoint='path/to/weights.pth')
```

---

### 3. `utils/model_utils.py`

**功能**：模型通用工具

**工具函数**：

| 函数                                         | 功能              |
| ------------------------------------------ | --------------- |
| `get_model_complexity(model, input_shape)` | 计算 FLOPs 和参数量   |
| `count_parameters(model, trainable_only)`  | 统计参数数量          |
| `freeze_module(module)`                    | 冻结模块参数          |
| `unfreeze_module(module)`                  | 解冻模块参数          |
| `freeze_bn(module)`                        | 冻结 BatchNorm 层  |
| `set_bn_eval(module)`                      | 设置 BN 为 eval 模式 |
| `fuse_conv_bn(model)`                      | 融合 Conv + BN    |
| `replace_module(model, old, new)`          | 替换模块            |
| `get_module_by_name(model, name)`          | 按名称获取模块         |
| `make_divisible(v, divisor)`               | 通道数对齐（8/16 的倍数） |
| `auto_pad(kernel_size, dilation)`          | 自动计算 padding    |

**复杂度分析输出**：

* FLOPs（浮点运算次数）
* MACs（乘加次数）
* 参数量（总/可训练）
* 内存占用估计

---

### 4. `utils/ckpt_convert.py`

**功能**：检查点格式转换

**转换能力**：

| 来源          | 目标    | 说明                  |
| ----------- | ----- | ------------------- |
| torchvision | 本框架   | torchvision 预训练权重适配 |
| timm        | 本框架   | timm 预训练权重适配        |
| mmcv/mmdet  | 本框架   | MMSeries 权重适配       |
| detectron2  | 本框架   | Detectron2 权重适配     |
| 本框架旧版       | 本框架新版 | 版本升级兼容              |

**转换函数**：

| 函数                                            | 功能             |
| --------------------------------------------- | -------------- |
| `convert_checkpoint(src, mapping)`            | 通用转换函数         |
| `convert_from_torchvision(state_dict, model)` | torchvision 转换 |
| `convert_from_timm(state_dict, model)`        | timm 转换        |
| `auto_convert(state_dict, model)`             | 自动检测并转换        |
| `remap_keys(state_dict, mapping)`             | 键名重映射          |
| `filter_by_prefix(state_dict, prefix)`        | 前缀过滤           |
| `strip_prefix(state_dict, prefix)`            | 移除前缀           |
| `add_prefix(state_dict, prefix)`              | 添加前缀           |

**加载策略**：

* `strict=True`：严格匹配
* `strict=False`：忽略不匹配
* 自动跳过形状不匹配
* 报告加载情况（matched/missing/unexpected）

---

## 模块对外接口

### 模型构建

```
# 完整模型构建
model = build_model(config.model)

# 子组件构建
backbone = build_backbone(config.model.backbone)
neck = build_neck(config.model.neck)
head = build_head(config.model.head)
```

### 模型使用

```
# 训练模式
losses = model(images, targets, mode='loss')

# 推理模式
predictions = model(images, mode='predict')

# 特征提取
features = model(images, mode='features')
```

### 模型工具

```
# 复杂度分析
flops, params = get_model_complexity(model, (3, 224, 224))

# 冻结骨干
freeze_module(model.backbone)

# 加载预训练
model.load_pretrained('path/to/weights.pth')
```

---

## 配置示例

```yaml
model:
  type: ImageClassifier
  backbone:
    type: ResNet
    depth: 50
    out_indices: [4]
    frozen_stages: 1
    norm_cfg:
      type: BN
      requires_grad: true
    init_cfg:
      type: Pretrained
      checkpoint: torchvision://resnet50
  neck:
    type: GlobalAveragePooling
  head:
    type: LinearClsHead
    num_classes: 1000
    in_channels: 2048
    loss:
      type: CrossEntropyLoss
      label_smoothing: 0.1
```

---

## 依赖关系

**被依赖方**：

* Training Engine - 模型训练
* Export Pipeline - 模型导出
* Visualization - 模型可视化

**依赖项**：

* `Registry` - 组件注册与构建
* `Losses` - 损失函数
* `torch.nn` - PyTorch 基础层
