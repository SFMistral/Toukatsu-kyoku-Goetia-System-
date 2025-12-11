# Augmentations 模块功能需求文档

## 模块概述

Augmentations 模块是系统的数据增强组件库，提供几何变换、光度变换、混合增强等丰富的数据增强能力。支持图像与标注（bbox/mask/keypoint）的同步变换，通过组合管道实现灵活的增强策略，兼容分类、检测、分割三大任务。

---

## 目录结构

```
augmentations/
├── __init__.py
├── builder.py               # 变换构建器
├── base_transform.py        # 变换基类
│
├── geometric/               # 几何变换
│   ├── __init__.py
│   ├── resize.py
│   ├── flip.py
│   ├── rotate.py
│   ├── crop.py
│   ├── affine.py
│   └── perspective.py
│
├── photometric/             # 光度变换
│   ├── __init__.py
│   ├── color_jitter.py
│   ├── normalize.py
│   ├── blur.py
│   ├── noise.py
│   └── histogram.py
│
├── mixing/                  # 混合增强
│   ├── __init__.py
│   ├── mixup.py
│   ├── cutmix.py
│   ├── mosaic.py
│   └── copypaste.py
│
├── formatting/              # 格式化处理
│   ├── __init__.py
│   ├── to_tensor.py
│   ├── pad.py
│   └── collect.py
│
└── pipelines/               # 增强流水线
    ├── __init__.py
    ├── compose.py
    ├── auto_augment.py
    └── preset_pipelines.py
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_transform(config)` - 变换构建函数
* `build_pipeline(configs)` - 管道构建函数
* `Compose` - 组合变换类
* `BaseTransform` - 变换基类
* 所有具体变换类

---

### 2. `builder.py`

**功能**：变换构建器，配置驱动的变换实例化

**核心职责**：

* 根据配置构建单个变换
* 根据配置列表构建变换管道
* 预设管道快速获取

**构建函数**：

| 函数                                | 功能                 |
| --------------------------------- | ------------------ |
| `build_transform(config)`         | 构建单个变换             |
| `build_pipeline(configs)`         | 构建变换管道（返回 Compose） |
| `get_preset_pipeline(name, task)` | 获取预设管道             |

**预设管道**：

| 名称                   | 说明            |
| -------------------- | ------------- |
| `imagenet_train`     | ImageNet 训练增强 |
| `imagenet_val`       | ImageNet 验证增强 |
| `coco_train`         | COCO 检测训练增强   |
| `coco_val`           | COCO 检测验证增强   |
| `segmentation_train` | 分割训练增强        |
| `segmentation_val`   | 分割验证增强        |

---

### 3. `base_transform.py`

**功能**：所有变换的抽象基类

**核心职责**：

* 定义变换统一接口
* 数据结构规范
* 标注同步变换框架
* 随机性控制

**基类接口**：

| 方法                                    | 功能           |
| ------------------------------------- | ------------ |
| `__call__(results)`                   | 应用变换（入口）     |
| `transform(results)`                  | 实际变换逻辑（子类实现） |
| `transform_image(image)`              | 图像变换         |
| `transform_bboxes(bboxes, ...)`       | 边界框变换        |
| `transform_masks(masks, ...)`         | 掩码变换         |
| `transform_keypoints(keypoints, ...)` | 关键点变换        |

**数据结构**：

`results` 字典包含：

| 键                  | 类型           | 说明             |
| ------------------ | ------------ | -------------- |
| `img`              | ndarray      | 图像数据           |
| `img_shape`        | tuple        | 图像形状 (H, W, C) |
| `ori_shape`        | tuple        | 原始形状           |
| `bboxes`           | ndarray      | 边界框 [N, 4]     |
| `labels`           | ndarray      | 类别标签           |
| `masks`            | ndarray/list | 分割掩码           |
| `keypoints`        | ndarray      | 关键点            |
| `transform_matrix` | ndarray      | 累积变换矩阵         |

**随机性控制**：

* 概率参数 `prob`
* 随机种子设置
* 可复现性支持

---

## 二、geometric/ 几何变换

### 1. `geometric/__init__.py`

**功能**：导出所有几何变换

---

### 2. `geometric/resize.py`

**功能**：尺寸调整变换

**变换类型**：

| 类                    | 说明     |
| -------------------- | ------ |
| `Resize`             | 固定尺寸缩放 |
| `RandomResize`       | 随机尺寸缩放 |
| `ResizeShortestEdge` | 短边缩放   |
| `ResizeLongestEdge`  | 长边缩放   |
| `LetterBox`          | 等比缩放填充 |

**Resize 参数**：

| 参数              | 说明                                |
| --------------- | --------------------------------- |
| `size`          | 目标尺寸 (H, W) 或单值                   |
| `scale`         | 缩放比例                              |
| `keep_ratio`    | 是否保持宽高比                           |
| `interpolation` | 插值方法：bilinear / nearest / bicubic |

**RandomResize 参数**：

| 参数           | 说明                |
| ------------ | ----------------- |
| `scales`     | 缩放比例范围 (min, max) |
| `sizes`      | 候选尺寸列表            |
| `keep_ratio` | 是否保持宽高比           |

**标注同步**：

* bbox：坐标按比例缩放
* mask：同步缩放
* keypoint：坐标按比例缩放

---

### 3. `geometric/flip.py`

**功能**：翻转变换

**变换类型**：

| 类                | 说明   |
| ---------------- | ---- |
| `RandomFlip`     | 随机翻转 |
| `HorizontalFlip` | 水平翻转 |
| `VerticalFlip`   | 垂直翻转 |

**参数**：

| 参数          | 说明                                    |
| ----------- | ------------------------------------- |
| `prob`      | 翻转概率                                  |
| `direction` | 翻转方向：horizontal / vertical / diagonal |

**标注同步**：

* bbox：坐标镜像翻转
* mask：同步翻转
* keypoint：坐标翻转 + 左右点交换（如左眼↔右眼）

---

### 4. `geometric/rotate.py`

**功能**：旋转变换

**变换类型**：

| 类                | 说明          |
| ---------------- | ----------- |
| `RandomRotate`   | 随机角度旋转      |
| `Rotate`         | 固定角度旋转      |
| `RandomRotate90` | 随机 90° 倍数旋转 |

**参数**：

| 参数             | 说明                |
| -------------- | ----------------- |
| `angle`        | 旋转角度或范围 (-30, 30) |
| `center`       | 旋转中心              |
| `scale`        | 缩放因子              |
| `border_mode`  | 边界填充模式            |
| `border_value` | 填充值               |
| `expand`       | 是否扩展画布以包含完整图像     |

**标注同步**：

* bbox：旋转后重新计算外接矩形
* mask：像素级旋转
* keypoint：坐标旋转变换

---

### 5. `geometric/crop.py`

**功能**：裁剪变换

**变换类型**：

| 类                   | 说明          |
| ------------------- | ----------- |
| `RandomCrop`        | 随机位置裁剪      |
| `CenterCrop`        | 中心裁剪        |
| `RandomResizedCrop` | 随机缩放裁剪      |
| `RandomCropNearBox` | 在目标附近裁剪（检测） |
| `CropInstance`      | 裁剪单个实例（分割）  |

**RandomCrop 参数**：

| 参数              | 说明          |
| --------------- | ----------- |
| `size`          | 裁剪尺寸 (H, W) |
| `padding`       | 填充大小        |
| `pad_if_needed` | 尺寸不足时填充     |
| `fill`          | 填充值         |
| `padding_mode`  | 填充模式        |

**RandomResizedCrop 参数**：

| 参数              | 说明                   |
| --------------- | -------------------- |
| `size`          | 输出尺寸                 |
| `scale`         | 裁剪面积比例范围 (0.08, 1.0) |
| `ratio`         | 裁剪宽高比范围 (3/4, 4/3)   |
| `interpolation` | 插值方法                 |

**标注同步**：

* bbox：坐标偏移 + 裁剪越界处理
* mask：同步裁剪
* keypoint：坐标偏移 + 可见性更新
* 越界目标过滤策略

---

### 6. `geometric/affine.py`

**功能**：仿射变换

**变换类型**：

| 类              | 说明     |
| -------------- | ------ |
| `RandomAffine` | 随机仿射变换 |
| `Affine`       | 固定仿射变换 |
| `ShearX`       | X 方向剪切 |
| `ShearY`       | Y 方向剪切 |
| `TranslateX`   | X 方向平移 |
| `TranslateY`   | Y 方向平移 |

**RandomAffine 参数**：

| 参数              | 说明     |
| --------------- | ------ |
| `degrees`       | 旋转角度范围 |
| `translate`     | 平移比例范围 |
| `scale`         | 缩放范围   |
| `shear`         | 剪切角度范围 |
| `interpolation` | 插值方法   |
| `fill`          | 填充值    |

**标注同步**：

* 通过仿射矩阵变换所有标注

---

### 7. `geometric/perspective.py`

**功能**：透视变换

**变换类型**：

| 类                   | 说明     |
| ------------------- | ------ |
| `RandomPerspective` | 随机透视变换 |
| `Perspective`       | 固定透视变换 |

**参数**：

| 参数                 | 说明   |
| ------------------ | ---- |
| `distortion_scale` | 扭曲程度 |
| `prob`             | 应用概率 |
| `interpolation`    | 插值方法 |
| `fill`             | 填充值  |

---

## 三、photometric/ 光度变换

### 1. `photometric/__init__.py`

**功能**：导出所有光度变换

---

### 2. `photometric/color_jitter.py`

**功能**：颜色抖动变换

**变换类型**：

| 类                  | 说明     |
| ------------------ | ------ |
| `ColorJitter`      | 综合颜色抖动 |
| `RandomBrightness` | 随机亮度   |
| `RandomContrast`   | 随机对比度  |
| `RandomSaturation` | 随机饱和度  |
| `RandomHue`        | 随机色调   |
| `RandomGamma`      | 随机伽马校正 |

**ColorJitter 参数**：

| 参数           | 说明      |
| ------------ | ------- |
| `brightness` | 亮度调整范围  |
| `contrast`   | 对比度调整范围 |
| `saturation` | 饱和度调整范围 |
| `hue`        | 色调调整范围  |

**特性**：

* 仅影响图像，不影响标注
* 参数可为单值或范围

---

### 3. `photometric/normalize.py`

**功能**：归一化变换

**变换类型**：

| 类             | 说明               |
| ------------- | ---------------- |
| `Normalize`   | 标准归一化            |
| `Denormalize` | 反归一化             |
| `ToFloat`     | 转为浮点型并缩放到 [0, 1] |

**Normalize 参数**：

| 参数       | 说明             |
| -------- | -------------- |
| `mean`   | 均值 (R, G, B)   |
| `std`    | 标准差 (R, G, B)  |
| `to_rgb` | 是否转换 BGR → RGB |

**预设值**：

| 数据集      | mean                  | std                   |
| -------- | --------------------- | --------------------- |
| ImageNet | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| COCO     | (0.471, 0.448, 0.408) | (0.234, 0.239, 0.242) |

---

### 4. `photometric/blur.py`

**功能**：模糊变换

**变换类型**：

| 类              | 说明         |
| -------------- | ---------- |
| `GaussianBlur` | 高斯模糊       |
| `MotionBlur`   | 运动模糊       |
| `MedianBlur`   | 中值模糊       |
| `RandomBlur`   | 随机模糊（多种方式） |

**GaussianBlur 参数**：

| 参数            | 说明               |
| ------------- | ---------------- |
| `kernel_size` | 卷积核大小            |
| `sigma`       | 标准差范围 (min, max) |
| `prob`        | 应用概率             |

---

### 5. `photometric/noise.py`

**功能**：噪声变换

**变换类型**：

| 类                 | 说明   |
| ----------------- | ---- |
| `GaussianNoise`   | 高斯噪声 |
| `SaltPepperNoise` | 椒盐噪声 |
| `PoissonNoise`    | 泊松噪声 |
| `RandomNoise`     | 随机噪声 |

**GaussianNoise 参数**：

| 参数     | 说明      |
| ------ | ------- |
| `mean` | 噪声均值    |
| `std`  | 噪声标准差范围 |
| `prob` | 应用概率    |

---

### 6. `photometric/histogram.py`

**功能**：直方图变换

**变换类型**：

| 类                       | 说明             |
| ----------------------- | -------------- |
| `HistogramEqualization` | 直方图均衡化         |
| `CLAHE`                 | 对比度受限自适应直方图均衡化 |
| `RandomHistogram`       | 随机直方图变换        |
| `AutoContrast`          | 自动对比度          |
| `Equalize`              | 均衡化            |
| `Posterize`             | 色调分离           |
| `Solarize`              | 曝光效果           |
| `Invert`                | 颜色反转           |

**CLAHE 参数**：

| 参数               | 说明    |
| ---------------- | ----- |
| `clip_limit`     | 对比度限制 |
| `tile_grid_size` | 网格大小  |
| `prob`           | 应用概率  |

---

## 四、mixing/ 混合增强

### 1. `mixing/__init__.py`

**功能**：导出所有混合增强

---

### 2. `mixing/mixup.py`

**功能**：Mixup 混合增强

**变换类型**：

| 类          | 说明                |
| ---------- | ----------------- |
| `Mixup`    | 标准 Mixup          |
| `CutMixup` | Mixup + CutMix 组合 |

**参数**：

| 参数            | 说明                          |
| ------------- | --------------------------- |
| `alpha`       | Beta 分布参数                   |
| `prob`        | 应用概率                        |
| `switch_prob` | CutMixup 中切换概率              |
| `mode`        | 混合模式：batch / pair / element |
| `correct_lam` | 是否校正 lambda                 |

**工作模式**：

* batch：同批次样本混合
* pair：成对样本混合
* element：逐元素混合

**标签处理**：

* 返回混合比例 lambda
* 软标签生成

---

### 3. `mixing/cutmix.py`

**功能**：CutMix 剪切混合

**变换类型**：

| 类           | 说明        |
| ----------- | --------- |
| `CutMix`    | 标准 CutMix |
| `ResizeMix` | 缩放混合      |

**参数**：

| 参数              | 说明             |
| --------------- | -------------- |
| `alpha`         | Beta 分布参数      |
| `prob`          | 应用概率           |
| `cutmix_minmax` | 剪切区域比例范围       |
| `correct_lam`   | 按实际面积校正 lambda |

**标签处理**：

* 按剪切区域面积计算混合比例
* 软标签生成

---

### 4. `mixing/mosaic.py`

**功能**：Mosaic 马赛克增强（YOLO 风格）

**变换类型**：

| 类         | 说明     |
| --------- | ------ |
| `Mosaic`  | 4 图马赛克 |
| `Mosaic9` | 9 图马赛克 |

**参数**：

| 参数                   | 说明     |
| -------------------- | ------ |
| `img_size`           | 输出图像尺寸 |
| `center_ratio_range` | 中心点范围  |
| `prob`               | 应用概率   |
| `pad_value`          | 填充值    |

**工作流程**：

1. 随机选择中心点
2. 4/9 张图像缩放填充到对应位置
3. 合并所有标注

**标注同步**：

* bbox：坐标变换 + 裁剪
* mask：拼接变换
* 越界目标过滤

---

### 5. `mixing/copypaste.py`

**功能**：Copy-Paste 实例复制粘贴

**变换类型**：

| 类                 | 说明      |
| ----------------- | ------- |
| `CopyPaste`       | 实例复制粘贴  |
| `SimpleCopyPaste` | 简化版复制粘贴 |

**参数**：

| 参数                  | 说明         |
| ------------------- | ---------- |
| `max_num_instances` | 最大粘贴实例数    |
| `prob`              | 应用概率       |
| `blend`             | 是否高斯混合边缘   |
| `sigma`             | 混合边缘 sigma |
| `scale_range`       | 实例缩放范围     |
| `flip_prob`         | 实例翻转概率     |

**工作流程**：

1. 从源图像提取实例（需要 mask）
2. 随机变换（缩放、翻转）
3. 粘贴到目标图像
4. 更新目标标注

**标注同步**：

* bbox：添加新实例框
* mask：添加新实例掩码
* 遮挡处理

---

## 五、formatting/ 格式化处理

### 1. `formatting/__init__.py`

**功能**：导出所有格式化变换

---

### 2. `formatting/to_tensor.py`

**功能**：张量转换

**变换类型**：

| 类               | 说明                     |
| --------------- | ---------------------- |
| `ToTensor`      | 图像转为 Tensor            |
| `ImageToTensor` | 图像转 Tensor (HWC → CHW) |
| `ToNumpy`       | 转为 NumPy 数组            |
| `ToDevice`      | 移动到指定设备                |

**ToTensor 参数**：

| 参数      | 说明       |
| ------- | -------- |
| `keys`  | 需要转换的键列表 |
| `dtype` | 目标数据类型   |

**处理内容**：

* 图像：HWC → CHW，uint8 → float32
* bbox：转为 Tensor
* mask：转为 Tensor
* label：转为 LongTensor

---

### 3. `formatting/pad.py`

**功能**：填充变换

**变换类型**：

| 类                | 说明       |
| ---------------- | -------- |
| `Pad`            | 固定填充     |
| `PadToSize`      | 填充到指定尺寸  |
| `PadToDivisible` | 填充到可整除尺寸 |
| `RandomPad`      | 随机填充     |

**Pad 参数**：

| 参数             | 说明                                       |
| -------------- | ---------------------------------------- |
| `size`         | 目标尺寸                                     |
| `size_divisor` | 尺寸除数                                     |
| `pad_val`      | 图像填充值                                    |
| `seg_pad_val`  | 分割图填充值                                   |
| `padding_mode` | 模式：constant / edge / reflect / symmetric |

**标注同步**：

* bbox：坐标偏移
* mask：同步填充

---

### 4. `formatting/collect.py`

**功能**：数据收集与打包

**变换类型**：

| 类              | 说明        |
| -------------- | --------- |
| `Collect`      | 收集指定键     |
| `PackInputs`   | 打包为模型输入格式 |
| `FormatBundle` | 格式化打包     |

**Collect 参数**：

| 参数          | 说明     |
| ----------- | ------ |
| `keys`      | 收集的键列表 |
| `meta_keys` | 元信息键列表 |

**PackInputs 功能**：

* 将分散的数据组织为统一结构
* 区分输入数据与目标数据
* 元信息打包

**输出结构**：

| 键              | 内容           |
| -------------- | ------------ |
| `inputs`       | 模型输入（图像）     |
| `data_samples` | 数据样本（标注、元信息） |

---

## 六、pipelines/ 增强流水线

### 1. `pipelines/__init__.py`

**功能**：导出所有管道类

---

### 2. `pipelines/compose.py`

**功能**：变换组合器

**组合类型**：

| 类              | 说明          |
| -------------- | ----------- |
| `Compose`      | 顺序组合        |
| `RandomChoice` | 随机选择一个变换    |
| `RandomApply`  | 随机决定是否应用    |
| `OneOf`        | 随机选择一个（带权重） |
| `SomeOf`       | 随机选择多个      |

**Compose 参数**：

| 参数           | 说明   |
| ------------ | ---- |
| `transforms` | 变换列表 |

**RandomChoice 参数**：

| 参数           | 说明     |
| ------------ | ------ |
| `transforms` | 候选变换列表 |
| `probs`      | 各变换概率  |

**OneOf 参数**：

| 参数           | 说明     |
| ------------ | ------ |
| `transforms` | 候选变换列表 |
| `probs`      | 各变换权重  |

**SomeOf 参数**：

| 参数           | 说明      |
| ------------ | ------- |
| `transforms` | 候选变换列表  |
| `n`          | 选择数量或范围 |
| `replace`    | 是否可重复选择 |

---

### 3. `pipelines/auto_augment.py`

**功能**：自动增强策略

**策略类型**：

| 类                    | 说明             |
| -------------------- | -------------- |
| `AutoAugment`        | 原始 AutoAugment |
| `RandAugment`        | RandAugment    |
| `TrivialAugmentWide` | TrivialAugment |
| `AugMix`             | AugMix         |

**AutoAugment 参数**：

| 参数        | 说明                             |
| --------- | ------------------------------ |
| `policy`  | 策略名称：imagenet / cifar10 / svhn |
| `hparams` | 超参数字典                          |

**RandAugment 参数**：

| 参数                   | 说明          |
| -------------------- | ----------- |
| `num_ops`            | 每次应用的操作数    |
| `magnitude`          | 操作强度 (0-10) |
| `num_magnitude_bins` | 强度分箱数       |
| `interpolation`      | 插值方法        |
| `fill`               | 填充值         |

**AugMix 参数**：

| 参数              | 说明             |
| --------------- | -------------- |
| `severity`      | 增强强度 (1-10)    |
| `mixture_width` | 混合宽度           |
| `chain_depth`   | 链式深度           |
| `alpha`         | Dirichlet 分布参数 |

**操作集**：

* 几何：旋转、剪切、平移
* 颜色：亮度、对比度、饱和度、色调
* 其他：锐化、后处理、均衡化

---

### 4. `pipelines/preset_pipelines.py`

**功能**：预设增强管道

**分类任务预设**：

| 管道                      | 说明            |
| ----------------------- | ------------- |
| `ImageNetTrainPipeline` | ImageNet 训练管道 |
| `ImageNetValPipeline`   | ImageNet 验证管道 |
| `CIFARTrainPipeline`    | CIFAR 训练管道    |

**ImageNetTrainPipeline 流程**：

1. RandomResizedCrop(224)
2. RandomHorizontalFlip
3. RandAugment（可选）
4. ToTensor
5. Normalize
6. RandomErasing（可选）

**检测任务预设**：

| 管道                  | 说明          |
| ------------------- | ----------- |
| `COCOTrainPipeline` | COCO 训练管道   |
| `COCOValPipeline`   | COCO 验证管道   |
| `YOLOTrainPipeline` | YOLO 风格训练管道 |

**YOLOTrainPipeline 流程**：

1. Mosaic
2. RandomAffine
3. MixUp（可选）
4. RandomHSV
5. RandomFlip
6. LetterBox
7. ToTensor
8. Normalize

**分割任务预设**：

| 管道                 | 说明     |
| ------------------ | ------ |
| `SegTrainPipeline` | 分割训练管道 |
| `SegValPipeline`   | 分割验证管道 |

**SegTrainPipeline 流程**：

1. RandomResize
2. RandomCrop
3. RandomFlip
4. PhotoMetricDistortion
5. Normalize
6. Pad
7. ToTensor

---

## 模块对外接口

### 变换构建

```
# 单个变换
transform = build_transform(dict(type='RandomFlip', prob=0.5))

# 变换管道
pipeline = build_pipeline([
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToTensor'),
])

# 预设管道
pipeline = get_preset_pipeline('imagenet_train')
```

### 变换使用

```
# 单样本变换
results = pipeline(results)

# 数据结构
results = {
    'img': image,
    'bboxes': bboxes,
    'labels': labels,
    'masks': masks,
    'img_shape': (H, W, C),
    'ori_shape': (H, W, C),
}

# 应用变换
results = pipeline(results)

# 获取变换后数据
transformed_img = results['img']
transformed_bboxes = results['bboxes']
```

### 混合增强使用

```
# 混合增强需要在 DataLoader collate 阶段使用
mixup = Mixup(alpha=0.8, prob=0.5)

# 在 batch 级别应用
mixed_images, mixed_labels, lam = mixup(batch_images, batch_labels)
```

---

## 配置示例

### 分类任务配置

```yaml
train_pipeline:
  - type: LoadImage
  - type: RandomResizedCrop
    size: 224
    scale: [0.08, 1.0]
    ratio: [0.75, 1.333]
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: RandAugment
    num_ops: 2
    magnitude: 9
  - type: ToTensor
  - type: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  - type: RandomErasing
    prob: 0.25

val_pipeline:
  - type: LoadImage
  - type: Resize
    size: 256
  - type: CenterCrop
    size: 224
  - type: ToTensor
  - type: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### 检测任务配置

```yaml
train_pipeline:
  - type: LoadImage
  - type: LoadAnnotations
    with_bbox: true
  - type: Mosaic
    img_size: 640
    prob: 1.0
  - type: RandomAffine
    degrees: 0
    translate: 0.1
    scale: [0.5, 1.5]
    shear: 0
  - type: MixUp
    prob: 0.15
    alpha: 8.0
  - type: RandomHSV
    hue: 0.015
    saturation: 0.7
    value: 0.4
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: LetterBox
    size: [640, 640]
  - type: FilterAnnotations
    min_bbox_size: 2
  - type: ToTensor
  - type: Normalize
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]
  - type: PackInputs

val_pipeline:
  - type: LoadImage
  - type: LetterBox
    size: [640, 640]
  - type: ToTensor
  - type: Normalize
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]
  - type: PackInputs
```

### 分割任务配置

```yaml
train_pipeline:
  - type: LoadImage
  - type: LoadAnnotations
    with_seg: true
  - type: RandomResize
    scales: [0.5, 2.0]
    keep_ratio: true
  - type: RandomCrop
    size: [512, 512]
    cat_max_ratio: 0.75
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: PhotoMetricDistortion
    brightness: 0.125
    contrast: 0.5
    saturation: 0.5
    hue: 18
  - type: Normalize
    mean: [123.675, 116.28, 103.53]
    std: [58.395, 57.12, 57.375]
  - type: PadToDivisible
    size_divisor: 32
    seg_pad_val: 255
  - type: ToTensor
  - type: PackInputs

val_pipeline:
  - type: LoadImage
  - type: Resize
    size: [2048, 512]
    keep_ratio: true
  - type: Normalize
    mean: [123.675, 116.28, 103.53]
    std: [58.395, 57.12, 57.375]
  - type: ToTensor
  - type: PackInputs
```

### 自动增强配置

```yaml
# RandAugment
train_pipeline:
  - type: LoadImage
  - type: RandomResizedCrop
    size: 224
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: RandAugment
    num_ops: 2
    magnitude: 9
    num_magnitude_bins: 31
  - type: ToTensor
  - type: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# AugMix
train_pipeline:
  - type: LoadImage
  - type: RandomResizedCrop
    size: 224
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: AugMix
    severity: 3
    mixture_width: 3
    chain_depth: -1
    alpha: 1.0
  - type: ToTensor
  - type: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### 随机选择配置

```yaml
train_pipeline:
  - type: LoadImage
  - type: OneOf
    transforms:
      - type: RandomResizedCrop
        size: 224
        scale: [0.08, 1.0]
      - type: Compose
        transforms:
          - type: Resize
            size: 256
          - type: RandomCrop
            size: 224
    probs: [0.7, 0.3]
  - type: RandomHorizontalFlip
    prob: 0.5
  - type: SomeOf
    n: 2
    transforms:
      - type: RandomBrightness
        limit: 0.2
      - type: RandomContrast
        limit: 0.2
      - type: GaussianBlur
        kernel_size: 3
      - type: GaussianNoise
        std: 0.1
  - type: ToTensor
  - type: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

---

## 标注同步规范

### 边界框同步

| 变换类型   | 同步方式            |
| ------ | --------------- |
| Resize | 坐标按比例缩放         |
| Flip   | 坐标镜像翻转          |
| Rotate | 旋转后计算新外接矩形      |
| Crop   | 坐标偏移 + 裁剪越界处理   |
| Affine | 仿射矩阵变换 + 重算外接矩形 |
| Pad    | 坐标偏移            |

**越界处理策略**：

| 策略                 | 说明        |
| ------------------ | --------- |
| `clip`             | 裁剪到图像边界   |
| `filter`           | 过滤完全越界的目标 |
| `filter_by_area`   | 按剩余面积比例过滤 |
| `filter_by_center` | 中心点越界则过滤  |

### 掩码同步

| 变换类型   | 同步方式               |
| ------ | ------------------ |
| Resize | 最近邻插值缩放            |
| Flip   | 像素级翻转              |
| Rotate | 像素级旋转              |
| Crop   | 区域裁剪               |
| Affine | 仿射变换（最近邻）          |
| Pad    | 边界填充（ignore_index） |

### 关键点同步

| 变换类型   | 同步方式         |
| ------ | ------------ |
| Resize | 坐标按比例缩放      |
| Flip   | 坐标翻转 + 左右点交换 |
| Rotate | 坐标旋转变换       |
| Crop   | 坐标偏移 + 可见性更新 |
| Affine | 仿射矩阵变换       |

---

## 变换参数规范

### 概率参数

所有随机变换支持 `prob` 参数：

* 范围：`[0.0, 1.0]`
* 默认值：各变换不同
* `prob=0`：不应用
* `prob=1`：始终应用

### 范围参数

支持多种格式：

* 单值：`brightness=0.2` → 范围 `[-0.2, 0.2]`
* 元组：`brightness=(0.8, 1.2)` → 范围 `[0.8, 1.2]`
* 列表：`scales=[256, 288, 320]` → 随机选择

### 插值方法

| 方法         | 值 | 适用场景     |
| ---------- | - | -------- |
| `nearest`  | 0 | 分割掩码     |
| `bilinear` | 1 | 通用图像（默认） |
| `bicubic`  | 2 | 高质量缩放    |
| `lanczos`  | 3 | 下采样      |

### 填充模式

| 模式          | 说明       |
| ----------- | -------- |
| `constant`  | 常数填充（默认） |
| `edge`      | 边缘复制     |
| `reflect`   | 反射填充     |
| `symmetric` | 对称填充     |

---

## 性能优化

### 批处理优化

| 特性     | 说明               |
| ------ | ---------------- |
| 批处理变换  | 部分变换支持批处理加速      |
| GPU 加速 | Kornia 后端 GPU 变换 |
| 多进程    | DataLoader 多进程并行 |

### 内存优化

| 特性   | 说明      |
| ---- | ------- |
| 原地操作 | 尽可能原地修改 |
| 延迟加载 | 大图按需加载  |
| 缓存控制 | 可配置缓存策略 |

---

## 可视化调试

### 增强效果可视化

| 功能                                      | 说明         |
| --------------------------------------- | ---------- |
| `visualize_pipeline(pipeline, image)`   | 可视化管道各步骤效果 |
| `visualize_batch(pipeline, images, n)`  | 可视化多次增强结果  |
| `compare_transforms(transforms, image)` | 对比不同变换效果   |

### 调试工具

| 功能                           | 说明      |
| ---------------------------- | ------- |
| `pipeline.debug_mode = True` | 启用调试模式  |
| 中间结果保存                       | 保存各步骤结果 |
| 变换参数记录                       | 记录随机参数  |

---

## 依赖关系

**被依赖方**：

* Dataset - 数据加载时应用增强
* DataLoader - 批处理增强（Mixup/CutMix）
* Training Engine - 训练过程增强控制

**依赖项**：

* `Registry` - 组件注册与构建
* `numpy` - 数组操作
* `opencv-python` - 图像处理
* `albumentations`（可选）- 高性能增强
* `kornia`（可选）- GPU 加速增强

---

## 扩展指南

### 自定义变换

**步骤**：

1. 继承 `BaseTransform`
2. 实现 `transform(results)` 方法
3. 注册到 `TRANSFORMS` 注册器
4. 配置文件中使用

**要求**：

* 修改 `results` 字典并返回
* 标注同步处理
* 更新 `img_shape` 等元信息

### 自定义管道

**步骤**：

1. 继承 `Compose` 或创建新类
2. 定义变换序列
3. 注册为预设管道
4. 配置文件中引用
