# Datasets 模块功能需求文档

## 模块概述

Datasets 模块是系统的数据集组件库，提供多种标准数据集格式支持、标注解析、数据采样等功能。支持 COCO、VOC、YOLO、ImageNet 等主流格式，提供分布式采样、类别平衡采样等策略，通过注册器管理实现配置驱动的数据集构建。

---

## 目录结构

```
datasets/
├── __init__.py              # 模块初始化
├── builder.py               # 数据集构建器
├── base_dataset.py          # 数据集基类
│
├── formats/                 # 数据集格式
│   ├── __init__.py
│   ├── coco.py              # COCO 格式
│   ├── voc.py               # VOC 格式
│   ├── yolo_format.py       # YOLO 格式
│   ├── imagenet.py          # ImageNet 格式
│   └── custom.py            # 自定义格式
│
├── parsers/                 # 标注解析器
│   ├── __init__.py
│   ├── annotation_parser.py # 解析器基类
│   ├── coco_parser.py       # COCO 解析器
│   ├── voc_parser.py        # VOC 解析器
│   └── yolo_parser.py       # YOLO 解析器
│
├── samplers/                # 数据采样器
│   ├── __init__.py
│   ├── distributed_sampler.py   # 分布式采样
│   ├── balanced_sampler.py      # 平衡采样
│   └── repeat_sampler.py        # 重复采样
│
└── utils/                   # 工具函数
    ├── __init__.py
    ├── collate.py           # 批处理整理
    ├── prefetcher.py        # 数据预取
    └── data_utils.py        # 通用工具
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_dataset(config)` - 数据集构建
* `build_dataloader(dataset, config)` - 数据加载器构建
* `DATASETS` - 数据集注册器
* `SAMPLERS` - 采样器注册器
* 所有数据集类、采样器类

---

### 2. `builder.py`

**功能**：数据集与数据加载器构建器

**核心职责**：

* 根据配置构建数据集
* 根据配置构建数据加载器
* 自动选择采样器
* 处理分布式场景

**构建函数**：

| 函数                                  | 功能      |
| ----------------------------------- | ------- |
| `build_dataset(config)`             | 构建数据集   |
| `build_dataloader(dataset, config)` | 构建数据加载器 |
| `build_sampler(dataset, config)`    | 构建采样器   |

**数据集配置参数**：

| 参数          | 说明     | 默认值   |
| ----------- | ------ | ----- |
| `type`      | 数据集类型  | 必填    |
| `data_root` | 数据根目录  | 必填    |
| `ann_file`  | 标注文件路径 | None  |
| `pipeline`  | 数据增强管道 | []    |
| `test_mode` | 是否测试模式 | False |

**数据加载器配置参数**：

| 参数                   | 说明      | 默认值   |
| -------------------- | ------- | ----- |
| `batch_size`         | 批大小     | 必填    |
| `num_workers`        | 工作进程数   | 4     |
| `shuffle`            | 是否打乱    | True  |
| `drop_last`          | 丢弃不完整批次 | False |
| `pin_memory`         | 锁页内存    | True  |
| `persistent_workers` | 持久化工作进程 | True  |
| `prefetch_factor`    | 预取因子    | 2     |
| `sampler`            | 采样器配置   | None  |
| `collate_fn`         | 整理函数    | None  |

---

### 3. `base_dataset.py`

**功能**：所有数据集的抽象基类

**核心职责**：

* 定义数据集统一接口
* 数据加载框架
* 缓存机制
* 数据增强管道应用

**基类接口**：

| 方法                           | 功能         |
| ---------------------------- | ---------- |
| `__len__()`                  | 返回数据集大小    |
| `__getitem__(idx)`           | 获取单个样本     |
| `load_annotations()`         | 加载标注（子类实现） |
| `prepare_data(idx)`          | 准备数据       |
| `get_ann_info(idx)`          | 获取标注信息     |
| `evaluate(results, metrics)` | 评估结果       |

**数据结构**：

`data_info` 字典包含：

| 键           | 类型      | 说明     |
| ----------- | ------- | ------ |
| `img_path`  | str     | 图像路径   |
| `img_id`    | int/str | 图像 ID  |
| `height`    | int     | 图像高度   |
| `width`     | int     | 图像宽度   |
| `instances` | list    | 实例标注列表 |

`instance` 字典包含：

| 键             | 类型           | 说明                   |
| ------------- | ------------ | -------------------- |
| `bbox`        | list         | 边界框 [x1, y1, x2, y2] |
| `bbox_label`  | int          | 类别标签                 |
| `mask`        | ndarray/dict | 分割掩码                 |
| `keypoints`   | list         | 关键点                  |
| `ignore_flag` | bool         | 是否忽略                 |

**缓存机制**：

| 参数               | 说明      | 默认值   |
| ---------------- | ------- | ----- |
| `serialize_data` | 是否序列化缓存 | True  |
| `lazy_init`      | 是否延迟初始化 | False |
| `max_refetch`    | 最大重试次数  | 1000  |

**过滤策略**：

| 方法              | 功能     |
| --------------- | ------ |
| `filter_data()` | 过滤无效数据 |

过滤条件：

* 无标注的图像（可配置保留）
* 损坏的图像
* 小于最小尺寸的图像

---

## 二、formats/ 数据集格式

### 1. `formats/__init__.py`

**功能**：导出所有数据集格式类

---

### 2. `formats/coco.py`

**功能**：COCO 格式数据集

**数据集类型**：

| 类                     | 说明           |
| --------------------- | ------------ |
| `CocoDataset`         | COCO 目标检测数据集 |
| `CocoSegDataset`      | COCO 实例分割数据集 |
| `CocoPanopticDataset` | COCO 全景分割数据集 |
| `CocoKeypointDataset` | COCO 关键点数据集  |

**CocoDataset 配置参数**：

| 参数                | 说明         | 默认值  |
| ----------------- | ---------- | ---- |
| `data_root`       | 数据根目录      | 必填   |
| `ann_file`        | 标注 JSON 文件 | 必填   |
| `img_prefix`      | 图像目录前缀     | ''   |
| `seg_prefix`      | 分割图目录前缀    | None |
| `filter_empty_gt` | 过滤空标注      | True |
| `min_size`        | 最小图像尺寸     | 32   |

**目录结构**：

```
data_root/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

**评估指标**：

* mAP (IoU=0.5:0.95)
* AP50, AP75
* AP_small, AP_medium, AP_large
* AR@1, AR@10, AR@100

---

### 3. `formats/voc.py`

**功能**：Pascal VOC 格式数据集

**数据集类型**：

| 类               | 说明          |
| --------------- | ----------- |
| `VOCDataset`    | VOC 目标检测数据集 |
| `VOCSegDataset` | VOC 语义分割数据集 |

**VOCDataset 配置参数**：

| 参数           | 说明              | 默认值                 |
| ------------ | --------------- | ------------------- |
| `data_root`  | 数据根目录           | 必填                  |
| `ann_file`   | 划分文件（train.txt） | 必填                  |
| `img_prefix` | 图像目录            | 'JPEGImages'        |
| `ann_prefix` | 标注目录            | 'Annotations'       |
| `seg_prefix` | 分割图目录           | 'SegmentationClass' |
| `year`       | 数据集年份           | 2012                |

**目录结构**：

```
data_root/
├── VOC2012/
│   ├── JPEGImages/
│   ├── Annotations/
│   ├── SegmentationClass/
│   └── ImageSets/
│       └── Main/
│           ├── train.txt
│           └── val.txt
```

**评估指标**：

* mAP (IoU=0.5)
* 各类别 AP

---

### 4. `formats/yolo_format.py`

**功能**：YOLO 格式数据集

**数据集类型**：

| 类                | 说明           |
| ---------------- | ------------ |
| `YOLODataset`    | YOLO 目标检测数据集 |
| `YOLOSegDataset` | YOLO 实例分割数据集 |

**YOLODataset 配置参数**：

| 参数          | 说明      | 默认值      |
| ----------- | ------- | -------- |
| `data_root` | 数据根目录   | 必填       |
| `img_dir`   | 图像目录    | 'images' |
| `label_dir` | 标签目录    | 'labels' |
| `split`     | 数据划分    | 'train'  |
| `classes`   | 类别列表或文件 | 必填       |

**目录结构**：

```
data_root/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── classes.txt
```

**标签格式**：

```
# 每行：class_id x_center y_center width height
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.2
```

**分割标签格式**：

```
# 每行：class_id x1 y1 x2 y2 ... xn yn
0 0.1 0.2 0.3 0.4 0.5 0.6 ...
```

---

### 5. `formats/imagenet.py`

**功能**：ImageNet 格式分类数据集

**数据集类型**：

| 类                 | 说明             |
| ----------------- | -------------- |
| `ImageNetDataset` | ImageNet 分类数据集 |
| `FolderDataset`   | 通用文件夹数据集       |

**ImageNetDataset 配置参数**：

| 参数           | 说明               | 默认值     |
| ------------ | ---------------- | ------- |
| `data_root`  | 数据根目录            | 必填      |
| `split`      | 数据划分：train / val | 'train' |
| `with_label` | 是否包含标签           | True    |

**FolderDataset 配置参数**：

| 参数           | 说明     | 默认值              |
| ------------ | ------ | ---------------- |
| `data_root`  | 数据根目录  | 必填               |
| `extensions` | 图像扩展名  | ['.jpg', '.png'] |
| `class_map`  | 类别映射文件 | None             |

**目录结构**：

```
data_root/
├── train/
│   ├── class_0/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_1/
└── val/
    ├── class_0/
    └── class_1/
```

**评估指标**：

* Top-1 Accuracy
* Top-5 Accuracy

---

### 6. `formats/custom.py`

**功能**：自定义格式数据集

**数据集类型**：

| 类               | 说明         |
| --------------- | ---------- |
| `CustomDataset` | 可扩展自定义数据集  |
| `CSVDataset`    | CSV 标注数据集  |
| `JSONDataset`   | JSON 标注数据集 |

**CustomDataset 配置参数**：

| 参数           | 说明    | 默认值  |
| ------------ | ----- | ---- |
| `data_root`  | 数据根目录 | 必填   |
| `ann_file`   | 标注文件  | 必填   |
| `img_prefix` | 图像前缀  | ''   |
| `classes`    | 类别列表  | None |
| `parser`     | 解析器配置 | None |

**CSVDataset 标注格式**：

```csv
image_path,x1,y1,x2,y2,class
img1.jpg,100,100,200,200,cat
img1.jpg,300,150,400,250,dog
```

**JSONDataset 标注格式**：

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

---

## 三、parsers/ 标注解析器

### 1. `parsers/__init__.py`

**功能**：导出所有解析器类

---

### 2. `parsers/annotation_parser.py`

**功能**：标注解析器基类

**基类接口**：

| 方法                           | 功能      |
| ---------------------------- | ------- |
| `parse(ann_file)`            | 解析标注文件  |
| `parse_annotation(ann_info)` | 解析单条标注  |
| `get_classes()`              | 获取类别列表  |
| `convert_to_standard(data)`  | 转换为标准格式 |

**标准格式**：

```python
{
    'img_id': str/int,
    'img_path': str,
    'height': int,
    'width': int,
    'instances': [
        {
            'bbox': [x1, y1, x2, y2],
            'bbox_label': int,
            'mask': ...,
            'ignore_flag': bool
        }
    ]
}
```

---

### 3. `parsers/coco_parser.py`

**功能**：COCO 格式解析器

**解析器类型**：

| 类            | 说明         |
| ------------ | ---------- |
| `CocoParser` | COCO 标注解析器 |

**解析内容**：

| 字段             | 来源                         |
| -------------- | -------------------------- |
| `img_id`       | images.id                  |
| `img_path`     | images.file_name           |
| `height`       | images.height              |
| `width`        | images.width               |
| `bbox`         | annotations.bbox (转为 xyxy) |
| `category_id`  | annotations.category_id    |
| `segmentation` | annotations.segmentation   |
| `area`         | annotations.area           |
| `iscrowd`      | annotations.iscrowd        |

**RLE 解码**：

* 支持 polygon 格式
* 支持 RLE 压缩格式
* 支持 COCO RLE 格式

---

### 4. `parsers/voc_parser.py`

**功能**：VOC XML 格式解析器

**解析器类型**：

| 类           | 说明          |
| ----------- | ----------- |
| `VocParser` | VOC XML 解析器 |

**解析内容**：

| 字段           | XML 路径                      |
| ------------ | --------------------------- |
| `img_path`   | annotation/filename         |
| `height`     | annotation/size/height      |
| `width`      | annotation/size/width       |
| `bbox`       | annotation/object/bndbox    |
| `class_name` | annotation/object/name      |
| `difficult`  | annotation/object/difficult |
| `truncated`  | annotation/object/truncated |

---

### 5. `parsers/yolo_parser.py`

**功能**：YOLO TXT 格式解析器

**解析器类型**：

| 类            | 说明         |
| ------------ | ---------- |
| `YoloParser` | YOLO 标注解析器 |

**解析内容**：

| 字段         | 说明         |
| ---------- | ---------- |
| `class_id` | 第 0 列      |
| `x_center` | 第 1 列（归一化） |
| `y_center` | 第 2 列（归一化） |
| `width`    | 第 3 列（归一化） |
| `height`   | 第 4 列（归一化） |
| `polygon`  | 第 5+ 列（分割） |

**坐标转换**：

* YOLO (cx, cy, w, h) → 标准 (x1, y1, x2, y2)
* 归一化坐标 → 绝对坐标

---

## 四、samplers/ 数据采样器

### 1. `samplers/__init__.py`

**功能**：导出所有采样器类

---

### 2. `samplers/distributed_sampler.py`

**功能**：分布式数据采样

**采样器类型**：

| 类                    | 说明     |
| -------------------- | ------ |
| `DistributedSampler` | 分布式采样器 |
| `InfiniteSampler`    | 无限采样器  |

**DistributedSampler 配置参数**：

| 参数             | 说明        | 默认值      |
| -------------- | --------- | -------- |
| `dataset`      | 数据集       | 必填       |
| `num_replicas` | 进程数       | None（自动） |
| `rank`         | 当前进程 rank | None（自动） |
| `shuffle`      | 是否打乱      | True     |
| `seed`         | 随机种子      | 0        |
| `drop_last`    | 丢弃不完整     | False    |

**数据划分**：

* 每个进程获取 `len(dataset) // num_replicas` 个样本
* 自动处理不整除情况
* 每个 epoch 自动设置不同随机种子

**InfiniteSampler 配置参数**：

| 参数        | 说明   | 默认值  |
| --------- | ---- | ---- |
| `dataset` | 数据集  | 必填   |
| `shuffle` | 是否打乱 | True |
| `seed`    | 随机种子 | 0    |

**特点**：

* 无限迭代，不以 epoch 为单位
* 适合按迭代数训练
* 自动处理分布式

---

### 3. `samplers/balanced_sampler.py`

**功能**：类别平衡采样

**采样器类型**：

| 类                         | 说明      |
| ------------------------- | ------- |
| `ClassBalancedSampler`    | 类别平衡采样  |
| `InstanceBalancedSampler` | 实例平衡采样  |
| `SquareRootSampler`       | 平方根平衡采样 |

**ClassBalancedSampler 配置参数**：

| 参数               | 说明    | 默认值  |
| ---------------- | ----- | ---- |
| `dataset`        | 数据集   | 必填   |
| `oversample_thr` | 过采样阈值 | 0.01 |
| `shuffle`        | 是否打乱  | True |

**采样权重计算**：

```
类别频率 f_c = 类别样本数 / 总样本数
重复因子 r_c = max(1, sqrt(阈值 / f_c))
```

**SquareRootSampler**：

* 样本权重与类别频率的平方根成反比
* 在均匀采样和实例平衡之间折中

---

### 4. `samplers/repeat_sampler.py`

**功能**：重复采样策略

**采样器类型**：

| 类                     | 说明      |
| --------------------- | ------- |
| `RepeatSampler`       | 固定重复采样  |
| `RepeatFactorSampler` | 因子重复采样  |
| `AspectRatioSampler`  | 宽高比分组采样 |

**RepeatSampler 配置参数**：

| 参数             | 说明   | 默认值  |
| -------------- | ---- | ---- |
| `dataset`      | 数据集  | 必填   |
| `repeat_times` | 重复次数 | 1    |
| `shuffle`      | 是否打乱 | True |

**RepeatFactorSampler 配置参数**：

| 参数           | 说明   | 默认值   |
| ------------ | ---- | ----- |
| `dataset`    | 数据集  | 必填    |
| `repeat_thr` | 重复阈值 | 0.001 |
| `shuffle`    | 是否打乱 | True  |

**重复因子计算**：

```
图像类别频率 f_i = 图像中类别的最大频率
重复因子 r_i = max(1, sqrt(阈值 / f_i))
```

**AspectRatioSampler 配置参数**：

| 参数                 | 说明    | 默认值 |
| ------------------ | ----- | --- |
| `dataset`          | 数据集   | 必填  |
| `batch_size`       | 批大小   | 必填  |
| `aspect_ratio_thr` | 宽高比阈值 | 1.0 |

**分组策略**：

* 按宽高比分组，同组图像一起批处理
* 减少 padding 开销
* 适合检测任务

---

## 五、utils/ 工具函数

### 1. `utils/__init__.py`

**功能**：导出所有工具函数

---

### 2. `utils/collate.py`

**功能**：批处理整理函数

**整理函数**：

| 函数                                         | 说明       |
| ------------------------------------------ | -------- |
| `default_collate(batch)`                   | 默认整理（分类） |
| `detection_collate(batch)`                 | 检测整理     |
| `segmentation_collate(batch)`              | 分割整理     |
| `multi_scale_collate(batch, size_divisor)` | 多尺度整理    |

**detection_collate 处理**：

* 图像 padding 到同一尺寸
* bbox 列表整理
* 返回 batch 索引

**输出结构**：

| 键           | 类型         | 说明               |
| ----------- | ---------- | ---------------- |
| `images`    | Tensor     | 批图像 [B, C, H, W] |
| `targets`   | list[dict] | 目标列表             |
| `img_metas` | list[dict] | 图像元信息            |

**segmentation_collate 处理**：

* 图像 padding
* 掩码 padding（使用 ignore_index）
* 保持对应关系

---

### 3. `utils/prefetcher.py`

**功能**：数据预取加速

**预取器类型**：

| 类                     | 说明        |
| --------------------- | --------- |
| `DataPrefetcher`      | CUDA 数据预取 |
| `BackgroundGenerator` | 后台生成器     |

**DataPrefetcher 配置参数**：

| 参数          | 说明     | 默认值    |
| ----------- | ------ | ------ |
| `loader`    | 数据加载器  | 必填     |
| `device`    | 目标设备   | 'cuda' |
| `normalize` | 是否归一化  | True   |
| `mean`      | 归一化均值  | None   |
| `std`       | 归一化标准差 | None   |

**预取流程**：

1. 后台线程加载下一批数据
2. 异步传输到 GPU
3. 可选在 GPU 上归一化
4. 训练时直接使用

**优化效果**：

* 隐藏数据传输延迟
* CPU-GPU 并行
* 减少训练空闲时间

---

### 4. `utils/data_utils.py`

**功能**：通用数据工具函数

**工具函数**：

| 函数                                         | 功能      |
| ------------------------------------------ | ------- |
| `get_classes(dataset_type)`                | 获取数据集类别 |
| `get_palette(dataset_type)`                | 获取调色板   |
| `read_image(path, backend)`                | 读取图像    |
| `verify_image(path)`                       | 验证图像有效性 |
| `compute_dataset_stats(dataset)`           | 计算数据集统计 |
| `split_dataset(dataset, ratios)`           | 划分数据集   |
| `merge_datasets(datasets)`                 | 合并数据集   |
| `convert_dataset_format(src, dst, format)` | 格式转换    |

**预定义类别**：

| 数据集        | 类别数  |
| ---------- | ---- |
| COCO       | 80   |
| VOC        | 20   |
| ImageNet   | 1000 |
| Cityscapes | 19   |
| ADE20K     | 150  |

**数据集统计**：

* 类别分布
* 图像尺寸分布
* bbox 尺寸分布
* 每图实例数分布

---

## 配置示例

### COCO 检测数据集

```yaml
train_dataset:
  type: CocoDataset
  data_root: data/coco
  ann_file: annotations/instances_train2017.json
  img_prefix: train2017
  filter_empty_gt: true
  pipeline:
    - type: LoadImage
    - type: LoadAnnotations
      with_bbox: true
    - type: RandomResize
      scales: [[480, 1333], [800, 1333]]
      keep_ratio: true
    - type: RandomFlip
      prob: 0.5
    - type: Normalize
      mean: [123.675, 116.28, 103.53]
      std: [58.395, 57.12, 57.375]
    - type: Pad
      size_divisor: 32
    - type: PackInputs
```

### ImageNet 分类数据集

```yaml
train_dataset:
  type: ImageNetDataset
  data_root: data/imagenet
  split: train
  pipeline:
    - type: LoadImage
    - type: RandomResizedCrop
      size: 224
    - type: RandomHorizontalFlip
    - type: RandAugment
      num_ops: 2
      magnitude: 9
    - type: ToTensor
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

### YOLO 格式数据集

```yaml
train_dataset:
  type: YOLODataset
  data_root: data/custom
  img_dir: images/train
  label_dir: labels/train
  classes: classes.txt
  pipeline:
    - type: LoadImage
    - type: LoadAnnotations
    - type: Mosaic
      img_size: 640
    - type: RandomAffine
    - type: LetterBox
      size: [640, 640]
    - type: PackInputs
```

### 数据加载器配置

```yaml
train_dataloader:
  batch_size: 16
  num_workers: 8
  shuffle: true
  pin_memory: true
  persistent_workers: true
  sampler:
    type: DistributedSampler
    shuffle: true
  collate_fn:
    type: detection_collate
```

### 类别平衡采样

```yaml
train_dataloader:
  batch_size: 16
  num_workers: 4
  sampler:
    type: ClassBalancedSampler
    oversample_thr: 0.01
```

### 重复采样配置

```yaml
train_dataloader:
  batch_size: 16
  num_workers: 4
  sampler:
    type: RepeatFactorSampler
    repeat_thr: 0.001
```

### 数据预取配置

```yaml
train_dataloader:
  batch_size: 16
  num_workers: 4
  prefetcher:
    type: DataPrefetcher
    device: cuda
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

---

## 数据集格式对照表

| 格式       | 标注类型     | 任务类型      | 标注存储  |
| -------- | -------- | --------- | ----- |
| COCO     | JSON     | 检测/分割/关键点 | 单文件   |
| VOC      | XML      | 检测/分割     | 每图一文件 |
| YOLO     | TXT      | 检测/分割     | 每图一文件 |
| ImageNet | 目录结构     | 分类        | 无单独标注 |
| Custom   | CSV/JSON | 自定义       | 单文件   |

---

## 采样器选择指南

| 场景    | 推荐采样器                | 说明         |
| ----- | -------------------- | ---------- |
| 单卡训练  | RandomSampler        | 默认随机采样     |
| 多卡训练  | DistributedSampler   | 自动数据划分     |
| 类别不平衡 | ClassBalancedSampler | 过采样少数类     |
| 长尾分布  | RepeatFactorSampler  | 按频率重复      |
| 按迭代训练 | InfiniteSampler      | 无限迭代       |
| 检测任务  | AspectRatioSampler   | 减少 padding |

---

## 依赖关系

**被依赖方**：

* Training Engine - 训练数据加载
* Evaluation - 评估数据加载
* Transforms - 数据增强应用

**依赖项**：

* `Registry` - 组件注册与构建
* `Transforms` - 数据增强管道
* `torch.utils.data` - PyTorch 数据加载
