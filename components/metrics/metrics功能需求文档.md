# Metrics 模块功能需求文档

## 模块概述

Metrics 模块是系统的评估指标库，提供分类、检测、分割三大任务的评估指标实现。采用累积式计算模式，支持分布式环境下的指标同步，通过注册器管理实现配置驱动的指标构建。

---

## 目录结构

```
metrics/
├── __init__.py
├── builder.py               # 指标构建器
├── base_metric.py           # 指标基类
│
├── classification/          # 分类指标
│   ├── __init__.py
│   ├── accuracy.py
│   ├── precision_recall.py
│   ├── f1_score.py
│   ├── confusion_matrix.py
│   ├── auc_roc.py
│   └── pr_curve.py
│
├── detection/               # 检测指标
│   ├── __init__.py
│   ├── mean_ap.py
│   ├── coco_metric.py
│   └── voc_metric.py
│
├── segmentation/            # 分割指标
│   ├── __init__.py
│   ├── iou.py
│   ├── dice_score.py
│   └── pixel_accuracy.py
│
└── utils/                   # 工具函数
    ├── __init__.py
    └── metric_utils.py
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_metric(config)` - 指标构建函数
* `build_metrics(configs)` - 批量构建
* `MetricCollection` - 指标集合类
* `BaseMetric` - 指标基类
* 所有具体指标类

---

### 2. `builder.py`

**功能**：指标构建器，配置驱动的指标实例化

**核心职责**：

* 根据配置构建单个指标
* 根据配置构建指标集合
* 自动处理指标参数
* 任务类型与指标匹配校验

**构建函数**：

| 函数                          | 功能                         |
| --------------------------- | -------------------------- |
| `build_metric(config)`      | 构建单个指标                     |
| `build_metrics(configs)`    | 构建多个指标，返回 MetricCollection |
| `get_default_metrics(task)` | 获取任务默认指标集                  |

**任务默认指标**：

| 任务             | 默认指标                                 |
| -------------- | ------------------------------------ |
| classification | Accuracy, Precision, Recall, F1Score |
| detection      | mAP, mAP50, mAP75                    |
| segmentation   | mIoU, Dice, PixelAccuracy            |

---

### 3. `base_metric.py`

**功能**：所有指标的抽象基类

**核心职责**：

* 定义指标统一接口
* 提供累积计算框架
* 支持分布式同步
* 状态管理

**基类接口**：

| 方法                       | 功能          |
| ------------------------ | ----------- |
| `update(preds, targets)` | 更新指标状态（单批次） |
| `compute()`              | 计算最终指标值     |
| `reset()`                | 重置指标状态      |
| `sync()`                 | 分布式状态同步     |
| `clone()`                | 克隆指标实例      |
| `to(device)`             | 移动到指定设备     |

**状态管理**：

* `add_state(name, default, dist_reduce_fx)` - 注册状态变量
* 状态自动跟踪与重置
* 分布式规约方式：sum / mean / cat / none

**计算模式**：

| 模式   | 说明                    |
| ---- | --------------------- |
| 累积模式 | 多次 update 后统一 compute |
| 即时模式 | 每次 update 返回当前批次结果    |

**分布式支持**：

* 自动检测分布式环境
* compute 前自动同步
* 支持自定义规约函数

---

## 二、classification/ 分类指标

### 1. `classification/__init__.py`

**功能**：导出所有分类指标

**导出内容**：

* 所有分类指标类
* 便捷函数

---

### 2. `classification/accuracy.py`

**功能**：准确率指标

**指标类型**：

| 类              | 说明        |
| -------------- | --------- |
| `Accuracy`     | 通用准确率     |
| `TopKAccuracy` | Top-K 准确率 |

**配置参数**：

| 参数            | 说明                                    |
| ------------- | ------------------------------------- |
| `topk`        | Top-K 值，支持多个如 (1, 5)                  |
| `threshold`   | 二分类阈值（默认 0.5）                         |
| `task`        | 任务类型：binary / multiclass / multilabel |
| `num_classes` | 类别数（多分类）                              |
| `average`     | 平均方式：micro / macro / weighted / none  |

**输出**：

* 单值：总体准确率
* 多值：各 Top-K 准确率字典

---

### 3. `classification/precision_recall.py`

**功能**：精确率与召回率

**指标类型**：

| 类             | 说明  |
| ------------- | --- |
| `Precision`   | 精确率 |
| `Recall`      | 召回率 |
| `Specificity` | 特异度 |

**配置参数**：

| 参数              | 说明                                    |
| --------------- | ------------------------------------- |
| `task`          | 任务类型：binary / multiclass / multilabel |
| `num_classes`   | 类别数                                   |
| `average`       | 平均方式：micro / macro / weighted / none  |
| `threshold`     | 分类阈值                                  |
| `ignore_index`  | 忽略的类别索引                               |
| `zero_division` | 除零处理：0 / 1 / warn                     |

**输出**：

* `average='none'`：各类别值列表
* 其他：单个聚合值

---

### 4. `classification/f1_score.py`

**功能**：F1 分数及变体

**指标类型**：

| 类            | 说明        |
| ------------ | --------- |
| `F1Score`    | F1 分数     |
| `FBetaScore` | F-Beta 分数 |

**配置参数**：

| 参数            | 说明                  |
| ------------- | ------------------- |
| `beta`        | Beta 值（F1 时 beta=1） |
| `task`        | 任务类型                |
| `num_classes` | 类别数                 |
| `average`     | 平均方式                |
| `threshold`   | 分类阈值                |

**输出**：

* 单值或各类别值列表

---

### 5. `classification/confusion_matrix.py`

**功能**：混淆矩阵

**指标类型**：

| 类                           | 说明      |
| --------------------------- | ------- |
| `ConfusionMatrix`           | 混淆矩阵    |
| `BinaryConfusionMatrix`     | 二分类混淆矩阵 |
| `MulticlassConfusionMatrix` | 多分类混淆矩阵 |
| `MultilabelConfusionMatrix` | 多标签混淆矩阵 |

**配置参数**：

| 参数            | 说明                             |
| ------------- | ------------------------------ |
| `num_classes` | 类别数                            |
| `normalize`   | 归一化方式：none / true / pred / all |
| `threshold`   | 分类阈值                           |

**输出**：

* 混淆矩阵张量 `[num_classes, num_classes]`
* 可提取：TP / TN / FP / FN

**扩展方法**：

| 方法                  | 功能           |
| ------------------- | ------------ |
| `plot()`            | 可视化混淆矩阵      |
| `get_tp_tn_fp_fn()` | 获取各统计量       |
| `to_dataframe()`    | 转为 DataFrame |

---

### 6. `classification/auc_roc.py`

**功能**：AUC-ROC 曲线指标

**指标类型**：

| 类          | 说明        |
| ---------- | --------- |
| `AUROC`    | ROC 曲线下面积 |
| `ROCCurve` | ROC 曲线数据  |

**配置参数**：

| 参数            | 说明                                   |
| ------------- | ------------------------------------ |
| `task`        | 任务类型                                 |
| `num_classes` | 类别数                                  |
| `average`     | 平均方式：micro / macro / weighted / none |
| `thresholds`  | 阈值数量或列表                              |
| `max_fpr`     | 部分 AUC 的最大 FPR                       |

**输出**：

* `AUROC`：AUC 值
* `ROCCurve`：(fpr, tpr, thresholds) 元组

---

### 7. `classification/pr_curve.py`

**功能**：PR 曲线指标

**指标类型**：

| 类                  | 说明       |
| ------------------ | -------- |
| `AveragePrecision` | 平均精度（AP） |
| `PRCurve`          | PR 曲线数据  |

**配置参数**：

| 参数            | 说明      |
| ------------- | ------- |
| `task`        | 任务类型    |
| `num_classes` | 类别数     |
| `average`     | 平均方式    |
| `thresholds`  | 阈值数量或列表 |

**输出**：

* `AveragePrecision`：AP 值
* `PRCurve`：(precision, recall, thresholds) 元组

---

## 三、detection/ 检测指标

### 1. `detection/__init__.py`

**功能**：导出所有检测指标

---

### 2. `detection/mean_ap.py`

**功能**：通用 mAP 计算

**指标类型**：

| 类        | 说明        |
| -------- | --------- |
| `MeanAP` | 通用 mAP 指标 |

**配置参数**：

| 参数                  | 说明                       |
| ------------------- | ------------------------ |
| `iou_thresholds`    | IoU 阈值列表，如 [0.5, 0.75]   |
| `class_names`       | 类别名称列表                   |
| `num_classes`       | 类别数                      |
| `box_format`        | 框格式：xyxy / xywh / cxcywh |
| `recall_thresholds` | 召回率采样点                   |
| `max_dets`          | 最大检测数量                   |

**输出指标**：

| 指标             | 说明              |
| -------------- | --------------- |
| `mAP`          | 所有 IoU 阈值的平均    |
| `mAP@0.5`      | IoU=0.5 时的 mAP  |
| `mAP@0.75`     | IoU=0.75 时的 mAP |
| `AP_per_class` | 各类别 AP          |
| `AR`           | 平均召回率           |

**输入格式**：

* 预测：`List[dict]`，每个 dict 含 boxes, scores, labels
* 真值：`List[dict]`，每个 dict 含 boxes, labels

---

### 3. `detection/coco_metric.py`

**功能**：COCO 标准评估指标

**指标类型**：

| 类                     | 说明        |
| --------------------- | --------- |
| `COCOMetric`          | COCO 官方评估 |
| `COCODetectionMetric` | 检测专用      |
| `COCOInstanceMetric`  | 实例分割专用    |

**配置参数**：

| 参数            | 说明                                |
| ------------- | --------------------------------- |
| `ann_file`    | COCO 标注文件路径（可选）                   |
| `metric`      | 评估类型：bbox / segm / keypoints      |
| `classwise`   | 是否输出各类别结果                         |
| `iou_thrs`    | IoU 阈值列表                          |
| `max_dets`    | 最大检测数量列表 [1, 10, 100]             |
| `area_ranges` | 面积范围：all / small / medium / large |

**输出指标**：

| 指标       | 说明               |
| -------- | ---------------- |
| `AP`     | IoU=0.50:0.95 平均 |
| `AP50`   | IoU=0.50         |
| `AP75`   | IoU=0.75         |
| `APs`    | 小目标 AP           |
| `APm`    | 中目标 AP           |
| `APl`    | 大目标 AP           |
| `AR@1`   | 最大 1 个检测的 AR     |
| `AR@10`  | 最大 10 个检测的 AR    |
| `AR@100` | 最大 100 个检测的 AR   |
| `ARs`    | 小目标 AR           |
| `ARm`    | 中目标 AR           |
| `ARl`    | 大目标 AR           |

**特性**：

* 兼容 pycocotools
* 支持直接传入标注或从文件加载
* 支持评估结果保存

---

### 4. `detection/voc_metric.py`

**功能**：Pascal VOC 评估指标

**指标类型**：

| 类           | 说明       |
| ----------- | -------- |
| `VOCMetric` | VOC 标准评估 |

**配置参数**：

| 参数              | 说明                |
| --------------- | ----------------- |
| `iou_threshold` | IoU 阈值（默认 0.5）    |
| `use_07_metric` | 是否使用 VOC07 11 点插值 |
| `class_names`   | 类别名称列表            |
| `difficult`     | 是否考虑 difficult 标记 |

**输出指标**：

| 指标             | 说明        |
| -------------- | --------- |
| `mAP`          | 所有类别平均 AP |
| `AP_per_class` | 各类别 AP 字典 |

**计算方式**：

* VOC07：11 点插值
* VOC10+：所有点插值（AUC）

---

## 四、segmentation/ 分割指标

### 1. `segmentation/__init__.py`

**功能**：导出所有分割指标

---

### 2. `segmentation/iou.py`

**功能**：IoU 相关指标

**指标类型**：

| 类              | 说明           |
| -------------- | ------------ |
| `IoU`          | 单类别 IoU      |
| `MeanIoU`      | 平均 IoU（mIoU） |
| `ClasswiseIoU` | 各类别 IoU      |

**配置参数**：

| 参数                   | 说明                            |
| -------------------- | ----------------------------- |
| `num_classes`        | 类别数                           |
| `ignore_index`       | 忽略的类别索引（如背景、边界）               |
| `average`            | 平均方式：micro / macro / weighted |
| `include_background` | 是否包含背景类                       |
| `reduce_labels`      | 是否将标签减 1（忽略 0）                |

**输出**：

| 指标              | 说明         |
| --------------- | ---------- |
| `mIoU`          | 所有类别平均 IoU |
| `IoU_per_class` | 各类别 IoU 列表 |
| `fwIoU`         | 频率加权 IoU   |

**输入格式**：

* 预测：分割掩码 `[B, H, W]` 或概率图 `[B, C, H, W]`
* 真值：标签掩码 `[B, H, W]`

---

### 3. `segmentation/dice_score.py`

**功能**：Dice 系数

**指标类型**：

| 类                 | 说明            |
| ----------------- | ------------- |
| `Dice`            | 单类别 Dice      |
| `MeanDice`        | 平均 Dice       |
| `GeneralizedDice` | 广义 Dice（类别加权） |

**配置参数**：

| 参数                   | 说明                             |
| -------------------- | ------------------------------ |
| `num_classes`        | 类别数                            |
| `ignore_index`       | 忽略的类别索引                        |
| `average`            | 平均方式                           |
| `include_background` | 是否包含背景类                        |
| `smooth`             | 平滑因子（避免除零）                     |
| `weight_type`        | 权重类型：uniform / square / simple |

**输出**：

* `Dice`：Dice 分数
* `Dice_per_class`：各类别 Dice

**特性**：

* 与 IoU 关系：Dice = 2 * IoU / (1 + IoU)
* 常用于医学图像分割

---

### 4. `segmentation/pixel_accuracy.py`

**功能**：像素级准确率

**指标类型**：

| 类                        | 说明        |
| ------------------------ | --------- |
| `PixelAccuracy`          | 全局像素准确率   |
| `MeanPixelAccuracy`      | 类别平均像素准确率 |
| `ClasswisePixelAccuracy` | 各类别像素准确率  |

**配置参数**：

| 参数              | 说明       |
| --------------- | -------- |
| `num_classes`   | 类别数      |
| `ignore_index`  | 忽略的类别索引  |
| `reduce_labels` | 是否将标签减 1 |

**输出**：

| 指标                   | 说明         |
| -------------------- | ---------- |
| `pixel_accuracy`     | 正确像素 / 总像素 |
| `mean_accuracy`      | 各类别准确率平均   |
| `accuracy_per_class` | 各类别准确率     |

---

## 五、utils/ 工具函数

### 1. `utils/__init__.py`

**功能**：导出所有工具函数

---

### 2. `utils/metric_utils.py`

**功能**：指标计算辅助工具

**工具函数**：

| 函数                                            | 功能           |
| --------------------------------------------- | ------------ |
| `box_iou(boxes1, boxes2)`                     | 计算框 IoU 矩阵   |
| `generalized_box_iou(boxes1, boxes2)`         | 计算 GIoU 矩阵   |
| `mask_iou(masks1, masks2)`                    | 计算掩码 IoU     |
| `nms(boxes, scores, threshold)`               | 非极大值抑制       |
| `soft_nms(boxes, scores, sigma)`              | 软 NMS        |
| `convert_box_format(boxes, from_fmt, to_fmt)` | 框格式转换        |
| `interpolate_pr_curve(precision, recall)`     | PR 曲线插值      |
| `compute_ap_from_pr(precision, recall)`       | 从 PR 曲线计算 AP |

**分布式工具**：

| 函数                          | 功能          |
| --------------------------- | ----------- |
| `all_gather_object(obj)`    | 收集所有进程的对象   |
| `reduce_tensor(tensor, op)` | 规约张量        |
| `sync_metric_state(states)` | 同步指标状态      |
| `is_distributed()`          | 检查是否分布式环境   |
| `get_world_size()`          | 获取进程数       |
| `get_rank()`                | 获取当前进程 rank |

**统计工具**：

| 函数                               | 功能            |
| -------------------------------- | ------------- |
| `safe_divide(a, b, zero_value)`  | 安全除法          |
| `to_onehot(labels, num_classes)` | 转为 one-hot 编码 |
| `flatten_dict(d, sep)`           | 展平嵌套字典        |
| `select_topk(scores, k)`         | 选择 Top-K      |

---

## 指标集合类

### `MetricCollection`

**功能**：管理多个指标的集合

**核心职责**：

* 统一管理多个指标
* 批量 update / compute / reset
* 前缀/后缀管理
* 避免重复计算共享状态

**接口**：

| 方法                       | 功能      |
| ------------------------ | ------- |
| `add_metrics(metrics)`   | 添加指标    |
| `update(preds, targets)` | 更新所有指标  |
| `compute()`              | 计算所有指标  |
| `reset()`                | 重置所有指标  |
| `clone(prefix)`          | 克隆并添加前缀 |
| `items()`                | 遍历指标    |
| `keys()`                 | 获取指标名列表 |

**配置参数**：

| 参数               | 说明         |
| ---------------- | ---------- |
| `metrics`        | 指标字典或列表    |
| `prefix`         | 指标名前缀      |
| `postfix`        | 指标名后缀      |
| `compute_groups` | 是否自动分组共享计算 |

---

## 模块对外接口

### 指标构建

```
# 单个指标
metric = build_metric(dict(type='Accuracy', topk=(1, 5)))

# 多个指标
metrics = build_metrics([
    dict(type='Accuracy'),
    dict(type='F1Score', average='macro'),
    dict(type='ConfusionMatrix', num_classes=10)
])

# 任务默认指标
metrics = get_default_metrics('classification')
```

### 指标使用

```
# 累积更新
for batch in dataloader:
    preds, targets = model(batch)
    metric.update(preds, targets)

# 计算结果
results = metric.compute()

# 重置状态
metric.reset()
```

### 分布式使用

```
# 自动同步
metric = Accuracy(sync_on_compute=True)

# 手动同步
metric.sync()
results = metric.compute()
```

---

## 配置示例

```yaml
metrics:
  train:
    - type: Accuracy
      topk: [1, 5]
  
  val:
    - type: Accuracy
      topk: [1, 5]
    - type: Precision
      average: macro
      num_classes: 10
    - type: Recall
      average: macro
      num_classes: 10
    - type: F1Score
      average: macro
      num_classes: 10
    - type: ConfusionMatrix
      num_classes: 10
```

```yaml
# 检测任务
metrics:
  val:
    - type: COCOMetric
      metric: bbox
      classwise: true
    - type: MeanAP
      iou_thresholds: [0.5, 0.75]
```

```yaml
# 分割任务
metrics:
  val:
    - type: MeanIoU
      num_classes: 21
      ignore_index: 255
    - type: MeanDice
      num_classes: 21
    - type: PixelAccuracy
      num_classes: 21
```

---

## 依赖关系

**被依赖方**：

* Training Engine - 训练过程评估
* Validation Pipeline - 验证评估
* Report Generator - 生成评估报告

**依赖项**：

* `Registry` - 组件注册与构建
* `torch.distributed` - 分布式同步
* `pycocotools`（可选）- COCO 评估
