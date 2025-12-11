# Compression 模块功能需求文档

## 模块概述

Compression 模块是系统的模型压缩组件库，提供量化、剪枝、知识蒸馏三大压缩技术。支持训练后量化与量化感知训练、结构化与非结构化剪枝、多种蒸馏策略，通过注册器管理实现配置驱动的压缩流程。

---

## 目录结构

```
compression/
├── __init__.py              # 模块初始化
├── builder.py               # 压缩组件构建器
│
├── quantization/            # 量化
│   ├── __init__.py
│   ├── ptq.py               # 训练后量化
│   ├── qat.py               # 量化感知训练
│   ├── calibrator.py        # 校准器
│   └── quant_utils.py       # 量化工具
│
├── pruning/                 # 剪枝
│   ├── __init__.py
│   ├── magnitude_pruner.py  # 幅值剪枝
│   ├── structured_pruner.py # 结构化剪枝
│   ├── sensitivity_analyzer.py  # 敏感度分析
│   └── pruning_scheduler.py # 剪枝调度
│
└── distillation/            # 知识蒸馏
    ├── __init__.py
    ├── base_distiller.py    # 蒸馏基类
    ├── feature_distiller.py # 特征蒸馏
    ├── logit_distiller.py   # Logit 蒸馏
    └── relation_distiller.py # 关系蒸馏
```

---

## 一、根目录文件

### 1. `__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_quantizer(config)` - 量化器构建
* `build_pruner(config)` - 剪枝器构建
* `build_distiller(config)` - 蒸馏器构建
* 所有压缩组件类
* 注册器

---

### 2. `builder.py`

**功能**：压缩组件统一构建器

**核心职责**：

* 根据配置构建各类压缩组件
* 组合多种压缩策略
* 验证配置合法性

**构建函数**：

| 函数                                          | 功能      |
| ------------------------------------------- | ------- |
| `build_quantizer(config, model)`            | 构建量化器   |
| `build_pruner(config, model)`               | 构建剪枝器   |
| `build_distiller(config, teacher, student)` | 构建蒸馏器   |
| `build_compression_pipeline(config, model)` | 构建压缩流水线 |

**压缩流水线**：

* 支持多种压缩技术组合
* 按顺序执行：剪枝 → 蒸馏 → 量化
* 可配置各阶段参数

---

## 二、quantization/ 量化模块

### 1. `quantization/__init__.py`

**功能**：导出量化相关组件

**导出内容**：

* PTQ / QAT 量化器
* 校准器
* 量化工具函数

---

### 2. `quantization/ptq.py`

**功能**：训练后量化（Post-Training Quantization）

**量化器类型**：

| 类                  | 说明         |
| ------------------ | ---------- |
| `PTQQuantizer`     | 通用 PTQ 量化器 |
| `DynamicQuantizer` | 动态量化器      |
| `StaticQuantizer`  | 静态量化器      |

**PTQQuantizer 配置参数**：

| 参数                        | 说明                             | 默认值         |
| ------------------------- | ------------------------------ | ----------- |
| `backend`                 | 后端：fbgemm / qnnpack / tensorrt | 'fbgemm'    |
| `dtype`                   | 量化类型：int8 / uint8 / fp16       | 'int8'      |
| `per_channel`             | 是否逐通道量化                        | True        |
| `symmetric`               | 是否对称量化                         | True        |
| `calibration_method`      | 校准方法                           | 'histogram' |
| `num_calibration_batches` | 校准批次数                          | 100         |

**量化流程**：

1. 模型准备（插入量化节点）
2. 校准（收集激活统计）
3. 转换（生成量化模型）

**支持的层**：

* Conv2d / Linear
* BatchNorm（融合到 Conv）
* ReLU / ReLU6
* Add / Concat

**接口方法**：

| 方法                             | 功能      |
| ------------------------------ | ------- |
| `prepare(model)`               | 准备模型    |
| `calibrate(model, dataloader)` | 校准模型    |
| `convert(model)`               | 转换为量化模型 |
| `quantize(model, dataloader)`  | 一键量化    |

---

### 3. `quantization/qat.py`

**功能**：量化感知训练（Quantization-Aware Training）

**量化器类型**：

| 类              | 说明         |
| -------------- | ---------- |
| `QATQuantizer` | 通用 QAT 量化器 |
| `LSQQuantizer` | LSQ 可学习量化  |

**QATQuantizer 配置参数**：

| 参数                 | 说明            | 默认值         |
| ------------------ | ------------- | ----------- |
| `backend`          | 量化后端          | 'fbgemm'    |
| `dtype`            | 量化类型          | 'int8'      |
| `per_channel`      | 逐通道量化         | True        |
| `symmetric`        | 对称量化          | True        |
| `freeze_bn_epochs` | BN 冻结起始 epoch | -1          |
| `observer`         | 观察器类型         | 'histogram' |

**LSQQuantizer 配置参数**：

| 参数            | 说明     | 默认值  |
| ------------- | ------ | ---- |
| `bits`        | 量化位宽   | 8    |
| `per_channel` | 逐通道量化  | True |
| `symmetric`   | 对称量化   | True |
| `learn_scale` | 学习缩放因子 | True |

**训练流程**：

1. 插入伪量化节点
2. 正常训练（梯度穿越）
3. 冻结 BN 统计量
4. 转换为真正量化模型

**接口方法**：

| 方法                          | 功能        |
| --------------------------- | --------- |
| `prepare(model)`            | 准备 QAT 模型 |
| `enable_fake_quant(model)`  | 启用伪量化     |
| `disable_fake_quant(model)` | 禁用伪量化     |
| `freeze_bn(model)`          | 冻结 BN     |
| `convert(model)`            | 转换为量化模型   |

---

### 4. `quantization/calibrator.py`

**功能**：量化校准器

**校准器类型**：

| 类                      | 说明         |
| ---------------------- | ---------- |
| `MinMaxCalibrator`     | 最小最大值校准    |
| `HistogramCalibrator`  | 直方图校准      |
| `EntropyCalibrator`    | 熵校准（KL 散度） |
| `PercentileCalibrator` | 百分位校准      |
| `MSECalibrator`        | MSE 最优校准   |

**MinMaxCalibrator 参数**：

| 参数            | 说明   | 默认值   |
| ------------- | ---- | ----- |
| `symmetric`   | 对称量化 | True  |
| `per_channel` | 逐通道  | False |

**HistogramCalibrator 参数**：

| 参数           | 说明    | 默认值   |
| ------------ | ----- | ----- |
| `num_bins`   | 直方图箱数 | 2048  |
| `percentile` | 百分位   | 99.99 |

**EntropyCalibrator 参数**：

| 参数                   | 说明    | 默认值  |
| -------------------- | ----- | ---- |
| `num_bins`           | 直方图箱数 | 2048 |
| `num_quantized_bins` | 量化箱数  | 128  |

**校准流程**：

1. 收集激活值统计
2. 计算最优量化参数
3. 设置 scale 和 zero_point

---

### 5. `quantization/quant_utils.py`

**功能**：量化工具函数

**工具函数**：

| 函数                                          | 功能      |
| ------------------------------------------- | ------- |
| `get_quantizable_modules(model)`            | 获取可量化模块 |
| `fuse_modules(model, modules_to_fuse)`      | 模块融合    |
| `auto_fuse(model)`                          | 自动融合    |
| `compute_scale_zp(min_val, max_val, dtype)` | 计算量化参数  |
| `quantize_tensor(tensor, scale, zp, dtype)` | 量化张量    |
| `dequantize_tensor(tensor, scale, zp)`      | 反量化张量   |
| `get_model_size(model, quantized)`          | 获取模型大小  |
| `benchmark_latency(model, input_shape)`     | 延迟基准测试  |

**模块融合规则**：

* Conv + BN → ConvBn
* Conv + BN + ReLU → ConvBnReLU
* Conv + ReLU → ConvReLU
* Linear + ReLU → LinearReLU

**精度评估**：

| 函数                                                 | 功能    |
| -------------------------------------------------- | ----- |
| `compare_weights(fp_model, quant_model)`           | 权重对比  |
| `compare_activations(fp_model, quant_model, data)` | 激活对比  |
| `compute_snr(original, quantized)`                 | 计算信噪比 |

---

## 三、pruning/ 剪枝模块

### 1. `pruning/__init__.py`

**功能**：导出剪枝相关组件

**导出内容**：

* 各类剪枝器
* 敏感度分析器
* 剪枝调度器

---

### 2. `pruning/magnitude_pruner.py`

**功能**：基于幅值的剪枝（非结构化）

**剪枝器类型**：

| 类                      | 说明        |
| ---------------------- | --------- |
| `MagnitudePruner`      | 幅值剪枝器     |
| `L1UnstructuredPruner` | L1 非结构化剪枝 |
| `RandomPruner`         | 随机剪枝      |

**MagnitudePruner 配置参数**：

| 参数               | 说明                                | 默认值            |
| ---------------- | --------------------------------- | -------------- |
| `sparsity`       | 目标稀疏度                             | 0.5            |
| `pruning_type`   | 类型：unstructured / semi_structured | 'unstructured' |
| `criterion`      | 准则：l1 / l2 / random               | 'l1'           |
| `exclude_layers` | 排除的层名                             | []             |
| `include_layers` | 包含的层名                             | None           |

**剪枝准则**：

* L1：按权重绝对值排序
* L2：按权重平方排序
* Random：随机选择

**接口方法**：

| 方法                               | 功能         |
| -------------------------------- | ---------- |
| `compute_mask(module, sparsity)` | 计算剪枝掩码     |
| `apply_mask(model)`              | 应用掩码       |
| `remove_mask(model)`             | 移除掩码（永久剪枝） |
| `get_sparsity(model)`            | 获取当前稀疏度    |
| `prune(model, sparsity)`         | 执行剪枝       |

**半结构化剪枝**：

* N:M 稀疏模式（如 2:4）
* 硬件友好
* 支持 Ampere GPU 加速

---

### 3. `pruning/structured_pruner.py`

**功能**：结构化剪枝

**剪枝器类型**：

| 类                  | 说明     |
| ------------------ | ------ |
| `StructuredPruner` | 结构化剪枝器 |
| `ChannelPruner`    | 通道剪枝   |
| `FilterPruner`     | 滤波器剪枝  |
| `LayerPruner`      | 层剪枝    |

**StructuredPruner 配置参数**：

| 参数               | 说明    | 默认值       |
| ---------------- | ----- | --------- |
| `sparsity`       | 目标稀疏度 | 0.5       |
| `pruning_dim`    | 剪枝维度  | 0（输出通道）   |
| `criterion`      | 重要性准则 | 'l1_norm' |
| `global_pruning` | 全局剪枝  | False     |
| `exclude_layers` | 排除的层  | []        |
| `round_to`       | 通道数对齐 | 8         |

**重要性准则**：

| 准则         | 说明          |
| ---------- | ----------- |
| `l1_norm`  | L1 范数       |
| `l2_norm`  | L2 范数       |
| `bn_scale` | BN 缩放因子     |
| `taylor`   | 泰勒展开        |
| `hrank`    | HRank（特征图秩） |

**接口方法**：

| 方法                                        | 功能      |
| ----------------------------------------- | ------- |
| `compute_importance(model)`               | 计算重要性分数 |
| `get_pruning_plan(model, sparsity)`       | 生成剪枝计划  |
| `prune(model, pruning_plan)`              | 执行剪枝    |
| `finetune_after_prune(model, dataloader)` | 剪枝后微调   |

**依赖处理**：

* 自动处理层间依赖
* 保持 shortcut 通道一致
* 更新 BN 层参数

---

### 4. `pruning/sensitivity_analyzer.py`

**功能**：剪枝敏感度分析

**分析器类型**：

| 类                     | 说明     |
| --------------------- | ------ |
| `SensitivityAnalyzer` | 敏感度分析器 |

**配置参数**：

| 参数               | 说明    | 默认值          |
| ---------------- | ----- | ------------ |
| `sparsity_range` | 稀疏度范围 | [0.1, 0.9]   |
| `num_steps`      | 采样步数  | 9            |
| `metric`         | 评估指标  | 'accuracy'   |
| `pruning_type`   | 剪枝类型  | 'structured' |

**分析流程**：

1. 逐层分析敏感度
2. 在不同稀疏度下评估性能
3. 生成敏感度曲线

**接口方法**：

| 方法                                    | 功能      |
| ------------------------------------- | ------- |
| `analyze(model, dataloader)`          | 执行敏感度分析 |
| `get_sensitivity_scores()`            | 获取敏感度分数 |
| `get_optimal_sparsity(target_metric)` | 获取最优稀疏度 |
| `plot_sensitivity()`                  | 绘制敏感度曲线 |
| `export_report(path)`                 | 导出分析报告  |

**输出内容**：

* 各层敏感度分数
* 稀疏度-精度曲线
* 推荐剪枝配置

---

### 5. `pruning/pruning_scheduler.py`

**功能**：渐进式剪枝调度

**调度器类型**：

| 类                    | 说明       |
| -------------------- | -------- |
| `PruningScheduler`   | 剪枝调度器    |
| `OneShotScheduler`   | 一次性剪枝    |
| `IterativeScheduler` | 迭代剪枝     |
| `CubicScheduler`     | 立方调度     |
| `AGPScheduler`       | AGP 渐进剪枝 |

**IterativeScheduler 配置参数**：

| 参数                 | 说明          | 默认值 |
| ------------------ | ----------- | --- |
| `initial_sparsity` | 初始稀疏度       | 0.0 |
| `target_sparsity`  | 目标稀疏度       | 0.5 |
| `start_epoch`      | 开始 epoch    | 0   |
| `end_epoch`        | 结束 epoch    | 100 |
| `frequency`        | 剪枝频率（epoch） | 1   |

**AGPScheduler（渐进剪枝）**：

稀疏度公式：

```
s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0)) ^ 3
```

**接口方法**：

| 方法                       | 功能       |
| ------------------------ | -------- |
| `step(epoch)`            | 更新剪枝状态   |
| `get_current_sparsity()` | 获取当前稀疏度  |
| `should_prune(epoch)`    | 判断是否需要剪枝 |
| `state_dict()`           | 获取状态     |
| `load_state_dict(state)` | 加载状态     |

---

## 四、distillation/ 知识蒸馏模块

### 1. `distillation/__init__.py`

**功能**：导出蒸馏相关组件

**导出内容**：

* 各类蒸馏器
* 蒸馏损失函数
* 工具函数

---

### 2. `distillation/base_distiller.py`

**功能**：蒸馏器基类

**核心职责**：

* 定义蒸馏统一接口
* 管理教师-学生模型
* 特征提取钩子
* 损失计算框架

**基类接口**：

| 方法                                 | 功能       |
| ---------------------------------- | -------- |
| `register_hooks(teacher, student)` | 注册特征提取钩子 |
| `forward(images, targets)`         | 前向计算     |
| `compute_distill_loss()`           | 计算蒸馏损失   |
| `get_teacher_features()`           | 获取教师特征   |
| `get_student_features()`           | 获取学生特征   |

**配置参数**：

| 参数                   | 说明     | 默认值  |
| -------------------- | ------ | ---- |
| `teacher`            | 教师模型配置 | 必填   |
| `student`            | 学生模型配置 | 必填   |
| `teacher_pretrained` | 教师权重路径 | 必填   |
| `freeze_teacher`     | 冻结教师模型 | True |
| `distill_weight`     | 蒸馏损失权重 | 1.0  |
| `task_weight`        | 任务损失权重 | 1.0  |

**损失组合**：

```
total_loss = task_weight * task_loss + distill_weight * distill_loss
```

---

### 3. `distillation/logit_distiller.py`

**功能**：Logit 级蒸馏（软标签蒸馏）

**蒸馏器类型**：

| 类                | 说明      |
| ---------------- | ------- |
| `LogitDistiller` | 标准 KD   |
| `DKDDistiller`   | 解耦知识蒸馏  |
| `DISTDistiller`  | DIST 蒸馏 |

**LogitDistiller 配置参数**：

| 参数            | 说明      | 默认值 |
| ------------- | ------- | --- |
| `temperature` | 蒸馏温度    | 4.0 |
| `alpha`       | 蒸馏损失权重  | 0.5 |
| `beta`        | 硬标签损失权重 | 0.5 |

**蒸馏损失**：

```
soft_loss = KL(student_soft, teacher_soft) * T^2
hard_loss = CE(student_logits, labels)
loss = alpha * soft_loss + beta * hard_loss
```

**DKDDistiller（解耦 KD）配置**：

| 参数              | 说明       | 默认值 |
| --------------- | -------- | --- |
| `temperature`   | 蒸馏温度     | 4.0 |
| `alpha`         | 目标类损失权重  | 1.0 |
| `beta`          | 非目标类损失权重 | 8.0 |
| `warmup_epochs` | 预热 epoch | 20  |

**适用场景**：

* 分类任务
* 教师学生结构相似
* 简单高效

---

### 4. `distillation/feature_distiller.py`

**功能**：特征级蒸馏

**蒸馏器类型**：

| 类                   | 说明                            |
| ------------------- | ----------------------------- |
| `FeatureDistiller`  | 通用特征蒸馏                        |
| `FitNetDistiller`   | FitNet（Hint Learning）         |
| `ATDistiller`       | Attention Transfer            |
| `FSPDistiller`      | Flow of Solution Procedure    |
| `OFDDistiller`      | Overhaul Feature Distillation |
| `ReviewKDDistiller` | Review KD                     |

**FeatureDistiller 配置参数**：

| 参数               | 说明     | 默认值   |
| ---------------- | ------ | ----- |
| `teacher_layers` | 教师特征层名 | 必填    |
| `student_layers` | 学生特征层名 | 必填    |
| `loss_type`      | 损失类型   | 'mse' |
| `projector`      | 投影器配置  | None  |
| `normalize`      | 是否归一化  | False |

**损失类型**：

| 类型       | 说明      |
| -------- | ------- |
| `mse`    | 均方误差    |
| `l1`     | L1 损失   |
| `cosine` | 余弦相似度损失 |
| `kl`     | KL 散度   |

**特征对齐投影器**：

* 通道对齐：1x1 Conv
* 空间对齐：插值
* 自动推断维度

**ATDistiller（注意力迁移）**：

| 参数          | 说明    | 默认值  |
| ----------- | ----- | ---- |
| `p`         | 范数类型  | 2    |
| `normalize` | 是否归一化 | True |

注意力图计算：

```
attention = mean(abs(feature) ^ p, dim=channel)
```

**适用场景**：

* 教师学生结构差异大
* 需要中间层指导
* 检测/分割任务

---

### 5. `distillation/relation_distiller.py`

**功能**：关系级蒸馏

**蒸馏器类型**：

| 类                   | 说明     |
| ------------------- | ------ |
| `RelationDistiller` | 通用关系蒸馏 |
| `RKDDistiller`      | 关系知识蒸馏 |
| `CRDDistiller`      | 对比表示蒸馏 |
| `PKTDistiller`      | 概率知识迁移 |
| `CCDistiller`       | 相关性一致性 |

**RKDDistiller 配置参数**：

| 参数                | 说明     | 默认值  |
| ----------------- | ------ | ---- |
| `distance_weight` | 距离损失权重 | 25.0 |
| `angle_weight`    | 角度损失权重 | 50.0 |

**关系类型**：

| 类型           | 说明      |
| ------------ | ------- |
| `distance`   | 样本间距离关系 |
| `angle`      | 样本间角度关系 |
| `similarity` | 相似度矩阵   |

**CRDDistiller（对比蒸馏）配置**：

| 参数              | 说明   | 默认值   |
| --------------- | ---- | ----- |
| `feature_dim`   | 特征维度 | 128   |
| `num_negatives` | 负样本数 | 16384 |
| `temperature`   | 温度   | 0.07  |
| `momentum`      | 队列动量 | 0.5   |

**适用场景**：

* 跨模态蒸馏
* 结构差异极大
* 自监督学习

---

## 配置示例

### 训练后量化

```yaml
compression:
  type: PTQQuantizer
  backend: tensorrt
  dtype: int8
  calibration_method: entropy
  num_calibration_batches: 100
```

### 量化感知训练

```yaml
compression:
  type: QATQuantizer
  backend: fbgemm
  dtype: int8
  per_channel: true
  freeze_bn_epochs: 80
  
  # 训练配置
  epochs: 100
  lr: 0.001
```

### 结构化剪枝

```yaml
compression:
  type: StructuredPruner
  sparsity: 0.5
  criterion: bn_scale
  global_pruning: true
  exclude_layers:
    - head
    - stem
  round_to: 8
  
  scheduler:
    type: AGPScheduler
    initial_sparsity: 0.0
    target_sparsity: 0.5
    start_epoch: 0
    end_epoch: 50
```

### 知识蒸馏

```yaml
compression:
  type: FeatureDistiller
  
  teacher:
    type: ResNet
    depth: 101
    pretrained: resnet101.pth
  
  student:
    type: ResNet
    depth: 18
  
  # 特征对齐
  teacher_layers: [layer1, layer2, layer3, layer4]
  student_layers: [layer1, layer2, layer3, layer4]
  
  # 损失配置
  loss_type: mse
  distill_weight: 1.0
  task_weight: 1.0
  
  # 额外添加 Logit 蒸馏
  logit_distill:
    temperature: 4.0
    alpha: 0.5
```

### 组合压缩

```yaml
compression:
  pipeline:
    # 阶段1：知识蒸馏
    - type: LogitDistiller
      temperature: 4.0
      epochs: 100
    
    # 阶段2：结构化剪枝
    - type: StructuredPruner
      sparsity: 0.5
      finetune_epochs: 20
    
    # 阶段3：量化
    - type: PTQQuantizer
      backend: tensorrt
      dtype: int8
```

---

## 依赖关系

**被依赖方**：

* Export Pipeline - 模型导出前压缩
* Training Engine - 蒸馏/QAT 训练

**依赖项**：

* `Registry` - 组件注册与构建
* `Models` - 模型定义
* `torch.quantization` - PyTorch 量化
* `torch.nn.utils.prune` - PyTorch 剪枝
