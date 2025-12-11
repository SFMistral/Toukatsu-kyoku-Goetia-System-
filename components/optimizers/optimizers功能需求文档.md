# Optimizers 模块功能需求文档

## 模块概述

Optimizers 模块是系统的优化器组件库，提供常用优化器的封装与扩展。支持参数分组、层级学习率衰减、权重衰减过滤等高级功能，通过注册器管理实现配置驱动的优化器构建。

---

## 目录结构

```
optimizers/
├── __init__.py          # 模块初始化
├── builder.py           # 优化器构建器
├── sgd.py               # SGD 优化器
├── adam.py              # Adam 优化器
├── adamw.py             # AdamW 优化器
├── lion.py              # Lion 优化器
└── layer_decay.py       # 层级学习率衰减
```

---

## 一、`__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_optimizer(config, model)` - 优化器构建函数
* `OPTIMIZERS` - 优化器注册器
* 所有优化器类
* 层级衰减工具函数

---

## 二、`builder.py`

**功能**：优化器构建器，配置驱动的优化器实例化

**核心职责**：

* 根据配置构建优化器
* 参数分组处理
* 权重衰减过滤
* 层级学习率设置

**构建函数**：

| 函数                                  | 功能     |
| ----------------------------------- | ------ |
| `build_optimizer(config, model)`    | 构建优化器  |
| `build_param_groups(model, config)` | 构建参数分组 |

**参数分组策略**：

| 策略    | 说明                 |
| ----- | ------------------ |
| 默认分组  | 所有参数同一组            |
| 无衰减分组 | bias、norm 层不应用权重衰减 |
| 层级衰减  | 不同层使用不同学习率         |
| 自定义分组 | 按参数名匹配分组           |

**权重衰减过滤**：

默认不应用权重衰减的参数：

* 所有 bias 参数
* LayerNorm / BatchNorm 的 weight 和 bias
* 位置编码参数
* CLS token 等特殊参数

**配置参数**：

| 参数                       | 说明        | 默认值  |
| ------------------------ | --------- | ---- |
| `type`                   | 优化器类型     | 必填   |
| `lr`                     | 基础学习率     | 必填   |
| `weight_decay`           | 权重衰减      | 0.0  |
| `no_weight_decay_params` | 不衰减的参数名模式 | []   |
| `param_groups`           | 自定义参数分组   | None |
| `layer_decay`            | 层级衰减配置    | None |

**自定义分组配置**：

```yaml
param_groups:
  - params: backbone
    lr_mult: 0.1
    weight_decay_mult: 1.0
  - params: head
    lr_mult: 1.0
    weight_decay_mult: 1.0
```

---

## 三、`sgd.py`

**功能**：SGD 优化器及变体

**优化器类型**：

| 类             | 说明                              |
| ------------- | ------------------------------- |
| `SGD`         | 标准 SGD（PyTorch 封装）              |
| `SGDP`        | SGD with Decoupled Weight Decay |
| `NesterovSGD` | Nesterov 动量 SGD                 |

**SGD 配置参数**：

| 参数             | 说明            | 默认值   |
| -------------- | ------------- | ----- |
| `lr`           | 学习率           | 必填    |
| `momentum`     | 动量            | 0.9   |
| `weight_decay` | 权重衰减（L2）      | 0.0   |
| `dampening`    | 动量抑制          | 0.0   |
| `nesterov`     | 是否使用 Nesterov | False |

**SGDP 特性**：

* 解耦权重衰减（类似 AdamW）
* 权重衰减不通过梯度应用
* 更稳定的正则化效果

**适用场景**：

* 大批量训练
* 图像分类基线
* 需要精确调优的场景

---

## 四、`adam.py`

**功能**：Adam 优化器及变体

**优化器类型**：

| 类          | 说明                               |
| ---------- | -------------------------------- |
| `Adam`     | 标准 Adam                          |
| `AdamP`    | Adam with Decoupled Weight Decay |
| `NAdam`    | Adam with Nesterov momentum      |
| `RAdam`    | Rectified Adam                   |
| `Adagrad`  | Adagrad 优化器                      |
| `Adadelta` | Adadelta 优化器                     |

**Adam 配置参数**：

| 参数             | 说明           | 默认值          |
| -------------- | ------------ | ------------ |
| `lr`           | 学习率          | 0.001        |
| `betas`        | 动量参数         | (0.9, 0.999) |
| `eps`          | 数值稳定性        | 1e-8         |
| `weight_decay` | 权重衰减（L2）     | 0.0          |
| `amsgrad`      | 是否使用 AMSGrad | False        |

**RAdam 特性**：

* 自适应学习率校正
* 无需学习率预热
* 训练更稳定

**NAdam 特性**：

* 结合 Nesterov 动量
* 更快收敛

**适用场景**：

* NLP 任务
* 小批量训练
* 默认推荐选择

---

## 五、`adamw.py`

**功能**：AdamW 优化器及变体

**优化器类型**：

| 类            | 说明              |
| ------------ | --------------- |
| `AdamW`      | AdamW（解耦权重衰减）   |
| `FusedAdamW` | 融合实现（更高效）       |
| `Adam8bit`   | 8-bit Adam（省显存） |

**AdamW 配置参数**：

| 参数             | 说明           | 默认值          |
| -------------- | ------------ | ------------ |
| `lr`           | 学习率          | 0.001        |
| `betas`        | 动量参数         | (0.9, 0.999) |
| `eps`          | 数值稳定性        | 1e-8         |
| `weight_decay` | 权重衰减         | 0.01         |
| `amsgrad`      | 是否使用 AMSGrad | False        |

**与 Adam 区别**：

* Adam：`grad = grad + weight_decay * param`
* AdamW：`param = param - lr * weight_decay * param`
* AdamW 权重衰减不依赖梯度缩放

**FusedAdamW 特性**：

* CUDA 内核融合
* 减少内存访问
* 需要 NVIDIA apex 或 DeepSpeed

**Adam8bit 特性**：

* 优化器状态 8-bit 量化
* 显存占用减少约 75%
* 需要 bitsandbytes 库

**适用场景**：

* Transformer 模型（推荐）
* 大规模预训练
* 需要正则化的场景

---

## 六、`lion.py`

**功能**：Lion 优化器（Google 最新优化器）

**优化器类型**：

| 类      | 说明       |
| ------ | -------- |
| `Lion` | Lion 优化器 |

**Lion 配置参数**：

| 参数             | 说明   | 默认值         |
| -------------- | ---- | ----------- |
| `lr`           | 学习率  | 0.0001      |
| `betas`        | 动量参数 | (0.9, 0.99) |
| `weight_decay` | 权重衰减 | 0.0         |

**算法特点**：

* 只使用动量的符号（sign）更新
* 内存效率更高（无二阶矩）
* 更新幅度统一

**更新规则**：

```
update = sign(beta1 * momentum + (1 - beta1) * grad)
param = param - lr * (update + weight_decay * param)
momentum = beta2 * momentum + (1 - beta2) * grad
```

**调优建议**：

* 学习率通常为 AdamW 的 3-10 倍小
* batch size 较大时效果更好
* 权重衰减可适当增大

**适用场景**：

* 视觉 Transformer
* 大批量训练
* 追求训练效率

---

## 七、`layer_decay.py`

**功能**：层级学习率衰减工具

**核心职责**：

* 为不同层设置不同学习率
* 支持多种衰减模式
* 自动识别模型层级结构

**工具函数**：

| 函数                                      | 功能         |
| --------------------------------------- | ---------- |
| `get_layer_decay_params(model, config)` | 获取层级衰减参数分组 |
| `get_num_layers(model)`                 | 获取模型层数     |
| `get_layer_id(name, num_layers)`        | 获取参数所属层 ID |

**配置参数**：

| 参数           | 说明       | 默认值          |
| ------------ | -------- | ------------ |
| `decay_rate` | 衰减率      | 0.75         |
| `decay_type` | 衰减类型     | 'layer_wise' |
| `num_layers` | 层数（自动检测） | None         |

**衰减类型**：

| 类型           | 说明    |
| ------------ | ----- |
| `layer_wise` | 逐层衰减  |
| `stage_wise` | 逐阶段衰减 |
| `uniform`    | 统一学习率 |

**衰减公式**：

```
layer_lr = base_lr * (decay_rate ^ (num_layers - layer_id))
```

**层级识别规则**：

| 模型类型   | 层级识别                            |
| ------ | ------------------------------- |
| ViT    | patch_embed → blocks.0-N → head |
| Swin   | patch_embed → stages.0-N → head |
| ResNet | conv1 → layer1-4 → fc           |
| 通用     | 按模块深度自动划分                       |

**分组输出**：

每组包含：

* `params`：参数列表
* `lr`：该组学习率
* `weight_decay`：权重衰减
* `layer_id`：层 ID（用于日志）

---

## 配置示例

### 基础配置

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.05
```

### 参数分组配置

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05
  # 自动过滤不需要权重衰减的参数
  no_weight_decay_params:
    - bias
    - norm
    - pos_embed
    - cls_token
```

### 层级学习率衰减

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05
  layer_decay:
    decay_rate: 0.75
    decay_type: layer_wise
    # 底层学习率更小，顶层学习率更大
```

### 自定义参数分组

```yaml
optimizer:
  type: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  param_groups:
    - params: backbone
      lr_mult: 0.1      # backbone 使用 0.1x 学习率
    - params: neck
      lr_mult: 0.5      # neck 使用 0.5x 学习率
    - params: head
      lr_mult: 1.0      # head 使用 1x 学习率
```

### 高效训练配置

```yaml
# 大模型省显存配置
optimizer:
  type: Adam8bit
  lr: 0.0001
  weight_decay: 0.01

# 或使用 Lion
optimizer:
  type: Lion
  lr: 0.0001
  betas: [0.9, 0.99]
  weight_decay: 0.1
```

### 微调配置

```yaml
# 预训练模型微调
optimizer:
  type: AdamW
  lr: 0.00001           # 较小学习率
  weight_decay: 0.01
  layer_decay:
    decay_rate: 0.65    # 较大衰减率
    decay_type: layer_wise
```

---

## 优化器选择指南

| 场景          | 推荐优化器              | 配置要点                 |
| ----------- | ------------------ | -------------------- |
| CNN 分类      | SGD + Momentum     | lr=0.1, momentum=0.9 |
| Transformer | AdamW              | lr=1e-4, wd=0.05     |
| 大批量训练       | Lion / LAMB        | 较大 weight_decay      |
| 微调          | AdamW + LayerDecay | 小 lr, 大 decay_rate   |
| 省显存         | Adam8bit           | 需要 bitsandbytes      |
| NLP 任务      | AdamW              | betas=(0.9, 0.98)    |

---

## 依赖关系

**被依赖方**：

* Training Engine - 训练过程优化
* Builder - 优化器构建

**依赖项**：

* `Registry` - 组件注册与构建
* `torch.optim` - PyTorch 优化器基类
* `apex`（可选）- FusedAdamW
* `bitsandbytes`（可选）- Adam8bit
