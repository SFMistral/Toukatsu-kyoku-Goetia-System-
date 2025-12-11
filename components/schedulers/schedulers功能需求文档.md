# Schedulers 模块功能需求文档

## 模块概述

Schedulers 模块是系统的学习率调度组件库，提供多种学习率调度策略。支持预热机制、周期性调度、自定义调度曲线等功能，可按 epoch 或 iteration 更新学习率，通过注册器管理实现配置驱动的调度器构建。

---

## 目录结构

```
schedulers/
├── __init__.py          # 模块初始化
├── builder.py           # 调度器构建器
├── step_lr.py           # 步进学习率
├── cosine_lr.py         # 余弦学习率
├── warmup.py            # 预热策略
├── poly_lr.py           # 多项式学习率
└── onecycle_lr.py       # OneCycle 学习率
```

---

## 一、`__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_scheduler(config, optimizer)` - 调度器构建函数
* `SCHEDULERS` - 调度器注册器
* 所有调度器类
* 预热包装器

---

## 二、`builder.py`

**功能**：调度器构建器，配置驱动的调度器实例化

**核心职责**：

* 根据配置构建调度器
* 自动包装预热策略
* 组合多个调度器
* 处理 epoch/iter 模式

**构建函数**：

| 函数                                               | 功能        |
| ------------------------------------------------ | --------- |
| `build_scheduler(config, optimizer)`             | 构建单个调度器   |
| `build_scheduler_with_warmup(config, optimizer)` | 构建带预热的调度器 |

**通用配置参数**：

| 参数                | 说明            | 默认值       |
| ----------------- | ------------- | --------- |
| `type`            | 调度器类型         | 必填        |
| `by_epoch`        | 按 epoch 更新    | True      |
| `warmup`          | 预热配置          | None      |
| `warmup_iters`    | 预热迭代数         | 0         |
| `warmup_ratio`    | 预热起始比例        | 0.001     |
| `warmup_by_epoch` | 预热按 epoch     | False     |
| `begin`           | 开始 epoch/iter | 0         |
| `end`             | 结束 epoch/iter | -1（到训练结束） |

**调度器组合**：

| 类                     | 说明        |
| --------------------- | --------- |
| `SequentialScheduler` | 顺序执行多个调度器 |
| `ChainedScheduler`    | 链式调度器     |

**SequentialScheduler 配置**：

```yaml
scheduler:
  type: SequentialScheduler
  schedulers:
    - type: LinearWarmup
      end: 5
    - type: CosineAnnealing
      begin: 5
      end: 100
```

---

## 三、`step_lr.py`

**功能**：步进式学习率调度

**调度器类型**：

| 类               | 说明     |
| --------------- | ------ |
| `StepLR`        | 固定步长衰减 |
| `MultiStepLR`   | 多里程碑衰减 |
| `ExponentialLR` | 指数衰减   |

**StepLR 配置参数**：

| 参数           | 说明          | 默认值 |
| ------------ | ----------- | --- |
| `step_size`  | 衰减步长（epoch） | 必填  |
| `gamma`      | 衰减因子        | 0.1 |
| `last_epoch` | 上次 epoch    | -1  |

**学习率公式**：

```
lr = base_lr * gamma ^ (epoch // step_size)
```

**MultiStepLR 配置参数**：

| 参数           | 说明       | 默认值 |
| ------------ | -------- | --- |
| `milestones` | 衰减里程碑列表  | 必填  |
| `gamma`      | 衰减因子     | 0.1 |
| `last_epoch` | 上次 epoch | -1  |

**学习率公式**：

```
lr = base_lr * gamma ^ (已过里程碑数量)
```

**ExponentialLR 配置参数**：

| 参数           | 说明           | 默认值 |
| ------------ | ------------ | --- |
| `gamma`      | 每 epoch 衰减因子 | 必填  |
| `last_epoch` | 上次 epoch     | -1  |

**学习率公式**：

```
lr = base_lr * gamma ^ epoch
```

**适用场景**：

* 经典 CNN 训练（ResNet 等）
* 需要明确控制衰减点
* 长期训练任务

---

## 四、`cosine_lr.py`

**功能**：余弦退火学习率调度

**调度器类型**：

| 类                             | 说明          |
| ----------------------------- | ----------- |
| `CosineAnnealingLR`           | 余弦退火        |
| `CosineAnnealingWarmRestarts` | 带热重启的余弦退火   |
| `CosineAnnealingWithMinLR`    | 带最小学习率的余弦退火 |

**CosineAnnealingLR 配置参数**：

| 参数           | 说明          | 默认值 |
| ------------ | ----------- | --- |
| `T_max`      | 周期长度（epoch） | 必填  |
| `eta_min`    | 最小学习率       | 0   |
| `last_epoch` | 上次 epoch    | -1  |

**学习率公式**：

```
lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
```

**CosineAnnealingWarmRestarts 配置参数**：

| 参数           | 说明       | 默认值 |
| ------------ | -------- | --- |
| `T_0`        | 首次周期长度   | 必填  |
| `T_mult`     | 周期倍增因子   | 1   |
| `eta_min`    | 最小学习率    | 0   |
| `last_epoch` | 上次 epoch | -1  |

**热重启特性**：

* 每个周期结束后重置学习率
* 周期可逐渐增长（T_mult > 1）
* 有助于跳出局部最优

**CosineAnnealingWithMinLR 配置参数**：

| 参数              | 说明      | 默认值  |
| --------------- | ------- | ---- |
| `T_max`         | 周期长度    | 必填   |
| `eta_min`       | 最小学习率   | 0    |
| `eta_min_ratio` | 最小学习率比例 | None |

**适用场景**：

* Transformer 训练（推荐）
* 需要平滑衰减
* 配合预热使用

---

## 五、`warmup.py`

**功能**：学习率预热策略

**预热类型**：

| 类                   | 说明    |
| ------------------- | ----- |
| `LinearWarmup`      | 线性预热  |
| `ExponentialWarmup` | 指数预热  |
| `ConstantWarmup`    | 常数预热  |
| `WarmupScheduler`   | 预热包装器 |

**LinearWarmup 配置参数**：

| 参数             | 说明         | 默认值   |
| -------------- | ---------- | ----- |
| `warmup_iters` | 预热迭代数      | 必填    |
| `warmup_ratio` | 起始学习率比例    | 0.001 |
| `by_epoch`     | 按 epoch 预热 | False |

**学习率公式**：

```
lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * iter / warmup_iters)
```

**ExponentialWarmup 配置参数**：

| 参数             | 说明      | 默认值   |
| -------------- | ------- | ----- |
| `warmup_iters` | 预热迭代数   | 必填    |
| `warmup_ratio` | 起始学习率比例 | 0.001 |

**学习率公式**：

```
lr = base_lr * warmup_ratio ^ (1 - iter / warmup_iters)
```

**ConstantWarmup 配置参数**：

| 参数             | 说明      | 默认值   |
| -------------- | ------- | ----- |
| `warmup_iters` | 预热迭代数   | 必填    |
| `warmup_ratio` | 常数学习率比例 | 0.001 |

**学习率**：

```
lr = base_lr * warmup_ratio  (iter < warmup_iters)
lr = base_lr                 (iter >= warmup_iters)
```

**WarmupScheduler（包装器）**：

| 参数             | 说明    |
| -------------- | ----- |
| `scheduler`    | 主调度器  |
| `warmup_type`  | 预热类型  |
| `warmup_iters` | 预热迭代数 |
| `warmup_ratio` | 起始比例  |

**功能**：

* 包装任意调度器添加预热
* 预热结束后切换到主调度器
* 支持预热按 iter、主调度按 epoch

---

## 六、`poly_lr.py`

**功能**：多项式学习率调度

**调度器类型**：

| 类          | 说明             |
| ---------- | -------------- |
| `PolyLR`   | 多项式衰减          |
| `LinearLR` | 线性衰减（Poly 的特例） |

**PolyLR 配置参数**：

| 参数            | 说明         | 默认值  |
| ------------- | ---------- | ---- |
| `total_iters` | 总迭代数       | 必填   |
| `power`       | 多项式幂次      | 1.0  |
| `eta_min`     | 最小学习率      | 0    |
| `by_epoch`    | 按 epoch 更新 | True |

**学习率公式**：

```
lr = (base_lr - eta_min) * (1 - iter / total_iters) ^ power + eta_min
```

**特殊情况**：

* `power=1.0`：线性衰减
* `power=0.9`：常用于语义分割
* `power=2.0`：平方衰减

**LinearLR 配置参数**：

| 参数             | 说明   | 默认值 |
| -------------- | ---- | --- |
| `start_factor` | 起始因子 | 1.0 |
| `end_factor`   | 结束因子 | 0.0 |
| `total_iters`  | 总迭代数 | 必填  |

**学习率公式**：

```
factor = start_factor + (end_factor - start_factor) * iter / total_iters
lr = base_lr * factor
```

**适用场景**：

* 语义分割（推荐）
* 需要平滑衰减到零
* 固定训练长度

---

## 七、`onecycle_lr.py`

**功能**：OneCycle 学习率调度

**调度器类型**：

| 类            | 说明          |
| ------------ | ----------- |
| `OneCycleLR` | OneCycle 调度 |
| `CyclicLR`   | 周期性学习率      |

**OneCycleLR 配置参数**：

| 参数                 | 说明                | 默认值   |
| ------------------ | ----------------- | ----- |
| `max_lr`           | 最大学习率             | 必填    |
| `total_steps`      | 总步数               | 必填    |
| `pct_start`        | 上升阶段比例            | 0.3   |
| `anneal_strategy`  | 退火策略：cos / linear | 'cos' |
| `div_factor`       | 初始学习率因子           | 25    |
| `final_div_factor` | 最终学习率因子           | 10000 |
| `three_phase`      | 是否三阶段             | False |

**两阶段模式**：

1. 上升阶段：`base_lr → max_lr`
2. 下降阶段：`max_lr → min_lr`

**三阶段模式**：

1. 上升阶段：`base_lr → max_lr`
2. 下降阶段：`max_lr → base_lr`
3. 退火阶段：`base_lr → min_lr`

**学习率计算**：

```
初始 lr = max_lr / div_factor
最终 lr = max_lr / (div_factor * final_div_factor)
```

**CyclicLR 配置参数**：

| 参数               | 说明                                      | 默认值          |
| ---------------- | --------------------------------------- | ------------ |
| `base_lr`        | 基础学习率                                   | 必填           |
| `max_lr`         | 最大学习率                                   | 必填           |
| `step_size_up`   | 上升步数                                    | 2000         |
| `step_size_down` | 下降步数                                    | None         |
| `mode`           | 模式：triangular / triangular2 / exp_range | 'triangular' |
| `gamma`          | exp_range 模式的衰减因子                       | 1.0          |
| `scale_fn`       | 自定义缩放函数                                 | None         |
| `cycle_momentum` | 是否同步调整动量                                | True         |

**模式说明**：

* `triangular`：三角波
* `triangular2`：每周期减半的三角波
* `exp_range`：指数衰减的三角波

**适用场景**：

* 快速训练
* 超收敛（Super-Convergence）
* 学习率范围测试

---

## 配置示例

### 基础配置

```yaml
# StepLR
scheduler:
  type: StepLR
  step_size: 30
  gamma: 0.1

# MultiStepLR
scheduler:
  type: MultiStepLR
  milestones: [30, 60, 90]
  gamma: 0.1

# CosineAnnealing
scheduler:
  type: CosineAnnealingLR
  T_max: 100
  eta_min: 0
```

### 带预热配置

```yaml
# 方式一：内置预热参数
scheduler:
  type: CosineAnnealingLR
  T_max: 100
  eta_min: 1.0e-6
  warmup: linear
  warmup_iters: 500
  warmup_ratio: 0.001

# 方式二：预热包装器
scheduler:
  type: WarmupScheduler
  warmup_type: linear
  warmup_iters: 500
  warmup_ratio: 0.001
  scheduler:
    type: CosineAnnealingLR
    T_max: 100
    eta_min: 1.0e-6
```

### 热重启配置

```yaml
scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 10          # 首周期 10 epochs
  T_mult: 2        # 周期翻倍
  eta_min: 1.0e-6
```

### OneCycle 配置

```yaml
scheduler:
  type: OneCycleLR
  max_lr: 0.01
  total_steps: 10000
  pct_start: 0.3
  anneal_strategy: cos
  div_factor: 25
  final_div_factor: 10000
```

### 多项式衰减配置

```yaml
# 语义分割常用
scheduler:
  type: PolyLR
  total_iters: 40000
  power: 0.9
  eta_min: 0
  by_epoch: false
```

### 组合调度器配置

```yaml
scheduler:
  type: SequentialScheduler
  schedulers:
    # 阶段1：线性预热
    - type: LinearLR
      start_factor: 0.001
      end_factor: 1.0
      total_iters: 5
      by_epoch: true
    # 阶段2：余弦退火
    - type: CosineAnnealingLR
      T_max: 95
      eta_min: 1.0e-6
      begin: 5
```

### 按迭代更新配置

```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 10000       # 总迭代数
  eta_min: 1.0e-6
  by_epoch: false    # 按迭代更新
  warmup: linear
  warmup_iters: 500
  warmup_ratio: 0.001
```

---

## 调度器选择指南

| 场景          | 推荐调度器                    | 配置要点                               |
| ----------- | ------------------------ | ---------------------------------- |
| CNN 分类      | MultiStepLR              | milestones 设为总 epoch 的 30%/60%/90% |
| Transformer | CosineAnnealing + Warmup | warmup 约 1-5 epochs                |
| 语义分割        | PolyLR                   | power=0.9, by_epoch=False          |
| 快速训练        | OneCycleLR               | max_lr 需要调参                        |
| 长期训练        | CosineWarmRestarts       | T_mult=2 逐渐增加周期                    |
| 微调          | CosineAnnealing          | 较小 T_max，无预热                       |

---

## 调度器接口规范

### 基本接口

| 方法                            | 功能        |
| ----------------------------- | --------- |
| `step(epoch=None)`            | 更新学习率     |
| `get_lr()`                    | 获取当前学习率列表 |
| `get_last_lr()`               | 获取最后一次学习率 |
| `state_dict()`                | 获取状态字典    |
| `load_state_dict(state_dict)` | 加载状态字典    |

### 更新时机

| 模式               | 调用位置                        |
| ---------------- | --------------------------- |
| `by_epoch=True`  | 每个 epoch 结束后调用 `step()`     |
| `by_epoch=False` | 每个 iteration 结束后调用 `step()` |

### 与优化器协作

```
# 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if not scheduler.by_epoch:
            scheduler.step()  # 按 iter 更新
    
    if scheduler.by_epoch:
        scheduler.step()      # 按 epoch 更新
```

---

## 依赖关系

**被依赖方**：

* Training Engine - 训练过程学习率调度
* Builder - 调度器构建

**依赖项**：

* `Registry` - 组件注册与构建
* `torch.optim.lr_scheduler` - PyTorch 调度器基类
* Optimizer - 关联的优化器实例
