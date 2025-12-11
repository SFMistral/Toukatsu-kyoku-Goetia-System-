# Hooks 模块功能需求文档

## 模块概述

Hooks 模块是训练过程的扩展机制，通过钩子函数在训练生命周期的关键节点注入自定义逻辑。支持检查点保存、日志记录、模型评估、早停、EMA、性能分析、可视化等功能，所有钩子通过优先级机制协调执行顺序。

---

## 目录结构

```
hooks/
├── __init__.py
├── builder.py               # 钩子构建器
├── base_hook.py             # 钩子基类
├── checkpoint_hook.py       # 检查点钩子
├── logger_hook.py           # 日志钩子
├── eval_hook.py             # 评估钩子
├── early_stopping_hook.py   # 早停钩子
├── ema_hook.py              # EMA 钩子
├── profiler_hook.py         # 性能分析钩子
└── visualization_hook.py    # 可视化钩子
```

---

## 一、`__init__.py`

**功能**：模块初始化，导出核心接口

**导出内容**：

* `build_hook(config)` - 钩子构建函数
* `build_hooks(configs)` - 批量构建
* `BaseHook` - 钩子基类
* `HOOKS` - 钩子注册器
* 所有具体钩子类

---

## 二、`builder.py`

**功能**：钩子构建器，配置驱动的钩子实例化

**核心职责**：

* 根据配置构建单个钩子
* 根据配置列表批量构建钩子
* 按优先级排序钩子列表
* 提供默认钩子集

**构建函数**：

| 函数                     | 功能        |
| ---------------------- | --------- |
| `build_hook(config)`   | 构建单个钩子    |
| `build_hooks(configs)` | 构建钩子列表并排序 |
| `get_default_hooks()`  | 获取默认钩子集   |

**优先级常量**：

| 优先级            | 值   | 说明     |
| -------------- | --- | ------ |
| `HIGHEST`      | 0   | 最高优先级  |
| `VERY_HIGH`    | 10  | 极高     |
| `HIGH`         | 30  | 高      |
| `ABOVE_NORMAL` | 40  | 较高     |
| `NORMAL`       | 50  | 正常（默认） |
| `BELOW_NORMAL` | 60  | 较低     |
| `LOW`          | 70  | 低      |
| `VERY_LOW`     | 90  | 极低     |
| `LOWEST`       | 100 | 最低优先级  |

**默认钩子优先级**：

| 钩子                | 优先级          |
| ----------------- | ------------ |
| EMAHook           | ABOVE_NORMAL |
| LoggerHook        | NORMAL       |
| CheckpointHook    | BELOW_NORMAL |
| EvalHook          | LOW          |
| EarlyStoppingHook | VERY_LOW     |

---

## 三、`base_hook.py`

**功能**：所有钩子的抽象基类

**核心职责**：

* 定义钩子生命周期接口
* 提供默认空实现
* 优先级管理
* 训练器引用

**生命周期方法**：

| 方法                               | 触发时机         |
| -------------------------------- | ------------ |
| `before_run(runner)`             | 训练开始前        |
| `after_run(runner)`              | 训练结束后        |
| `before_epoch(runner)`           | 每个 epoch 开始前 |
| `after_epoch(runner)`            | 每个 epoch 结束后 |
| `before_train_epoch(runner)`     | 训练 epoch 开始前 |
| `after_train_epoch(runner)`      | 训练 epoch 结束后 |
| `before_val_epoch(runner)`       | 验证 epoch 开始前 |
| `after_val_epoch(runner)`        | 验证 epoch 结束后 |
| `before_train_iter(runner)`      | 训练迭代开始前      |
| `after_train_iter(runner)`       | 训练迭代结束后      |
| `before_val_iter(runner)`        | 验证迭代开始前      |
| `after_val_iter(runner)`         | 验证迭代结束后      |
| `before_save_checkpoint(runner)` | 保存检查点前       |
| `after_load_checkpoint(runner)`  | 加载检查点后       |

**基类属性**：

| 属性         | 说明    |
| ---------- | ----- |
| `priority` | 钩子优先级 |
| `runner`   | 训练器引用 |

**辅助方法**：

| 方法                          | 功能              |
| --------------------------- | --------------- |
| `every_n_epochs(runner, n)` | 判断是否每 n 个 epoch |
| `every_n_iters(runner, n)`  | 判断是否每 n 个迭代     |
| `is_last_epoch(runner)`     | 判断是否最后一个 epoch  |
| `is_last_iter(runner)`      | 判断是否最后一个迭代      |
| `get_triggered_stages()`    | 获取钩子触发的阶段列表     |

---

## 四、`checkpoint_hook.py`

**功能**：检查点保存与管理

**核心职责**：

* 定期保存模型检查点
* 管理检查点数量
* 保存最佳模型
* 支持断点续训

**钩子类**：

| 类                | 说明      |
| ---------------- | ------- |
| `CheckpointHook` | 检查点保存钩子 |

**配置参数**：

| 参数               | 说明                  | 默认值            |
| ---------------- | ------------------- | -------------- |
| `interval`       | 保存间隔（epoch）         | 1              |
| `by_epoch`       | 按 epoch 还是 iter     | True           |
| `save_optimizer` | 是否保存优化器状态           | True           |
| `save_scheduler` | 是否保存调度器状态           | True           |
| `max_keep_ckpts` | 最大保留数量              | 5              |
| `save_last`      | 是否保存最后一个            | True           |
| `save_best`      | 是否保存最佳模型            | True           |
| `best_metric`    | 最佳模型判断指标            | 'accuracy'     |
| `rule`           | 比较规则：greater / less | 'greater'      |
| `out_dir`        | 保存目录                | None（使用工作目录）   |
| `filename_tmpl`  | 文件名模板               | 'epoch_{}.pth' |

**保存内容**：

| 键                | 内容                 |
| ---------------- | ------------------ |
| `meta`           | 元信息（时间、epoch、iter） |
| `state_dict`     | 模型权重               |
| `optimizer`      | 优化器状态              |
| `scheduler`      | 调度器状态              |
| `ema_state_dict` | EMA 模型权重（如有）       |

**触发时机**：

* `after_train_epoch`：按 epoch 保存
* `after_train_iter`：按 iter 保存
* `after_val_epoch`：保存最佳模型

**文件管理**：

* 自动删除旧检查点
* 保留最新 N 个
* best 和 last 不计入数量限制

---

## 五、`logger_hook.py`

**功能**：训练日志记录

**核心职责**：

* 记录训练指标
* 输出到多种后端
* 格式化日志输出
* 学习率记录

**钩子类**：

| 类                 | 说明                  |
| ----------------- | ------------------- |
| `LoggerHook`      | 通用日志钩子              |
| `TensorBoardHook` | TensorBoard 日志      |
| `WandbHook`       | Weights & Biases 日志 |
| `MLflowHook`      | MLflow 日志           |

**LoggerHook 配置**：

| 参数                    | 说明           | 默认值  |
| --------------------- | ------------ | ---- |
| `interval`            | 日志间隔（iter）   | 50   |
| `by_epoch`            | 按 epoch 统计   | True |
| `log_metric_by_epoch` | 指标按 epoch 记录 | True |
| `ignore_last`         | 忽略最后不完整间隔    | True |

**TensorBoardHook 配置**：

| 参数               | 说明      | 默认值   |
| ---------------- | ------- | ----- |
| `log_dir`        | 日志目录    | None  |
| `interval`       | 记录间隔    | 50    |
| `log_graph`      | 是否记录模型图 | False |
| `log_images`     | 是否记录图像  | False |
| `image_interval` | 图像记录间隔  | 1000  |

**WandbHook 配置**：

| 参数          | 说明      | 默认值   |
| ----------- | ------- | ----- |
| `project`   | W&B 项目名 | None  |
| `name`      | 运行名称    | None  |
| `config`    | 配置字典    | None  |
| `log_model` | 是否上传模型  | False |
| `interval`  | 记录间隔    | 50    |
| `log_code`  | 是否记录代码  | False |

**记录内容**：

| 类型  | 内容                 |
| --- | ------------------ |
| 标量  | loss、accuracy、lr 等 |
| 图像  | 预测结果、特征图           |
| 直方图 | 权重分布、梯度分布          |
| 文本  | 配置、摘要              |
| 模型  | 检查点文件              |

**触发时机**：

* `before_run`：初始化日志后端
* `after_train_iter`：记录训练指标
* `after_val_epoch`：记录验证指标
* `after_run`：关闭日志后端

---

## 六、`eval_hook.py`

**功能**：模型评估钩子

**核心职责**：

* 定期触发模型评估
* 计算评估指标
* 记录评估结果
* 支持多数据集评估

**钩子类**：

| 类              | 说明      |
| -------------- | ------- |
| `EvalHook`     | 评估钩子    |
| `DistEvalHook` | 分布式评估钩子 |

**配置参数**：

| 参数               | 说明              | 默认值        |
| ---------------- | --------------- | ---------- |
| `dataloader`     | 验证数据加载器         | 必填         |
| `interval`       | 评估间隔（epoch）     | 1          |
| `by_epoch`       | 按 epoch 还是 iter | True       |
| `start`          | 开始评估的 epoch     | 0          |
| `save_best`      | 是否触发保存最佳        | True       |
| `metric`         | 评估指标名           | 'accuracy' |
| `rule`           | 比较规则            | 'greater'  |
| `efficient_test` | 是否节省内存          | False      |
| `test_fn`        | 自定义测试函数         | None       |

**分布式评估**：

* 多 GPU 数据并行评估
* 结果自动聚合
* 仅主进程输出结果

**触发时机**：

* `after_train_epoch`：按 epoch 评估
* `after_train_iter`：按 iter 评估

**输出**：

* 评估指标字典
* 更新 runner 的 best 状态

---

## 七、`early_stopping_hook.py`

**功能**：早停机制

**核心职责**：

* 监控指标变化
* 判断是否停止训练
* 支持 patience 机制
* 可选恢复最佳模型

**钩子类**：

| 类                   | 说明   |
| ------------------- | ---- |
| `EarlyStoppingHook` | 早停钩子 |

**配置参数**：

| 参数                     | 说明                  | 默认值        |
| ---------------------- | ------------------- | ---------- |
| `monitor`              | 监控的指标名              | 'val_loss' |
| `patience`             | 容忍次数                | 10         |
| `min_delta`            | 最小改善量               | 0.0        |
| `mode`                 | 模式：min / max / auto | 'auto'     |
| `baseline`             | 基准值                 | None       |
| `restore_best_weights` | 是否恢复最佳权重            | True       |
| `check_finite`         | 检查是否为有限值            | True       |
| `stopping_threshold`   | 达到阈值立即停止            | None       |
| `divergence_threshold` | 发散阈值立即停止            | None       |

**状态追踪**：

| 状态              | 说明        |
| --------------- | --------- |
| `best_score`    | 最佳指标值     |
| `best_epoch`    | 最佳 epoch  |
| `wait_count`    | 等待计数      |
| `stopped_epoch` | 停止的 epoch |

**模式说明**：

* `min`：指标越小越好（如 loss）
* `max`：指标越大越好（如 accuracy）
* `auto`：根据指标名自动判断

**触发时机**：

* `after_val_epoch`：检查是否早停

**停止条件**：

* `wait_count >= patience`
* `current_score >= stopping_threshold`（max 模式）
* `current_score >= divergence_threshold`（发散检测）
* 指标为 NaN 或 Inf

---

## 八、`ema_hook.py`

**功能**：指数移动平均模型

**核心职责**：

* 维护 EMA 模型
* 更新 EMA 权重
* 评估时使用 EMA
* 检查点保存 EMA

**钩子类**：

| 类         | 说明     |
| --------- | ------ |
| `EMAHook` | EMA 钩子 |

**配置参数**：

| 参数               | 说明          | 默认值    |
| ---------------- | ----------- | ------ |
| `momentum`       | EMA 动量      | 0.9999 |
| `interval`       | 更新间隔（iter）  | 1      |
| `warm_up`        | 预热迭代数       | 100    |
| `resume_from`    | 恢复 EMA 的检查点 | None   |
| `update_buffers` | 是否更新 buffer | False  |

**EMA 更新公式**：

```
ema_weight = momentum * ema_weight + (1 - momentum) * current_weight
```

**动态动量**：

* 预热阶段动量线性增加
* `current_momentum = min(momentum, (1 + iter) / (warm_up + iter))`

**触发时机**：

* `before_run`：初始化 EMA 模型
* `after_train_iter`：更新 EMA 权重
* `before_val_epoch`：切换到 EMA 模型
* `after_val_epoch`：切换回原模型
* `before_save_checkpoint`：保存 EMA 状态

**模型切换**：

* 验证时自动使用 EMA 模型
* 训练时使用原始模型
* 支持手动切换

---

## 九、`profiler_hook.py`

**功能**：性能分析与监控

**核心职责**：

* GPU/CPU 性能分析
* 内存使用监控
* 瓶颈识别
* 性能报告生成

**钩子类**：

| 类                    | 说明                  |
| -------------------- | ------------------- |
| `ProfilerHook`       | PyTorch Profiler 钩子 |
| `MemoryProfilerHook` | 内存分析钩子              |
| `IterTimerHook`      | 迭代计时钩子              |

**ProfilerHook 配置**：

| 参数                    | 说明              | 默认值   |
| --------------------- | --------------- | ----- |
| `by_epoch`            | 按 epoch 分析      | False |
| `profile_iters`       | 分析的迭代数          | 100   |
| `schedule`            | 调度配置            | 见下文   |
| `on_trace_ready`      | 追踪回调            | None  |
| `record_shapes`       | 记录张量形状          | True  |
| `profile_memory`      | 分析内存            | True  |
| `with_stack`          | 记录调用栈           | False |
| `with_flops`          | 计算 FLOPs        | True  |
| `export_chrome_trace` | 导出 Chrome 追踪    | True  |
| `export_tensorboard`  | 导出到 TensorBoard | True  |

**调度配置**：

| 参数       | 说明      |
| -------- | ------- |
| `wait`   | 跳过迭代数   |
| `warmup` | 预热迭代数   |
| `active` | 活跃分析迭代数 |
| `repeat` | 重复次数    |

**MemoryProfilerHook 配置**：

| 参数               | 说明        | 默认值  |
| ---------------- | --------- | ---- |
| `interval`       | 记录间隔      | 50   |
| `log_gpu`        | 记录 GPU 内存 | True |
| `log_cpu`        | 记录 CPU 内存 | True |
| `warn_threshold` | 警告阈值（GB）  | None |

**IterTimerHook 配置**：

| 参数         | 说明       | 默认值  |
| ---------- | -------- | ---- |
| `interval` | 统计间隔     | 50   |
| `log_eta`  | 记录预计剩余时间 | True |

**输出内容**：

| 类型    | 内容               |
| ----- | ---------------- |
| 时间    | 数据加载、前向、反向、优化器耗时 |
| 内存    | GPU/CPU 峰值、当前使用  |
| 吞吐    | 样本/秒、迭代/秒        |
| 调用栈   | 耗时函数调用栈          |
| FLOPs | 计算量统计            |

**触发时机**：

* `before_run`：初始化 profiler
* `before_train_iter`：开始计时
* `after_train_iter`：记录耗时
* `after_run`：生成报告

---

## 十、`visualization_hook.py`

**功能**：训练过程可视化

**核心职责**：

* 预测结果可视化
* 特征图可视化
* 注意力图可视化
* 梯度可视化

**钩子类**：

| 类                      | 说明      |
| ---------------------- | ------- |
| `VisualizationHook`    | 可视化钩子   |
| `DetVisualizationHook` | 检测可视化钩子 |
| `SegVisualizationHook` | 分割可视化钩子 |

**VisualizationHook 配置**：

| 参数            | 说明          | 默认值          |
| ------------- | ----------- | ------------ |
| `interval`    | 可视化间隔（iter） | 500          |
| `draw_gt`     | 是否绘制真值      | True         |
| `draw_pred`   | 是否绘制预测      | True         |
| `show`        | 是否显示窗口      | False        |
| `save`        | 是否保存文件      | True         |
| `out_dir`     | 保存目录        | None         |
| `max_samples` | 每次最大样本数     | 16           |
| `score_thr`   | 置信度阈值（检测）   | 0.3          |
| `backend`     | 可视化后端       | 'matplotlib' |

**DetVisualizationHook 扩展**：

| 参数            | 说明     |
| ------------- | ------ |
| `class_names` | 类别名称列表 |
| `palette`     | 颜色调色板  |
| `bbox_color`  | 边框颜色   |
| `text_color`  | 文本颜色   |
| `thickness`   | 线条粗细   |

**SegVisualizationHook 扩展**：

| 参数            | 说明     |
| ------------- | ------ |
| `class_names` | 类别名称列表 |
| `palette`     | 分割调色板  |
| `opacity`     | 叠加透明度  |
| `show_edge`   | 是否显示边缘 |

**可视化内容**：

| 类型   | 内容            |
| ---- | ------------- |
| 预测结果 | 分类标签、检测框、分割掩码 |
| 特征图  | 中间层特征热力图      |
| 注意力  | 注意力权重可视化      |
| CAM  | 类激活图          |
| 梯度   | 梯度热力图         |

**触发时机**：

* `after_train_iter`：训练可视化
* `after_val_iter`：验证可视化

---

## 配置示例

### 完整钩子配置

```yaml
hooks:
  # 检查点钩子
  - type: CheckpointHook
    interval: 1
    max_keep_ckpts: 3
    save_best: true
    best_metric: accuracy
    rule: greater

  # 日志钩子
  - type: LoggerHook
    interval: 50

  # TensorBoard 钩子
  - type: TensorBoardHook
    log_dir: tensorboard_logs
    interval: 100
    log_graph: true

  # 评估钩子
  - type: EvalHook
    interval: 1
    save_best: true
    metric: accuracy

  # 早停钩子
  - type: EarlyStoppingHook
    monitor: val_loss
    patience: 10
    mode: min
    restore_best_weights: true

  # EMA 钩子
  - type: EMAHook
    momentum: 0.9999
    warm_up: 500

  # 可视化钩子
  - type: VisualizationHook
    interval: 500
    max_samples: 8
    save: true
```

### 分布式训练配置

```yaml
hooks:
  - type: CheckpointHook
    interval: 1
    max_keep_ckpts: 3
    # 仅主进程保存
    
  - type: LoggerHook
    interval: 50
    # 仅主进程输出

  - type: DistEvalHook
    interval: 1
    # 分布式评估

  - type: EMAHook
    momentum: 0.9999
    # 所有进程同步更新
```

### 性能分析配置

```yaml
hooks:
  - type: ProfilerHook
    profile_iters: 100
    schedule:
      wait: 10
      warmup: 10
      active: 80
    record_shapes: true
    profile_memory: true
    export_chrome_trace: true

  - type: MemoryProfilerHook
    interval: 100
    warn_threshold: 10.0

  - type: IterTimerHook
    interval: 50
```

---

## 执行顺序

### 训练流程钩子调用

```
before_run
├── [epoch loop]
│   ├── before_epoch
│   ├── before_train_epoch
│   │   ├── [iter loop]
│   │   │   ├── before_train_iter
│   │   │   └── after_train_iter (按优先级顺序)
│   │   └── after_train_epoch
│   │
│   ├── before_val_epoch
│   │   ├── [iter loop]
│   │   │   ├── before_val_iter
│   │   │   └── after_val_iter
│   │   └── after_val_epoch
│   │
│   └── after_epoch
└── after_run
```

### 优先级执行规则

* **before_xxx**：优先级值**小**的先执行
* **after_xxx**：优先级值**大**的先执行（逆序）

---

## 依赖关系

**被依赖方**：

* Training Engine - 训练过程调用钩子
* Runner - 注册和管理钩子

**依赖项**：

* `Registry` - 钩子注册与构建
* `Logger` - 日志输出
* `Checkpoint` - 检查点保存
* `torch.profiler` - 性能分析
