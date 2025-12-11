```
Toukatsukyoku-Goetia-System/
│
├── 📁 master/                                          # Master端（本地中控）
│   ├── __init__.py
│   ├── master_main.py
│   │
│   ├── 📁 node_manager/                                # 节点管理
│   │   ├── __init__.py
│   │   ├── node_registry.py
│   │   ├── node_status.py
│   │   ├── node_selector.py
│   │   ├── node_health_checker.py
│   │   └── gpu_resource_pool.py
│   │
│   ├── 📁 task_scheduler/                              # 任务调度
│   │   ├── __init__.py
│   │   ├── task_queue.py
│   │   ├── task_dispatcher.py
│   │   ├── task_tracker.py
│   │   ├── task_priority.py
│   │   ├── task_retry.py
│   │   └── time_estimator.py
│   │
│   ├── 📁 connection/                                  # 通信管理
│   │   ├── __init__.py
│   │   ├── websocket_server.py
│   │   ├── connection_pool.py
│   │   ├── message_handler.py
│   │   └── heartbeat_monitor.py
│   │
│   ├── 📁 data_aggregator/                             # 数据聚合
│   │   ├── __init__.py
│   │   ├── metric_collector.py
│   │   ├── log_aggregator.py
│   │   ├── checkpoint_manager.py
│   │   ├── result_processor.py
│   │   └── experiment_comparator.py
│   │
│   ├── 📁 packager/                                    # 任务打包
│   │   ├── __init__.py
│   │   ├── dependency_analyzer.py
│   │   ├── code_extractor.py
│   │   ├── package_builder.py
│   │   └── package_sender.py
│   │
│   ├── 📁 protocol/                                    # 通信协议
│   │   ├── __init__.py
│   │   ├── message_types.py
│   │   ├── message_serializer.py
│   │   └── protocol_handler.py
│   │
│   ├── 📁 security/                                    # 安全认证
│   │   ├── __init__.py
│   │   ├── token_manager.py
│   │   ├── agent_authenticator.py
│   │   └── encryption.py
│   │
│   └── 📁 report_generator/                            # 报告生成
│       ├── __init__.py
│       ├── llm_client.py
│       ├── report_builder.py
│       ├── markdown_exporter.py
│       ├── pdf_exporter.py
│       └── 📁 templates/
│           ├── experiment_report.md
│           └── comparison_report.md
│
├── 📁 agent/                                           # Agent端（部署到云服务器）
│   ├── __init__.py
│   ├── agent_main.py
│   ├── config.yaml
│   │
│   ├── 📁 connection/                                  # 通信模块
│   │   ├── __init__.py
│   │   ├── websocket_client.py
│   │   ├── heartbeat.py
│   │   ├── reconnector.py
│   │   └── message_sender.py
│   │
│   ├── 📁 executor/                                    # 执行模块
│   │   ├── __init__.py
│   │   ├── task_receiver.py
│   │   ├── package_handler.py
│   │   ├── dynamic_loader.py
│   │   ├── training_executor.py
│   │   ├── process_manager.py
│   │   └── gpu_manager.py
│   │
│   ├── 📁 reporter/                                    # 上报模块
│   │   ├── __init__.py
│   │   ├── environment_reporter.py
│   │   ├── metric_reporter.py
│   │   ├── progress_reporter.py
│   │   ├── log_streamer.py
│   │   ├── checkpoint_uploader.py
│   │   └── visualization_reporter.py
│   │
│   ├── 📁 data_handler/                                # 数据处理
│   │   ├── __init__.py
│   │   ├── downloader.py
│   │   ├── extractor.py
│   │   ├── path_mapper.py
│   │   ├── validator.py
│   │   └── cleaner.py
│   │
│   ├── 📁 runtime_core/                                # 运行时核心
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   ├── base_trainer.py
│   │   ├── base_evaluator.py
│   │   ├── registry.py
│   │   ├── hook_manager.py
│   │   ├── checkpoint_handler.py
│   │   ├── mixed_precision.py
│   │   ├── distributed.py
│   │   ├── gradient_accumulation.py
│   │   ├── seed_manager.py
│   │   └── utils.py
│   │
│   ├── requirements.txt
│   └── install.sh
│
├── 📁 components/                                      # 组件库（Master本地存储）
│   │
│   ├── 📁 models/                                      # 模型组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_model.py
│   │   │
│   │   ├── 📁 backbones/
│   │   │   ├── __init__.py
│   │   │   ├── resnet.py
│   │   │   ├── efficientnet.py
│   │   │   ├── vit.py
│   │   │   ├── swin_transformer.py
│   │   │   ├── convnext.py
│   │   │   ├── mobilenet.py
│   │   │   └── darknet.py
│   │   │
│   │   ├── 📁 necks/
│   │   │   ├── __init__.py
│   │   │   ├── fpn.py
│   │   │   ├── pan.py
│   │   │   ├── bifpn.py
│   │   │   └── yolo_neck.py
│   │   │
│   │   ├── 📁 heads/
│   │   │   ├── __init__.py
│   │   │   ├── cls_head.py
│   │   │   ├── det_head.py
│   │   │   ├── seg_head.py
│   │   │   └── inst_seg_head.py
│   │   │
│   │   ├── 📁 detectors/
│   │   │   ├── __init__.py
│   │   │   ├── base_detector.py
│   │   │   ├── faster_rcnn.py
│   │   │   ├── yolo.py
│   │   │   ├── fcos.py
│   │   │   ├── detr.py
│   │   │   └── retinanet.py
│   │   │
│   │   ├── 📁 segmentors/
│   │   │   ├── __init__.py
│   │   │   ├── base_segmentor.py
│   │   │   ├── unet.py
│   │   │   ├── deeplabv3.py
│   │   │   ├── segformer.py
│   │   │   ├── pspnet.py
│   │   │   └── mask_rcnn.py
│   │   │
│   │   ├── 📁 classifiers/
│   │   │   ├── __init__.py
│   │   │   ├── base_classifier.py
│   │   │   └── image_classifier.py
│   │   │
│   │   ├── 📁 layers/
│   │   │   ├── __init__.py
│   │   │   ├── conv_module.py
│   │   │   ├── norm_layers.py
│   │   │   ├── activation.py
│   │   │   ├── attention.py
│   │   │   ├── drop.py
│   │   │   ├── position_encoding.py
│   │   │   └── blocks.py
│   │   │
│   │   └── 📁 utils/
│   │       ├── __init__.py
│   │       ├── weight_init.py
│   │       ├── model_utils.py
│   │       └── ckpt_convert.py
│   │
│   ├── 📁 datasets/                                    # 数据集组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_dataset.py
│   │   │
│   │   ├── 📁 formats/
│   │   │   ├── __init__.py
│   │   │   ├── coco.py
│   │   │   ├── voc.py
│   │   │   ├── yolo_format.py
│   │   │   ├── imagenet.py
│   │   │   └── custom.py
│   │   │
│   │   ├── 📁 parsers/
│   │   │   ├── __init__.py
│   │   │   ├── annotation_parser.py
│   │   │   ├── coco_parser.py
│   │   │   ├── voc_parser.py
│   │   │   └── yolo_parser.py
│   │   │
│   │   ├── 📁 samplers/
│   │   │   ├── __init__.py
│   │   │   ├── distributed_sampler.py
│   │   │   ├── balanced_sampler.py
│   │   │   └── repeat_sampler.py
│   │   │
│   │   └── 📁 utils/
│   │       ├── __init__.py
│   │       ├── collate.py
│   │       ├── prefetcher.py
│   │       └── data_utils.py
│   │
│   ├── 📁 augmentations/                               # 数据增强组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_transform.py
│   │   │
│   │   ├── 📁 geometric/
│   │   │   ├── __init__.py
│   │   │   ├── resize.py
│   │   │   ├── flip.py
│   │   │   ├── rotate.py
│   │   │   ├── crop.py
│   │   │   ├── affine.py
│   │   │   └── perspective.py
│   │   │
│   │   ├── 📁 photometric/
│   │   │   ├── __init__.py
│   │   │   ├── color_jitter.py
│   │   │   ├── normalize.py
│   │   │   ├── blur.py
│   │   │   ├── noise.py
│   │   │   └── histogram.py
│   │   │
│   │   ├── 📁 mixing/
│   │   │   ├── __init__.py
│   │   │   ├── mixup.py
│   │   │   ├── cutmix.py
│   │   │   ├── mosaic.py
│   │   │   └── copypaste.py
│   │   │
│   │   ├── 📁 formatting/
│   │   │   ├── __init__.py
│   │   │   ├── to_tensor.py
│   │   │   ├── pad.py
│   │   │   └── collect.py
│   │   │
│   │   └── 📁 pipelines/
│   │       ├── __init__.py
│   │       ├── compose.py
│   │       ├── auto_augment.py
│   │       └── preset_pipelines.py
│   │
│   ├── 📁 losses/                                      # 损失函数组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_loss.py
│   │   │
│   │   ├── 📁 classification/
│   │   │   ├── __init__.py
│   │   │   ├── cross_entropy.py
│   │   │   ├── focal_loss.py
│   │   │   ├── label_smooth.py
│   │   │   └── asymmetric_loss.py
│   │   │
│   │   ├── 📁 detection/
│   │   │   ├── __init__.py
│   │   │   ├── iou_loss.py
│   │   │   ├── smooth_l1.py
│   │   │   ├── gfocal_loss.py
│   │   │   └── varifocal_loss.py
│   │   │
│   │   ├── 📁 segmentation/
│   │   │   ├── __init__.py
│   │   │   ├── dice_loss.py
│   │   │   ├── lovasz_loss.py
│   │   │   ├── boundary_loss.py
│   │   │   └── ohem_loss.py
│   │   │
│   │   ├── 📁 distillation/
│   │   │   ├── __init__.py
│   │   │   ├── kd_loss.py
│   │   │   ├── feature_loss.py
│   │   │   └── relation_loss.py
│   │   │
│   │   └── 📁 utils/
│   │       ├── __init__.py
│   │       ├── loss_utils.py
│   │       └── weight_reduce.py
│   │
│   ├── 📁 metrics/                                     # 评估指标组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_metric.py
│   │   │
│   │   ├── 📁 classification/
│   │   │   ├── __init__.py
│   │   │   ├── accuracy.py
│   │   │   ├── precision_recall.py
│   │   │   ├── f1_score.py
│   │   │   ├── confusion_matrix.py
│   │   │   ├── auc_roc.py
│   │   │   └── pr_curve.py
│   │   │
│   │   ├── 📁 detection/
│   │   │   ├── __init__.py
│   │   │   ├── mean_ap.py
│   │   │   ├── coco_metric.py
│   │   │   └── voc_metric.py
│   │   │
│   │   ├── 📁 segmentation/
│   │   │   ├── __init__.py
│   │   │   ├── iou.py
│   │   │   ├── dice_score.py
│   │   │   └── pixel_accuracy.py
│   │   │
│   │   └── 📁 utils/
│   │       ├── __init__.py
│   │       └── metric_utils.py
│   │
│   ├── 📁 optimizers/                                  # 优化器组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── adamw.py
│   │   ├── lion.py
│   │   └── layer_decay.py
│   │
│   ├── 📁 schedulers/                                  # 学习率调度组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── step_lr.py
│   │   ├── cosine_lr.py
│   │   ├── warmup.py
│   │   ├── poly_lr.py
│   │   └── onecycle_lr.py
│   │
│   ├── 📁 hooks/                                       # 训练钩子组件
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── base_hook.py
│   │   ├── checkpoint_hook.py
│   │   ├── logger_hook.py
│   │   ├── eval_hook.py
│   │   ├── early_stopping_hook.py
│   │   ├── ema_hook.py
│   │   ├── profiler_hook.py
│   │   └── visualization_hook.py
│   │
│   └── 📁 compression/                                 # 模型压缩组件
│       ├── __init__.py
│       ├── builder.py
│       │
│       ├── 📁 quantization/
│       │   ├── __init__.py
│       │   ├── ptq.py
│       │   ├── qat.py
│       │   ├── calibrator.py
│       │   └── quant_utils.py
│       │
│       ├── 📁 pruning/
│       │   ├── __init__.py
│       │   ├── magnitude_pruner.py
│       │   ├── structured_pruner.py
│       │   ├── sensitivity_analyzer.py
│       │   └── pruning_scheduler.py
│       │
│       └── 📁 distillation/
│           ├── __init__.py
│           ├── base_distiller.py
│           ├── feature_distiller.py
│           ├── logit_distiller.py
│           └── relation_distiller.py
│
├── 📁 export/                                          # 模型导出
│   ├── __init__.py
│   ├── base_exporter.py
│   ├── onnx_exporter.py
│   ├── tensorrt_exporter.py
│   ├── openvino_exporter.py
│   ├── coreml_exporter.py
│   ├── ncnn_exporter.py
│   │
│   ├── 📁 optimizers/
│   │   ├── __init__.py
│   │   ├── onnx_optimizer.py
│   │   ├── graph_optimizer.py
│   │   └── shape_inference.py
│   │
│   ├── 📁 validators/
│   │   ├── __init__.py
│   │   ├── accuracy_validator.py
│   │   ├── performance_validator.py
│   │   └── consistency_checker.py
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       ├── input_spec.py
│       └── export_utils.py
│
├── 📁 visualization/                                   # 可视化模块
│   ├── __init__.py
│   │
│   ├── 📁 feature_maps/
│   │   ├── __init__.py
│   │   ├── activation_extractor.py
│   │   ├── feature_visualizer.py
│   │   └── cam_visualizer.py
│   │
│   ├── 📁 gradients/
│   │   ├── __init__.py
│   │   ├── gradient_extractor.py
│   │   ├── gradient_visualizer.py
│   │   └── gradient_flow.py
│   │
│   ├── 📁 statistics/
│   │   ├── __init__.py
│   │   ├── activation_stats.py
│   │   ├── gradient_stats.py
│   │   ├── weight_stats.py
│   │   └── distribution_plotter.py
│   │
│   ├── 📁 metrics/
│   │   ├── __init__.py
│   │   ├── curve_plotter.py
│   │   ├── confusion_matrix_plotter.py
│   │   ├── pr_curve_plotter.py
│   │   ├── roc_curve_plotter.py
│   │   └── loss_landscape.py
│   │
│   ├── 📁 model/
│   │   ├── __init__.py
│   │   ├── architecture_visualizer.py
│   │   ├── lineage_tracker.py
│   │   └── lineage_graph.py
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       ├── color_utils.py
│       └── plot_utils.py
│
├── 📁 analysis/                                        # 分析模块
│   ├── __init__.py
│   │
│   ├── 📁 profiler/
│   │   ├── __init__.py
│   │   ├── model_profiler.py
│   │   ├── flops_counter.py
│   │   ├── params_counter.py
│   │   ├── memory_profiler.py
│   │   └── latency_profiler.py
│   │
│   ├── 📁 benchmark/
│   │   ├── __init__.py
│   │   ├── inference_benchmark.py
│   │   ├── throughput_benchmark.py
│   │   └── benchmark_reporter.py
│   │
│   └── 📁 comparison/
│       ├── __init__.py
│       ├── experiment_comparator.py
│       ├── config_differ.py
│       └── metric_comparator.py
│
├── 📁 experiment/                                      # 实验管理
│   ├── __init__.py
│   │
│   ├── 📁 tracker/
│   │   ├── __init__.py
│   │   ├── experiment_tracker.py
│   │   ├── metric_tracker.py
│   │   ├── artifact_tracker.py
│   │   └── lineage_tracker.py
│   │
│   ├── 📁 reproducer/
│   │   ├── __init__.py
│   │   ├── environment_capturer.py
│   │   ├── command_generator.py
│   │   ├── seed_manager.py
│   │   └── reproduce_validator.py
│   │
│   ├── 📁 storage/
│   │   ├── __init__.py
│   │   ├── experiment_storage.py
│   │   ├── checkpoint_storage.py
│   │   ├── artifact_storage.py
│   │   └── cloud_sync.py
│   │
│   └── 📁 id_generator/
│       ├── __init__.py
│       ├── experiment_id.py
│       └── path_resolver.py
│
├── 📁 logging/                                         # 日志系统
│   ├── __init__.py
│   ├── logger.py
│   ├── formatters.py
│   ├── handlers.py
│   │
│   ├── 📁 writers/
│   │   ├── __init__.py
│   │   ├── file_writer.py
│   │   ├── console_writer.py
│   │   └── json_writer.py
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       ├── log_parser.py
│       └── log_aggregator.py
│
├── 📁 configs/                                         # 配置系统（Hydra）
│   ├── __init__.py
│   │
│   ├── 📁 hydra/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── config_composer.py
│   │   ├── config_validator.py
│   │   ├── override_parser.py
│   │   └── interpolation.py
│   │
│   ├── 📁 schemas/
│   │   ├── __init__.py
│   │   ├── model_schema.py
│   │   ├── dataset_schema.py
│   │   ├── training_schema.py
│   │   ├── augmentation_schema.py
│   │   └── export_schema.py
│   │
│   ├── 📁 templates/
│   │   ├── 📁 classification/
│   │   │   ├── resnet50_imagenet.yaml
│   │   │   ├── vit_base.yaml
│   │   │   └── efficientnet_b0.yaml
│   │   │
│   │   ├── 📁 detection/
│   │   │   ├── yolov8_coco.yaml
│   │   │   ├── faster_rcnn_coco.yaml
│   │   │   └── fcos_coco.yaml
│   │   │
│   │   └── 📁 segmentation/
│   │       ├── unet_medical.yaml
│   │       ├── deeplabv3_cityscapes.yaml
│   │       └── mask_rcnn_coco.yaml
│   │
│   ├── 📁 defaults/
│   │   ├── model.yaml
│   │   ├── dataset.yaml
│   │   ├── optimizer.yaml
│   │   ├── scheduler.yaml
│   │   ├── augmentation.yaml
│   │   ├── training.yaml
│   │   ├── export.yaml
│   │   └── compression.yaml
│   │
│   ├── 📁 groups/
│   │   ├── backbone/
│   │   ├── neck/
│   │   ├── head/
│   │   ├── loss/
│   │   └── metric/
│   │
│   ├── master_config.yaml
│   └── system_config.yaml
│
├── 📁 registry/                                        # 全局注册器
│   ├── __init__.py
│   ├── registry.py
│   ├── model_registry.py
│   ├── dataset_registry.py
│   ├── loss_registry.py
│   ├── metric_registry.py
│   ├── augmentation_registry.py
│   ├── optimizer_registry.py
│   ├── scheduler_registry.py
│   ├── hook_registry.py
│   ├── exporter_registry.py
│   └── component_scanner.py
│
├── 📁 api/                                             # API层
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── 📁 routes/
│   │   ├── __init__.py
│   │   ├── task_routes.py
│   │   ├── node_routes.py
│   │   ├── model_routes.py
│   │   ├── dataset_routes.py
│   │   ├── metric_routes.py
│   │   ├── experiment_routes.py
│   │   ├── export_routes.py
│   │   ├── visualization_routes.py
│   │   ├── report_routes.py
│   │   ├── comparison_routes.py
│   │   ├── auth_routes.py
│   │   └── system_routes.py
│   │
│   ├── 📁 schemas/
│   │   ├── __init__.py
│   │   ├── task_schema.py
│   │   ├── node_schema.py
│   │   ├── model_schema.py
│   │   ├── dataset_schema.py
│   │   ├── experiment_schema.py
│   │   ├── export_schema.py
│   │   └── response_schema.py
│   │
│   ├── 📁 services/
│   │   ├── __init__.py
│   │   ├── task_service.py
│   │   ├── node_service.py
│   │   ├── component_service.py
│   │   ├── experiment_service.py
│   │   ├── export_service.py
│   │   ├── report_service.py
│   │   ├── comparison_service.py
│   │   └── file_service.py
│   │
│   ├── 📁 websocket/
│   │   ├── __init__.py
│   │   ├── ws_manager.py
│   │   ├── ws_handlers.py
│   │   └── ws_events.py
│   │
│   └── 📁 middleware/
│       ├── __init__.py
│       ├── auth.py
│       ├── cors.py
│       ├── rate_limit.py
│       └── error_handler.py
│
├── 📁 webui/                                           # Web界面
│   ├── 📁 public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── 📁 assets/
│   │
│   ├── 📁 src/
│   │   ├── main.js
│   │   ├── App.vue
│   │   │
│   │   ├── 📁 views/
│   │   │   ├── Dashboard.vue
│   │   │   ├── TaskCreate.vue
│   │   │   ├── TaskList.vue
│   │   │   ├── TaskDetail.vue
│   │   │   ├── TaskMonitor.vue
│   │   │   ├── NodeList.vue
│   │   │   ├── NodeDetail.vue
│   │   │   ├── GpuResourcePool.vue
│   │   │   ├── ModelLibrary.vue
│   │   │   ├── DatasetManager.vue
│   │   │   ├── ExperimentList.vue
│   │   │   ├── ExperimentDetail.vue
│   │   │   ├── ExperimentCompare.vue
│   │   │   ├── ExportCenter.vue
│   │   │   ├── ReportViewer.vue
│   │   │   ├── VisualizationCenter.vue
│   │   │   ├── Settings.vue
│   │   │   └── Login.vue
│   │   │
│   │   ├── 📁 components/
│   │   │   ├── 📁 common/
│   │   │   │   ├── Header.vue
│   │   │   │   ├── Sidebar.vue
│   │   │   │   ├── Footer.vue
│   │   │   │   ├── Loading.vue
│   │   │   │   ├── Modal.vue
│   │   │   │   ├── Notification.vue
│   │   │   │   ├── Breadcrumb.vue
│   │   │   │   ├── SearchBar.vue
│   │   │   │   ├── Pagination.vue
│   │   │   │   └── EmptyState.vue
│   │   │   │
│   │   │   ├── 📁 task/
│   │   │   │   ├── TaskTypeSelector.vue
│   │   │   │   ├── ModelSelector.vue
│   │   │   │   ├── BackboneSelector.vue
│   │   │   │   ├── NeckSelector.vue
│   │   │   │   ├── HeadSelector.vue
│   │   │   │   ├── DatasetConfig.vue
│   │   │   │   ├── MultiDatasetInput.vue
│   │   │   │   ├── AugmentationConfig.vue
│   │   │   │   ├── AugmentationPipeline.vue
│   │   │   │   ├── LossSelector.vue
│   │   │   │   ├── MetricSelector.vue
│   │   │   │   ├── OptimizerConfig.vue
│   │   │   │   ├── SchedulerConfig.vue
│   │   │   │   ├── HyperParamConfig.vue
│   │   │   │   ├── DistributedConfig.vue
│   │   │   │   ├── MixedPrecisionConfig.vue
│   │   │   │   ├── ExportFormatSelector.vue
│   │   │   │   ├── CompressionConfig.vue
│   │   │   │   ├── QuantizationConfig.vue
│   │   │   │   ├── PruningConfig.vue
│   │   │   │   ├── DistillationConfig.vue
│   │   │   │   ├── PretrainedWeightSelector.vue
│   │   │   │   ├── NodeSelector.vue
│   │   │   │   ├── TaskSummary.vue
│   │   │   │   ├── TaskProgress.vue
│   │   │   │   └── ConfigYamlPreview.vue
│   │   │   │
│   │   │   ├── 📁 monitor/
│   │   │   │   ├── TrainingDashboard.vue
│   │   │   │   ├── PerformanceDashboard.vue
│   │   │   │   ├── MetricChart.vue
│   │   │   │   ├── MultiMetricChart.vue
│   │   │   │   ├── LossChart.vue
│   │   │   │   ├── LearningRateChart.vue
│   │   │   │   ├── ResourceChart.vue
│   │   │   │   ├── GpuUtilizationChart.vue
│   │   │   │   ├── MemoryUsageChart.vue
│   │   │   │   ├── ThroughputChart.vue
│   │   │   │   ├── LogViewer.vue
│   │   │   │   ├── RealTimeLog.vue
│   │   │   │   ├── ProgressBar.vue
│   │   │   │   ├── EtaDisplay.vue
│   │   │   │   ├── CheckpointList.vue
│   │   │   │   ├── BestCheckpointBadge.vue
│   │   │   │   └── TimeRangeSelector.vue
│   │   │   │
│   │   │   ├── 📁 experiment/
│   │   │   │   ├── ExperimentCard.vue
│   │   │   │   ├── ExperimentTable.vue
│   │   │   │   ├── ExperimentFilter.vue
│   │   │   │   ├── ExperimentTimeline.vue
│   │   │   │   ├── ExperimentIdBadge.vue
│   │   │   │   ├── ReproduceCommand.vue
│   │   │   │   ├── EnvironmentInfo.vue
│   │   │   │   └── ArtifactList.vue
│   │   │   │
│   │   │   ├── 📁 compare/
│   │   │   │   ├── ExperimentSelector.vue
│   │   │   │   ├── MetricCompareChart.vue
│   │   │   │   ├── ParallelCurveChart.vue
│   │   │   │   ├── ConfigDiffViewer.vue
│   │   │   │   ├── ConfigHighlighter.vue
│   │   │   │   ├── CompareTable.vue
│   │   │   │   └── DragZoomChart.vue
│   │   │   │
│   │   │   ├── 📁 visualization/
│   │   │   │   ├── FeatureMapViewer.vue
│   │   │   │   ├── GradientFlowChart.vue
│   │   │   │   ├── ActivationStatsChart.vue
│   │   │   │   ├── WeightDistribution.vue
│   │   │   │   ├── ConfusionMatrixViewer.vue
│   │   │   │   ├── PrCurveViewer.vue
│   │   │   │   ├── RocCurveViewer.vue
│   │   │   │   ├── CamViewer.vue
│   │   │   │   ├── ModelArchitectureGraph.vue
│   │   │   │   └── ModelLineageGraph.vue
│   │   │   │
│   │   │   ├── 📁 export/
│   │   │   │   ├── ExportFormatCard.vue
│   │   │   │   ├── ExportProgress.vue
│   │   │   │   ├── ExportResult.vue
│   │   │   │   ├── ValidationResult.vue
│   │   │   │   ├── PerformanceTable.vue
│   │   │   │   └── DownloadButton.vue
│   │   │   │
│   │   │   ├── 📁 report/
│   │   │   │   ├── ReportPreview.vue
│   │   │   │   ├── ReportGenerator.vue
│   │   │   │   ├── MarkdownViewer.vue
│   │   │   │   ├── PdfViewer.vue
│   │   │   │   └── ReportDownload.vue
│   │   │   │
│   │   │   ├── 📁 node/
│   │   │   │   ├── NodeCard.vue
│   │   │   │   ├── NodeStatus.vue
│   │   │   │   ├── GpuInfo.vue
│   │   │   │   ├── GpuResourcePoolChart.vue
│   │   │   │   ├── NodeHealthIndicator.vue
│   │   │   │   └── ResourceAllocationChart.vue
│   │   │   │
│   │   │   ├── 📁 analysis/
│   │   │   │   ├── ModelProfiler.vue
│   │   │   │   ├── FlopsDisplay.vue
│   │   │   │   ├── ParamsDisplay.vue
│   │   │   │   ├── LatencyDisplay.vue
│   │   │   │   ├── MemoryDisplay.vue
│   │   │   │   └── BenchmarkTable.vue
│   │   │   │
│   │   │   └── 📁 charts/
│   │   │       ├── LineChart.vue
│   │   │       ├── AreaChart.vue
│   │   │       ├── BarChart.vue
│   │   │       ├── ScatterChart.vue
│   │   │       ├── HeatmapChart.vue
│   │   │       ├── PieChart.vue
│   │   │       ├── RadarChart.vue
│   │   │       ├── GaugeChart.vue
│   │   │       ├── TreeChart.vue
│   │   │       └── ChartZoomPlugin.vue
│   │   │
│   │   ├── 📁 composables/
│   │   │   ├── useWebSocket.js
│   │   │   ├── useTask.js
│   │   │   ├── useExperiment.js
│   │   │   ├── useNode.js
│   │   │   ├── useChart.js
│   │   │   ├── useCompare.js
│   │   │   ├── useNotification.js
│   │   │   └── useTheme.js
│   │   │
│   │   ├── 📁 store/
│   │   │   ├── index.js
│   │   │   ├── task.js
│   │   │   ├── node.js
│   │   │   ├── experiment.js
│   │   │   ├── component.js
│   │   │   ├── comparison.js
│   │   │   ├── visualization.js
│   │   │   ├── report.js
│   │   │   └── user.js
│   │   │
│   │   ├── 📁 api/
│   │   │   ├── index.js
│   │   │   ├── task.js
│   │   │   ├── node.js
│   │   │   ├── experiment.js
│   │   │   ├── component.js
│   │   │   ├── export.js
│   │   │   ├── visualization.js
│   │   │   ├── report.js
│   │   │   ├── comparison.js
│   │   │   └── websocket.js
│   │   │
│   │   ├── 📁 router/
│   │   │   └── index.js
│   │   │
│   │   ├── 📁 utils/
│   │   │   ├── request.js
│   │   │   ├── formatter.js
│   │   │   ├── validator.js
│   │   │   ├── chartHelper.js
│   │   │   ├── colorHelper.js
│   │   │   ├── dateHelper.js
│   │   │   ├── downloadHelper.js
│   │   │   └── configHelper.js
│   │   │
│   │   ├── 📁 constants/
│   │   │   ├── index.js
│   │   │   ├── taskTypes.js
│   │   │   ├── nodeStatus.js
│   │   │   ├── chartColors.js
│   │   │   └── exportFormats.js
│   │   │
│   │   └── 📁 styles/
│   │       ├── main.scss
│   │       ├── variables.scss
│   │       ├── mixins.scss
│   │       ├── components.scss
│   │       ├── charts.scss
│   │       └── themes/
│   │           ├── light.scss
│   │           └── dark.scss
│   │
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── README.md
│
├── 📁 database/                                        # 数据库层
│   ├── __init__.py
│   ├── connection.py
│   ├── cloud_connection.py
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── task.py
│   │   ├── node.py
│   │   ├── experiment.py
│   │   ├── metric_record.py
│   │   ├── checkpoint.py
│   │   ├── artifact.py
│   │   ├── environment_snapshot.py
│   │   ├── config_snapshot.py
│   │   ├── comparison_record.py
│   │   ├── export_record.py
│   │   ├── report.py
│   │   ├── user.py
│   │   └── system_log.py
│   │
│   ├── 📁 repositories/
│   │   ├── __init__.py
│   │   ├── task_repository.py
│   │   ├── node_repository.py
│   │   ├── experiment_repository.py
│   │   ├── metric_repository.py
│   │   ├── checkpoint_repository.py
│   │   ├── artifact_repository.py
│   │   ├── config_repository.py
│   │   ├── export_repository.py
│   │   ├── report_repository.py
│   │   ├── user_repository.py
│   │   └── csv_exporter.py
│   │
│   ├── 📁 migrations/
│   │   ├── __init__.py
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── 📁 versions/
│   │       └── .gitkeep
│   │
│   └── 📁 seeds/
│       ├── __init__.py
│       ├── initial_data.py
│       └── demo_data.py
│
├── 📁 tools/                                           # 工具脚本
│   ├── start_master.py
│   ├── start_api.py
│   ├── start_webui.py
│   ├── export_model.py
│   ├── validate_config.py
│   ├── generate_agent_package.py
│   ├── init_database.py
│   ├── sync_cloud_db.py
│   ├── generate_report.py
│   ├── benchmark_model.py
│   ├── compare_experiments.py
│   ├── export_csv.py
│   ├── cleanup_experiments.py
│   └── migrate_database.py
│
├── 📁 tests/                                           # 测试
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── 📁 unit/
│   │   ├── __init__.py
│   │   ├── 📁 components/
│   │   │   ├── test_models.py
│   │   │   ├── test_datasets.py
│   │   │   ├── test_losses.py
│   │   │   ├── test_metrics.py
│   │   │   ├── test_augmentations.py
│   │   │   └── test_compression.py
│   │   │
│   │   ├── 📁 master/
│   │   │   ├── test_node_manager.py
│   │   │   ├── test_task_scheduler.py
│   │   │   ├── test_packager.py
│   │   │   └── test_report_generator.py
│   │   │
│   │   ├── 📁 agent/
│   │   │   ├── test_executor.py
│   │   │   ├── test_reporter.py
│   │   │   ├── test_data_handler.py
│   │   │   └── test_runtime_core.py
│   │   │
│   │   ├── 📁 export/
│   │   │   ├── test_onnx_exporter.py
│   │   │   ├── test_tensorrt_exporter.py
│   │   │   └── test_validators.py
│   │   │
│   │   ├── 📁 configs/
│   │   │   ├── test_hydra_loader.py
│   │   │   └── test_config_validator.py
│   │   │
│   │   ├── 📁 visualization/
│   │   │   ├── test_feature_maps.py
│   │   │   └── test_metrics_plotter.py
│   │   │
│   │   ├── 📁 experiment/
│   │   │   ├── test_tracker.py
│   │   │   └── test_reproducer.py
│   │   │
│   │   └── 📁 api/
│   │       ├── test_routes.py
│   │       └── test_services.py
│   │
│   ├── 📁 integration/
│   │   ├── __init__.py
│   │   ├── test_master_agent.py
│   │   ├── test_task_flow.py
│   │   ├── test_training_pipeline.py
│   │   ├── test_export_pipeline.py
│   │   ├── test_experiment_tracking.py
│   │   ├── test_report_generation.py
│   │   └── test_api_websocket.py
│   │
│   ├── 📁 e2e/
│   │   ├── __init__.py
│   │   ├── test_classification_task.py
│   │   ├── test_detection_task.py
│   │   ├── test_segmentation_task.py
│   │   ├── test_distributed_training.py
│   │   ├── test_full_workflow.py
│   │   └── test_webui_interaction.py
│   │
│   └── 📁 fixtures/
│       ├── __init__.py
│       ├── sample_configs.py
│       ├── sample_datasets.py
│       ├── mock_nodes.py
│       └── mock_metrics.py
│
├── 📁 docs/                                            # 文档
│   ├── 📁 user_guide/
│   │   ├── getting_started.md
│   │   ├── installation.md
│   │   ├── quick_start.md
│   │   ├── task_creation.md
│   │   ├── monitoring.md
│   │   ├── experiment_management.md
│   │   ├── model_export.md
│   │   ├── report_generation.md
│   │   └── faq.md
│   │
│   ├── 📁 developer_guide/
│   │   ├── architecture.md
│   │   ├── contributing.md
│   │   ├── code_style.md
│   │   ├── adding_models.md
│   │   ├── adding_datasets.md
│   │   ├── adding_losses.md
│   │   ├── adding_metrics.md
│   │   ├── adding_augmentations.md
│   │   ├── adding_exporters.md
│   │   ├── plugin_development.md
│   │   └── testing.md
│   │
│   ├── 📁 api_reference/
│   │   ├── rest_api.md
│   │   ├── websocket_api.md
│   │   ├── components_api.md
│   │   ├── registry_api.md
│   │   └── config_schema.md
│   │
│   ├── 📁 deployment/
│   │   ├── local_deployment.md
│   │   ├── docker_deployment.md
│   │   ├── cloud_deployment.md
│   │   ├── agent_deployment.md
│   │   ├── database_setup.md
│   │   ├── security_configuration.md
│   │   └── scaling.md
│   │
│   ├── 📁 tutorials/
│   │   ├── classification_tutorial.md
│   │   ├── detection_tutorial.md
│   │   ├── segmentation_tutorial.md
│   │   ├── distributed_training.md
│   │   ├── model_compression.md
│   │   ├── custom_dataset.md
│   │   └── experiment_comparison.md
│   │
│   └── 📁 data_format/
│       ├── dataset_format.md
│       ├── annotation_format.md
│       ├── config_format.md
│       ├── checkpoint_format.md
│       └── export_format.md
│
├── 📁 scripts/                                         # 部署脚本
│   ├── install_master.sh
│   ├── install_agent.sh
│   ├── install_all.sh
│   ├── start_all.sh
│   ├── stop_all.sh
│   ├── backup_database.sh
│   ├── restore_database.sh
│   ├── setup_cloud_db.sh
│   │
│   ├── 📁 docker/
│   │   ├── docker-compose.yaml
│   │   ├── docker-compose.dev.yaml
│   │   ├── docker-compose.prod.yaml
│   │   ├── Dockerfile.master
│   │   ├── Dockerfile.agent
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.webui
│   │   └── .dockerignore
│   │
│   └── 📁 k8s/
│       ├── namespace.yaml
│       ├── master-deployment.yaml
│       ├── api-deployment.yaml
│       ├── webui-deployment.yaml
│       ├── database-statefulset.yaml
│       ├── redis-deployment.yaml
│       ├── configmap.yaml
│       ├── secrets.yaml
│       ├── ingress.yaml
│       └── hpa.yaml
│
├── 📁 experiments/                                     # 实验存储目录
│   ├── .gitkeep
│   └── README.md
│
├── 📁 pretrained/                                      # 预训练权重目录
│   ├── .gitkeep
│   └── README.md
│
├── .gitignore
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
├── setup.py
├── setup.cfg
├── requirements.txt
├── requirements-dev.txt
├── requirements-agent.txt
├── MANIFEST.in
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```
