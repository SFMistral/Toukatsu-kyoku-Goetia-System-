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