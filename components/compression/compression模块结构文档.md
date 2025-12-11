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