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