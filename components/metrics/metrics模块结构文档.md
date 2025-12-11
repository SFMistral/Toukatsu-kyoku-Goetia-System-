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