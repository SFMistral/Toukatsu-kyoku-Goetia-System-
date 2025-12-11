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