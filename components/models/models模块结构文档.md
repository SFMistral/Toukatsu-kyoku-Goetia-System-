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