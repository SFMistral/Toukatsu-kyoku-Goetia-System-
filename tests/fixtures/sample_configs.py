"""
示例配置数据
"""

SAMPLE_MODEL_CONFIG = {
    "type": "ImageClassifier",
    "backbone": {
        "type": "ResNet",
        "depth": 50,
        "pretrained": True,
        "frozen_stages": -1,
    },
    "neck": None,
    "head": {
        "type": "LinearClsHead",
        "num_classes": 1000,
        "in_channels": 2048,
        "loss": {
            "type": "CrossEntropyLoss",
            "weight": 1.0,
        },
    },
}

SAMPLE_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "num_workers": 4,
    "distributed": {
        "enabled": False,
        "backend": "nccl",
    },
    "mixed_precision": {
        "enabled": True,
        "dtype": "fp16",
    },
    "gradient_clip": {
        "enabled": True,
        "max_norm": 1.0,
    },
}

SAMPLE_OPTIMIZER_CONFIG = {
    "type": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
}

SAMPLE_FULL_CONFIG = {
    "model": SAMPLE_MODEL_CONFIG,
    "training": SAMPLE_TRAINING_CONFIG,
    "optimizer": SAMPLE_OPTIMIZER_CONFIG,
    "dataset": {
        "type": "ImageNet",
        "data_root": "data/imagenet",
        "num_classes": 1000,
    },
}
