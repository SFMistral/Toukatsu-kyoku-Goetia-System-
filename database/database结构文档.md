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