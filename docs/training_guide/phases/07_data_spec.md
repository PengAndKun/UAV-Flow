## 训练数据规范

### 目录结构

```
UAV-Flow/
└── training_data/
    ├── house_references/                ← Phase 1: 参考图
    │   ├── house_1/
    │   │   ├── ref_front.jpg
    │   │   ├── ref_left45.jpg
    │   │   ├── ref_right45.jpg
    │   │   ├── ref_close.jpg
    │   │   └── ref_high.jpg
    │   ├── house_2/
    │   └── house_3/
    │
    ├── door_detection/                  ← Phase 2: 门检测标注
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   ├── labels/
    │   │   ├── train/
    │   │   └── val/
    │   └── door_dataset.yaml
    │
    ├── trajectories/                    ← Phase 3-4: 轨迹数据
    │   ├── outdoor_nav/
    │   │   ├── ep_001.jsonl
    │   │   └── ...
    │   ├── entry/
    │   │   ├── ep_001.jsonl
    │   │   └── ...
    │   └── indoor_search/
    │       ├── ep_001.jsonl
    │       └── ...
    │
    ├── distillation/                    ← Phase 5: 蒸馏数据
    │   ├── distillation_dataset_v1.jsonl
    │   └── distillation_dataset_v2.jsonl
    │
    └── artifacts/                       ← 训练产物
        ├── house_reference_features.npy
        ├── door_detector_v1/
        │   └── best.pt
        ├── indoor_search_mlp_v1.json
        └── unified_policy_v1.json
```

### 数据版本管理

每次重大数据变更时创建新版本：

```
distillation_dataset_v1.jsonl  ← 初始版本
distillation_dataset_v2.jsonl  ← 加入第一轮迭代数据
distillation_dataset_v3.jsonl  ← 加入第二轮迭代数据
```

旧版本不删除，用于对比实验。

