
sweep_config = {
    "method": "bayes",  # 搜索策略: "grid", "random", or "bayes"
    "metric": {
        "name": "test_acc_quar",  # 目标优化指标
        "goal": "maximize"  # 优化方向: "maximize" or "minimize"
    },
    "parameters": {
        "hidden_dim": {
            "values": [32, 64, 128, 256, 512, 1024]
        },
        "n_heads": {
            "values": [2, 4, 8, 16]
        },
        "n_layers": {
            "values": [2, 4, 8, 12, 16, 24]
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.00001,
            "max": 0.01
        },
        "batch_size": {
            "values": [8, 32, 128, 256, 512]  # 离散值搜索
        },
        "dataset": {
            "values": ['./data/alldata_pseudo_multigene_with_features_0617.pt']
        },
        "loss_function": {
            "values": ["criterion1", "criterion2", "criterion3", "criterion4", "criterion5"]
        },
    }
}
