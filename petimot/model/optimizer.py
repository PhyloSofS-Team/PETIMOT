import torch


def get_optimizer(parameters, optimizer_name, learning_rate, **kwargs):

    optimizer_name = optimizer_name.lower()
    optimizer_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
    }

    if optimizer_name not in optimizer_map:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}. "
            f"Valid options: {list(optimizer_map.keys())}"
        )

    defaults = {
        "adam": {"weight_decay": 0.0, "amsgrad": False},
        "adamw": {"weight_decay": 0.01},
        "sgd": {"momentum": 0.9, "nesterov": True},
        "rmsprop": {"alpha": 0.99, "momentum": 0.0},
    }.get(optimizer_name, {})

    final_params = {**defaults, **kwargs}

    return optimizer_map[optimizer_name](parameters, lr=learning_rate, **final_params)
