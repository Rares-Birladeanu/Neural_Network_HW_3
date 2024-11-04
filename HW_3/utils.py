import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def setup_optimizer(config, model):
    if config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                         weight_decay=config['weight_decay'], nesterov=config['nesterov'])
    elif config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
                             momentum=config['momentum'])
    else:
        raise ValueError("Optimizer not supported.")


def setup_scheduler(config, optimizer):
    if config['scheduler'] == 'StepLR':
        return StepLR(optimizer, step_size=7, gamma=0.1)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, patience=3)
    elif config['scheduler'] == 'None':
        return None
