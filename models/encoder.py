import torch.nn as nn
from models.resnet import get_resnet

def get_encoder(model_name: str, num_layers: int, lr: float, temperature: float):
    '''
    Chooses an encoder from the given models.

    Parameters:
        model_name (String): The encoder name. It can be one of the following
                             possible models [ResNet18, ResNet34, ResNet50, Forward-Forward]
        num_layers (int): number of hidden layers
        lr (int): the learning rate
        temperature (float): the used temperature value

    Returns:
        encoder (nn.Module): the chosen model.
    '''

    # List of all possible encoders
    encoders = ["resnet18", "resnet34", "resnet50", "forward-forward"]

    if model_name not in encoders:
        raise KeyError(f"{model_name} is not a valid encoder name")

    # ResNet 18/34/50
    if model_name in ["resnet18", "resnet34", "resnet50"]:
        encoder = get_resnet(name=model_name)
    else:
      pass
        #encoder = FF_Net(num_layers, lr, temperature, device=device)

    print(f'Encoder ===> {model_name}')

    return nn.Sequential(*list(encoder.children())[:-1])