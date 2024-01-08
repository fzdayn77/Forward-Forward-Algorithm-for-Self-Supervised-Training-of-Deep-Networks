from configs import CONFIGS
from models.forward_forward import FF_Net
from models.resnet import get_resnet

def get_encoder(model_name, num_hidden_layers, num_epochs, lr):
    '''
    Chooses an encoder from the given models.

    Parameters:
        model_name (String): The encoder name. It can be one of the following
                             possible models [ResNet18, ResNet34, ResNet50, Forward-Forward]
        num_layers (int): number of hidden layers
        lr (int): the learning rate
        temperature (float): the used temperature value

    Returns:
        encoder (nn.Module): the chosen model
    '''

    # List of all possible encoders
    encoders = ["resnet18", "resnet34", "resnet50", "forward-forward"]

    if model_name not in encoders:
        raise KeyError(f"{model_name} is not a valid encoder name")

    # ResNet 18/34/50
    if model_name in ["resnet18", "resnet34", "resnet50"]:
        encoder = get_resnet(name=model_name)
    else:
      encoder = FF_Net(in_features=CONFIGS['in_features'], out_features=CONFIGS['out_features'], 
                       num_hidden_layers=num_hidden_layers, num_epochs=num_epochs, lr=lr, device=DEVICE)

    print(f'Encoder ===> {model_name}')

    return encoder