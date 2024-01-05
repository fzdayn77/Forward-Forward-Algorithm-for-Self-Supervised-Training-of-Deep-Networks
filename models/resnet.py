import torchvision

def get_resnet(name: str, pretrained=False):
    """
    Imports a ResNet model considering the given model name.

    Parameters:
        name (string): the ResNet model name. It must be one of the allowed
                       ResNet valid names.
        pretrained (boolean): Default is False. Allows us to choose an already 
                              pretrained model or not.

    Returns:
        resnets[name]: the chosen ResNet model
    """
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid model name !!")
    
    return resnets[name]