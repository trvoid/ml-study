import torch
import torch.nn as nn
import timm

def get_model(model_name, pretrained=True, num_classes=10):
    """
    Create a model using timm library.
    
    Args:
        model_name (str): Name of the model architecture
        pretrained (bool): Whether to load pretrained weights
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: The requested model
    """
    
    # Map user-friendly names to timm model names
    model_mapping = {
        'efficientnet': 'efficientnet_b0',
        'mobilenet': 'mobilenetv3_small_100',
        'wideresnet': 'wide_resnet28_10',
        'vit': 'vit_base_patch16_224',
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_mapping.keys())}")
        
    timm_name = model_mapping[model_name]
    
    # Create model using timm
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    
    return model
