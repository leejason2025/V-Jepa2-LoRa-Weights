import torch
from typing import Optional


dependencies = ['torch']


def vjepa2_ac_droid(pretrained: bool = True, checkpoint_step: int = 1000, map_location: str = 'cpu'):
    """
    V-JEPA2-AC model fine-tuned on DROID dataset with LoRA

    Args:
        pretrained: If True, load the fine-tuned checkpoint
        checkpoint_step: Which checkpoint step to load 
        map_location: Device to load the model on (default: 'cpu')

    Returns:
        Dictionary containing:
            - 'checkpoint': Full checkpoint dict with model_state_dict, optimizer_state_dict, etc.
            - 'global_step': Training step number
            - 'epoch': Training epoch
            - 'lora_config': LoRA configuration used

    Example:
        >>> result = torch.hub.load('leejason2025/vjepa2-droid', 'vjepa2_ac_droid')
        >>> checkpoint = result['checkpoint']
        >>> model_state = checkpoint['model_state_dict']
        >>> print(f"Loaded checkpoint from step {result['global_step']}")
    """
    if not pretrained:
        raise ValueError("Only pretrained checkpoint is available")

    # Download checkpoint
    checkpoint_url = f'https://github.com/leejason2025/vjepa2-droid/releases/download/v1.0/checkpoint_step_{checkpoint_step}.pt'

    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url,
        map_location=map_location,
        weights_only=False
    )

    # Extract metadata
    result = {
        'checkpoint': checkpoint,
        'global_step': checkpoint['global_step'],
        'epoch': checkpoint['epoch'],
        'lora_config': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.05,
            'use_rslora': True,
            'target_modules': [
                'predictor_blocks.*.attn.qkv',
                'predictor_blocks.*.attn.proj',
                'predictor_blocks.*.mlp.fc1',
                'predictor_blocks.*.mlp.fc2'
            ]
        }
    }

    return result
