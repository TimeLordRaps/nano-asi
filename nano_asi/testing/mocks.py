import torch
import torch.nn as nn
from typing import Optional

class MockModel(nn.Module):
    """
    Comprehensive mock model for testing LoRA and other model-related functionalities
    """
    def __init__(
        self, 
        input_dim: int = 512, 
        output_dim: int = 512, 
        num_layers: int = 4,
        max_seq_length: int = 2048
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, input_dim)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_layers)
        ])
        
        # Add attributes that might be checked during testing
        self.max_seq_length = max_seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Add a config attribute with necessary properties
        self.config = type('MockConfig', (), {
            'max_position_embeddings': max_seq_length,
            'update': lambda x: None,
            'unsloth_version': '2024.12.4'
        })()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass for mock model
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_input_embeddings(self):
        """
        Mock method to simulate model's embedding layer retrieval
        """
        return self.embed_tokens
    
    def __getattr__(self, name):
        """
        Ensure max_seq_length and config are always accessible
        """
        if name == 'max_seq_length':
            return 2048
        elif name == 'config':
            return self.config
        elif name == 'max_position_embeddings':
            return self.config.max_position_embeddings
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
