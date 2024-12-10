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
