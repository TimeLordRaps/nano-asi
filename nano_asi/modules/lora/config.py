from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter generation."""
    input_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 512
    num_layers: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
    ])
    num_diffusion_steps: int = 500
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    scheduler_type: str = "cosine"
    use_8bit_quantization: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    device: Optional[str] = "auto"
    
    def __post_init__(self):
        """
        Post-initialization method to handle device configuration.
        
        Automatically selects the appropriate device if 'auto' is specified.
        """
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
