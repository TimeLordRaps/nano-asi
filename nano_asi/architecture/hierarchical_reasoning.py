import torch
import torch.nn as nn
from typing import Dict, List, Any

class HierarchicalReasoningLayer(nn.Module):
    """
    Hierarchical Reasoning Layer with Multi-Frame Processing
    
    Processes different frames of reasoning with specialized attention
    """
    def __init__(
        self, 
        input_dim: int = 512,
        num_heads: int = 16,
        num_frames: int = 4
    ):
        super().__init__()
        
        # Frame-specific attention heads
        self.frame_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads // num_frames) 
            for _ in range(num_frames)
        ])
        
        # Cross-frame attention heads
        self.cross_frame_attention = nn.ModuleList([
            nn.MultiheadAttention(input_dim, 1) 
            for _ in range(10)  # 4 single-frame, 3 pairwise, 2 three-group, 1 full combination
        ])
        
        # Frame output projections
        self.frame_output_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_frames)
        ])
        
    def forward(
        self, 
        frame_tokens: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning layer
        
        Args:
            frame_tokens: List of tokens for each reasoning frame
        
        Returns:
            Dictionary of processed frame tokens and combined representations
        """
        # Process single-frame attention
        single_frame_outputs = []
        for i, (tokens, attention_head, projection) in enumerate(
            zip(frame_tokens, self.frame_attention_heads, self.frame_output_projections)
        ):
            attn_output, _ = attention_head(tokens, tokens, tokens)
            projected_output = projection(attn_output)
            single_frame_outputs.append(projected_output)
        
        # Cross-frame attention processing
        cross_frame_outputs = []
        
        # Pairwise frame attention
        pairwise_indices = [(0, 1), (1, 2), (2, 3)]
        for i, (frame1_idx, frame2_idx) in enumerate(pairwise_indices):
            cross_attn, _ = self.cross_frame_attention[i](
                single_frame_outputs[frame1_idx], 
                single_frame_outputs[frame2_idx], 
                single_frame_outputs[frame2_idx]
            )
            cross_frame_outputs.append(cross_attn)
        
        # Three-group frame attention
        three_group_indices = [(0, 1, 2), (1, 2, 3)]
        for i, group in enumerate(three_group_indices, start=3):
            group_tokens = torch.stack([single_frame_outputs[idx] for idx in group])
            cross_attn, _ = self.cross_frame_attention[i](
                group_tokens.mean(dim=0), 
                group_tokens, 
                group_tokens
            )
            cross_frame_outputs.append(cross_attn)
        
        # Full combination attention
        full_tokens = torch.stack(single_frame_outputs)
        full_cross_attn, _ = self.cross_frame_attention[-1](
            full_tokens.mean(dim=0), 
            full_tokens, 
            full_tokens
        )
        cross_frame_outputs.append(full_cross_attn)
        
        return {
            'single_frame_outputs': single_frame_outputs,
            'cross_frame_outputs': cross_frame_outputs,
            'combined_representation': torch.mean(torch.stack(cross_frame_outputs), dim=0)
        }
import torch
import torch.nn as nn
from typing import Dict, List, Any

class HierarchicalReasoningLayer(nn.Module):
    """
    Hierarchical Reasoning Layer with Multi-Frame Processing
    
    Processes different frames of reasoning with specialized attention
    """
    def __init__(
        self, 
        input_dim: int = 512,
        num_heads: int = 16,
        num_frames: int = 4
    ):
        super().__init__()
        
        # Frame-specific attention heads
        self.frame_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads // num_frames) 
            for _ in range(num_frames)
        ])
        
        # Cross-frame attention heads
        self.cross_frame_attention = nn.ModuleList([
            nn.MultiheadAttention(input_dim, 1) 
            for _ in range(10)  # 4 single-frame, 3 pairwise, 2 three-group, 1 full combination
        ])
        
        # Frame output projections
        self.frame_output_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_frames)
        ])
        
    def forward(
        self, 
        frame_tokens: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning layer
        
        Args:
            frame_tokens: List of tokens for each reasoning frame
        
        Returns:
            Dictionary of processed frame tokens and combined representations
        """
        # Process single-frame attention
        single_frame_outputs = []
        for i, (tokens, attention_head, projection) in enumerate(
            zip(frame_tokens, self.frame_attention_heads, self.frame_output_projections)
        ):
            attn_output, _ = attention_head(tokens, tokens, tokens)
            projected_output = projection(attn_output)
            single_frame_outputs.append(projected_output)
        
        # Cross-frame attention processing
        cross_frame_outputs = []
        
        # Pairwise frame attention
        pairwise_indices = [(0, 1), (1, 2), (2, 3)]
        for i, (frame1_idx, frame2_idx) in enumerate(pairwise_indices):
            cross_attn, _ = self.cross_frame_attention[i](
                single_frame_outputs[frame1_idx], 
                single_frame_outputs[frame2_idx], 
                single_frame_outputs[frame2_idx]
            )
            cross_frame_outputs.append(cross_attn)
        
        # Three-group frame attention
        three_group_indices = [(0, 1, 2), (1, 2, 3)]
        for i, group in enumerate(three_group_indices, start=3):
            group_tokens = torch.stack([single_frame_outputs[idx] for idx in group])
            cross_attn, _ = self.cross_frame_attention[i](
                group_tokens.mean(dim=0), 
                group_tokens, 
                group_tokens
            )
            cross_frame_outputs.append(cross_attn)
        
        # Full combination attention
        full_tokens = torch.stack(single_frame_outputs)
        full_cross_attn, _ = self.cross_frame_attention[-1](
            full_tokens.mean(dim=0), 
            full_tokens, 
            full_tokens
        )
        cross_frame_outputs.append(full_cross_attn)
        
        return {
            'single_frame_outputs': single_frame_outputs,
            'cross_frame_outputs': cross_frame_outputs,
            'combined_representation': torch.mean(torch.stack(cross_frame_outputs), dim=0)
        }
import torch
import torch.nn as nn
from typing import Dict, List, Any

class HierarchicalReasoningLayer(nn.Module):
    """
    Hierarchical Reasoning Layer with Multi-Frame Processing
    
    Processes different frames of reasoning with specialized attention
    """
    def __init__(
        self, 
        input_dim: int = 512,
        num_heads: int = 16,
        num_frames: int = 4
    ):
        super().__init__()
        
        # Frame-specific attention heads
        self.frame_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads // num_frames) 
            for _ in range(num_frames)
        ])
        
        # Cross-frame attention heads
        self.cross_frame_attention = nn.ModuleList([
            nn.MultiheadAttention(input_dim, 1) 
            for _ in range(10)  # 4 single-frame, 3 pairwise, 2 three-group, 1 full combination
        ])
        
        # Frame output projections
        self.frame_output_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_frames)
        ])
        
    def forward(
        self, 
        frame_tokens: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning layer
        
        Args:
            frame_tokens: List of tokens for each reasoning frame
        
        Returns:
            Dictionary of processed frame tokens and combined representations
        """
        # Process single-frame attention
        single_frame_outputs = []
        for i, (tokens, attention_head, projection) in enumerate(
            zip(frame_tokens, self.frame_attention_heads, self.frame_output_projections)
        ):
            attn_output, _ = attention_head(tokens, tokens, tokens)
            projected_output = projection(attn_output)
            single_frame_outputs.append(projected_output)
        
        # Cross-frame attention processing
        cross_frame_outputs = []
        
        # Pairwise frame attention
        pairwise_indices = [(0, 1), (1, 2), (2, 3)]
        for i, (frame1_idx, frame2_idx) in enumerate(pairwise_indices):
            cross_attn, _ = self.cross_frame_attention[i](
                single_frame_outputs[frame1_idx], 
                single_frame_outputs[frame2_idx], 
                single_frame_outputs[frame2_idx]
            )
            cross_frame_outputs.append(cross_attn)
        
        # Three-group frame attention
        three_group_indices = [(0, 1, 2), (1, 2, 3)]
        for i, group in enumerate(three_group_indices, start=3):
            group_tokens = torch.stack([single_frame_outputs[idx] for idx in group])
            cross_attn, _ = self.cross_frame_attention[i](
                group_tokens.mean(dim=0), 
                group_tokens, 
                group_tokens
            )
            cross_frame_outputs.append(cross_attn)
        
        # Full combination attention
        full_tokens = torch.stack(single_frame_outputs)
        full_cross_attn, _ = self.cross_frame_attention[-1](
            full_tokens.mean(dim=0), 
            full_tokens, 
            full_tokens
        )
        cross_frame_outputs.append(full_cross_attn)
        
        return {
            'single_frame_outputs': single_frame_outputs,
            'cross_frame_outputs': cross_frame_outputs,
            'combined_representation': torch.mean(torch.stack(cross_frame_outputs), dim=0)
        }
import torch
import torch.nn as nn
from typing import Dict, List, Any

class HierarchicalReasoningLayer(nn.Module):
    """
    Hierarchical Reasoning Layer with Multi-Frame Processing
    
    Processes different frames of reasoning with specialized attention
    """
    def __init__(
        self, 
        input_dim: int = 512,
        num_heads: int = 16,
        num_frames: int = 4
    ):
        super().__init__()
        
        # Frame-specific attention heads
        self.frame_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads // num_frames) 
            for _ in range(num_frames)
        ])
        
        # Cross-frame attention heads
        self.cross_frame_attention = nn.ModuleList([
            nn.MultiheadAttention(input_dim, 1) 
            for _ in range(10)  # 4 single-frame, 3 pairwise, 2 three-group, 1 full combination
        ])
        
        # Frame output projections
        self.frame_output_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_frames)
        ])
        
    def forward(
        self, 
        frame_tokens: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning layer
        
        Args:
            frame_tokens: List of tokens for each reasoning frame
        
        Returns:
            Dictionary of processed frame tokens and combined representations
        """
        # Process single-frame attention
        single_frame_outputs = []
        for i, (tokens, attention_head, projection) in enumerate(
            zip(frame_tokens, self.frame_attention_heads, self.frame_output_projections)
        ):
            attn_output, _ = attention_head(tokens, tokens, tokens)
            projected_output = projection(attn_output)
            single_frame_outputs.append(projected_output)
        
        # Cross-frame attention processing
        cross_frame_outputs = []
        
        # Pairwise frame attention
        pairwise_indices = [(0, 1), (1, 2), (2, 3)]
        for i, (frame1_idx, frame2_idx) in enumerate(pairwise_indices):
            cross_attn, _ = self.cross_frame_attention[i](
                single_frame_outputs[frame1_idx], 
                single_frame_outputs[frame2_idx], 
                single_frame_outputs[frame2_idx]
            )
            cross_frame_outputs.append(cross_attn)
        
        # Three-group frame attention
        three_group_indices = [(0, 1, 2), (1, 2, 3)]
        for i, group in enumerate(three_group_indices, start=3):
            group_tokens = torch.stack([single_frame_outputs[idx] for idx in group])
            cross_attn, _ = self.cross_frame_attention[i](
                group_tokens.mean(dim=0), 
                group_tokens, 
                group_tokens
            )
            cross_frame_outputs.append(cross_attn)
        
        # Full combination attention
        full_tokens = torch.stack(single_frame_outputs)
        full_cross_attn, _ = self.cross_frame_attention[-1](
            full_tokens.mean(dim=0), 
            full_tokens, 
            full_tokens
        )
        cross_frame_outputs.append(full_cross_attn)
        
        return {
            'single_frame_outputs': single_frame_outputs,
            'cross_frame_outputs': cross_frame_outputs,
            'combined_representation': torch.mean(torch.stack(cross_frame_outputs), dim=0)
        }
import torch
import torch.nn as nn
from typing import Dict, List, Any

class HierarchicalReasoningLayer(nn.Module):
    """
    Hierarchical Reasoning Layer with Multi-Frame Processing
    
    Processes different frames of reasoning with specialized attention
    """
    def __init__(
        self, 
        input_dim: int = 512,
        num_heads: int = 16,
        num_frames: int = 4
    ):
        super().__init__()
        
        # Frame-specific attention heads
        self.frame_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads // num_frames) 
            for _ in range(num_frames)
        ])
        
        # Cross-frame attention heads
        self.cross_frame_attention = nn.ModuleList([
            nn.MultiheadAttention(input_dim, 1) 
            for _ in range(10)  # 4 single-frame, 3 pairwise, 2 three-group, 1 full combination
        ])
        
        # Frame output projections
        self.frame_output_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_frames)
        ])
        
    def forward(
        self, 
        frame_tokens: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning layer
        
        Args:
            frame_tokens: List of tokens for each reasoning frame
        
        Returns:
            Dictionary of processed frame tokens and combined representations
        """
        # Process single-frame attention
        single_frame_outputs = []
        for i, (tokens, attention_head, projection) in enumerate(
            zip(frame_tokens, self.frame_attention_heads, self.frame_output_projections)
        ):
            attn_output, _ = attention_head(tokens, tokens, tokens)
            projected_output = projection(attn_output)
            single_frame_outputs.append(projected_output)
        
        # Cross-frame attention processing
        cross_frame_outputs = []
        
        # Pairwise frame attention
        pairwise_indices = [(0, 1), (1, 2), (2, 3)]
        for i, (frame1_idx, frame2_idx) in enumerate(pairwise_indices):
            cross_attn, _ = self.cross_frame_attention[i](
                single_frame_outputs[frame1_idx], 
                single_frame_outputs[frame2_idx], 
                single_frame_outputs[frame2_idx]
            )
            cross_frame_outputs.append(cross_attn)
        
        # Three-group frame attention
        three_group_indices = [(0, 1,2), (1, 2, 3)]
        for i, group in enumerate(three_group_indices, start=3):
            group_tokens = torch.stack([single_frame_outputs[idx] for idx in group])
            cross_attn, _ = self.cross_frame_attention[i](
                group_tokens.mean(dim=0), 
                group_tokens, 
                group_tokens
            )
            cross_frame_outputs.append(cross_attn)
        
        # Full combination attention
        full_tokens = torch.stack(single_frame_outputs)
        full_cross_attn, _ = self.cross_frame_attention[-1](
            full_tokens.mean(dim=0), 
            full_tokens, 
            full_tokens
        )
        cross_frame_outputs.append(full_cross_attn)
        
        return {
            'single_frame_outputs': single_frame_outputs,
            'cross_frame_outputs': cross_frame_outputs,
            'combined_representation': torch.mean(torch.stack(cross_frame_outputs), dim=0)
        }
