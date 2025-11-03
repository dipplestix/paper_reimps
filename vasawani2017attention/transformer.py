import torch.nn as nn
import torch
from attention import Attention
from glu import GLU


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-layer normalization, multi-head attention, and GLU feed-forward network.
    
    This implementation follows modern transformer design with pre-norm architecture,
    which applies layer normalization before each sub-block rather than after.
    
    Args:
        hidden_size (int): Size of input and output embeddings
        n_heads (int): Number of attention heads
        glu_dim_multiplier (int, optional): Multiplier for GLU intermediate dimension. Defaults to 2.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, 
                 hidden_size: int, 
                 n_heads: int, 
                 glu_dim_multiplier: int = 2, 
                 dropout_p: float = 0.0,
                 use_qk_norm: bool = False,
                 use_rotary: bool = True,
                 use_glu: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(n_heads, 
                              hidden_size, 
                              dropout_p=dropout_p, 
                              use_qk_norm=use_qk_norm, 
                              use_rotary=use_rotary)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        if use_glu:
            # Directly calculate intermediate dimension
            intermediate_dim = hidden_size * glu_dim_multiplier
            self.ffn = GLU(hidden_size, intermediate_dim, hidden_size, dropout_p=dropout_p)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size*4, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size*4, hidden_size, bias=True),
                nn.Dropout(dropout_p)
            )
    
    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Apply transformer block operations: attention and GLU with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs for attention. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Pre-norm approach: normalize, then apply module, then add residual
        norm_x = self.ln1(x)
        x = x + self.attn(norm_x, 
                          attention_mask=attention_mask, 
                          position_ids=position_ids)
        norm_x = self.ln2(x)
        x = x + self.ffn(norm_x)
        return x

class Transformer(nn.Module):
    """
    Full transformer model with token embeddings, optional positional embeddings, and multiple layers.
    
    This implementation uses pre-norm transformer blocks and a final layer normalization.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_size (int): Size of the hidden/embedding dimensions
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer blocks
        glu_dim_multiplier (int, optional): Multiplier for GLU intermediate dimension. Defaults to 2.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        max_seq_length (int, optional): Maximum sequence length for positional embeddings. Defaults to 2048.
        use_positional_embedding (bool, optional): Whether to use positional embeddings. Defaults to True.
    """
    def __init__(self, 
                 vocab_size: int, 
                 hidden_size: int, 
                 n_heads: int, 
                 n_layers: int, 
                 glu_dim_multiplier: int = 2, 
                 dropout_p: float = 0.0, 
                 max_seq_length: int = 2048, 
                 use_positional_embedding: bool = False,
                 use_qk_norm: bool = False,
                 use_rotary: bool = True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_length, hidden_size) if use_positional_embedding else None
        self.max_seq_length = max_seq_length
        
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_size, 
                             n_heads, 
                             glu_dim_multiplier, 
                             dropout_p, 
                             use_qk_norm, 
                             use_rotary) 
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_size) 
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Apply the transformer model to the input.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
                If None but positional embeddings are enabled, sequential positions will be used.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = x.size()
        x = self.token_embedding(x)
        
        # Handle positional embeddings
        if self.positional_embedding is not None:
            # Generate default position_ids if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                # Ensure positions don't exceed maximum length
                position_ids = torch.clamp(position_ids, 0, self.max_seq_length - 1)
            
            x = x + self.positional_embedding(position_ids)
        
        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            
        # Final layer norm
        x = self.ln(x)
        out = self.out(x)
        return out

