import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from rotary_embedding_torch import RotaryEmbedding


class Attention(nn.Module):
    """
    Multi-head attention module with support for grouped-query attention (GQA).
    
    This implements the attention mechanism described in "Attention Is All You Need" 
    (Vaswani et al., 2017) with additional support for:
    - Grouped Query Attention (GQA): Allows fewer key/value heads than query heads
    - Query/Key normalization: Optional RMSNorm applied to queries and keys
    - Dropout: Attention dropout during training
    - Rotary Position Embeddings (RoPE): Enhances position awareness
    
    Args:
        n_heads (int): Number of attention heads
        hidden_size (int): Size of the input and output embeddings
        n_kv_heads (int, optional): Number of key/value heads (for GQA). Defaults to n_heads.
        dropout_p (float, optional): Dropout probability. Defaults to 0.
        attn_bias (bool, optional): Whether to include bias in linear projections. Defaults to False.
        use_qk_norm (bool, optional): Whether to apply normalization to queries and keys. Defaults to False.
        causal (bool, optional): Whether to apply causal mask. Defaults to True.
        use_rotary (bool, optional): Whether to use rotary position embeddings. Defaults to False.
        max_seq_length (int, optional): Maximum sequence length for rotary embeddings. Defaults to 2048.
        rope_theta (float, optional): Base frequency for rotary embeddings. Defaults to 10000.0.
    """
    def __init__(self,
                 n_heads: int,
                 hidden_size: int,
                 n_kv_heads: int = None,
                 dropout_p: float = 0,
                 attn_bias: bool = False,
                 use_qk_norm: bool = False,
                 causal: bool = True,
                 use_rotary: bool = False,
                 rope_theta: float = 10000.0,
                 ):

        super().__init__()

        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")
        
        if n_kv_heads is None:
            self.n_kv_heads = n_heads
        else:
            self.n_kv_heads = n_kv_heads
        
        if n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
            
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.head_dim = self.hidden_size // self.n_heads
        self.n_key_value_groups = self.n_heads // self.n_kv_heads
        self.causal = causal
        self.use_rotary = use_rotary
        
        # Initialize projection matrices
        self.q = nn.Linear(hidden_size, n_heads*self.head_dim, bias=attn_bias)
        self.k = nn.Linear(hidden_size, self.n_kv_heads*self.head_dim, bias=attn_bias)
        self.v = nn.Linear(hidden_size, self.n_kv_heads*self.head_dim, bias=attn_bias)
        self.o = nn.Linear(n_heads*self.head_dim, hidden_size, bias=attn_bias)

        # Optional normalization for queries and keys
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
        # Initialize rotary embeddings if enabled
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                theta=rope_theta,
            )

    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Apply multi-head attention on the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, 1, seq_len, seq_len].
                                                   Defaults to None.
            position_ids (torch.Tensor, optional): Position ids of shape [batch_size, seq_len]. 
                                                 Defaults to None (sequential positions).
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        bsz, q_len, _ = x.size()

        # Create position ids if not provided
        if position_ids is None and self.use_rotary:
            position_ids = torch.arange(q_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        # Project inputs to queries, keys, and values
        q = self.q(x)  # [batch, seq, n_heads*head_dim]
        k = self.k(x)  # [batch, seq, n_kv_heads*head_dim]
        v = self.v(x)  # [batch, seq, n_kv_heads*head_dim]

        # Reshape for multi-head attention [batch, heads, seq, dim]
        q = q.view(bsz, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, q_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, q_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply rotary position embeddings if enabled
        if self.use_rotary:
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k, position_ids=position_ids)
            
        # Apply normalization (if needed)
        if not isinstance(self.q_norm, nn.Identity):
            # Reshape for normalization
            q = q.permute(0, 2, 1, 3)  # [batch, seq, heads, dim]
            k = k.permute(0, 2, 1, 3)  # [batch, seq, heads, dim]
            
            # Apply normalization
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            # Reshape back for attention
            q = q.permute(0, 2, 1, 3)  # [batch, heads, seq, dim]
            k = k.permute(0, 2, 1, 3)  # [batch, heads, seq, dim]

        # Repeat keys and values for grouped-query attention
        if self.n_key_value_groups > 1:
            k = repeat_kv(k, self.n_key_value_groups)
            v = repeat_kv(v, self.n_key_value_groups)

        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal if attention_mask is None else False
        )

        # Reshape and project output
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch, seq, heads, dim]
        attn_output = attn_output.view(bsz, q_len, self.n_heads * self.head_dim)
        attn_output = self.o(attn_output)

        return attn_output


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads in grouped-query attention.
    
    This function expands the key or value tensors when using grouped-query attention where
    the number of key/value heads is less than the number of query heads.
    
    Args:
        hidden_states (torch.Tensor): Input tensor of shape [batch, num_key_value_heads, seq_len, head_dim]
        n_rep (int): Number of times to repeat each key/value head
        
    Returns:
        torch.Tensor: Repeated tensor of shape [batch, num_attention_heads, seq_len, head_dim]
                     where num_attention_heads = num_key_value_heads * n_rep
    
    Note:
        This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)
        Adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
