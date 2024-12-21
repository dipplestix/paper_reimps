import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,
                 n_heads: int,
                 n_kv_heads: int,
                 hidden_size: int,
                 dropout_p: float = 0,
                 attn_bias: bool = False
                 ):

        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.head_dim = self.hidden_size // self.n_heads
        self.n_key_value_groups = self.n_heads // self.n_kv_heads

        self.q = nn.Linear(hidden_size, n_heads*self.head_dim, bias=attn_bias)
        self.k = nn.Linear(hidden_size, n_kv_heads*self.head_dim, bias=attn_bias)
        self.v = nn.Linear(hidden_size, n_kv_heads*self.head_dim, bias=attn_bias)
        self.o = nn.Linear(hidden_size, hidden_size, bias=attn_bias)

    def forward(self, x):
        bsz, q_len, _ = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.n_key_value_groups)
        v = repeat_kv(v, self.n_key_value_groups)

        attn_output = F.scaled_dot_product_attention(q, k, v,
                                                     dropout_p=self.dropout_p if self.training else 0.0,
                                                     is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o(attn_output)

        return attn_output


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
