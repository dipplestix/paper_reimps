import torch.nn as nn


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) implementation based on the architecture used in modern transformers
    like LLaMA and Mistral.
    
    This version uses a fused projection followed by chunking for the gate and content paths:
    1. A single linear projection creates both gate and content representations
    2. The gate path applies an activation function (SiLU/Swish by default)
    3. The content path passes through unchanged
    4. The two paths are combined with element-wise multiplication
    5. A final projection transforms the result to the desired output dimension
    
    This implementation follows the optimized SwiGLU variant used in state-of-the-art models,
    offering improved efficiency through fused operations.
    
    Args:
        input_dim (int): Dimension of the input features
        intermediate_dim (int): Dimension of the intermediate representation (hidden dimension)
        output_dim (int): Dimension of the output features
        activation (nn.Module, optional): Activation function for the gate path. 
                                          Defaults to nn.SiLU (Swish activation).
        use_bias (bool, optional): Whether to include bias terms in the linear projections.
                                  Defaults to False.
        dropout_p (float, optional): Dropout probability applied after gating. Defaults to 0.0.
    """
    def __init__(self, 
                 input_dim, 
                 intermediate_dim, 
                 output_dim, 
                 activation=nn.SiLU,
                 use_bias=False,
                 dropout_p=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_p)

        # Fused projection for both gate and content paths (more efficient)
        self.fused_gate_up_proj = nn.Linear(input_dim, 2 * intermediate_dim, bias=use_bias)

        # Project gated representation to output dimension
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=use_bias)
        
    def forward(self, x):
        # Single fused projection for both gate and content
        fused = self.fused_gate_up_proj(x)
        
        # Split the fused projection into gate and content parts
        gate, up = fused.chunk(2, dim=-1)

        # Apply activation to the gate path
        gate = self.activation(gate)
        
        # Element-wise multiplication of gate and content
        gated_output = up * gate
        
        # Apply dropout after gating (standard practice in modern transformers)
        gated_output = self.dropout(gated_output)
        
        # Project to output dimension
        out = self.down_proj(gated_output)

        return out
