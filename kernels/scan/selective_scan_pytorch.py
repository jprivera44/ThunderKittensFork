import torch
import torch.nn.functional as F
from typing import Optional

def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference implementation matching Mamba's selective_scan_cuda.
    
    This implements the recurrence:
    x[t] = A_bar[t] * x[t-1] + B_bar[t] * u[t]
    y[t] = C[t] * x[t] + D * u[t]
    
    where:
    A_bar[t] = exp(softplus(delta[t] + delta_bias) * A)
    B_bar[t] = softplus(delta[t] + delta_bias) * B[t]
    
    Args:
        u: (batch, dim, seq_len) - input sequence
        delta: (batch, dim, seq_len) - time steps
        A: (dim, dstate) - state matrix (should be negative for stability)
        B: (batch, dstate, seq_len) - input-to-state matrix
        C: (batch, dstate, seq_len) - state-to-output matrix
        D: (dim,) - feedthrough parameter (optional)
        z: (batch, dim, seq_len) - multiplicative gating (optional)
        delta_bias: (dim,) - bias for delta (optional)
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return the last hidden state
        
    Returns:
        y: (batch, dim, seq_len) - output
        last_state: (batch, dim, dstate) - if return_last_state=True
    """
    # Get dimensions
    batch, dim, seq_len = u.shape
    dstate = A.shape[1]
    
    # Apply delta bias if provided
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    
    # Apply softplus to delta if requested
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Initialize state
    x = torch.zeros(batch, dim, dstate, dtype=u.dtype, device=u.device)
    
    # Prepare output tensor
    ys = []
    
    # Main recurrence loop
    for i in range(seq_len):
        # Get current inputs
        u_t = u[:, :, i]  # (batch, dim)
        delta_t = delta[:, :, i]  # (batch, dim)
        B_t = B[:, :, i]  # (batch, dstate)
        C_t = C[:, :, i]  # (batch, dstate)
        
        # Discretize A and B
        # A_bar = exp(delta * A)
        deltaA = torch.exp(delta_t[:, :, None] * A[None, :, :])  # (batch, dim, dstate)
        
        # B_bar = delta * B
        deltaB_u = delta_t[:, :, None] * B_t[:, None, :] * u_t[:, :, None]  # (batch, dim, dstate)
        
        # State update: x[t] = A_bar * x[t-1] + B_bar * u[t]
        x = deltaA * x + deltaB_u
        
        # Output: y[t] = C[t] * x[t]
        y = torch.sum(x * C_t[:, None, :], dim=-1)  # (batch, dim)
        
        # Add feedthrough if provided
        if D is not None:
            y = y + D * u_t
        
        # Apply multiplicative gating if provided
        if z is not None:
            y = y * F.silu(z[:, :, i])
        
        ys.append(y)
    
    # Stack outputs
    y = torch.stack(ys, dim=2)  # (batch, dim, seq_len)
    
    if return_last_state:
        return y, x
    else:
        return y


def selective_scan_easy(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified version for testing basic functionality.
    No D, z, or delta_bias parameters.
    """
    return selective_scan_ref(
        u, delta, A, B, C,
        D=None, z=None, delta_bias=None,
        delta_softplus=True, return_last_state=False
    )